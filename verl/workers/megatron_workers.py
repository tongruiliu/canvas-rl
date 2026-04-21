# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Megatron worker entry points for canvas-rl.

This ports the training-side Megatron runtime boundary from verl far enough to
initialize actor/reference MCore modules, compute log probabilities, and run PPO
actor updates. The rollout engine is intentionally still a separate patch layer:
canvas-rl's current rollout path is FSDP->vLLM specific and needs a Megatron
weight-export/sync sharding manager before generation can run.
"""

import datetime
import os
import random
from dataclasses import asdict, is_dataclass
from typing import Literal, Optional

import numpy as np
import torch
import torch.distributed as dist
from codetiming import Timer
from megatron.core import parallel_state as mpu
from omegaconf import OmegaConf
from transformers import AutoConfig

from ..protocol import DataProto
from ..single_controller.base import Worker
from ..single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from ..utils.dataset import process_image, process_video
from ..utils.device import get_device_id, get_nccl_backend, get_torch_device
from ..utils.flops_counter import FlopsCounter
from ..utils.fs import copy_to_local
from ..utils.megatron.optimizer import (
    get_megatron_last_lr,
    get_megatron_optimizer,
    get_megatron_optimizer_param_scheduler,
    init_megatron_optim_config,
)
from ..utils.megatron_utils import (
    McoreModuleWrapperConfig,
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    make_megatron_module,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
    register_megatron_training_hooks,
)
from ..utils.model import get_generation_config, load_mcore_dist_weights, update_model_config
from ..utils.model_utils import print_model_size
from ..utils.tokenizer import get_processor, get_tokenizer
from ..utils.torch_dtypes import PrecisionType
from .actor.megatron_actor import MegatronPPOActor


_ROLLOUT_PENDING_MESSAGE = """
Megatron actor/ref training runtime is wired, but rollout generation is not yet
ported for canvas-rl.

Next required patch layer:
1. add Megatron->vLLM/SGLang weight export and sync;
2. preserve canvas-rl's tool/multi-turn rollout request path;
3. route rollout DP/TP/PP dispatch through the inference mesh.
"""

_CRITIC_PENDING_MESSAGE = """
Megatron critic runtime is not ported yet. Actor/ref logprob and update are
available; value-model initialization and update need the next critic patch.
"""


def _set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if get_torch_device().device_count() > 0:
        from megatron.core import tensor_parallel

        tensor_parallel.model_parallel_cuda_manual_seed(seed)


def _get_context_parallel_rank() -> int:
    get_rank = getattr(mpu, "get_context_parallel_rank", None)
    return get_rank() if get_rank is not None else 0


def _as_plain_dict(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if is_dataclass(value):
        return asdict(value)
    return dict(value)


def _mean_metric(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu()
        return value.item() if value.numel() == 1 else value.mean().item()
    if isinstance(value, list):
        flat = []
        for item in value:
            if isinstance(item, torch.Tensor):
                item = item.detach().float().cpu()
                flat.append(item.item() if item.numel() == 1 else item.mean().item())
            elif np.isscalar(item):
                flat.append(item)
        return float(np.mean(flat)) if flat else value
    if np.isscalar(value):
        return float(value)
    return value


class AsyncActorRolloutRefWorker(Worker):
    """Actor/rollout/ref worker for Megatron actor/ref training operations."""

    def __init__(
        self,
        config,
        role: Literal["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"],
    ):
        super().__init__()
        self.config = config
        self.role = role
        self._cache = {}

        self._has_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._has_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._has_ref = self.role in ["ref", "actor_rollout_ref"]
        if self.config.actor.disable_kl:
            self._has_ref = False

        self._lora_rank = self.config.actor.model.lora.rank
        self._is_lora = self._lora_rank > 0

        if self._has_actor or self._has_ref:
            self._init_distributed()
            self._register_actor_mesh()
        if self._has_rollout:
            # Until rollout sync is ported, keep rollout routed to all workers so
            # calls fail with the explicit rollout message instead of dispatcher errors.
            self._register_dispatch_collect_info(mesh_name="rollout", dp_rank=self.rank, is_collect=True)

        self._is_offload_param = self.config.actor.megatron.param_offload
        self._is_offload_grad = self.config.actor.megatron.grad_offload
        self._is_offload_optimizer = self.config.actor.megatron.optimizer_offload
        self._ref_is_offload_param = self.config.ref.megatron.param_offload

        _set_random_seed(self.config.actor.megatron.seed)

    def _init_distributed(self) -> None:
        if not dist.is_initialized():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            dist.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=int(getattr(self.config, "nccl_timeout", 600))),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
            get_torch_device().set_device(local_rank)

        if not mpu.model_parallel_is_initialized():
            megatron_config = self.config.actor.megatron if self._has_actor else self.config.ref.megatron
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=megatron_config.tensor_model_parallel_size,
                pipeline_model_parallel_size=megatron_config.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=megatron_config.virtual_pipeline_model_parallel_size,
                use_sharp=False,
                context_parallel_size=megatron_config.context_parallel_size,
                expert_model_parallel_size=megatron_config.expert_model_parallel_size,
                expert_tensor_parallel_size=megatron_config.expert_tensor_parallel_size,
                nccl_communicator_config_path=None,
            )

    def _register_actor_mesh(self) -> None:
        is_collect = (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
            and _get_context_parallel_rank() == 0
        )
        self._register_dispatch_collect_info(mesh_name="actor", dp_rank=mpu.get_data_parallel_rank(), is_collect=is_collect)

    def _actor_runtime_config(self):
        actor_cfg = self.config.actor
        dp_size = mpu.get_data_parallel_world_size()
        ppo_mini_batch_size = max(1, actor_cfg.global_batch_size * self.config.rollout.n // dp_size)
        ppo_micro_batch_size = actor_cfg.micro_batch_size_per_device_for_update
        return OmegaConf.create(
            {
                "ppo_mini_batch_size": ppo_mini_batch_size,
                "ppo_micro_batch_size_per_gpu": ppo_micro_batch_size,
                "ppo_epochs": actor_cfg.ppo_epochs,
                "shuffle": False,
                "data_loader_seed": None,
                "clip_ratio": actor_cfg.clip_ratio_high,
                "clip_ratio_low": actor_cfg.clip_ratio_low,
                "clip_ratio_high": actor_cfg.clip_ratio_high,
                "clip_ratio_c": actor_cfg.clip_ratio_dual,
                "entropy_coeff": 0.0,
                "loss_agg_mode": actor_cfg.loss_avg_mode,
                "policy_loss": {"loss_mode": "vanilla" if actor_cfg.loss_type == "default" else actor_cfg.loss_type},
                "use_kl_loss": actor_cfg.use_kl_loss,
                "kl_loss_type": actor_cfg.kl_penalty,
                "kl_loss_coef": actor_cfg.kl_coef,
                "recompute_old_log_prob": True,
                "use_dynamic_bsz": False,
                "ppo_max_token_len_per_gpu": None,
                "rollout_n": self.config.rollout.n,
                "ulysses_sequence_parallel_size": 1,
                "profiler": {"tool": None},
                "router_replay": {"mode": "disabled"},
                "megatron": _as_plain_dict(actor_cfg.megatron),
            }
        )

    def _ref_runtime_config(self):
        ref_cfg = self.config.ref
        return OmegaConf.create(
            {
                "ppo_mini_batch_size": self.config.actor.micro_batch_size_per_device_for_experience,
                "ppo_micro_batch_size_per_gpu": ref_cfg.micro_batch_size_per_device_for_experience,
                "ppo_epochs": 1,
                "shuffle": False,
                "data_loader_seed": None,
                "clip_ratio": self.config.actor.clip_ratio_high,
                "entropy_coeff": 0.0,
                "loss_agg_mode": self.config.actor.loss_avg_mode,
                "policy_loss": {"loss_mode": "vanilla"},
                "use_kl_loss": False,
                "kl_loss_type": "kl",
                "kl_loss_coef": 0.0,
                "recompute_old_log_prob": True,
                "use_dynamic_bsz": False,
                "ppo_max_token_len_per_gpu": None,
                "ulysses_sequence_parallel_size": 1,
                "profiler": {"tool": None},
                "router_replay": {"mode": "disabled"},
                "megatron": _as_plain_dict(ref_cfg.megatron),
            }
        )

    def _optimizer_config(self):
        optim_cfg = self.config.actor.optim
        training_steps = max(1, int(optim_cfg.training_steps if optim_cfg.training_steps > 0 else 1))
        warmup_steps = optim_cfg.lr_warmup_steps
        if warmup_steps is None:
            warmup_steps = int(optim_cfg.lr_warmup_ratio * training_steps)
        min_lr = 0.0
        if optim_cfg.min_lr_ratio is not None:
            min_lr = optim_cfg.lr * optim_cfg.min_lr_ratio
        return OmegaConf.create(
            {
                "optimizer": "adam",
                "lr": optim_cfg.lr,
                "min_lr": min_lr,
                "clip_grad": self.config.actor.max_grad_norm,
                "weight_decay": optim_cfg.weight_decay,
                "total_training_steps": training_steps,
                "lr_decay_steps": training_steps,
                "lr_warmup_steps": warmup_steps,
                "lr_warmup_steps_ratio": None,
                "lr_warmup_init": 0.0,
                "lr_decay_style": optim_cfg.lr_scheduler_type,
                "weight_decay_incr_style": "constant",
                "use_checkpoint_opt_param_scheduler": False,
                "override_optimizer_config": {"adam_beta1": optim_cfg.betas[0], "adam_beta2": optim_cfg.betas[1]},
                "lr_wsd_decay_steps": None,
                "lr_wsd_decay_style": "exponential",
            }
        )

    def _init_hf_config_and_tf_config(self, megatron_config) -> None:
        model_config = self.config.actor.model
        model_path = model_config.model_path
        tokenizer_path = model_config.tokenizer_path or model_path
        trust_remote_code = model_config.trust_remote_code

        self.local_path = copy_to_local(model_path)
        local_tokenizer_path = copy_to_local(tokenizer_path)
        self.tokenizer = get_tokenizer(local_tokenizer_path, trust_remote_code=trust_remote_code, use_fast=True)
        self.processor = get_processor(local_tokenizer_path, trust_remote_code=trust_remote_code, use_fast=True)

        hf_config = AutoConfig.from_pretrained(
            self.local_path,
            trust_remote_code=trust_remote_code,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        override_config_kwargs = _as_plain_dict(model_config.override_config)
        override_config_kwargs.update(
            {
                "bos_token_id": self.tokenizer.bos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
        )
        update_model_config(hf_config, override_config_kwargs=override_config_kwargs)

        self.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)
        self.architectures = getattr(hf_config, "architectures", None) or []
        self.generation_config = get_generation_config(self.local_path, trust_remote_code=trust_remote_code)

        if not megatron_config.use_mbridge:
            raise NotImplementedError("canvas-rl Megatron worker currently requires megatron.use_mbridge=True.")

        override_transformer_config = _as_plain_dict(megatron_config.override_transformer_config)
        from ..models.mcore.config_converter import mapping_string_to_attn_backend
        from ..models.mcore.mbridge import AutoBridge

        override_transformer_config = mapping_string_to_attn_backend(override_transformer_config)
        bridge = AutoBridge.from_config(hf_config, dtype=self.dtype)
        bridge.set_extra_args(**override_transformer_config)
        tf_config = bridge.config
        tf_config.fp16 = self.dtype == torch.float16
        tf_config.bf16 = self.dtype == torch.bfloat16

        self.bridge = bridge
        self.provider = None
        self.vanilla_bridge = True
        self.hf_config = hf_config
        self.tf_config = tf_config

        self.print_rank0(f"Model config after override: {hf_config}")
        self.print_rank0(f"TF config: {tf_config}")

    def _build_actor_model_optimizer(self):
        megatron_cfg = self.config.actor.megatron
        self._init_hf_config_and_tf_config(megatron_cfg)
        wrap_config = McoreModuleWrapperConfig(
            is_value_model=False,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            wrap_with_ddp=True,
            use_distributed_optimizer=megatron_cfg.use_distributed_optimizer,
        )
        actor_module, updated_tf_config = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
            hf_config=self.hf_config,
            bridge=self.bridge,
            provider=self.provider,
            override_model_config=_as_plain_dict(self.config.actor.model.override_config),
            override_ddp_config=_as_plain_dict(megatron_cfg.override_ddp_config),
            peft_cls=None,
            peft_config=None,
        )
        self.tf_config = updated_tf_config

        if self.config.actor.load_weight:
            if megatron_cfg.use_dist_checkpointing:
                load_mcore_dist_weights(
                    actor_module,
                    megatron_cfg.dist_checkpointing_path,
                    is_value_model=False,
                    prefix=megatron_cfg.dist_checkpointing_prefix,
                )
            else:
                self.bridge.load_weights(actor_module, self.local_path)

        if self.rank == 0:
            print_model_size(actor_module[0])

        optim_config = self._optimizer_config()
        optim_config_megatron = init_megatron_optim_config(
            optim_config,
            use_distributed_optimizer=wrap_config.use_distributed_optimizer,
            fp16=self.dtype == torch.float16,
        )
        actor_optimizer = get_megatron_optimizer(model=actor_module, config=optim_config_megatron)
        actor_optimizer_scheduler = get_megatron_optimizer_param_scheduler(optimizer=actor_optimizer, config=optim_config)
        register_megatron_training_hooks(actor_module, actor_optimizer)
        return actor_module, actor_optimizer, actor_optimizer_scheduler, self.hf_config

    def _build_ref_model(self):
        megatron_cfg = self.config.ref.megatron
        # Actor and ref share tokenizer/config path, but the ref Megatron config may
        # differ in offload/checkpoint options; rebuild through the same bridge path.
        self._init_hf_config_and_tf_config(megatron_cfg)
        wrap_config = McoreModuleWrapperConfig(
            is_value_model=False,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            wrap_with_ddp=False,
            use_distributed_optimizer=megatron_cfg.use_distributed_optimizer,
        )
        ref_module, updated_tf_config = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
            hf_config=self.hf_config,
            bridge=self.bridge,
            provider=self.provider,
            override_model_config=_as_plain_dict(self.config.actor.model.override_config),
            override_ddp_config=None,
        )
        self.tf_config = updated_tf_config

        if self.config.ref.load_weight:
            if megatron_cfg.use_dist_checkpointing:
                load_mcore_dist_weights(
                    ref_module,
                    megatron_cfg.dist_checkpointing_path,
                    is_value_model=False,
                    prefix=megatron_cfg.dist_checkpointing_prefix,
                )
            else:
                self.bridge.load_weights(ref_module, self.local_path)
        return ref_module, self.hf_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self._is_lora:
            raise NotImplementedError("Megatron actor/ref path does not support LoRA in canvas-rl yet.")

        self.param_dtype = PrecisionType.to_dtype(self.config.actor.megatron.dtype)
        self.dtype = self.param_dtype

        if self._has_actor:
            (
                self.actor_module,
                self.actor_optimizer,
                self.actor_optimizer_scheduler,
                self.actor_model_config,
            ) = self._build_actor_model_optimizer()
            if self._is_offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
            if self._is_offload_optimizer:
                offload_megatron_optimizer(self.actor_optimizer)

            self.actor_runtime_config = self._actor_runtime_config()
            self.actor = MegatronPPOActor(
                config=self.actor_runtime_config,
                model_config=self.actor_model_config,
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                actor_module=self.actor_module,
                actor_optimizer=self.actor_optimizer,
            )
            self.flops_counter = FlopsCounter(self.actor_model_config)

        if self._has_ref:
            self.ref_module, self.ref_model_config = self._build_ref_model()
            if self._ref_is_offload_param:
                offload_megatron_model_to_cpu(self.ref_module)

            self.ref_runtime_config = self._ref_runtime_config()
            self.ref_policy = MegatronPPOActor(
                config=self.ref_runtime_config,
                model_config=self.ref_model_config,
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                actor_module=self.ref_module,
                actor_optimizer=None,
            )

        get_torch_device().empty_cache()

    def _process_multi_modal_inputs(self, data: DataProto) -> None:
        if "multi_modal_inputs" in data.non_tensor_batch:
            return
        if "multi_modal_data" not in data.non_tensor_batch:
            return
        if self.processor is None:
            raise ValueError("multi_modal_data is present but model processor is not available.")

        if "uid" in self._cache and not np.all(data.non_tensor_batch["uid"] == self._cache["uid"]):
            self._cache.clear()

        if "multi_modal_inputs" not in self._cache:
            min_pixels = data.meta_info["min_pixels"]
            max_pixels = data.meta_info["max_pixels"]
            video_fps = data.meta_info["video_fps"]
            batch_multi_modal_inputs = []
            multi_modal_inputs_cache = {}
            for index, multi_modal_data in zip(data.non_tensor_batch["uid"], data.non_tensor_batch["multi_modal_data"]):
                if index not in multi_modal_inputs_cache:
                    images, videos = [], []
                    for image in multi_modal_data.get("images", []):
                        images.append(process_image(image, min_pixels, max_pixels))
                    for video in multi_modal_data.get("videos", []):
                        videos.append(process_video(video, min_pixels, max_pixels, video_fps))

                    if images:
                        multi_modal_inputs = dict(self.processor.image_processor(images=images, return_tensors="pt"))
                    elif videos:
                        multi_modal_inputs = dict(
                            self.processor.image_processor(images=None, videos=videos, return_tensors="pt")
                        )
                    else:
                        multi_modal_inputs = {}
                    multi_modal_inputs_cache[index] = multi_modal_inputs
                batch_multi_modal_inputs.append(multi_modal_inputs_cache[index])

            self._cache["uid"] = data.non_tensor_batch["uid"]
            self._cache["multi_modal_inputs"] = np.array(batch_multi_modal_inputs, dtype=object)

        data.non_tensor_batch["multi_modal_inputs"] = self._cache["multi_modal_inputs"]

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: DataProto):
        assert self._has_actor
        self._process_multi_modal_inputs(data)
        data = data.to(get_device_id())

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        if self._is_offload_optimizer:
            load_megatron_optimizer(self.actor_optimizer)

        data.meta_info["micro_batch_size"] = self.config.actor.micro_batch_size_per_device_for_update
        with Timer(name="update_policy", logger=None) as timer:
            metrics = self.actor.update_policy(dataloader=self.actor.make_minibatch_iterator(data=data))
        delta_time = timer.last

        if "global_token_num" in data.meta_info:
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(data.meta_info["global_token_num"], delta_time)
            metrics["perf/mfu_actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
        metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        metrics["actor/lr"] = get_megatron_last_lr(self.actor_optimizer)
        self.actor_optimizer_scheduler.step(1)

        output = DataProto(
            non_tensor_batch={
                key: np.array([_mean_metric(value)] if np.isscalar(_mean_metric(value)) else [_mean_metric(value)], dtype=object)
                for key, value in metrics.items()
            }
        )

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)

        get_torch_device().empty_cache()
        return output.to("cpu")

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_log_probs(self, data: DataProto):
        assert self._has_actor
        self._process_multi_modal_inputs(data)
        data = data.to(get_device_id())

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module, load_grad=False)

        data.meta_info["micro_batch_size"] = self.config.actor.micro_batch_size_per_device_for_experience
        data.meta_info["max_token_len"] = None
        data.meta_info["use_dynamic_bsz"] = False
        data.meta_info["temperature"] = self.config.rollout.temperature

        output, _, _ = self.actor.compute_log_prob(data=data, calculate_entropy=False)
        output = DataProto.from_dict(
            tensors={"old_log_probs": output},
            meta_info={"temperature": self.config.rollout.temperature},
        )

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)

        get_torch_device().empty_cache()
        return output.to("cpu")

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_ref_log_probs(self, data: DataProto):
        assert self._has_ref
        self._process_multi_modal_inputs(data)
        data = data.to(get_device_id())

        if self._ref_is_offload_param:
            load_megatron_model_to_gpu(self.ref_module, load_grad=False)

        data.meta_info["micro_batch_size"] = self.config.ref.micro_batch_size_per_device_for_experience
        data.meta_info["max_token_len"] = None
        data.meta_info["use_dynamic_bsz"] = False
        data.meta_info["temperature"] = self.config.rollout.temperature

        output, _, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
        output = DataProto.from_dict(tensors={"ref_log_probs": output})

        if self._ref_is_offload_param:
            offload_megatron_model_to_cpu(self.ref_module)

        get_torch_device().empty_cache()
        return output.to("cpu")

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    def generate_sequences(self, prompts: DataProto):
        raise NotImplementedError(_ROLLOUT_PENDING_MESSAGE)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prepare_rollout_engine(self):
        raise NotImplementedError(_ROLLOUT_PENDING_MESSAGE)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_rollout_engine(self):
        raise NotImplementedError(_ROLLOUT_PENDING_MESSAGE)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path: str, save_model_only: bool = False):
        raise NotImplementedError("Megatron checkpoint save is not ported in canvas-rl yet.")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path: Optional[str]):
        if path is None:
            if self._has_actor and self._is_offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
            if self._has_actor and self._is_offload_optimizer:
                offload_megatron_optimizer(self.actor_optimizer)
            return
        raise NotImplementedError("Megatron checkpoint load is not ported in canvas-rl yet.")


class CriticWorker(Worker):
    """Critic worker placeholder for the Megatron backend."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._register_dispatch_collect_info(mesh_name="critic", dp_rank=self.rank, is_collect=True)

    def _raise_pending(self):
        raise NotImplementedError(_CRITIC_PENDING_MESSAGE)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self._raise_pending()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    def compute_values(self, data: DataProto):
        self._raise_pending()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    def update_critic(self, data: DataProto):
        self._raise_pending()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path: str, save_model_only: bool = False):
        self._raise_pending()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path: str):
        self._raise_pending()
