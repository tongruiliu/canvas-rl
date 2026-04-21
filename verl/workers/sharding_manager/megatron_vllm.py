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
"""Megatron actor to vLLM sharding manager."""

import inspect
from typing import Iterable

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from vllm import LLM
from vllm.distributed import parallel_state as vllm_ps

from ...protocol import DataProto, all_gather_data_proto
from ...utils.megatron_utils import load_megatron_model_to_gpu, offload_megatron_model_to_cpu
from ...utils.model_utils import print_gpu_memory_usage
from .base import BaseShardingManager


class MegatronVLLMShardingManager(BaseShardingManager):
    """Synchronize Megatron actor weights into canvas-rl's colocated vLLM engine."""

    def __init__(
        self,
        actor_module: list[torch.nn.Module],
        inference_engine: LLM,
        bridge,
        device_mesh: DeviceMesh,
        use_param_offload: bool,
    ):
        self.actor_module = actor_module
        self.inference_engine = inference_engine
        self.bridge = bridge
        self.device_mesh = device_mesh
        self.use_param_offload = use_param_offload
        self.loaded = False

        self.world_size = dist.get_world_size()
        self.tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()
        self.tp_group = vllm_ps.get_tensor_model_parallel_group().device_group

        self.freed_bytes = 0
        self.torch_random_states = torch.cuda.get_rng_state()
        gen_dp_rank = self.device_mesh["dp"].get_local_rank()
        torch.cuda.manual_seed(gen_dp_rank + 1000)
        self.gen_random_states = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(self.torch_random_states)

    def _make_weight_iterator(self) -> Iterable[tuple[str, torch.Tensor]]:
        if self.bridge is None:
            raise NotImplementedError("Megatron->vLLM sync currently requires mbridge weight export.")
        return self.bridge.export_weights(self.actor_module)

    def _sync_weight_to_vllm(self) -> None:
        if self.use_param_offload:
            load_megatron_model_to_gpu(self.actor_module, load_grad=False)

        print_gpu_memory_usage("Before export Megatron weights in sharding manager")
        model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
        model.load_weights(self._make_weight_iterator())
        print_gpu_memory_usage("After sync Megatron weights to vLLM in sharding manager")

        if self.use_param_offload:
            offload_megatron_model_to_cpu(self.actor_module)

        torch.cuda.empty_cache()

    def load_vllm_and_sync_weights(self) -> None:
        torch.cuda.empty_cache()
        assert self.loaded is False, "vLLM engine has already been loaded"
        self.loaded = True

        print_gpu_memory_usage("Before vLLM wake up in Megatron sharding manager")
        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=["weights"])
        else:
            self.inference_engine.wake_up()

        self._sync_weight_to_vllm()

        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=["kv_cache"])

        self.torch_random_states = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(self.gen_random_states)
        print_gpu_memory_usage("After vLLM wake up in Megatron sharding manager")

    def offload_vllm(self) -> None:
        assert self.loaded is True, "vLLM engine has not been loaded"
        self.loaded = False

        print_gpu_memory_usage("Before vLLM offload in Megatron sharding manager")
        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]
        self.inference_engine.sleep(level=1)
        free_bytes_after_sleep = torch.cuda.mem_get_info()[0]
        self.freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep

        for module in self.actor_module:
            module.train()
        torch.cuda.empty_cache()

        self.gen_random_states = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(self.torch_random_states)
        print_gpu_memory_usage("After vLLM offload in Megatron sharding manager")

    def preprocess_data(self, data: DataProto) -> DataProto:
        all_gather_data_proto(data, size=self.tp_size, group=self.tp_group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        if self.tp_size > 1:
            data = data.chunk(chunks=self.tp_size)[self.tp_rank]
        return data
