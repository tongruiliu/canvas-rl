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

import asyncio
import copy
import json

import os
from contextlib import contextmanager
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from ...utils.vllm_utils import VLLMHijack
from .base import BaseRollout
from .config import RolloutConfig

from ...tools import ToolResponse, initialize_tools_from_config
from .tool_parser import ToolParser


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    # repeat the elements, supports both tensor and numpy array
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None

def _process_multi_modal_data(
    multi_modal_data: dict[str, Any],
    min_pixels: int,
    max_pixels: int,
    video_fps: float,
    return_video_metadata: bool = False,
) -> dict[str, Any]:
    # may convert image path to image object
    images, videos = [], []
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            videos.append(
                process_video(
                    video,
                    min_pixels,
                    max_pixels,
                    video_fps,
                    return_metadata=return_video_metadata,
                )
            )

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None


class vLLMRollout(BaseRollout):
    # processor: mllm, tokenizer: nlp
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        **kwargs,
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.pad_token_id = tokenizer.pad_token_id
        self.return_video_metadata = processor is not None and "Qwen3VLProcessor" in processor.__class__.__name__
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        
        self.tools = {}
        self.tool_schemas = []
        self.tool_parser = None
        if self.config.multi_turn.enable:
            tool_list = []
            if self.config.multi_turn.tool_config_path:
                tool_list = initialize_tools_from_config(self.config.multi_turn.tool_config_path)
            self.tools = {tool.name: tool for tool in tool_list}
            self.tool_schemas = [tool.tool_schema for tool in tool_list]
            self.tool_parser = ToolParser.get_tool_parser(self.config.multi_turn.format, tokenizer)

        engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        VLLMHijack.hijack()

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy" if not self.lora_kwargs else "safetensors",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)
    
    def _apply_chat_template_from_raw_prompt(self, raw_prompt: list[dict[str, Any]]) -> str:
        # agentic process
        if self.processor is not None:
            return self.processor.apply_chat_template(raw_prompt, add_generation_prompt=True, tokenize=False)
        return self.tokenizer.apply_chat_template(raw_prompt, add_generation_prompt=True, tokenize=False)
    
    def _build_agentic_vllm_inputs(
        self,
        batch_raw_prompt: np.ndarray,
        batch_multi_modal_data: Optional[np.ndarray],
        meta_info: dict[str, Any],
    ) -> list[dict[str, Any]]:
        vllm_inputs = []
        for idx, raw_prompt in enumerate(batch_raw_prompt):
            prompt_text = self._apply_chat_template_from_raw_prompt(list(raw_prompt))
            prompt_token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

            # Keep prompt-side truncation aligned with current prompt_length contract.
            if len(prompt_token_ids) > self.config.prompt_length:
                prompt_token_ids = prompt_token_ids[: self.config.prompt_length]

            vllm_input = {"prompt_token_ids": prompt_token_ids}
            if batch_multi_modal_data is not None:
                multi_modal_data = batch_multi_modal_data[idx]
                vllm_input["multi_modal_data"] = _process_multi_modal_data(
                    multi_modal_data,
                    meta_info["min_pixels"],
                    meta_info["max_pixels"],
                    meta_info["video_fps"],
                    return_video_metadata=self.return_video_metadata,
                )
            vllm_inputs.append(vllm_input)

        return vllm_inputs
    
    def _build_lora_requests(self, batch_size: int) -> Optional[list[LoRARequest]]:
        if not self.lora_kwargs:
            return None

        lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
        if len(lora_int_ids) == 0:
            return None

        lora_int_id = lora_int_ids[0]
        return [
            LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
        ] * batch_size
    
    def _run_async_tool(self, coroutine):
        """
        Run async tool hooks from the synchronous rollout worker.
        coroutine is function that can pause and recover, often definited by async def.
        coroutine can not run directly, should use event loop.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coroutine)
            finally:
                new_loop.close()
        return asyncio.run(coroutine)
    
    def _truncate_tool_text(self, text: Optional[str]) -> Optional[str]:
        """Truncate long tool outputs using the current multi_turn config."""
        if text is None:
            return None

        max_len = self.config.multi_turn.max_tool_response_length
        if len(text) <= max_len:
            return text

        side = self.config.multi_turn.tool_response_truncate_side
        if side == "left":
            return text[:max_len] + "...(truncated)"
        if side == "right":
            return "(truncated)..." + text[-max_len:]

        half = max_len // 2
        return text[:half] + "...(truncated)..." + text[-half:]
    
    def _call_tool(
        self,
        function_call,
    ) -> tuple[ToolResponse, float, dict]:
        """Execute one parsed tool call and return a normalized ToolResponse."""
        if function_call.name not in self.tools:
            return ToolResponse(text=f"Unknown tool: {function_call.name}"), 0.0, {}

        tool = self.tools[function_call.name]
        try:
            tool_args = json.loads(function_call.arguments)
        except Exception as exc:
            return ToolResponse(text=f"Invalid tool arguments: {exc}"), 0.0, {}

        instance_id, _ = self._run_async_tool(tool.create())
        try:
            tool_response, _, _ = self._run_async_tool(
                tool.execute(instance_id=instance_id, parameters=tool_args)
            )
        except Exception as exc:
            return ToolResponse(text=f"Error when executing tool: {exc}"), 0.0, {}
        finally:
            self._run_async_tool(tool.release(instance_id))

        tool_response.text = self._truncate_tool_text(tool_response.text)
        return tool_response, 0.0, {}
    
    def _tool_response_to_message(self, tool_response: ToolResponse) -> dict[str, Any]:
        """Convert one ToolResponse into a tool-role chat message."""
        if tool_response.image or tool_response.video:
            content = []
            if tool_response.image:
                content.append({"type": "image"})
            if tool_response.video:
                content.append({"type": "video"})
            if tool_response.text:
                content.append({"type": "text", "text": tool_response.text})
            return {"role": "tool", "content": content}

        return {"role": "tool", "content": tool_response.text or ""}
    
    def _merge_multi_modal_data(
        self,
        current_multi_modal_data: Optional[dict[str, Any]],
        tool_response: ToolResponse,
    ) -> Optional[dict[str, Any]]:
        """Merge tool-returned images/videos into the current sample state."""
        if not tool_response.image and not tool_response.video:
            return current_multi_modal_data

        if current_multi_modal_data is None:
            merged = {}
        else:
            merged = copy.deepcopy(current_multi_modal_data)

        if tool_response.image:
            # if "images" is not existed in merged, create; else, do nothing.
            merged.setdefault("images", [])
            merged["images"].extend(tool_response.image)
        if tool_response.video:
            merged.setdefault("videos", [])
            merged["videos"].extend(tool_response.video)

        return merged

    def _run_agentic_sample_once(
        self,
        raw_prompt: list[dict[str, Any]],
        multi_modal_data: Optional[dict[str, Any]],
        meta_info: dict[str, Any],
    ) -> tuple[list[int], list[int], Optional[dict[str, Any]], dict[str, Any]]:
        """Run one full agentic trajectory for a single sample and return trajectory metadata."""
        messages = list(raw_prompt)
        current_multi_modal_data = copy.deepcopy(multi_modal_data) if multi_modal_data is not None else None
        response_token_chunks: list[int] = []
        # 1:model generate, 0:tool return.
        response_mask_chunks: list[int] = []
        assistant_turns = 0
        user_turns = 0
        tool_call_count = 0
        tool_failure_count = 0
        stop_reason = "max_response_length"

        while len(response_mask_chunks) < self.config.response_length:
            assistant_turns += 1
            if (
                self.config.multi_turn.max_assistant_turns is not None
                and assistant_turns > self.config.multi_turn.max_assistant_turns
            ):
                stop_reason = "max_assistant_turns"
                break

            prompt_text = self._apply_chat_template_from_raw_prompt(messages)
            prompt_token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            if len(prompt_token_ids) > self.config.prompt_length:
                prompt_token_ids = prompt_token_ids[: self.config.prompt_length]
            
            vllm_input = {"prompt_token_ids": prompt_token_ids}
            if current_multi_modal_data is not None:
                vllm_input["multi_modal_data"] = _process_multi_modal_data(
                    current_multi_modal_data,
                    meta_info["min_pixels"],
                    meta_info["max_pixels"],
                    meta_info["video_fps"],
                    return_video_metadata=self.return_video_metadata,
                )

            with self.update_sampling_params(**meta_info):
                old_n = self.sampling_params.n
                self.sampling_params.n = 1
                try:
                    completion = self.inference_engine.generate(
                        prompts=[vllm_input],
                        sampling_params=self.sampling_params,
                        lora_request=self._build_lora_requests(1),
                        use_tqdm=False,
                    )[0]
                finally:
                    self.sampling_params.n = old_n
            
            generated_ids = completion.outputs[0].token_ids
            response_token_chunks.extend(generated_ids)
            response_mask_chunks.extend([1] * len(generated_ids))

            if len(response_mask_chunks) >= self.config.response_length:
                stop_reason = "max_response_length"
                break

            if self.tool_parser is None:
                stop_reason = "no_tool_parser"
                break

            assistant_text, function_calls = self.tool_parser.extract_tool_calls(generated_ids)
            messages.append({"role": "assistant", "content": assistant_text})

            if not function_calls:
                stop_reason = "finished_no_tool_call"
                break

            tool_messages = []
            for function_call in function_calls[: self.config.multi_turn.max_parallel_calls]:
                tool_call_count += 1
                tool_response, _, _ = self._call_tool(function_call)
                if tool_response.text and tool_response.text.startswith("Error when executing tool:"):
                    tool_failure_count += 1
                tool_message = self._tool_response_to_message(tool_response)
                tool_messages.append(tool_message)
                current_multi_modal_data = self._merge_multi_modal_data(current_multi_modal_data, tool_response)

            if not tool_messages:
                stop_reason = "empty_tool_messages"
                break

            user_turns += 1
            if self.config.multi_turn.max_user_turns is not None and user_turns > self.config.multi_turn.max_user_turns:
                stop_reason = "max_user_turns"
                break

            # Re-encode only the tool messages as observation tokens.
            tool_prompt_text = self._apply_chat_template_from_raw_prompt(tool_messages)
            tool_token_ids = self.tokenizer.encode(tool_prompt_text, add_special_tokens=False)
            response_token_chunks.extend(tool_token_ids)
            response_mask_chunks.extend([0] * len(tool_token_ids))
            messages.extend(tool_messages)
        
        trajectory_info = {
            "num_turns": assistant_turns + user_turns + 1,
            "assistant_turns": assistant_turns,
            "user_turns": user_turns,
            "tool_calls": tool_call_count,
            "tool_failures": tool_failure_count,
            "stop_reason": stop_reason,
        }

        return (
            response_token_chunks[: self.config.response_length],
            response_mask_chunks[: self.config.response_length],
            current_multi_modal_data,
            trajectory_info,
        )
    
    def _use_agentic_rollout(self, prompts: DataProto) -> bool:
        if self.config.agent.enable:
            return True
        if self.config.multi_turn.enable:
            return True
        if "raw_prompt" in prompts.non_tensor_batch:
            return self.config.agent.default_agent_loop != "single_turn_agent" and self.config.agent.enable
        return False
    
    def _pad_response_ids_and_masks(
        self,
        response_id_list: list[list[int]],
        response_mask_list: list[list[int]],
        device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad token ids and response masks to the configured response length."""
        padded_response_ids = VF.pad_2d_list_to_length(
            response_id_list, self.pad_token_id, max_length=self.config.response_length
        ).to(device)
        padded_response_masks = VF.pad_2d_list_to_length(
            response_mask_list, 0, max_length=self.config.response_length
        ).to(device)
        return padded_response_ids, padded_response_masks
    
    def _build_multi_modal_inputs_cache(self, batch_multi_modal_data: list[Optional[dict[str, Any]]]) -> np.ndarray:
        """Convert final multimodal sample state into cached processor inputs for actor/ref scoring."""
        cached = []
        for multi_modal_data in batch_multi_modal_data:
            if multi_modal_data is None:
                cached.append({})
                continue

            images, videos = [], []
            if "images" in multi_modal_data:
                images.extend(multi_modal_data["images"])
            if "videos" in multi_modal_data:
                videos.extend(multi_modal_data["videos"])

            if len(images) != 0:
                cached.append(dict(self.processor.image_processor(images=images, return_tensors="pt")))
            elif len(videos) != 0:
                cached.append(dict(self.processor.image_processor(images=None, videos=videos, return_tensors="pt")))
            else:
                cached.append({})

        return np.array(cached, dtype=object)
    
    def _summarize_agentic_batch(self, traj_infos: list[dict[str, Any]]) -> dict[str, float]:
        """Summarize one agentic rollout batch into scalar metrics."""
        if len(traj_infos) == 0:
            return {}

        return {
            "agentic/avg_num_turns": float(np.mean([info["num_turns"] for info in traj_infos])),
            "agentic/avg_assistant_turns": float(np.mean([info["assistant_turns"] for info in traj_infos])),
            "agentic/avg_user_turns": float(np.mean([info["user_turns"] for info in traj_infos])),
            "agentic/avg_tool_calls": float(np.mean([info["tool_calls"] for info in traj_infos])),
            "agentic/avg_tool_failures": float(np.mean([info["tool_failures"] for info in traj_infos])),
        }
    
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        if self._use_agentic_rollout(prompts):
            return self._generate_sequences_agentic(prompts)
        return self._generate_sequences_single_turn(prompts)

    @torch.no_grad()
    def _generate_sequences_single_turn(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": _process_multi_modal_data(
                            multi_modal_data,
                            prompts.meta_info["min_pixels"],
                            prompts.meta_info["max_pixels"],
                            prompts.meta_info["video_fps"],
                            return_video_metadata=self.return_video_metadata,
                        ),
                    }
                )
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
            completions: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=self.use_tqdm,
            )
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)
    
    @torch.no_grad()
    def _generate_sequences_agentic(self, prompts: DataProto) -> DataProto:
        """Run the message-first agentic rollout across the whole batch."""
        input_ids: torch.Tensor = prompts.batch["input_ids"]
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt = non_tensor_batch.pop("raw_prompt")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        non_tensor_batch.pop("raw_prompt_ids", None)

        if batch_size != len(batch_raw_prompt):
            raise RuntimeError("Agentic rollout batch is inconsistent with raw_prompt size.")
        
        rollout_n = prompts.meta_info.get("n", self.config.n)
        all_response_ids = []
        all_response_masks = []
        all_multi_modal_data = []
        all_raw_prompts = []
        all_traj_infos = []

        for sample_idx in range(batch_size):
            raw_prompt = list(batch_raw_prompt[sample_idx])
            sample_mm = None if batch_multi_modal_data is None else batch_multi_modal_data[sample_idx]

            for _ in range(rollout_n):
                sample_response_ids, sample_response_mask, sample_final_mm, sample_traj_info = self._run_agentic_sample_once(
                    raw_prompt=raw_prompt,
                    multi_modal_data=sample_mm,
                    meta_info=prompts.meta_info,
                )
                all_response_ids.append(sample_response_ids)
                all_response_masks.append(sample_response_mask)
                all_multi_modal_data.append(sample_final_mm)
                all_raw_prompts.append(raw_prompt)
                all_traj_infos.append(sample_traj_info)
        
        response_ids, response_mask = self._pad_response_ids_and_masks(
            all_response_ids, all_response_masks, input_ids.device
        )

        if rollout_n > 1:
            input_ids = _repeat_interleave(input_ids, rollout_n)
            attention_mask = _repeat_interleave(attention_mask, rollout_n)
            position_ids = _repeat_interleave(position_ids, rollout_n)
            batch_size = batch_size * rollout_n

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        attention_mask = torch.cat((attention_mask, response_mask.to(attention_mask.dtype)), dim=-1)

        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask.to(attention_mask.dtype),
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        
        output_non_tensor_batch = {
            "raw_prompt": np.array(all_raw_prompts, dtype=object),
        }

        output_non_tensor_batch["agentic_num_turns"] = np.array(
            [info["num_turns"] for info in all_traj_infos], dtype=np.int32
        )
        output_non_tensor_batch["agentic_assistant_turns"] = np.array(
            [info["assistant_turns"] for info in all_traj_infos], dtype=np.int32
        )
        output_non_tensor_batch["agentic_user_turns"] = np.array(
            [info["user_turns"] for info in all_traj_infos], dtype=np.int32
        )
        output_non_tensor_batch["agentic_tool_calls"] = np.array(
            [info["tool_calls"] for info in all_traj_infos], dtype=np.int32
        )
        output_non_tensor_batch["agentic_tool_failures"] = np.array(
            [info["tool_failures"] for info in all_traj_infos], dtype=np.int32
        )
        output_non_tensor_batch["agentic_stop_reason"] = np.array(
            [info["stop_reason"] for info in all_traj_infos], dtype=object
        )

        if any(mm is not None for mm in all_multi_modal_data):
            output_non_tensor_batch["multi_modal_data"] = np.array(all_multi_modal_data, dtype=object)
        
        if self.processor is not None and any(mm is not None for mm in all_multi_modal_data):
            output_non_tensor_batch["multi_modal_inputs"] = self._build_multi_modal_inputs_cache(all_multi_modal_data)
        
        meta_info = dict(prompts.meta_info)
        meta_info["agentic_metrics"] = self._summarize_agentic_batch(all_traj_infos)

        return DataProto(batch=batch, non_tensor_batch=output_non_tensor_batch, meta_info=meta_info)