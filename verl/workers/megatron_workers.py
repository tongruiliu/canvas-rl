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
"""Megatron worker entry points.

This file intentionally starts as a thin runtime boundary.  canvas-rl's public
trainer can now route `worker.actor.strategy=megatron` here without touching the
existing FSDP path; the full Megatron training implementation is ported in the
next layer because it depends on a large set of verl MCore utilities, optimizer
helpers, checkpoint managers, and Megatron-to-vLLM weight synchronization.
"""

from typing import Literal

from ..protocol import DataProto
from ..single_controller.base import Worker
from ..single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register


_MEGATRON_PENDING_MESSAGE = """
Megatron strategy is wired through the canvas-rl trainer, but the runtime worker
implementation has not been ported yet.

Next required patch layer:
1. port verl.models.mcore and verl.utils.megatron* helpers;
2. port MegatronPPOActor / Megatron critic implementations;
3. port MegatronCheckpointManager;
4. replace FSDP->vLLM weight sync with Megatron->vLLM weight sync while keeping
   canvas-rl's tool/multi-turn vLLM rollout.
"""


class _MegatronRuntimePendingMixin:
    def _raise_pending(self):
        raise NotImplementedError(_MEGATRON_PENDING_MESSAGE)


class AsyncActorRolloutRefWorker(Worker, _MegatronRuntimePendingMixin):
    """Actor/rollout/ref worker placeholder for the Megatron backend."""

    def __init__(
        self,
        config,
        role: Literal["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"],
    ):
        super().__init__()
        self.config = config
        self.role = role

        # Register a valid mesh so WorkerGroup's lazy ND dispatcher can query it.
        # The real Megatron worker will replace this with mpu data-parallel ranks
        # and collect masks for TP/PP/CP/EP groups.
        self._register_dispatch_collect_info(mesh_name="actor", dp_rank=self.rank, is_collect=True)
        self._register_dispatch_collect_info(mesh_name="rollout", dp_rank=self.rank, is_collect=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self._raise_pending()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: DataProto):
        self._raise_pending()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_log_probs(self, data: DataProto):
        self._raise_pending()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_ref_log_probs(self, data: DataProto):
        self._raise_pending()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    def generate_sequences(self, prompts: DataProto):
        self._raise_pending()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prepare_rollout_engine(self):
        self._raise_pending()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_rollout_engine(self):
        self._raise_pending()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path: str, save_model_only: bool = False):
        self._raise_pending()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path: str):
        self._raise_pending()


class CriticWorker(Worker, _MegatronRuntimePendingMixin):
    """Critic worker placeholder for the Megatron backend."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._register_dispatch_collect_info(mesh_name="critic", dp_rank=self.rank, is_collect=True)

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
