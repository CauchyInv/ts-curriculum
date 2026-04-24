# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from tensordict import TensorDict
from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.utils.model import compute_position_id_with_mask
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=3, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        # legacy reward model implementation
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # Initialize teacher-student related variables (optional)
        curri_method = self.config.data.get("curri_method", None)
        if curri_method == "teacher_student":
            self.teacher_model = self.config.data.get("teacher_model", "none")
            self.ts_version = self.config.data.get("ts_version", "none")
            use_adaptive_cfg = self.config.data.get("use_adaptive", False)
            if isinstance(use_adaptive_cfg, str):
                use_adaptive_flag = use_adaptive_cfg.strip().lower() in {"1", "true", "yes", "y", "on"}
            else:
                use_adaptive_flag = bool(use_adaptive_cfg)
            self.v8_use_adaptive = bool(self.ts_version == "v8" and use_adaptive_flag)
            v8_mix44444_adaptive_cfg = self.config.data.get("v8_mix44444_adaptive", False)
            if isinstance(v8_mix44444_adaptive_cfg, str):
                v8_mix44444_adaptive_flag = v8_mix44444_adaptive_cfg.strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "y",
                    "on",
                }
            else:
                v8_mix44444_adaptive_flag = bool(v8_mix44444_adaptive_cfg)
            self.v8_mix44444_adaptive = bool(self.ts_version == "v8" and v8_mix44444_adaptive_flag)
            self.student_answer_history = {}
            self.teacher_hint_dict = {}
            self.crafted_wrong_answer = {}
            self.teacher_lemmas = {}
            # Initialize old_rewards (will be updated during training)
            # Default batch size is 128, but we'll initialize dynamically
            self.old_rewards = None
            self.hint_generator = None
            # Initialize metric tracking variables for teacher_student
            # Legacy hard split for openr1_mixed_no_choice_no_think.parquet
            self._legacy_hard_problem_list = [
                0, 1, 2, 3, 4, 5, 6, 7, 8,
                73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
                107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                121, 122, 123, 124, 125, 126, 127
            ]
            # omni_128/int_hard_1024 are all-hard datasets.
            self._all_hard_dataset_basenames = {"omni_128.parquet", "hard_1024.parquet"}
            train_files_cfg = self.config.data.get("train_files", None)
            train_files_list = train_files_cfg if isinstance(train_files_cfg, (list, tuple)) else [train_files_cfg]
            auto_all_hard = any(
                tf is not None and os.path.basename(str(tf)) in self._all_hard_dataset_basenames
                for tf in train_files_list
            )
            # Explicit override from config: +data.all_problems_are_hard=true/false
            # If not provided, fallback to auto detection.
            all_hard_override = self.config.data.get("all_problems_are_hard", None)
            if all_hard_override is None:
                self._all_problems_are_hard = auto_all_hard
            elif isinstance(all_hard_override, bool):
                self._all_problems_are_hard = all_hard_override
            elif isinstance(all_hard_override, str):
                self._all_problems_are_hard = all_hard_override.strip().lower() in {"1", "true", "yes", "y", "on"}
            else:
                self._all_problems_are_hard = bool(all_hard_override)
            # For all-hard datasets this will be overwritten to [0..len(train_dataset)-1]
            # after dataloader creation.
            self.hard_problem_list = self._legacy_hard_problem_list.copy()
            self.problems_solved_in_state_0 = set()
            self.problems_solved_in_state_0_hard = set()
            self.problems_solved_in_state_0_medium = set()
            self.problems_solved_hard_full_subproblem = set()
            self.problems_solved_medium_full_subproblem = set()
            self.current_reward_dict_avg = {}
            self.current_reward_dict_avg_in_state_0 = {}
            # v4: global subproblem solved counts (problem_id set per subproblem_index 1..4)
            self._global_subproblem_solved = {1: set(), 2: set(), 3: set(), 4: set()}
            # v8 adaptive curriculum state: problem_id -> current t (1..4), default starts at t=4.
            self._v8_problem_t_state = {}
            # v8 mix44444 adaptive state:
            # problem_id -> {1: bool(active), 2: bool(active), 3: bool(active)}
            self._v8_mix44444_q_active = {}
        else:
            self.teacher_model = None
            self.ts_version = None
            self.v8_use_adaptive = False
            self.v8_mix44444_adaptive = False
            self.student_answer_history = None
            self.teacher_hint_dict = None
            self.crafted_wrong_answer = None
            self.teacher_lemmas = None
            self.old_rewards = None
            self.hint_generator = None
            self._legacy_hard_problem_list = None
            self._all_hard_dataset_basenames = None
            self._all_problems_are_hard = False
            self.hard_problem_list = None
            self.problems_solved_in_state_0 = None
            self.problems_solved_in_state_0_hard = None
            self.problems_solved_in_state_0_medium = None
            self.current_reward_dict_avg = None
            self.current_reward_dict_avg_in_state_0 = None
            self._global_subproblem_solved = None
            self._v8_problem_t_state = None
            self._v8_mix44444_q_active = None

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # Finalize hard/medium split for teacher-student after dataset is available.
        curri_method = self.config.data.get("curri_method", None)
        if curri_method == "teacher_student" and self._all_problems_are_hard:
            # Assume contiguous problem_id from 0..N-1 for all-hard datasets.
            # This matches omni_128 and int hard_1024 datasets.
            self.hard_problem_list = list(range(len(self.train_dataset)))
        if curri_method == "teacher_student" and self.v8_use_adaptive:
            problem_ids_for_state = []
            try:
                if hasattr(self.train_dataset, "dataframe") and "problem_id" in self.train_dataset.dataframe.column_names:
                    problem_ids_for_state = [str(x) for x in self.train_dataset.dataframe["problem_id"]]
            except Exception:
                problem_ids_for_state = []
            if len(problem_ids_for_state) == 0:
                problem_ids_for_state = [str(i) for i in range(len(self.train_dataset))]
            # Keep insertion order and initialize to t=4.
            self._v8_problem_t_state = {pid: 4 for pid in dict.fromkeys(problem_ids_for_state)}
            print(f"[v8_adaptive] initialized state for {len(self._v8_problem_t_state)} problems at t=4")
        if curri_method == "teacher_student" and self.v8_mix44444_adaptive:
            problem_ids_for_mix44444 = []
            try:
                if hasattr(self.train_dataset, "dataframe") and "problem_id" in self.train_dataset.dataframe.column_names:
                    problem_ids_for_mix44444 = [str(x) for x in self.train_dataset.dataframe["problem_id"]]
            except Exception:
                problem_ids_for_mix44444 = []
            if len(problem_ids_for_mix44444) == 0:
                problem_ids_for_mix44444 = [str(i) for i in range(len(self.train_dataset))]
            self._v8_mix44444_q_active = {
                pid: {1: True, 2: True, 3: True} for pid in dict.fromkeys(problem_ids_for_mix44444)
            }
            print(
                f"[v8_mix44444_adaptive] initialized q1/q2/q3 active state for "
                f"{len(self._v8_mix44444_q_active)} problems"
            )

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid", "problem_id"}) & batch.non_tensor_batch.keys()
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _process_teacher_student_prompts(self, batch: DataProto) -> Optional[DataProto]:
        """
        Process teacher-student prompts by mixing different prompt variants.
        This method handles v1-v5 versions of teacher-student curriculum learning.
        
        IMPORTANT: generate_sequences uses raw_prompt (messages) from non_tensor_batch,
        not input_ids from batch. So we need to generate mixed raw_prompt, not input_ids.
        
        Args:
            batch: The original batch with all input_ids variants (not modified)
            
        Returns:
            DataProto with expanded raw_prompt (messages) in non_tensor_batch if mixing is enabled,
            None otherwise. Does not modify the original batch.
        """
        curri_method = self.config.data.get("curri_method", None)
        if curri_method != "teacher_student":
            return None
        
        # Check if batch has the required input_ids variants
        batch_size = len(batch.batch)
        n = self.config.actor_rollout_ref.rollout.n
        
        # Get old_rewards if available (from batch or from self.old_rewards)
        old_rewards = None
        # if "old_rewards" in batch.batch:
        #     old_rewards = batch.batch["old_rewards"]
        
        # Get original raw_prompt (messages) from batch
        if "raw_prompt" not in batch.non_tensor_batch:
            return None
        
        original_raw_prompts = batch.non_tensor_batch["raw_prompt"]
        mixed_raw_prompts = None
        mixed_reward_models = None  # For v4: store mixed reward_models with correct ground_truth
        teacher_student_mixed_flags = None  # Mark which samples are subproblem/hint (True) vs original (False)
        mixed_subproblem_indices = None  # For v4: 0=原题, 1..4=subproblem1..4
        
        # Process different ts_version variants
        # Now we directly use raw_prompt_* fields from dataset instead of decoding input_ids
        if self.ts_version == "nurl":
            mixed_raw_prompts = []
            mixed_reward_models = []
            teacher_student_mixed_flags = []
            nurl_hints = []
            nurl_questions = []
            reward_models = batch.non_tensor_batch.get("reward_model", None)
            hints_arr = batch.non_tensor_batch.get("nurl_hint", None)
            questions_arr = batch.non_tensor_batch.get("nurl_question", None)
            for i in range(batch_size):
                original_messages = original_raw_prompts[i]
                original_reward_model = reward_models[i] if reward_models is not None else {}
                hint = str(hints_arr[i]) if hints_arr is not None and i < len(hints_arr) else ""
                question = str(questions_arr[i]) if questions_arr is not None and i < len(questions_arr) else ""
                mixed_raw_prompts.extend([original_messages] * n)
                mixed_reward_models.extend([original_reward_model.copy() for _ in range(n)])
                teacher_student_mixed_flags.extend([False] * n)
                nurl_hints.extend([hint] * n)
                nurl_questions.extend([question] * n)
        elif "raw_prompt_hint" in batch.non_tensor_batch:
            # v1/v2: Mix hint and no-hint prompts
            hint_n = n // 2
            no_hint_n = n - hint_n
            
            mixed_raw_prompts = []
            teacher_student_mixed_flags = []
            hint_prompts = batch.non_tensor_batch["raw_prompt_hint"]
            
            for i in range(batch_size):
                original_messages = original_raw_prompts[i]
                if old_rewards is not None and old_rewards[i].item() != 0.0:
                    # If old_reward != 0, use only no_hint (original messages)
                    current_messages = [original_messages] * n
                    current_flags = [False] * n  # All original (in_state_0)
                else:
                    # Mix hint and no_hint
                    hint_messages = hint_prompts[i]
                    current_messages = [hint_messages] * hint_n + [original_messages] * no_hint_n
                    current_flags = [True] * hint_n + [False] * no_hint_n  # hint=True, original=False
                
                mixed_raw_prompts.extend(current_messages)
                teacher_student_mixed_flags.extend(current_flags)
            
        elif "raw_prompt_crafted" in batch.non_tensor_batch:
            # v3: Mix crafted, lemmas, and raw prompts
            hint_n = 2
            no_hint_n = 8
            
            mixed_raw_prompts = []
            teacher_student_mixed_flags = []
            crafted_prompts = batch.non_tensor_batch["raw_prompt_crafted"]
            lemma1_prompts = batch.non_tensor_batch.get("raw_prompt_lemma1", None)
            lemma2_prompts = batch.non_tensor_batch.get("raw_prompt_lemma2", None)
            lemma3_prompts = batch.non_tensor_batch.get("raw_prompt_lemma3", None)
            
            for i in range(batch_size):
                original_messages = original_raw_prompts[i]
                if old_rewards is not None and old_rewards[i].item() != 0.0:
                    # If old_reward != 0, use only raw (original messages)
                    current_messages = [original_messages] * n
                    current_flags = [False] * n  # All original (in_state_0)
                else:
                    # Mix crafted, lemmas, and raw
                    crafted_messages = crafted_prompts[i]
                    lemma1_messages = lemma1_prompts[i] if lemma1_prompts is not None else original_messages
                    lemma2_messages = lemma2_prompts[i] if lemma2_prompts is not None else original_messages
                    lemma3_messages = lemma3_prompts[i] if lemma3_prompts is not None else original_messages
                    
                    current_messages = (
                        [crafted_messages] * hint_n +
                        [lemma1_messages] * hint_n +
                        [lemma2_messages] * hint_n +
                        [lemma3_messages] * hint_n +
                        [original_messages] * no_hint_n
                    )
                    current_flags = (
                        [True] * (hint_n * 4) +  # crafted, lemma1, lemma2, lemma3 are all True
                        [False] * no_hint_n  # original is False (in_state_0)
                    )
                
                mixed_raw_prompts.extend(current_messages)
                teacher_student_mixed_flags.extend(current_flags)
            
        elif "raw_prompt_subproblem1" in batch.non_tensor_batch:
            # v4: Mix subproblems and raw prompts. v4_subproblem_mode: 0=4orig+sub1,2,3,4各1; 1=4orig+4sub4; 2=8sub4
            sub_n = 1
            v4_mode = self.config.data.get("v4_subproblem_mode", 0)

            mixed_raw_prompts = []
            mixed_reward_models = []
            teacher_student_mixed_flags = []
            mixed_subproblem_indices = []
            sub1_prompts = batch.non_tensor_batch["raw_prompt_subproblem1"]
            sub2_prompts = batch.non_tensor_batch.get("raw_prompt_subproblem2", None)
            sub3_prompts = batch.non_tensor_batch.get("raw_prompt_subproblem3", None)
            sub4_prompts = batch.non_tensor_batch.get("raw_prompt_subproblem4", None)

            # Get reward_model from batch if it exists
            reward_models = batch.non_tensor_batch.get("reward_model", None)

            for i in range(batch_size):
                original_messages = original_raw_prompts[i]
                original_reward_model = reward_models[i] if reward_models is not None else {}

                if old_rewards is not None and old_rewards[i].item() != 0.0:
                    # If old_reward != 0, use only raw (original messages)
                    current_messages = [original_messages] * n
                    current_reward_models = [original_reward_model.copy() for _ in range(n)]
                    current_flags = [False] * n
                    current_subproblem_indices = [0] * n
                else:
                    sub1_messages = sub1_prompts[i]
                    sub2_messages = sub2_prompts[i] if sub2_prompts is not None else original_messages
                    sub3_messages = sub3_prompts[i] if sub3_prompts is not None else original_messages
                    sub4_messages = sub4_prompts[i] if sub4_prompts is not None else original_messages
                    gt_orig = original_reward_model.get("ground_truth", "")
                    gt_sub1 = original_reward_model.get("ground_truth_sub1", gt_orig)
                    gt_sub2 = original_reward_model.get("ground_truth_sub2", gt_orig)
                    gt_sub3 = original_reward_model.get("ground_truth_sub3", gt_orig)
                    gt_sub4 = original_reward_model.get("ground_truth_sub4", gt_orig)

                    if v4_mode == 1:
                        # MODE 1: 4 sub4 + 4 original (j=0..3 sub4, j=4..7 original)
                        current_messages = [sub4_messages] * 4 + [original_messages] * 4
                        current_reward_models = []
                        for j in range(n):
                            new_rm = {"style": "rule", "ground_truth": gt_sub4 if j < 4 else gt_orig}
                            current_reward_models.append(new_rm)
                        current_flags = [True] * 4 + [False] * 4
                        current_subproblem_indices = [4] * 4 + [0] * 4
                    elif v4_mode == 2:
                        # MODE 2: 8 sub4
                        current_messages = [sub4_messages] * n
                        current_reward_models = [{"style": "rule", "ground_truth": gt_sub4} for _ in range(n)]
                        current_flags = [True] * n
                        current_subproblem_indices = [4] * n
                    elif v4_mode == 3:
                        # MODE 3: 4 sub1 + 4 sub2 (j=0..3 sub1, j=4..7 sub2)
                        current_messages = [sub1_messages] * 4 + [sub2_messages] * 4
                        current_reward_models = []
                        for j in range(n):
                            new_rm = {"style": "rule", "ground_truth": gt_sub1 if j < 4 else gt_sub2}
                            current_reward_models.append(new_rm)
                        current_flags = [True] * n
                        current_subproblem_indices = [1] * 4 + [2] * 4
                    # MODE 4: 2 sub1 + 2 sub2 + 2 sub3 + 2 sub4 + 8 original
                    elif v4_mode == 4:
                        current_messages = [sub1_messages] * 2 + [sub2_messages] * 2 + [sub3_messages] * 2 + [sub4_messages] * 2 + [original_messages] * 8
                        current_reward_models = []
                        for j in range(n):
                            new_reward_model = {"style": "rule"}
                            if j < 2:
                                new_reward_model["ground_truth"] = gt_sub1
                            elif j < 4:
                                new_reward_model["ground_truth"] = gt_sub2
                            elif j < 6:
                                new_reward_model["ground_truth"] = gt_sub3
                            elif j < 8:
                                new_reward_model["ground_truth"] = gt_sub4
                            else:
                                new_reward_model["ground_truth"] = gt_orig
                            current_reward_models.append(new_reward_model)
                        current_flags = [True] * 2 + [True] * 2 + [True] * 2 + [True] * 2 + [False] * 8
                        current_subproblem_indices = [1] * 2 + [2] * 2 + [3] * 2 + [4] * 2 + [0] * 8
                    else:
                        # MODE 0 (default): 4 original + sub1,2,3,4 各1道
                        current_messages = (
                            [sub1_messages] * sub_n
                            + [sub2_messages] * sub_n
                            + [sub3_messages] * sub_n
                            + [sub4_messages] * sub_n
                            + [original_messages] * (n - 4 * sub_n)
                        )
                        current_reward_models = []
                        for j in range(n):
                            new_reward_model = {"style": "rule"}
                            if j < sub_n:
                                new_reward_model["ground_truth"] = gt_sub1
                            elif j < 2 * sub_n:
                                new_reward_model["ground_truth"] = gt_sub2
                            elif j < 3 * sub_n:
                                new_reward_model["ground_truth"] = gt_sub3
                            elif j < 4 * sub_n:
                                new_reward_model["ground_truth"] = gt_sub4
                            else:
                                new_reward_model["ground_truth"] = gt_orig
                            current_reward_models.append(new_reward_model)
                        current_flags = [True] * (4 * sub_n) + [False] * (n - 4 * sub_n)
                        current_subproblem_indices = [1] * sub_n + [2] * sub_n + [3] * sub_n + [4] * sub_n + [0] * (n - 4 * sub_n)

                mixed_raw_prompts.extend(current_messages)
                mixed_reward_models.extend(current_reward_models)
                teacher_student_mixed_flags.extend(current_flags)
                mixed_subproblem_indices.extend(current_subproblem_indices)
            
        elif "raw_prompt_subproblems" in batch.non_tensor_batch:
            subproblems_prompts = batch.non_tensor_batch["raw_prompt_subproblems"]
            reward_models = batch.non_tensor_batch.get("reward_model", None)

            # v7 supports richer per-rollout mixing. Keep v5 behavior unchanged.
            if self.ts_version in ("v7", "v8"):
                v7_t_mix_mode = str(self.config.data.get("v7_t_mix_mode", "legacy_sub8")).lower()
                v7_prompt_mode = str(self.config.data.get("v7_prompt_mode", "explicit_t")).lower()
                v7_1_problem_match = str(self.config.data.get("v7_1_problem_match", "v7")).lower()
                # v8 + mix44 curriculum level control:
                # data.v7_curri_level=j means mix44 curriculum branch uses t=j (j in 1..4).
                # Default j=4 preserves original behavior.
                v7_curri_level_cfg = self.config.data.get("v7_curri_level", 4)
                try:
                    v7_curri_level = int(v7_curri_level_cfg)
                except (TypeError, ValueError):
                    v7_curri_level = 4
                v7_curri_level = max(1, min(4, v7_curri_level))

                mixed_raw_prompts = []
                mixed_reward_models = []
                teacher_student_mixed_flags = []
                v7_t_values = []
                v7_prompt_mode_values = []
                v8_mix_group_values = []

                prompt_keys = {
                    1: "raw_prompt_subproblems_t1_unified" if v7_prompt_mode == "unified" else "raw_prompt_subproblems_t1",
                    2: "raw_prompt_subproblems_t2_unified" if v7_prompt_mode == "unified" else "raw_prompt_subproblems_t2",
                    3: "raw_prompt_subproblems_t3_unified" if v7_prompt_mode == "unified" else "raw_prompt_subproblems_t3",
                    4: "raw_prompt_subproblems_t4_unified" if v7_prompt_mode == "unified" else "raw_prompt_subproblems_t4",
                }

                prompt_arrays = {}
                for t in [1, 2, 3, 4]:
                    key = prompt_keys[t]
                    prompt_arrays[t] = batch.non_tensor_batch.get(key, None)
                # Optional single-question original-template prompts for mix44444.
                prompt_q_orig_arrays = {
                    1: batch.non_tensor_batch.get("raw_prompt_subproblems_q1_orig", None),
                    2: batch.non_tensor_batch.get("raw_prompt_subproblems_q2_orig", None),
                    3: batch.non_tensor_batch.get("raw_prompt_subproblems_q3_orig", None),
                    4: batch.non_tensor_batch.get("raw_prompt_subproblems_q4_orig", None),
                }

                for i in range(batch_size):
                    original_messages = original_raw_prompts[i]
                    current_problem_id = None
                    if "problem_id" in batch.non_tensor_batch:
                        try:
                            current_problem_id = str(batch.non_tensor_batch["problem_id"][i])
                        except Exception:
                            current_problem_id = None
                    original_reward_model = reward_models[i] if reward_models is not None else {}
                    if not isinstance(original_reward_model, dict):
                        original_reward_model = (
                            original_reward_model.__dict__ if hasattr(original_reward_model, "__dict__") else {}
                        )

                    if v7_t_mix_mode == "balanced_2222" and n == 8:
                        t_plan = [4, 4, 3, 3, 2, 2, 1, 1]
                    elif v7_t_mix_mode == "balanced_2222":
                        base = n // 4
                        rem = n % 4
                        t_plan = [4] * base + [3] * base + [2] * base + [1] * base
                        t_plan.extend([4, 3, 2, 1][:rem])
                    elif v7_t_mix_mode == "mix62" and n == 8:
                        # Six t=4 samples + two t=1 samples.
                        t_plan = [4, 4, 4, 4, 4, 4, 1, 1]
                    elif v7_t_mix_mode == "mix62":
                        # Generalize by ratio t4:t1 = 3:1
                        base = n // 4
                        rem = n % 4
                        t_plan = [4] * (3 * base) + [1] * base
                        t_plan.extend([4, 4, 4, 1][:rem])
                    elif v7_t_mix_mode == "mix44" and n == 8:
                        # Four curriculum samples + four original samples.
                        # For v8 adaptive mode, curriculum t is per-problem state.
                        # Use t=0 as an internal sentinel for original samples.
                        if self.ts_version == "v8" and self.v8_use_adaptive and current_problem_id is not None:
                            adaptive_t = int(self._v8_problem_t_state.get(current_problem_id, 4))
                            adaptive_t = max(1, min(4, adaptive_t))
                            t_plan = [adaptive_t, adaptive_t, adaptive_t, adaptive_t, 0, 0, 0, 0]
                        else:
                            mix44_curr_t = v7_curri_level if self.ts_version == "v8" else 4
                            t_plan = [mix44_curr_t, mix44_curr_t, mix44_curr_t, mix44_curr_t, 0, 0, 0, 0]
                    elif v7_t_mix_mode == "mix44":
                        # Generalize by ratio t_curriculum:orig = 1:1 (using t=0 for original).
                        base = n // 2
                        rem = n % 2
                        if self.ts_version == "v8" and self.v8_use_adaptive and current_problem_id is not None:
                            adaptive_t = int(self._v8_problem_t_state.get(current_problem_id, 4))
                            adaptive_t = max(1, min(4, adaptive_t))
                            t_plan = [adaptive_t] * base + [0] * base
                            t_plan.extend([adaptive_t, 0][:rem])
                        else:
                            mix44_curr_t = v7_curri_level if self.ts_version == "v8" else 4
                            t_plan = [mix44_curr_t] * base + [0] * base
                            t_plan.extend([mix44_curr_t, 0][:rem])
                    elif v7_t_mix_mode == "balanced_11114" and n == 8:
                        # One sample for t=4/3/2/1 plus four GRPO-style original samples.
                        # Use t=0 as an internal sentinel for original samples.
                        t_plan = [4, 3, 2, 1, 0, 0, 0, 0]
                    elif v7_t_mix_mode == "balanced_11114":
                        # Generalize by ratio t4:t3:t2:t1:orig = 1:1:1:1:4
                        base = n // 8
                        rem = n % 8
                        t_plan = [4] * base + [3] * base + [2] * base + [1] * base + [0] * (4 * base)
                        t_plan.extend([4, 3, 2, 1, 0, 0, 0, 0][:rem])
                    elif v7_t_mix_mode == "mix44444" and n == 20:
                        # v8 mix44444:
                        # - 4x t4 curriculum samples (mixed=True)
                        # - 4x q1 + 4x q2 + 4x q3 + 4x q4 single-question samples (mixed=False)
                        # Use negative sentinels -1..-4 for q1..q4 single-question branches.
                        if self.ts_version == "v8" and self.v8_mix44444_adaptive and current_problem_id is not None:
                            q_state = self._v8_mix44444_q_active.get(current_problem_id, None)
                            if not isinstance(q_state, dict):
                                q_state = {1: True, 2: True, 3: True}
                                self._v8_mix44444_q_active[current_problem_id] = q_state
                            active_q = [q for q in [1, 2, 3] if bool(q_state.get(q, True))]
                            inactive_q_count = 3 - len(active_q)
                            # Replace each removed qi block with 2x t4 + 2x q4.
                            t4_count = 4 + 2 * inactive_q_count
                            q4_count = 4 + 2 * inactive_q_count
                            t_plan = [4] * t4_count
                            for q in [1, 2, 3]:
                                if q in active_q:
                                    t_plan.extend([-q] * 4)
                            t_plan.extend([-4] * q4_count)
                        else:
                            t_plan = [4, 4, 4, 4] + ([-1] * 4) + ([-2] * 4) + ([-3] * 4) + ([-4] * 4)
                    elif v7_t_mix_mode == "mix44444":
                        # Generalized ratio 1:1:1:1:1 across [t4, q1, q2, q3, q4].
                        base = n // 5
                        rem = n % 5
                        t_plan = [4] * base + ([-1] * base) + ([-2] * base) + ([-3] * base) + ([-4] * base)
                        t_plan.extend([4, -1, -2, -3, -4][:rem])
                    else:
                        # Legacy: all v7 mixed samples use full 4-problem prompt.
                        t_plan = [4] * n

                    gt_orig = original_reward_model.get("ground_truth", "")
                    gt_sub1 = original_reward_model.get("ground_truth_sub1", gt_orig)
                    gt_sub2 = original_reward_model.get("ground_truth_sub2", gt_orig)
                    gt_sub3 = original_reward_model.get("ground_truth_sub3", gt_orig)
                    gt_sub4 = original_reward_model.get("ground_truth_sub4", gt_orig)
                    gt_all = [gt_sub1, gt_sub2, gt_sub3, gt_sub4]

                    for t in t_plan:
                        t = int(t)
                        mix_group_id = 0
                        stored_t = t
                        if t <= -1:
                            # mix44444 single-question GRPO branches: q1..q4.
                            abs_q = min(4, max(1, -t))
                            prompt_arr_q = prompt_q_orig_arrays.get(abs_q, None)
                            if prompt_arr_q is not None:
                                current_messages = prompt_arr_q[i]
                            else:
                                current_messages = original_messages
                            current_flag = False
                            rm_variant = original_reward_model.copy()
                            rm_variant["ground_truth"] = gt_all[abs_q - 1]
                            rm_variant["v8_mix44444_group"] = abs_q
                            prompt_mode_value = "grpo"
                            mix_group_id = abs_q
                            stored_t = 0
                        elif t == 0:
                            # Explicit original branch (used by balanced_11114).
                            current_messages = original_messages
                            current_flag = False
                            rm_variant = original_reward_model.copy()
                            prompt_mode_value = "grpo"
                            mix_group_id = 4  # original/full question bucket
                        elif (
                            t == 1
                            and v7_1_problem_match == "grpo"
                            and v7_t_mix_mode != "balanced_11114"
                            and not (self.ts_version == "v8" and self.v8_use_adaptive)
                            and not (
                                self.ts_version == "v8"
                                and v7_t_mix_mode == "mix44"
                                and v7_curri_level == 1
                            )
                        ):
                            # Match GRPO distribution for the 1-problem branch.
                            current_messages = original_messages
                            current_flag = False
                            rm_variant = original_reward_model.copy()
                            prompt_mode_value = "grpo"
                            mix_group_id = 4
                        else:
                            prompt_arr = prompt_arrays.get(t, None)
                            if prompt_arr is not None:
                                current_messages = prompt_arr[i]
                            else:
                                current_messages = subproblems_prompts[i]
                            current_flag = True
                            prompt_mode_value = v7_prompt_mode

                            q_start = 5 - t  # original subproblem index start (1-based)
                            selected_gt = gt_all[q_start - 1:4]
                            rm_variant = original_reward_model.copy()
                            for local_idx in [1, 2, 3, 4]:
                                if local_idx <= t:
                                    rm_variant[f"ground_truth_sub{local_idx}"] = selected_gt[local_idx - 1]
                                else:
                                    rm_variant[f"ground_truth_sub{local_idx}"] = ""
                            rm_variant["num_problems"] = t
                            rm_variant["v7_q_start"] = q_start
                            rm_variant["v7_t"] = t
                            rm_variant["v7_prompt_mode"] = prompt_mode_value
                            rm_variant["ground_truth"] = selected_gt[-1] if len(selected_gt) > 0 else gt_orig

                        mixed_raw_prompts.append(current_messages)
                        mixed_reward_models.append(rm_variant)
                        teacher_student_mixed_flags.append(current_flag)
                        v7_t_values.append(stored_t)
                        v7_prompt_mode_values.append(prompt_mode_value)
                        v8_mix_group_values.append(mix_group_id)
            else:
                # v5: Mix subproblems (all-in-one) and raw prompts
                sub_n = 8

                mixed_raw_prompts = []
                mixed_reward_models = []
                teacher_student_mixed_flags = []

                for i in range(batch_size):
                    original_messages = original_raw_prompts[i]
                    original_reward_model = reward_models[i] if reward_models is not None else {}

                    if old_rewards is not None and old_rewards[i].item()!= 0.0:
                        # If old_reward == 1, use only raw (original messages)
                        current_messages = [original_messages] * n
                        current_flags = [False] * n  # All original (in_state_0)
                    else:
                        # Mix subproblems and raw
                        subproblems_messages = subproblems_prompts[i]
                        current_messages = [subproblems_messages] * sub_n + [original_messages] * (n - sub_n)
                        current_flags = [True] * sub_n + [False] * (n - sub_n)  # subproblems=True, original=False

                    # For v5, all rollouts use the same reward_model (with all ground_truth_subX fields)
                    # This is used by truncate_response_by_subproblem_correctness to check subproblem correctness
                    # No need to set different ground_truth for different rollouts
                    current_reward_models = [original_reward_model.copy() for _ in range(n)]

                    mixed_raw_prompts.extend(current_messages)
                    mixed_reward_models.extend(current_reward_models)
                    teacher_student_mixed_flags.extend(current_flags)
        
        # If mixing was performed, create and return a DataProto with the mixed raw_prompt
        if mixed_raw_prompts is not None:
            
            # Check if we have mixed_reward_models (for v4)
            non_tensor_batch_dict = {
                "raw_prompt": np.array(mixed_raw_prompts, dtype=object)
            }
            
            # Add mixed_reward_models if it exists (for v4 with subproblems)
            if mixed_reward_models is not None and len(mixed_reward_models) > 0:
                non_tensor_batch_dict["reward_model"] = np.array(mixed_reward_models, dtype=object)
            
            # Add teacher_student_mixed flags: True for subproblem/hint, False for original (in_state_0)
            if teacher_student_mixed_flags is not None and len(teacher_student_mixed_flags) > 0:
                non_tensor_batch_dict["teacher_student_mixed"] = np.array(teacher_student_mixed_flags, dtype=bool)
            # v4: subproblem_index 0=原题, 1..4=subproblem1..4
            if mixed_subproblem_indices is not None and len(mixed_subproblem_indices) > 0:
                non_tensor_batch_dict["subproblem_index"] = np.array(mixed_subproblem_indices, dtype=np.int32)
            if self.ts_version in ("v7", "v8"):
                if 'v7_t_values' in locals() and len(v7_t_values) > 0:
                    non_tensor_batch_dict["v7_t"] = np.array(v7_t_values, dtype=np.int32)
                if 'v7_prompt_mode_values' in locals() and len(v7_prompt_mode_values) > 0:
                    non_tensor_batch_dict["v7_prompt_mode"] = np.array(v7_prompt_mode_values, dtype=object)
                if 'v8_mix_group_values' in locals() and len(v8_mix_group_values) > 0:
                    non_tensor_batch_dict["v8_mix_group"] = np.array(v8_mix_group_values, dtype=np.int32)
            if self.ts_version == "nurl":
                if "nurl_hints" in locals() and len(nurl_hints) > 0:
                    non_tensor_batch_dict["nurl_hint"] = np.array(nurl_hints, dtype=object)
                if "nurl_questions" in locals() and len(nurl_questions) > 0:
                    non_tensor_batch_dict["nurl_question"] = np.array(nurl_questions, dtype=object)
            
            # Add problem_id: expand original problem_id to match mixed_raw_prompts order
            if "problem_id" in batch.non_tensor_batch:
                problem_ids = batch.non_tensor_batch["problem_id"]
                mixed_problem_ids = []
                for i in range(batch_size):
                    original_problem_id = problem_ids[i]
                    # Repeat problem_id n times to match mixed_raw_prompts
                    mixed_problem_ids.extend([original_problem_id] * n)
                non_tensor_batch_dict["problem_id"] = np.array(mixed_problem_ids, dtype=object)
            
            mixed_data = DataProto(
                batch=None,
                non_tensor_batch=non_tensor_batch_dict,
                meta_info={"teacher_student_mixed": True}
            )
            # Remove None values
            mixed_data.non_tensor_batch = {k: v for k, v in mixed_data.non_tensor_batch.items() if v is not None}
            
            return mixed_data
        
        return None

    def _apply_nurl_hint_to_repeated_prompts(
        self, repeated_gen_batch: DataProto, hard_mask: torch.Tensor, rollout_n: int
    ) -> int:
        """Inject NuRL offline hint for hard groups into first (rollout_n-1) rollouts in-place.

        Returns the number of modified rollouts.
        """
        if "raw_prompt" not in repeated_gen_batch.non_tensor_batch:
            return 0
        hints = repeated_gen_batch.non_tensor_batch.get("nurl_hint", None)
        if hints is None:
            return 0
        raw_prompts = repeated_gen_batch.non_tensor_batch["raw_prompt"]
        modified = 0
        modified_indices = []
        hard_list = hard_mask.detach().cpu().tolist()
        group_size = int(max(1, rollout_n))
        for sample_idx, is_hard in enumerate(hard_list):
            if not bool(is_hard):
                continue
            base_idx = sample_idx * group_size
            if base_idx >= len(hints):
                continue
            hint = str(hints[base_idx]) if hints[base_idx] is not None else ""
            if not hint.strip():
                continue
            append_text = f"\n\nYou might find the following hint helpful:\n{hint}"
            for ridx in range(max(0, group_size - 1)):
                idx = base_idx + ridx
                if idx >= len(raw_prompts):
                    break
                msgs = raw_prompts[idx]
                if isinstance(msgs, np.ndarray):
                    msgs = msgs.tolist()
                if isinstance(msgs, tuple):
                    msgs = list(msgs)
                if not isinstance(msgs, list):
                    continue
                new_msgs = deepcopy(msgs)
                user_pos = None
                for j in range(len(new_msgs) - 1, -1, -1):
                    m = new_msgs[j]
                    if isinstance(m, dict) and m.get("role") == "user":
                        user_pos = j
                        break
                if user_pos is None:
                    continue
                user_content = str(new_msgs[user_pos].get("content", ""))
                if append_text in user_content:
                    continue
                new_msgs[user_pos] = {**new_msgs[user_pos], "content": user_content + append_text}
                raw_prompts[idx] = new_msgs
                modified += 1
                modified_indices.append(idx)
        repeated_gen_batch.non_tensor_batch["raw_prompt"] = raw_prompts
        if modified_indices:
            self._retokenize_modified_nurl_prompts(repeated_gen_batch, modified_indices)
        return modified

    def _retokenize_modified_nurl_prompts(self, repeated_gen_batch: DataProto, modified_indices: list[int]) -> None:
        """Re-tokenize modified NuRL prompts in-place so second rollout uses injected hints.

        This runs only in ts_version=nurl path and updates prompt-side tensors:
        prompts/input_ids/attention_mask/position_ids (+ raw_prompt_ids if present).
        """
        if "raw_prompt" not in repeated_gen_batch.non_tensor_batch:
            return
        if repeated_gen_batch.batch is None:
            return
        if len(modified_indices) == 0:
            return

        # Keep indices deterministic and unique for stable debugging.
        modified_indices = sorted(set(int(i) for i in modified_indices if i is not None))

        max_prompt_length = int(self.config.data.max_prompt_length)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        raw_prompts = repeated_gen_batch.non_tensor_batch["raw_prompt"]

        for idx in modified_indices:
            if idx < 0 or idx >= len(raw_prompts):
                continue

            msgs = raw_prompts[idx]
            if isinstance(msgs, np.ndarray):
                msgs = msgs.tolist()
            if isinstance(msgs, tuple):
                msgs = list(msgs)
            if not isinstance(msgs, list):
                continue

            try:
                prompt_ids = self.tokenizer.apply_chat_template(
                    msgs,
                    tokenize=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                ).squeeze(0).to(torch.long)
            except Exception:
                continue

            # Keep the tail to preserve generation prompt and injected hint if overlong.
            if prompt_ids.numel() > max_prompt_length:
                prompt_ids = prompt_ids[-max_prompt_length:]

            def _build_left_padded(width: int):
                width = int(max(1, width))
                ids = prompt_ids
                if ids.numel() > width:
                    ids = ids[-width:]
                seq_len_local = int(ids.numel())
                pad_len_local = max(0, width - seq_len_local)
                if pad_len_local > 0:
                    pad_local = torch.full((pad_len_local,), int(pad_id), dtype=torch.long)
                    padded_ids_local = torch.cat([pad_local, ids], dim=0)
                else:
                    padded_ids_local = ids
                attn_local = torch.zeros(width, dtype=torch.long)
                pos_local = torch.zeros(width, dtype=torch.long)
                if seq_len_local > 0:
                    attn_local[-seq_len_local:] = 1
                    pos_local[-seq_len_local:] = torch.arange(seq_len_local, dtype=torch.long)
                return padded_ids_local, attn_local, pos_local, seq_len_local

            # Base prompt-width tensors (legacy behavior).
            padded_prompt_ids, new_attention_mask, new_position_ids, seq_len = _build_left_padded(max_prompt_length)

            # Update tensor batch fields used by rollout engines.
            if "prompts" in repeated_gen_batch.batch:
                dst = repeated_gen_batch.batch["prompts"]
                width = int(dst.shape[-1]) if dst.dim() >= 2 else max_prompt_length
                ids_w, _, _, _ = _build_left_padded(width)
                repeated_gen_batch.batch["prompts"][idx] = ids_w.to(device=dst.device, dtype=dst.dtype)
            if "input_ids" in repeated_gen_batch.batch:
                dst = repeated_gen_batch.batch["input_ids"]
                width = int(dst.shape[-1]) if dst.dim() >= 2 else max_prompt_length
                ids_w, _, _, _ = _build_left_padded(width)
                repeated_gen_batch.batch["input_ids"][idx] = ids_w.to(device=dst.device, dtype=dst.dtype)
            if "attention_mask" in repeated_gen_batch.batch:
                dst = repeated_gen_batch.batch["attention_mask"]
                width = int(dst.shape[-1]) if dst.dim() >= 2 else max_prompt_length
                _, attn_w, _, _ = _build_left_padded(width)
                repeated_gen_batch.batch["attention_mask"][idx] = attn_w.to(device=dst.device, dtype=dst.dtype)
            if "position_ids" in repeated_gen_batch.batch:
                dst = repeated_gen_batch.batch["position_ids"]
                # Some models may carry 2D/3D position ids; only update compatible 1D per sample.
                if dst[idx].dim() == 1:
                    width = int(dst.shape[-1]) if dst.dim() >= 2 else max_prompt_length
                    _, _, pos_w, _ = _build_left_padded(width)
                    repeated_gen_batch.batch["position_ids"][idx] = pos_w.to(device=dst.device, dtype=dst.dtype)

            # Keep non-tensor raw prompt ids for downstream debug/inspection if present.
            if "raw_prompt_ids" in repeated_gen_batch.non_tensor_batch:
                try:
                    arr = repeated_gen_batch.non_tensor_batch["raw_prompt_ids"]
                    if isinstance(arr, np.ndarray):
                        arr = arr.tolist()
                    arr[idx] = prompt_ids.tolist()
                    repeated_gen_batch.non_tensor_batch["raw_prompt_ids"] = np.array(arr, dtype=object)
                except Exception:
                    pass

    def _init_teacher_model(self):
        """Initialize teacher model hint generator."""
        from verl.trainer.ppo.own_utils import BatchHintGenerator, DynamicVLLMTeacherGenerator
        import logging
        logger = logging.getLogger(__name__)
        
        if self.teacher_model == "api":
            api_key = self.config.data.get("teacher_api_key", None)
            if api_key is None:
                logger.warning("teacher_model is 'api' but teacher_api_key not provided, skipping hint generation")
                return
            max_workers = self.config.data.get("teacher_max_workers", 16)
            self.hint_generator = BatchHintGenerator(api_key=api_key, max_workers=max_workers)
            print(f"Initialized BatchHintGenerator with max_workers={max_workers}")
        elif self.teacher_model == "local":
            model_path = self.config.data.get("teacher_model_path", None)
            if model_path is None:
                logger.warning("teacher_model is 'local' but teacher_model_path not provided, skipping hint generation")
                return
            tensor_parallel_size = self.config.data.get("teacher_tensor_parallel_size", 4)
            gpu_ids = self.config.data.get("teacher_gpu_ids", [4, 5, 6, 7])
            self.hint_generator = DynamicVLLMTeacherGenerator.remote(
                model_path=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_ids=gpu_ids
            )
            print(f"Initialized DynamicVLLMTeacherGenerator with model_path={model_path}, tensor_parallel_size={tensor_parallel_size}, gpu_ids={gpu_ids}")
        else:
            logger.warning(f"teacher_model '{self.teacher_model}' not implemented, skipping hint generation")

    def _update_old_rewards(self, batch: DataProto, reward_tensor: torch.Tensor):
        """Update old_rewards based on current batch rewards.
        
        This matches the logic in mix_trainer.py:
        - Reshape rewards to (batch_size, n) where batch_size is the number of unique problems
        - Sum rewards across n rollouts for each problem
        """
        if "problem_id" not in batch.non_tensor_batch:
            return
        
        # Group rewards by uid (each uid corresponds to one problem with n rollouts)
        uids = batch.non_tensor_batch.get("uid", None)
        if uids is None:
            return
        
        problem_ids = batch.non_tensor_batch["problem_id"]
        current_rewards = reward_tensor.sum(-1)
        n = self.config.actor_rollout_ref.rollout.n
        
        # Get unique uids and problem_ids
        uids_list = uids.tolist()
        unique_uids_in_order = list(dict.fromkeys(uids_list))
        unique_uids = np.array(unique_uids_in_order, dtype=object)
        unique_problem_ids = [problem_ids[uids == uid][0] for uid in unique_uids]
        
        # Initialize old_rewards if needed (dynamic size by observed problem_id).
        if self.old_rewards is None:
            max_problem_id = max(int(pid) for pid in unique_problem_ids)
            self.old_rewards = torch.zeros(max_problem_id + 1, device=reward_tensor.device, dtype=reward_tensor.dtype)
        
        # Update old_rewards for each problem_id
        # Similar to mix_trainer.py: reshaped_rewards.sum(dim=1)
        for i, uid in enumerate(unique_uids):
            uid_mask = uids == uid
            uid_rewards = current_rewards[uid_mask]
            current_problem_id = str(unique_problem_ids[i])
            pid_int = int(current_problem_id)
            
            # Extend old_rewards if needed
            if pid_int >= len(self.old_rewards):
                new_size = pid_int + 1
                new_old_rewards = torch.zeros(new_size, device=self.old_rewards.device, dtype=self.old_rewards.dtype)
                new_old_rewards[:len(self.old_rewards)] = self.old_rewards
                self.old_rewards = new_old_rewards
            
            # Sum rewards for this problem (n rollouts per problem)
            old_reward_value = uid_rewards.sum()
            self.old_rewards[pid_int] = old_reward_value

    def _save_rollout_samples(self, batch: DataProto, reward_tensor: torch.Tensor, epoch: int):
        """Save rollout samples to specified directory."""
        from verl.trainer.ppo.own_utils import save_problem_rollouts
        output_dir = self.config.trainer.get("rollout_samples_dir", None)
        if output_dir is None:
            return
        if "problem_id" not in batch.non_tensor_batch:
            return
        
        success_value = 1
        
        problem_ids = batch.non_tensor_batch["problem_id"]
        uids = batch.non_tensor_batch.get("uid", None)
        
        if uids is None:
            return
        
        # Get unique uids and problem_ids
        uids_list = uids.tolist()
        unique_uids_in_order = list(dict.fromkeys(uids_list))
        unique_uids = np.array(unique_uids_in_order, dtype=object)
        unique_problem_ids = [problem_ids[uids == uid][0] for uid in unique_uids]
        
        # Create output directory (matching mix_trainer.py format)
        epoch_output_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
        
        # Save samples for each problem
        for i, uid in enumerate(unique_uids):
            uid_mask = uids == uid
            uid_rewards = reward_tensor[uid_mask].sum(-1)
            current_problem_id = str(unique_problem_ids[i])
            save_problem_rollouts(
                output_dir=epoch_output_dir,
                current_problem_id=current_problem_id,
                uid_rewards=uid_rewards,
                problem_ids=problem_ids.tolist(),
                success_value=success_value,
                batch=batch,
                tokenizer=self.tokenizer
            )

    def _update_student_answer_history(self, batch: DataProto, reward_tensor: torch.Tensor):
        """Update student answer history for teacher_student curriculum learning."""
        from verl.trainer.ppo.own_utils import current_student_answer
        
        if "problem_id" not in batch.non_tensor_batch:
            return
        
        problem_ids = batch.non_tensor_batch["problem_id"]
        uids = batch.non_tensor_batch.get("uid", None)
        
        if uids is None:
            return
        
        # Get unique uids and problem_ids
        uids_list = uids.tolist()
        unique_uids_in_order = list(dict.fromkeys(uids_list))
        unique_uids = np.array(unique_uids_in_order, dtype=object)
        unique_problem_ids = [problem_ids[uids == uid][0] for uid in unique_uids]
        
        # Update student_answer_history for each problem
        for i, uid in enumerate(unique_uids):
            uid_mask = uids == uid
            uid_rewards = reward_tensor[uid_mask].sum(-1)
            current_problem_id = str(unique_problem_ids[i])
            
            self.student_answer_history[current_problem_id] = current_student_answer(
                current_problem_id=current_problem_id,
                uid_rewards=uid_rewards,
                problem_ids=problem_ids.tolist(),
                batch=batch,
                tokenizer=self.tokenizer
            )

    def _get_in_state_0_indices(self, batch: DataProto, uid_mask: torch.Tensor, current_answer: dict = None) -> int:
        """
        Identify the starting index of in_state_0 samples (samples using original input_ids).
        
        In teacher_student setting, a group contains multiple rollouts with different prompts.
        The samples using original input_ids typically have the shortest prompt length and are placed at the end.
        This function finds the starting index by comparing prompt lengths, matching the logic in mix_trainer.py.
        
        Args:
            batch: DataProto containing batch data
            uid_mask: Boolean mask indicating which samples belong to the current uid/group
            current_answer: Optional dict from current_student_answer (for compatibility with mix_trainer.py logic)
        
        Returns:
            int: Starting index of in_state_0 samples (all samples from this index onwards are in_state_0)
        """
        # If current_answer is provided, use the same logic as mix_trainer.py
        if current_answer is not None:
            length_list = [current_answer.get(f"rollout_{j+1}", {}).get("data", {}).get("prompts", {}).get("length", 0) 
                          for j in range(len(current_answer))]
            if len(length_list) == 0:
                return 0
            len_min = length_list[-1]
            tmp_k = len(length_list) - length_list.count(len_min)
            # Match mix_trainer.py logic exactly: if tmp_k != 0, set to 8
            # Note: This seems to be a hardcoded value in mix_trainer.py
            if tmp_k != 0:
                tmp_k = 8
            return tmp_k
        
        # Fallback: compute from batch directly
        if "prompts" not in batch.batch:
            return 0
        
        # Get prompt lengths for all samples in this group
        prompts = batch.batch["prompts"][uid_mask]
        prompt_lengths = []
        for prompt in prompts:
            # Count non-padding tokens
            if self.tokenizer.pad_token_id is not None:
                length = (prompt != self.tokenizer.pad_token_id).sum().item()
            else:
                length = prompt.numel()
            prompt_lengths.append(length)
        
        if len(prompt_lengths) == 0:
            return 0
        
        # Match mix_trainer.py logic: len_min = length_list[-1] (last sample's length)
        len_min = prompt_lengths[-1]
        
        # Count how many samples have length == len_min
        # tmp_k is the index where samples with length == len_min start
        # tmp_k = len(uid_rewards) - length_list.count(len_min)
        tmp_k = len(prompt_lengths) - prompt_lengths.count(len_min)
        
        # Fallback logic from mix_trainer.py: if tmp_k != 0, set to n_samples
        if tmp_k != 0:
            n = self.config.actor_rollout_ref.rollout.n
            tmp_k = n  # Use n_samples as fallback (mix_trainer.py uses 8, but we use n)
        
        return tmp_k

    def _compute_teacher_student_metrics(self, batch: DataProto, reward_tensor: torch.Tensor, metrics: dict):
        """
        Compute teacher_student specific metrics including in_state_0 metrics.
        
        This function computes:
        - batch/solved_hard: average reward for hard problems
        - batch/solved_medium: average reward for medium problems
        - batch/solved_hard_in_state_0: average reward for hard problems in state_0
        - batch/solved_medium_in_state_0: average reward for medium problems in state_0
        - global/total_solved_in_state_0: total problems solved in state_0
        - global/total_solved_in_state_0_hard: total hard problems solved in state_0
        - global/total_solved_in_state_0_medium: total medium problems solved in state_0
        - batch/newly_solved_in_state_0: newly solved problems in state_0 in this batch
        - batch/solved_in_state_0: total solved problems in state_0 in this batch
        """
        
        if "problem_id" not in batch.non_tensor_batch:
            return
        
        problem_ids = batch.non_tensor_batch["problem_id"]
        uids = batch.non_tensor_batch.get("uid", None)
        
        if uids is None:
            return
        
        # Get unique uids and problem_ids
        uids_list = uids.tolist()
        unique_uids_in_order = list(dict.fromkeys(uids_list))
        unique_uids = np.array(unique_uids_in_order, dtype=object)
        unique_problem_ids = [problem_ids[uids == uid][0] for uid in unique_uids]
        
        # Determine success and fail values based on reward implementation version
        reward_impl_version = self.config.data.get("reward_impl_version", 0)
        if reward_impl_version == 0:
            fail_value = 0
            success_value = 1
        elif reward_impl_version == 1:
            fail_value = -0.5
            success_value = 1
        elif reward_impl_version in [2, 3, 4]:
            fail_value = 0
            success_value = 1
        else:
            fail_value = 0
            success_value = 1
        
        batch_newly_solved_count = 0
        solved_in_state_0_count = 0
        solve_all_count = 0  # Count of problem groups where all rollouts succeeded
        solve_none_count = 0  # Count of problem groups where no rollouts succeeded
        
        # Process each problem
        for i, uid in enumerate(unique_uids):
            uid_mask = uids == uid
            uid_rewards = reward_tensor[uid_mask].sum(-1)
            current_problem_id = str(unique_problem_ids[i])
            
            # Get in_state_0 rewards using teacher_student_mixed flag if available
            # teacher_student_mixed=False means in_state_0 (original problem)
            # Note: in_state_0 samples may not be contiguous in the group due to shuffling
            if "teacher_student_mixed" in batch.non_tensor_batch:
                teacher_student_mixed_flags = batch.non_tensor_batch["teacher_student_mixed"][uid_mask]
                # Find rewards where teacher_student_mixed is False (in_state_0)
                in_state_0_mask = ~teacher_student_mixed_flags  # False means in_state_0
                # Directly select rewards using boolean mask (indices may be non-contiguous)
                uid_rewards_in_state_0 = uid_rewards[in_state_0_mask]
            else:
                # Fallback: use old logic with current_student_answer and tmp_k
                from verl.trainer.ppo.own_utils import current_student_answer
                current_answer = current_student_answer(
                    current_problem_id=current_problem_id,
                    uid_rewards=uid_rewards,
                    problem_ids=problem_ids.tolist(),
                    batch=batch,
                    tokenizer=self.tokenizer
                )
                tmp_k = self._get_in_state_0_indices(batch, uid_mask, current_answer=current_answer)
                uid_rewards_in_state_0 = uid_rewards[tmp_k:]
            
            # Update current_reward_dict_avg_in_state_0
            if len(uid_rewards_in_state_0) == 0:
                self.current_reward_dict_avg_in_state_0[current_problem_id] = 0.0
            else:
                self.current_reward_dict_avg_in_state_0[current_problem_id] = uid_rewards_in_state_0.sum().item() / len(uid_rewards_in_state_0)
            
            # Check if solved in state_0
            # Use >= instead of == to handle floating point precision issues
            # Also check if reward is close to success_value (within 1e-6 tolerance)
            tolerance = 1e-6
            if len(uid_rewards_in_state_0) > 0:
                # Check if any reward is >= success_value (with tolerance for floating point)
                is_once_solved = (uid_rewards_in_state_0 >= success_value - tolerance).any()
            else:
                # If uid_rewards_in_state_0 is empty, check all uid_rewards as fallback
                # This handles cases where tmp_k calculation might be incorrect
                is_once_solved = False
                # if len(uid_rewards) > 0:
                #     is_once_solved = (uid_rewards >= success_value - tolerance).any()
                # else:
                #     is_once_solved = False
            
            if is_once_solved:
                solved_in_state_0_count += 1
                if current_problem_id not in self.problems_solved_in_state_0:
                    batch_newly_solved_count += 1
                self.problems_solved_in_state_0.add(current_problem_id)
                if int(current_problem_id) in self.hard_problem_list:
                    self.problems_solved_in_state_0_hard.add(current_problem_id)
                else:
                    self.problems_solved_in_state_0_medium.add(current_problem_id)
            
            # Update current_reward_dict_avg
            self.current_reward_dict_avg[current_problem_id] = uid_rewards.sum().item() / len(uid_rewards)
            
            # Check if all rollouts in this group succeeded or all failed
            # tolerance is already defined above (line 1141)
            is_success = (uid_rewards >= success_value - tolerance)
            if is_success.all():
                solve_all_count += 1
            elif not is_success.any():
                solve_none_count += 1
        
        # Compute batch metrics
        hard_problem_list = self.hard_problem_list if self.hard_problem_list is not None else []
        hard_problem_set = set(hard_problem_list)

        if len(hard_problem_list) > 0:
            hard_rewards = [self.current_reward_dict_avg.get(str(pid), 0.0) for pid in hard_problem_list]
            metrics['batch/solved_hard'] = sum(hard_rewards) / len(hard_problem_list)

            hard_rewards_in_state_0 = [self.current_reward_dict_avg_in_state_0.get(str(pid), 0.0) for pid in hard_problem_list]
            metrics['batch/solved_hard_in_state_0'] = sum(hard_rewards_in_state_0) / len(hard_problem_list)

        # Compute medium problem metrics dynamically by problem_id range.
        # For all-hard datasets (omni_128/int hard_1024), this becomes empty as expected.
        current_reward_problem_ids = []
        for pid_str in self.current_reward_dict_avg.keys():
            try:
                current_reward_problem_ids.append(int(pid_str))
            except Exception:
                continue
        max_known_pid = -1
        if len(hard_problem_list) > 0:
            max_known_pid = max(max_known_pid, max(hard_problem_list))
        if len(current_reward_problem_ids) > 0:
            max_known_pid = max(max_known_pid, max(current_reward_problem_ids))

        medium_problem_list = []
        if max_known_pid >= 0:
            medium_problem_list = [pid for pid in range(max_known_pid + 1) if pid not in hard_problem_set]
        if len(medium_problem_list) > 0:
            medium_rewards = [self.current_reward_dict_avg.get(str(pid), 0.0) for pid in medium_problem_list]
            metrics['batch/solved_medium'] = sum(medium_rewards) / len(medium_problem_list)
            
            medium_rewards_in_state_0 = [self.current_reward_dict_avg_in_state_0.get(str(pid), 0.0) for pid in medium_problem_list]
            metrics['batch/solved_medium_in_state_0'] = sum(medium_rewards_in_state_0) / len(medium_problem_list)
        
        # Compute global metrics
        metrics['global/total_solved_in_state_0'] = len(self.problems_solved_in_state_0)
        metrics['global/total_solved_in_state_0_hard'] = len(self.problems_solved_in_state_0_hard)
        metrics['global/total_solved_in_state_0_medium'] = len(self.problems_solved_in_state_0_medium)
        metrics['batch/newly_solved_in_state_0'] = batch_newly_solved_count
        metrics['batch/solved_in_state_0'] = solved_in_state_0_count
        
        # Compute batch/solve_none and batch/solve_all metrics
        # Count of problem groups where all rollouts succeeded or all failed
        metrics['batch/solve_all'] = solve_all_count
        metrics['batch/solve_none'] = solve_none_count
        
        # v4: batch/global subproblem_index solved metrics (0=原题, 1..4=subproblem1..4)
        if (
            self._global_subproblem_solved is not None
            and "subproblem_index" in batch.non_tensor_batch
            and "problem_id" in batch.non_tensor_batch
        ):
            sub_idx = batch.non_tensor_batch["subproblem_index"]
            if isinstance(sub_idx, np.ndarray):
                sub_idx = torch.from_numpy(sub_idx).to(device=reward_tensor.device)
            elif not isinstance(sub_idx, torch.Tensor):
                sub_idx = torch.tensor(sub_idx, device=reward_tensor.device, dtype=torch.long)
            per_sample_reward = reward_tensor.sum(dim=-1)  # (batch_size,)
            solved = per_sample_reward > 0
            problem_ids = batch.non_tensor_batch["problem_id"]
            for k in [1, 2, 3, 4]:
                mask = (sub_idx == k) & solved
                metrics[f"batch/subproblem{k}_solved"] = mask.sum().item()
                # Update global: distinct problem_ids that are subproblem k and solved at least once
                if isinstance(problem_ids, np.ndarray):
                    for i in range(len(problem_ids)):
                        if i < mask.shape[0] and mask[i].item():
                            self._global_subproblem_solved[k].add(str(problem_ids[i]))
                else:
                    for i, pid in enumerate(problem_ids):
                        if i < mask.shape[0] and mask[i].item():
                            self._global_subproblem_solved[k].add(str(pid))
            for k in [1, 2, 3, 4]:
                metrics[f"global/subproblem{k}_solved"] = len(self._global_subproblem_solved[k])
            print(f"global/subproblem1_solved: {self._global_subproblem_solved[1]}")
            print(f"global/subproblem2_solved: {self._global_subproblem_solved[2]}")
            print(f"global/subproblem3_solved: {self._global_subproblem_solved[3]}")
            print(f"global/subproblem4_solved: {self._global_subproblem_solved[4]}")
        # Compute subproblem_correct_count metrics for v5 (truncation version)
        if (self.ts_version == 'v5' or self.ts_version == 'v7' or self.ts_version == 'v8') and "subproblem_correct_count" in batch.non_tensor_batch:
            subproblem_correct_counts = batch.non_tensor_batch["subproblem_correct_count"]
            # Convert to list for safe None handling
            if isinstance(subproblem_correct_counts, np.ndarray):
                subproblem_correct_counts = subproblem_correct_counts.tolist()
            elif not isinstance(subproblem_correct_counts, list):
                subproblem_correct_counts = list(subproblem_correct_counts)
            
            # Filter non-original problems (subproblem_correct_count is not None)
            non_original_counts = [count for count in subproblem_correct_counts if count is not None]
            extract_failed_count = sum(1 for count in non_original_counts if count == -1)

            # v7 metrics:
            # - Keep batch/subproblem_0..4_solved but reinterpret as absolute question index solved counts.
            #   0 means parse succeeded but solved none (k==0).
            # - Add batch/v7_t{t}_k{kk}_count (kk=-1..4).
            if self.ts_version in ("v7", "v8"):
                reward_models_arr = batch.non_tensor_batch.get("reward_model", None)
                v7_t_arr = batch.non_tensor_batch.get("v7_t", None)
                if isinstance(v7_t_arr, np.ndarray):
                    v7_t_arr = v7_t_arr.tolist()
                elif v7_t_arr is not None and not isinstance(v7_t_arr, list):
                    v7_t_arr = list(v7_t_arr)

                abs_solved_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
                t_k_counts = {(t, kk): 0 for t in [1, 2, 3, 4] for kk in [-1, 0, 1, 2, 3, 4]}

                for idx, k_val in enumerate(subproblem_correct_counts):
                    if k_val is None:
                        continue
                    try:
                        k_int = int(k_val)
                    except Exception:
                        continue

                    # Determine t and absolute question start index.
                    t_int = None
                    if v7_t_arr is not None and idx < len(v7_t_arr) and v7_t_arr[idx] is not None:
                        try:
                            t_int = int(v7_t_arr[idx])
                        except Exception:
                            t_int = None
                    q_start = None
                    if reward_models_arr is not None and idx < len(reward_models_arr):
                        rm = reward_models_arr[idx]
                        if not isinstance(rm, dict):
                            rm = rm.__dict__ if hasattr(rm, "__dict__") else {}
                        if t_int is None:
                            try:
                                t_int = int(rm.get("num_problems", 4))
                            except Exception:
                                t_int = 4
                        try:
                            q_start = int(rm.get("v7_q_start", 5 - int(t_int)))
                        except Exception:
                            q_start = None
                    if t_int is None:
                        t_int = 4
                    t_int = max(1, min(4, int(t_int)))
                    if q_start is None:
                        q_start = 5 - t_int
                    q_start = max(1, min(4, int(q_start)))

                    # New requested metric: per-t per-k counts.
                    if (t_int, k_int) in t_k_counts:
                        t_k_counts[(t_int, k_int)] += 1

                    # Absolute subproblem solved counts:
                    # k == 0 means parse succeeded but no subproblem solved.
                    if k_int == 0:
                        abs_solved_counts[0] += 1
                    elif k_int > 0:
                        solved_until = min(4, q_start + k_int - 1)
                        for abs_q in range(q_start, solved_until + 1):
                            abs_solved_counts[abs_q] += 1

                metrics['batch/extract_failed'] = extract_failed_count
                metrics['batch/subproblem_0_solved'] = abs_solved_counts[0]
                metrics['batch/subproblem_1_solved'] = abs_solved_counts[1]
                metrics['batch/subproblem_2_solved'] = abs_solved_counts[2]
                metrics['batch/subproblem_3_solved'] = abs_solved_counts[3]
                metrics['batch/subproblem_4_solved'] = abs_solved_counts[4]

                for t in [1, 2, 3, 4]:
                    for kk in [-1, 0, 1, 2, 3, 4]:
                        metrics[f"batch/v7_t{t}_k{kk}_count"] = t_k_counts[(t, kk)]

                # v8 mix44444 adaptive progression (next-step effect):
                # For each problem and qi in {1,2,3}, if all 4 qi samples in this step
                # have reward==1.0, mark qi inactive from next step.
                if (
                    self.ts_version == "v8"
                    and self.v8_mix44444_adaptive
                    and self._v8_mix44444_q_active is not None
                    and str(self.config.data.get("v7_t_mix_mode", "")).lower() == "mix44444"
                ):
                    problem_ids_arr = batch.non_tensor_batch.get("problem_id", None)
                    v8_mix_group_arr = batch.non_tensor_batch.get("v8_mix_group", None)

                    if isinstance(problem_ids_arr, np.ndarray):
                        problem_ids_list = [str(x) for x in problem_ids_arr.tolist()]
                    elif problem_ids_arr is not None:
                        problem_ids_list = [str(x) for x in list(problem_ids_arr)]
                    else:
                        problem_ids_list = []

                    if isinstance(v8_mix_group_arr, np.ndarray):
                        mix_group_list = [int(x) for x in v8_mix_group_arr.tolist()]
                    elif v8_mix_group_arr is not None:
                        mix_group_list = [int(x) for x in list(v8_mix_group_arr)]
                    else:
                        mix_group_list = []

                    per_sample_reward = reward_tensor.sum(dim=-1)
                    grouped_q_rewards = defaultdict(lambda: {1: [], 2: [], 3: []})
                    total_len = min(len(problem_ids_list), len(mix_group_list), int(per_sample_reward.shape[0]))
                    for idx in range(total_len):
                        qid = int(mix_group_list[idx])
                        if qid not in (1, 2, 3):
                            continue
                        pid = problem_ids_list[idx]
                        grouped_q_rewards[pid][qid].append(float(per_sample_reward[idx].item()))

                    deactivated = {1: 0, 2: 0, 3: 0}
                    tol = 1e-6
                    for pid, q_rewards_dict in grouped_q_rewards.items():
                        if pid not in self._v8_mix44444_q_active:
                            self._v8_mix44444_q_active[pid] = {1: True, 2: True, 3: True}
                        for qid in [1, 2, 3]:
                            if not self._v8_mix44444_q_active[pid].get(qid, True):
                                continue
                            rewards_q = q_rewards_dict.get(qid, [])
                            if len(rewards_q) >= 4 and all(float(r) >= 1.0 - tol for r in rewards_q):
                                self._v8_mix44444_q_active[pid][qid] = False
                                deactivated[qid] += 1

                    if sum(deactivated.values()) > 0:
                        print(
                            "[v8_mix44444_adaptive] deactivated this step: "
                            f"q1={deactivated[1]}, q2={deactivated[2]}, q3={deactivated[3]}"
                        )

                    active_counts = {1: 0, 2: 0, 3: 0}
                    for st in self._v8_mix44444_q_active.values():
                        for qid in [1, 2, 3]:
                            if bool(st.get(qid, True)):
                                active_counts[qid] += 1
                    metrics["global/v8_mix44444_active_q1_problems"] = active_counts[1]
                    metrics["global/v8_mix44444_active_q2_problems"] = active_counts[2]
                    metrics["global/v8_mix44444_active_q3_problems"] = active_counts[3]
                    metrics["batch/v8_mix44444_deactivated_q1"] = deactivated[1]
                    metrics["batch/v8_mix44444_deactivated_q2"] = deactivated[2]
                    metrics["batch/v8_mix44444_deactivated_q3"] = deactivated[3]

                # v8 adaptive progression (next-step effect):
                # For each problem currently at t>1 in curriculum branch, if all its curriculum samples
                # in this step satisfy k>=1 (first local part solved), move to t-1 from next step.
                if self.ts_version == "v8" and self.v8_use_adaptive and self._v8_problem_t_state is not None:
                    problem_ids_arr = batch.non_tensor_batch.get("problem_id", None)
                    teacher_student_mixed = batch.non_tensor_batch.get("teacher_student_mixed", None)
                    if isinstance(problem_ids_arr, np.ndarray):
                        problem_ids_list = [str(x) for x in problem_ids_arr.tolist()]
                    elif problem_ids_arr is not None:
                        problem_ids_list = [str(x) for x in list(problem_ids_arr)]
                    else:
                        problem_ids_list = []

                    if isinstance(teacher_student_mixed, np.ndarray):
                        mixed_list = [bool(x) for x in teacher_student_mixed.tolist()]
                    elif teacher_student_mixed is not None:
                        mixed_list = [bool(x) for x in list(teacher_student_mixed)]
                    else:
                        mixed_list = []

                    grouped_k = defaultdict(list)
                    grouped_t = defaultdict(list)
                    total_len = min(len(subproblem_correct_counts), len(problem_ids_list), len(mixed_list))
                    for idx in range(total_len):
                        if not mixed_list[idx]:
                            continue
                        pid = problem_ids_list[idx]
                        try:
                            k_int = int(subproblem_correct_counts[idx])
                        except Exception:
                            continue

                        t_int = None
                        if v7_t_arr is not None and idx < len(v7_t_arr) and v7_t_arr[idx] is not None:
                            try:
                                t_int = int(v7_t_arr[idx])
                            except Exception:
                                t_int = None
                        if t_int is None and reward_models_arr is not None and idx < len(reward_models_arr):
                            rm = reward_models_arr[idx]
                            if not isinstance(rm, dict):
                                rm = rm.__dict__ if hasattr(rm, "__dict__") else {}
                            try:
                                t_int = int(rm.get("v7_t", rm.get("num_problems", 4)))
                            except Exception:
                                t_int = None
                        if t_int is None:
                            t_int = int(self._v8_problem_t_state.get(pid, 4))
                        t_int = max(1, min(4, int(t_int)))
                        grouped_k[pid].append(k_int)
                        grouped_t[pid].append(t_int)

                    moved_pids = 0
                    for pid, k_list in grouped_k.items():
                        if len(k_list) == 0:
                            continue
                        t_list = grouped_t.get(pid, [])
                        if len(t_list) == 0:
                            continue
                        current_t = int(t_list[0])
                        if any(int(t) != current_t for t in t_list):
                            continue
                        if current_t <= 1:
                            self._v8_problem_t_state[pid] = 1
                            continue
                        # Need all curriculum samples in this step to have first local subproblem solved.
                        # Under current extraction, this is equivalent to k>=1.
                        if all(int(kv) >= 1 for kv in k_list):
                            next_t = max(1, current_t - 1)
                            prev_t = int(self._v8_problem_t_state.get(pid, current_t))
                            if next_t < prev_t:
                                self._v8_problem_t_state[pid] = next_t
                                moved_pids += 1
                            else:
                                self._v8_problem_t_state[pid] = min(prev_t, next_t)

                    if moved_pids > 0:
                        print(f"[v8_adaptive] moved {moved_pids} problems to easier t in next step")

                    t_state_counts = {1: 0, 2: 0, 3: 0, 4: 0}
                    for t_val in self._v8_problem_t_state.values():
                        try:
                            t_key = max(1, min(4, int(t_val)))
                        except Exception:
                            t_key = 4
                        t_state_counts[t_key] += 1
                    for t in [1, 2, 3, 4]:
                        metrics[f"global/problems_at_t{t}"] = t_state_counts[t]
            else:
                # Legacy behavior for v5 / old v7 setup.
                subproblem_0_solved = sum(1 for count in non_original_counts if count == 0)
                subproblem_1_solved = sum(1 for count in non_original_counts if count >= 1)
                subproblem_2_solved = sum(1 for count in non_original_counts if count >= 2)
                subproblem_3_solved = sum(1 for count in non_original_counts if count >= 3)
                subproblem_4_solved = sum(1 for count in non_original_counts if count >= 4)

                metrics['batch/extract_failed'] = extract_failed_count
                metrics['batch/subproblem_0_solved'] = subproblem_0_solved
                metrics['batch/subproblem_1_solved'] = subproblem_1_solved
                metrics['batch/subproblem_2_solved'] = subproblem_2_solved
                metrics['batch/subproblem_3_solved'] = subproblem_3_solved
                metrics['batch/subproblem_4_solved'] = subproblem_4_solved
            
            # Track problems with k=4 (all subproblems correct)
            # Note: k=4 means all 4 subproblems are correct, even though truncated=False for k=4
            if "problem_id" in batch.non_tensor_batch:
                problem_ids_arr = batch.non_tensor_batch["problem_id"]
                
                # Convert to list/array for safe indexing
                if isinstance(problem_ids_arr, np.ndarray):
                    problem_ids_arr = problem_ids_arr.tolist()
                elif not isinstance(problem_ids_arr, list):
                    problem_ids_arr = list(problem_ids_arr)
                
                # Iterate through all samples
                for idx in range(len(subproblem_correct_counts)):
                    # Check if subproblem_correct_count==4 (all subproblems correct)
                    if (idx < len(subproblem_correct_counts) and subproblem_correct_counts[idx] == 4 and
                        idx < len(problem_ids_arr)):
                        problem_id = str(problem_ids_arr[idx])
                        # Check if it's a hard or medium problem
                        if int(problem_id) in self.hard_problem_list:
                            self.problems_solved_hard_full_subproblem.add(problem_id)
                        else:
                            self.problems_solved_medium_full_subproblem.add(problem_id)
            
            # Add global metrics for full subproblem solved problems
            metrics['global/total_solved_hard_full_subproblem'] = len(self.problems_solved_hard_full_subproblem)
            metrics['global/total_solved_medium_full_subproblem'] = len(self.problems_solved_medium_full_subproblem)
            print(f"total_solved_hard_full_subproblem: {self.problems_solved_hard_full_subproblem}")
            print(f"total_solved_medium_full_subproblem: {self.problems_solved_medium_full_subproblem}")

        # Emit adaptive t-state metrics even when the current batch has no subproblem stats.
        if self.ts_version == "v8" and self.v8_use_adaptive and self._v8_problem_t_state is not None:
            t_state_counts = {1: 0, 2: 0, 3: 0, 4: 0}
            for t_val in self._v8_problem_t_state.values():
                try:
                    t_key = max(1, min(4, int(t_val)))
                except Exception:
                    t_key = 4
                t_state_counts[t_key] += 1
            for t in [1, 2, 3, 4]:
                metrics[f"global/problems_at_t{t}"] = t_state_counts[t]
        if self.ts_version == "v8" and self.v8_mix44444_adaptive and self._v8_mix44444_q_active is not None:
            active_counts = {1: 0, 2: 0, 3: 0}
            for st in self._v8_mix44444_q_active.values():
                for qid in [1, 2, 3]:
                    if bool(st.get(qid, True)):
                        active_counts[qid] += 1
            metrics["global/v8_mix44444_active_q1_problems"] = active_counts[1]
            metrics["global/v8_mix44444_active_q2_problems"] = active_counts[2]
            metrics["global/v8_mix44444_active_q3_problems"] = active_counts[3]

        print(f"problem_solved_in_state_0: {self.problems_solved_in_state_0}")
        print(f"problem_solved_in_state_0_hard: {self.problems_solved_in_state_0_hard}")
        print(f"problem_solved_in_state_0_medium: {self.problems_solved_in_state_0_medium}")
        print(f"  - batch/solved_hard: {metrics.get('batch/solved_hard', 'N/A')}")
        print(f"  - batch/solved_medium: {metrics.get('batch/solved_medium', 'N/A')}")
        print(f"  - batch/solved_hard_in_state_0: {metrics.get('batch/solved_hard_in_state_0', 'N/A')}")
        print(f"  - batch/solved_medium_in_state_0: {metrics.get('batch/solved_medium_in_state_0', 'N/A')}")
        print(f"  - global/total_solved_in_state_0: {metrics.get('global/total_solved_in_state_0', 'N/A')}")
        print(f"  - batch/newly_solved_in_state_0: {metrics.get('batch/newly_solved_in_state_0', 'N/A')}")
        print(f"  - batch/solved_in_state_0: {metrics.get('batch/solved_in_state_0', 'N/A')}")

    def _generate_teacher_hints(self):
        """Generate teacher hints based on student answer history."""
        from verl.trainer.ppo.own_utils import BatchHintGenerator
        
        if self.hint_generator is None:
            return
        
        if not self.student_answer_history:
            return
        
        # Prepare problem_dict and tgt_dict for hint generation
        problem_dict = {}
        tgt_dict = {}
        
        for problem_id, answer_history in self.student_answer_history.items():
            if not answer_history:
                continue
            # Get raw_input_ids and tgt_input_ids from the first rollout
            rollout_1 = answer_history.get("rollout_1", {})
            if rollout_1:
                data = rollout_1.get("data", {})
                if "raw_input_ids" in data:
                    problem_dict[problem_id] = data["raw_input_ids"]["text"]
                if "tgt_input_ids" in data:
                    tgt_dict[problem_id] = data["tgt_input_ids"]["text"]
        
        if not problem_dict:
            return
        
        # Generate hints based on ts_version
        if self.ts_version == 'v1':
            if isinstance(self.hint_generator, BatchHintGenerator):
                self.teacher_hint_dict = self.hint_generator.generate_hints_batch(
                    student_answer_history=self.student_answer_history
                )
        elif self.ts_version == 'v2':
            if hasattr(self.hint_generator, 'generate_crafted_wrong_answers_batch'):
                if ray.is_initialized():
                    self.crafted_wrong_answer = ray.get(
                        self.hint_generator.generate_crafted_wrong_answers_batch.remote(
                            problems=problem_dict,
                            ground_truth_answers=tgt_dict
                        )
                    )
                else:
                    self.crafted_wrong_answer = self.hint_generator.generate_crafted_wrong_answers_batch(
                        problems=problem_dict,
                        ground_truth_answers=tgt_dict
                    )
        elif self.ts_version == 'v3':
            if hasattr(self.hint_generator, 'generate_crafted_wrong_answers_batch'):
                if ray.is_initialized():
                    self.crafted_wrong_answer = ray.get(
                        self.hint_generator.generate_crafted_wrong_answers_batch.remote(
                            problems=problem_dict,
                            ground_truth_answers=tgt_dict
                        )
                    )
                    self.teacher_lemmas = ray.get(
                        self.hint_generator.generate_guided_lemmas_batch.remote(
                            problems=problem_dict,
                            ground_truth_answers=tgt_dict,
                            num_lemmas=3
                        )
                    )
                else:
                    self.crafted_wrong_answer = self.hint_generator.generate_crafted_wrong_answers_batch(
                        problems=problem_dict,
                        ground_truth_answers=tgt_dict
                    )
                    self.teacher_lemmas = self.hint_generator.generate_guided_lemmas_batch(
                        problems=problem_dict,
                        ground_truth_answers=tgt_dict,
                        num_lemmas=3
                    )
        
        # Update dataset with new hints (if dataset supports it)
        # Note: This requires the dataset to have an update_teacher_hints method
        # If the dataset doesn't support it, hints will be used in the next epoch
        if hasattr(self.train_dataset, 'update_teacher_hints'):
            self.train_dataset.update_teacher_hints(
                teacher_hint_dict=self.teacher_hint_dict,
                crafted_wrong_answer=self.crafted_wrong_answer,
                teacher_lemmas=self.teacher_lemmas
            )

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store original inputs (use prompts from generation output, same as training rollout)
            # This ensures the chat template is correctly applied
            if "prompts" in test_output_gen_batch.batch:
                input_texts = self.tokenizer.batch_decode(
                    test_output_gen_batch.batch["prompts"], skip_special_tokens=False
                )
            else:
                # Fallback: decode from input_ids (may not have chat template applied correctly)
                input_ids = test_batch.batch["input_ids"]
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=False) for ids in input_ids]
            sample_inputs.extend(input_texts)

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=False) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        avg_core_sources = {"olympiad_bench", "minerva", "math", "amc", "aime"}
        avg_score_metric2vals = defaultdict(list)
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val
                    # Aggregate pass@k-like metrics across 5 core data sources:
                    # val-core/avg_score/mean@k
                    if data_source in avg_core_sources and metric_name.startswith("mean@"):
                        if (var_name == core_var) and (metric_sec == "val-core"):
                            avg_score_metric2vals[metric_name].append(float(metric_val))

        for metric_name, vals in avg_score_metric2vals.items():
            if len(vals) > 0:
                metric_dict[f"val-core/avg_score/{metric_name}"] = float(np.mean(vals))

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        # for legacy discriminative reward model, we create a reward model worker here
        # for reward loop discriminative reward model, we create a reward loop manager here
        if not self.use_reward_loop:
            # legacy reward model only handle reward-model based scenario
            if self.use_rm:
                # we create a RM here
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
                )
                self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
        else:
            # reward loop handle hybrid reward scenario (rule, disrm, genrm, ...)
            can_reward_loop_parallelize = self.config.actor_rollout_ref.rollout.mode == "async" and (
                not self.use_rm or self.config.reward_model.enable_resource_pool
            )
            # judge if we can asynchronously parallelize reward model with actor rollout
            # two condition that we can parallelize reward model with actor rollout:
            # 1. reward model is not enabled (rule-based reward can parallelize)
            # 2. reward model is enabled but extra resource pool is enabled
            # If we cannot parallelize, we should enable synchronous mode here, and launch a reward loop manager here
            # else for parallelize mode, we launch a reward worker for each rollout worker (in agent loop, not here)
            if not can_reward_loop_parallelize:
                from verl.experimental.reward import RewardLoopManager

                self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                self.reward_loop_manager = RewardLoopManager(
                    config=self.config,
                    rm_resource_pool=resource_pool,
                )

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
                rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            else:
                rm_resource_pool = None

            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                rm_resource_pool=rm_resource_pool,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                workload_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (workload_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=False, compute_loss=False)
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)
        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # Add old_rewards to batch if teacher_student is enabled
                curri_method = self.config.data.get("curri_method", None)
                if curri_method == "teacher_student" and self.old_rewards is not None:
                    # Map old_rewards to batch items based on problem_id
                    if "problem_id" in batch.non_tensor_batch:
                        problem_ids = batch.non_tensor_batch["problem_id"]
                        batch_old_rewards = []
                        for pid in problem_ids:
                            pid_int = int(pid)
                            if pid_int < len(self.old_rewards):
                                batch_old_rewards.append(self.old_rewards[pid_int].item())
                            else:
                                batch_old_rewards.append(0.0)
                        batch.batch["old_rewards"] = torch.tensor(batch_old_rewards, device=batch.batch["input_ids"].device, dtype=torch.float32)
                
                # Process teacher-student prompts if enabled (before _get_gen_batch)
                mixed_data = self._process_teacher_student_prompts(batch)
                gen_batch = self._get_gen_batch(batch)
                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                # If teacher-student mixing was performed, update gen_batch with mixed raw_prompt
                if mixed_data is not None:
                    # First, repeat gen_batch to match the expanded length
                    gen_batch = gen_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                    )
                    # Replace raw_prompt, reward_model, and teacher_student_mixed with mixed versions (which have correct order)
                    gen_batch.non_tensor_batch["raw_prompt"] = mixed_data.non_tensor_batch["raw_prompt"]
                    if "reward_model" in mixed_data.non_tensor_batch:
                        gen_batch.non_tensor_batch["reward_model"] = mixed_data.non_tensor_batch["reward_model"]
                    if "teacher_student_mixed" in mixed_data.non_tensor_batch:
                        gen_batch.non_tensor_batch["teacher_student_mixed"] = mixed_data.non_tensor_batch["teacher_student_mixed"]
                    if "v7_t" in mixed_data.non_tensor_batch:
                        gen_batch.non_tensor_batch["v7_t"] = mixed_data.non_tensor_batch["v7_t"]
                    if "v7_prompt_mode" in mixed_data.non_tensor_batch:
                        gen_batch.non_tensor_batch["v7_prompt_mode"] = mixed_data.non_tensor_batch["v7_prompt_mode"]
                    if "v8_mix_group" in mixed_data.non_tensor_batch:
                        gen_batch.non_tensor_batch["v8_mix_group"] = mixed_data.non_tensor_batch["v8_mix_group"]
                    if "nurl_hint" in mixed_data.non_tensor_batch:
                        gen_batch.non_tensor_batch["nurl_hint"] = mixed_data.non_tensor_batch["nurl_hint"]
                    if "nurl_question" in mixed_data.non_tensor_batch:
                        gen_batch.non_tensor_batch["nurl_question"] = mixed_data.non_tensor_batch["nurl_question"]
                    gen_batch_output = gen_batch
                else:
                    # No mixing, repeat as usual
                    gen_batch_output = gen_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                    )

                is_last_step = self.global_steps >= self.total_training_steps
                nurl_reward_batch_before = None
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                    
                    # For teacher_student v5/v6: process responses based on subproblem correctness
                    curri_method = self.config.data.get("curri_method", None)
                    ts_version = self.config.data.get("ts_version", None)
                    if curri_method == "teacher_student" and ts_version == "v5" and mixed_data is not None:
                        print("Truncating responses based on subproblem correctness (v5)")
                        from verl.trainer.ppo.own_utils import truncate_response_by_subproblem_correctness
                        # Get USE_SUBPROBLEM_PROMPT from config, default to False
                        use_subproblem_prompt = self.config.data.get("use_subproblem_prompt", False)
                        gen_batch_output = truncate_response_by_subproblem_correctness(
                            gen_batch_output=gen_batch_output,
                            mixed_data=mixed_data,
                            tokenizer=self.tokenizer,
                            reward_fn=self.reward_fn,
                            use_subproblem_prompt=use_subproblem_prompt
                        )
                    elif curri_method == "teacher_student" and ts_version in ("v7", "v8") and mixed_data is not None:
                        print(f"Marking response tokens based on subproblem correctness ({ts_version})")
                        from verl.trainer.ppo.own_utils import mark_response_tokens_by_subproblem_correctness
                        # Get USE_SUBPROBLEM_PROMPT from config, default to False
                        use_subproblem_prompt = self.config.data.get("use_subproblem_prompt", False)
                        v7_format_mode = self.config.data.get("v7_format_mode", "subproblem")
                        v7_num_problems = self.config.data.get("v7_num_problems", 4)
                        if v7_num_problems == "auto":
                            v7_num_problems = 4
                        try:
                            v7_num_problems = int(v7_num_problems)
                        except Exception:
                            v7_num_problems = 4
                        v7_require_strict_eos = self.config.data.get("v7_require_strict_eos", True)
                        v7_parse_fail_policy = self.config.data.get("v7_parse_fail_policy", "hard")
                        gen_batch_output = mark_response_tokens_by_subproblem_correctness(
                            gen_batch_output=gen_batch_output,
                            mixed_data=mixed_data,
                            tokenizer=self.tokenizer,
                            reward_fn=self.reward_fn,
                            use_subproblem_prompt=use_subproblem_prompt,
                            format_mode=v7_format_mode,
                            num_problems=v7_num_problems,
                            require_strict_eos=v7_require_strict_eos,
                            parse_fail_policy=v7_parse_fail_policy,
                        )
                        
                        # Debug: Print token marking statistics (only for teacher_student_mixed samples)
                        if "token_correctness_mask" in gen_batch_output.batch:
                            token_correctness_mask = gen_batch_output.batch["token_correctness_mask"]
                            responses = gen_batch_output.batch["responses"]
                            batch_size = responses.shape[0]
                            response_length = responses.shape[1]
                            
                            # Get mixed indices to only count teacher_student_mixed samples
                            mixed_indices_for_debug = []
                            if mixed_data is not None and "teacher_student_mixed" in mixed_data.non_tensor_batch:
                                teacher_student_mixed_flags = mixed_data.non_tensor_batch["teacher_student_mixed"]
                                if isinstance(teacher_student_mixed_flags, np.ndarray):
                                    mixed_indices_for_debug = np.where(teacher_student_mixed_flags)[0]
                            
                            # Count tokens per sample (only for mixed samples)
                            correct_tokens_per_sample = (token_correctness_mask > 0.5).sum(dim=1)  # (batch_size,)
                            wrong_tokens_per_sample = (token_correctness_mask <= 0.5).sum(dim=1)  # (batch_size,)
                            
                            # Get k value distribution
                            if "subproblem_correct_count" in gen_batch_output.non_tensor_batch:
                                subproblem_correct_count = gen_batch_output.non_tensor_batch["subproblem_correct_count"]
                                k_distribution = {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
                                for k_val in subproblem_correct_count:
                                    if k_val is not None and k_val in k_distribution:
                                        k_distribution[k_val] += 1
                                
                                print(f"[v7_debug] Token marking completed: batch_size={batch_size}, response_length={response_length}")
                                print(f"[v7_debug] K value distribution (all samples): {k_distribution}")
                                
                                # Statistics for mixed samples only
                                if len(mixed_indices_for_debug) > 0:
                                    mixed_indices_tensor = torch.tensor(mixed_indices_for_debug, device=token_correctness_mask.device)
                                    mixed_correct_tokens = correct_tokens_per_sample[mixed_indices_tensor]
                                    mixed_wrong_tokens = wrong_tokens_per_sample[mixed_indices_tensor]
                                   
                                    print(f"[v7_debug] Mixed samples only ({len(mixed_indices_for_debug)} samples):")
                                    print(f"[v7_debug]   Correct tokens per sample: min={mixed_correct_tokens.min().item()}, "
                                          f"max={mixed_correct_tokens.max().item()}, "
                                          f"mean={mixed_correct_tokens.float().mean().item():.2f}, "
                                          f"median={mixed_correct_tokens.float().median().item():.2f}")
                                    print(f"[v7_debug]   Wrong tokens per sample: min={mixed_wrong_tokens.min().item()}, "
                                          f"max={mixed_wrong_tokens.max().item()}, "
                                          f"mean={mixed_wrong_tokens.float().mean().item():.2f}, "
                                          f"median={mixed_wrong_tokens.float().median().item():.2f}")
                                    
                                    # K distribution for mixed samples only
                                    mixed_k_distribution = {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
                                    for idx in mixed_indices_for_debug:
                                        k_val = subproblem_correct_count[idx]
                                        if k_val is not None and k_val in mixed_k_distribution:
                                            mixed_k_distribution[k_val] += 1
                                    print(f"[v7_debug]   K value distribution (mixed samples only): {mixed_k_distribution}")
                                else:
                                    print(f"[v7_debug] No mixed samples found for statistics")
                                
                                # Also print all samples statistics for reference
                                print(f"[v7_debug] All samples statistics (includes non-mixed samples):")
                                print(f"[v7_debug]   Correct tokens per sample: min={correct_tokens_per_sample.min().item()}, "
                                      f"max={correct_tokens_per_sample.max().item()}, "
                                      f"mean={correct_tokens_per_sample.float().mean().item():.2f}, "
                                      f"median={correct_tokens_per_sample.float().median().item():.2f}")
                                print(f"[v7_debug]   Wrong tokens per sample: min={wrong_tokens_per_sample.min().item()}, "
                                      f"max={wrong_tokens_per_sample.max().item()}, "
                                      f"mean={wrong_tokens_per_sample.float().mean().item():.2f}, "
                                      f"median={wrong_tokens_per_sample.float().median().item():.2f}")
                            else:
                                print(f"[v7_debug] Token marking completed: batch_size={batch_size}, response_length={response_length}")
                                if len(mixed_indices_for_debug) > 0:
                                    mixed_indices_tensor = torch.tensor(mixed_indices_for_debug, device=token_correctness_mask.device)
                                    mixed_correct_tokens = correct_tokens_per_sample[mixed_indices_tensor]
                                    mixed_wrong_tokens = wrong_tokens_per_sample[mixed_indices_tensor]
                                    print(f"[v7_debug] Mixed samples only ({len(mixed_indices_for_debug)} samples): "
                                          f"Total correct tokens={mixed_correct_tokens.sum().item()}, "
                                          f"Total wrong tokens={mixed_wrong_tokens.sum().item()}")
                                print(f"[v7_debug] All samples: Total correct tokens={correct_tokens_per_sample.sum().item()}, "
                                      f"Total wrong tokens={wrong_tokens_per_sample.sum().item()}")

                    # NuRL stage2: detect all-fail groups on initial rollout,
                    # inject offline abstract hint into first (n-1) rollouts, and regenerate in-place.
                    if curri_method == "teacher_student" and ts_version == "nurl" and mixed_data is not None:
                        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
                        if rollout_n > 1:
                            with marked_timer("nurl_reward_before_hint", timing_raw, color="yellow"):
                                batch_before_hint = batch.repeat(repeat_times=rollout_n, interleave=True)
                                batch_before_hint.batch.pop("input_ids", None)
                                batch_before_hint.batch.pop("attention_mask", None)
                                batch_before_hint.batch.pop("position_ids", None)
                                batch_before_hint = batch_before_hint.union(gen_batch_output)
                                if "reward_model" in mixed_data.non_tensor_batch:
                                    batch_before_hint.non_tensor_batch["reward_model"] = mixed_data.non_tensor_batch["reward_model"]
                                reward_tensor_before, _ = compute_reward(batch_before_hint, self.reward_fn)
                                nurl_reward_batch_before = reward_tensor_before.sum(-1).reshape(-1, rollout_n).mean(1)
                            hard_mask = nurl_reward_batch_before == 0
                            solve_none_before = int(hard_mask.sum().item())
                            solve_all_before = int((nurl_reward_batch_before == 1).sum().item())
                            metrics["batch/solve_none_before_hint_injection"] = solve_none_before
                            metrics["batch/solve_all_before_hint_injection"] = solve_all_before
                            if solve_none_before > 0:
                                retry_gen_batch = deepcopy(gen_batch_output)
                                # NuRL async rollout path may drop non-tensor fields.
                                # Backfill required metadata so hint injection + reward scoring are deterministic.
                                expected_len = len(retry_gen_batch.non_tensor_batch.get("raw_prompt", []))
                                if expected_len > 0:
                                    def _backfill_key(key: str):
                                        if key in retry_gen_batch.non_tensor_batch and len(retry_gen_batch.non_tensor_batch[key]) == expected_len:
                                            return
                                        src_arr = None
                                        for src in (gen_batch.non_tensor_batch, mixed_data.non_tensor_batch):
                                            if key in src and len(src[key]) == expected_len:
                                                src_arr = src[key]
                                                break
                                        if src_arr is None:
                                            for src in (gen_batch.non_tensor_batch, mixed_data.non_tensor_batch):
                                                if key in src:
                                                    base_arr = src[key]
                                                    if len(base_arr) * rollout_n == expected_len:
                                                        expanded = []
                                                        for v in base_arr:
                                                            expanded.extend([v] * rollout_n)
                                                        src_arr = np.array(expanded, dtype=object)
                                                        break
                                        if src_arr is not None:
                                            retry_gen_batch.non_tensor_batch[key] = np.array(src_arr, dtype=object)

                                    for key in ("nurl_hint", "nurl_question", "data_source", "ability", "reward_model", "extra_info", "problem_id"):
                                        _backfill_key(key)

                                modified_rollouts = self._apply_nurl_hint_to_repeated_prompts(
                                    retry_gen_batch, hard_mask=hard_mask, rollout_n=rollout_n
                                )
                                metrics["batch/nurl_hint_injected_rollouts"] = int(modified_rollouts)
                                if modified_rollouts > 0:
                                    with marked_timer("nurl_regen", timing_raw, color="red"):
                                        if not self.async_rollout_mode:
                                            gen_batch_output = self.actor_rollout_wg.generate_sequences(retry_gen_batch)
                                        else:
                                            gen_batch_output = self.async_rollout_manager.generate_sequences(retry_gen_batch)
                                        # Keep rerolled batch metadata complete for downstream reward manager / save_rollout.
                                        regen_len = len(gen_batch_output.non_tensor_batch.get("raw_prompt", []))
                                        if regen_len > 0:
                                            for key in ("nurl_hint", "nurl_question", "data_source", "ability", "reward_model", "extra_info", "problem_id"):
                                                if key in gen_batch_output.non_tensor_batch and len(gen_batch_output.non_tensor_batch[key]) == regen_len:
                                                    continue
                                                if key in retry_gen_batch.non_tensor_batch and len(retry_gen_batch.non_tensor_batch[key]) == regen_len:
                                                    gen_batch_output.non_tensor_batch[key] = np.array(
                                                        retry_gen_batch.non_tensor_batch[key], dtype=object
                                                    )
                                        timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                                        gen_batch_output.meta_info.pop("timing", None)
                    
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    rm_scores = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # If teacher-student mixing was performed, remove input_ids, attention_mask, position_ids from batch
                    # to avoid overwriting the mixed data in gen_batch_output
                    if mixed_data is not None:
                        batch.batch.pop("input_ids", None)
                        batch.batch.pop("attention_mask", None)
                        batch.batch.pop("position_ids", None)
                    batch = batch.union(gen_batch_output)
                    # Update reward_model and teacher_student_mixed if they exist in mixed_data
                    # This ensures reward_model and teacher_student_mixed order matches the mixed prompts order
                    if mixed_data is not None:
                        if "reward_model" in mixed_data.non_tensor_batch:
                            batch.non_tensor_batch["reward_model"] = mixed_data.non_tensor_batch["reward_model"]
                                    
                        if "teacher_student_mixed" in mixed_data.non_tensor_batch:
                            batch.non_tensor_batch["teacher_student_mixed"] = mixed_data.non_tensor_batch["teacher_student_mixed"]
                        if "v7_t" in mixed_data.non_tensor_batch:
                            batch.non_tensor_batch["v7_t"] = mixed_data.non_tensor_batch["v7_t"]
                        if "v7_prompt_mode" in mixed_data.non_tensor_batch:
                            batch.non_tensor_batch["v7_prompt_mode"] = mixed_data.non_tensor_batch["v7_prompt_mode"]
                        if "v8_mix_group" in mixed_data.non_tensor_batch:
                            batch.non_tensor_batch["v8_mix_group"] = mixed_data.non_tensor_batch["v8_mix_group"]
                        if "nurl_hint" in mixed_data.non_tensor_batch:
                            batch.non_tensor_batch["nurl_hint"] = mixed_data.non_tensor_batch["nurl_hint"]
                        if "nurl_question" in mixed_data.non_tensor_batch:
                            batch.non_tensor_batch["nurl_question"] = mixed_data.non_tensor_batch["nurl_question"]
                        if "subproblem_index" in mixed_data.non_tensor_batch:
                            batch.non_tensor_batch["subproblem_index"] = np.array(
                                mixed_data.non_tensor_batch["subproblem_index"], dtype=np.int32
                            )
                    # Option B (v4): Luffy-style in-place mutation so the exact batch reward sees is updated.
                    # This is robust even if union() or worker returned a different reward_model array.
                    if mixed_data is not None and self.ts_version == "v4" and "reward_model" in batch.non_tensor_batch:
                        n_rollout = self.config.actor_rollout_ref.rollout.n
                        batch_size = len(batch) // n_rollout
                        reward_models = batch.non_tensor_batch["reward_model"]
                        sub_n = 1
                        v4_mode = self.config.data.get("v4_subproblem_mode", 0)
                        for i in range(batch_size):
                            orig_rm = reward_models[i * n_rollout]
                            if not isinstance(orig_rm, dict):
                                orig_rm = orig_rm.__dict__ if hasattr(orig_rm, "__dict__") else {}
                            gt_orig = orig_rm.get("ground_truth", "")
                            gt_sub1 = orig_rm.get("ground_truth_sub1", gt_orig)
                            gt_sub2 = orig_rm.get("ground_truth_sub2", gt_orig)
                            gt_sub3 = orig_rm.get("ground_truth_sub3", gt_orig)
                            gt_sub4 = orig_rm.get("ground_truth_sub4", gt_orig)
                            for j in range(n_rollout):
                                idx = i * n_rollout + j
                                new_rm = {"style": "rule"}
                                if v4_mode == 1:
                                    new_rm["ground_truth"] = gt_sub4 if j < 4 else gt_orig
                                elif v4_mode == 2:
                                    new_rm["ground_truth"] = gt_sub4
                                elif v4_mode == 3:
                                    new_rm["ground_truth"] = gt_sub1 if j < 4 else gt_sub2
                                else:
                                    if j < sub_n:
                                        new_rm["ground_truth"] = gt_sub1
                                    elif j < 2 * sub_n:
                                        new_rm["ground_truth"] = gt_sub2
                                    elif j < 3 * sub_n:
                                        new_rm["ground_truth"] = gt_sub3
                                    elif j < 4 * sub_n:
                                        new_rm["ground_truth"] = gt_sub4
                                    else:
                                        new_rm["ground_truth"] = gt_orig
                                batch.non_tensor_batch["reward_model"][idx] = new_rm
                    # Note: After _balance_batch, teacher_student_mixed will be reordered along with other fields,
                    # so the correspondence is maintained

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    print(f"batch.batch keys: {batch.batch.keys()}")
                    print(f"batch.non_tensor_batch keys: {batch.non_tensor_batch.keys()}")
                    print(f"batch.meta_info keys: {batch.meta_info.keys()}")
                    print(f"batch.meta_info['reward_extra_keys']: {batch.meta_info['reward_extra_keys']}")
                    print(f"batch.non_tensor_batch['reward_model']: {batch.non_tensor_batch['reward_model']}")
                    print(f"batch.non_tensor_batch['reward_extra_info']: {batch.non_tensor_batch['reward_extra_info']}")
                    batch.non_tensor_batch.pop("reward_extra_info", None)
                    print(f"self.use_rm: {self.use_rm}")
                    print(f"batch.batch['rm_scores']: {batch.batch['rm_scores']}")
                    if not self.use_rm and "rm_scores" in batch.batch.keys():
                        batch.batch.pop("rm_scores", None)
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            if not self.use_reward_loop:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                            else:
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                        batch.batch["token_level_scores"] = reward_tensor
                        if curri_method == "teacher_student" and ts_version == "nurl":
                            rollout_n = int(self.config.actor_rollout_ref.rollout.n)
                            if rollout_n > 0 and reward_tensor.size(0) % rollout_n == 0:
                                nurl_reward_batch_after = reward_tensor.sum(-1).reshape(-1, rollout_n).mean(1)
                                metrics["batch/solve_none_after_hint_injection"] = int(
                                    (nurl_reward_batch_after == 0).sum().item()
                                )
                                metrics["batch/solve_all_after_hint_injection"] = int(
                                    (nurl_reward_batch_after == 1).sum().item()
                                )
                        
                        # For v5: Set reward=1.0 for successfully truncated samples
                        # This must be done after reward computation but before _update_old_rewards and _save_rollout_samples
                        # to ensure these functions use the modified reward values
                        if curri_method == "teacher_student" and ts_version == "v5":
                            # Get all teacher_student_mixed samples and set reward based on subproblem_correct_count
                            if "teacher_student_mixed" in batch.non_tensor_batch:
                                teacher_student_mixed_flags = batch.non_tensor_batch["teacher_student_mixed"]
                                if isinstance(teacher_student_mixed_flags, np.ndarray):
                                    mixed_indices = np.where(teacher_student_mixed_flags)[0]
                                    if len(mixed_indices) > 0:
                                        # Get subproblem_correct_count (k values) for each sample
                                        subproblem_correct_count = batch.non_tensor_batch.get("subproblem_correct_count", None)
                                        truncated_valid_lengths = batch.non_tensor_batch.get("truncated_valid_lengths", None)
                                        
                                        # reward_tensor shape: (batch_size, response_length)
                                        # reward_tensor only contains response part, not prompt
                                        response_length = reward_tensor.shape[1]
                                        response_mask = batch.batch.get("response_mask", None)
                                        
                                        # Statistics for each k value
                                        k_counts = {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
                                        
                                        for idx in mixed_indices:
                                            idx = int(idx)
                                            
                                            # Get k value (number of correct subproblems) for this sample
                                            k = None
                                            if subproblem_correct_count is not None:
                                                if isinstance(subproblem_correct_count, np.ndarray) and idx < len(subproblem_correct_count):
                                                    k = subproblem_correct_count[idx]
                                                elif hasattr(subproblem_correct_count, '__getitem__'):
                                                    k = subproblem_correct_count[idx]
                                            
                                            # Calculate reward: k=-1 or k=0 -> reward=0.0, otherwise reward=k/4
                                            if k is not None and isinstance(k, (int, float)):
                                                if k == -1 or k == 0:
                                                    reward_value = 0.0
                                                elif 1 <= k <= 4:
                                                    if k == 1:
                                                        reward_value = 0.1
                                                    if k == 2:
                                                        reward_value = 0.2
                                                    if k == 3:
                                                        reward_value = 0.5
                                                    if k == 4:
                                                        reward_value = 1.0
                                                else:
                                                    # Invalid k value, skip this sample
                                                    continue
                                            else:
                                                # k is None or invalid type, skip this sample
                                                continue
                                            
                                            # First, zero out all rewards for this sample to ensure clean state
                                            reward_tensor[idx, :] = 0.0
                                            
                                            # Find the last valid token position
                                            last_valid_idx = None
                                            
                                            # Try to use truncated_valid_lengths first (for truncated samples)
                                            if truncated_valid_lengths is not None and idx < len(truncated_valid_lengths):
                                                valid_length = truncated_valid_lengths[idx]
                                                if valid_length is not None and valid_length > 0:
                                                    last_valid_idx = int(valid_length) - 1
                                                    if last_valid_idx < 0 or last_valid_idx >= response_length:
                                                        last_valid_idx = None
                                            
                                            # Fallback to response_mask if truncated_valid_lengths not available or invalid
                                            if last_valid_idx is None and response_mask is not None:
                                                mask_bool = response_mask[idx].bool()
                                                valid_indices = torch.where(mask_bool)[0]
                                                if len(valid_indices) > 0:
                                                    last_valid_idx = valid_indices[-1].item()
                                                    if last_valid_idx >= response_length:
                                                        last_valid_idx = None
                                            
                                            # Last resort: use the last token of the response
                                            if last_valid_idx is None:
                                                if response_length > 0:
                                                    last_valid_idx = response_length - 1
                                                else:
                                                    # Skip if response_length is 0
                                                    continue
                                            
                                            # Set reward on the last valid token
                                            reward_tensor[idx, last_valid_idx] = reward_value
                                            
                                            # Update statistics
                                            if k in k_counts:
                                                k_counts[k] += 1
                                        
                                        # Update token_level_scores with modified reward_tensor
                                        batch.batch["token_level_scores"] = reward_tensor
                                        
                                        # Print statistics for each k value
                                        for k in [-1, 0, 1, 2, 3, 4]:
                                            count = k_counts[k]
                                            if k == -1 or k == 0:
                                                reward_value = 0.0
                                            else:
                                                if k == 1:
                                                    reward_value = 0.1
                                                if k == 2:
                                                    reward_value = 0.2
                                                if k == 3:
                                                    reward_value = 0.5
                                                if k == 4:
                                                    reward_value = 1.0
                                            print(f"[v5_reward] Set reward as {reward_value} based on k={k} for {count} teacher_student_mixed samples")
                        elif curri_method == "teacher_student" and ts_version == "v7":
                            # For v7: Set reward only on the last valid token (last token of k-th subproblem)
                            # Similar to v5, but v7 doesn't truncate the response
                            # We use token_correctness_mask to find the last correct token position
                            if "teacher_student_mixed" in batch.non_tensor_batch:
                                teacher_student_mixed_flags = batch.non_tensor_batch["teacher_student_mixed"]
                                if isinstance(teacher_student_mixed_flags, np.ndarray):
                                    mixed_indices = np.where(teacher_student_mixed_flags)[0]
                                    if len(mixed_indices) > 0:
                                        # Get subproblem_correct_count (k values) for each sample
                                        subproblem_correct_count = batch.non_tensor_batch.get("subproblem_correct_count", None)
                                        token_correctness_mask = batch.batch.get("token_correctness_mask", None)
                                        reward_models = batch.non_tensor_batch.get("reward_model", None)
                                        v7_reward_map_mode = str(self.config.data.get("v7_reward_map_mode", "legacy_k")).lower()
                                        
                                        # reward_tensor shape: (batch_size, response_length)
                                        # reward_tensor only contains response part, not prompt
                                        response_length = reward_tensor.shape[1]
                                        response_mask = batch.batch.get("response_mask", None)
                                        
                                        # Statistics for each k value
                                        k_counts = {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
                                        
                                        for idx in mixed_indices:
                                            idx = int(idx)
                                            
                                            # Get k value (number of correct subproblems) for this sample
                                            k = None
                                            if subproblem_correct_count is not None:
                                                if isinstance(subproblem_correct_count, np.ndarray) and idx < len(subproblem_correct_count):
                                                    k = subproblem_correct_count[idx]
                                                elif hasattr(subproblem_correct_count, '__getitem__'):
                                                    k = subproblem_correct_count[idx]
                                            
                                            # Calculate reward: k=-1 or k=0 -> reward=0.0, otherwise reward=k/4
                                            if k is not None and isinstance(k, (int, float)):
                                                if k == -1 or k == 0:
                                                    reward_value = 0.0
                                                elif 1 <= k <= 4:
                                                    if v7_reward_map_mode == "absolute_difficulty":
                                                        q_start = 1
                                                        if reward_models is not None and idx < len(reward_models):
                                                            rm = reward_models[idx]
                                                            if not isinstance(rm, dict):
                                                                rm = rm.__dict__ if hasattr(rm, "__dict__") else {}
                                                            try:
                                                                q_start = int(rm.get("v7_q_start", 1))
                                                            except Exception:
                                                                q_start = 1
                                                        abs_q_idx = max(1, min(4, q_start + int(k) - 1))
                                                        if abs_q_idx == 1:
                                                            reward_value = 0.1
                                                        elif abs_q_idx == 2:
                                                            reward_value = 0.2
                                                        elif abs_q_idx == 3:
                                                            reward_value = 0.5
                                                        else:
                                                            reward_value = 1.0
                                                    else:
                                                        if k == 1:
                                                            reward_value = 0.1
                                                        if k == 2:
                                                            reward_value = 0.2
                                                        if k == 3:
                                                            reward_value = 0.5
                                                        if k == 4:
                                                            reward_value = 1.0
                                                else:
                                                    # Invalid k value, skip this sample
                                                    continue
                                            else:
                                                # k is None or invalid type, skip this sample
                                                continue
                                            
                                            # First, zero out all rewards for this sample to ensure clean state
                                            reward_tensor[idx, :] = 0.0
                                            
                                            # Find the last valid token position (last token of k-th subproblem)
                                            last_valid_idx = None
                                            
                                            # Use token_correctness_mask to find the last correct token
                                            if token_correctness_mask is not None:
                                                # token_correctness_mask: 1 for correct tokens, 0 for wrong tokens
                                                correctness_mask = token_correctness_mask[idx].bool()
                                                correct_indices = torch.where(correctness_mask)[0]
                                                if len(correct_indices) > 0:
                                                    # Last correct token is the last token of k-th subproblem
                                                    last_valid_idx = correct_indices[-1].item()
                                                    if last_valid_idx >= response_length:
                                                        last_valid_idx = None
                                            
                                            # Fallback to response_mask if token_correctness_mask not available or invalid
                                            if last_valid_idx is None and response_mask is not None:
                                                mask_bool = response_mask[idx].bool()
                                                valid_indices = torch.where(mask_bool)[0]
                                                if len(valid_indices) > 0:
                                                    last_valid_idx = valid_indices[-1].item()
                                                    if last_valid_idx >= response_length:
                                                        last_valid_idx = None
                                            
                                            # Last resort: use the last token of the response
                                            if last_valid_idx is None:
                                                if response_length > 0:
                                                    last_valid_idx = response_length - 1
                                                else:
                                                    # Skip if response_length is 0
                                                    continue
                                            
                                            # Set reward on the last valid token (last token of k-th subproblem)
                                            reward_tensor[idx, last_valid_idx] = reward_value
                                            
                                            # Update statistics
                                            if k in k_counts:
                                                k_counts[k] += 1
                                        
                                        # Update token_level_scores with modified reward_tensor
                                        batch.batch["token_level_scores"] = reward_tensor
                                        
                                        # Print statistics for each k value
                                        total_mixed_samples = len(mixed_indices)
                                        print(f"[v7_debug] Reward setting: Processing {total_mixed_samples} teacher_student_mixed samples")
                                        for k in [-1, 0, 1, 2, 3, 4]:
                                            count = k_counts[k]
                                            if count > 0:
                                                print(f"[v7_reward] mode={v7_reward_map_mode}, k={k}, count={count} "
                                                      f"({count/total_mixed_samples*100:.1f}%)")
                                        
                                        # Debug: Print reward distribution statistics
                                        if token_correctness_mask is not None:
                                            # Count how many samples have correct/wrong tokens
                                            samples_with_correct = (token_correctness_mask[mixed_indices] > 0.5).any(dim=1).sum().item()
                                            samples_with_wrong = (token_correctness_mask[mixed_indices] <= 0.5).any(dim=1).sum().item()
                                            samples_all_correct = ((token_correctness_mask[mixed_indices] > 0.5).all(dim=1)).sum().item()
                                            samples_all_wrong = ((token_correctness_mask[mixed_indices] <= 0.5).all(dim=1)).sum().item()
                                            
                                            print(f"[v7_debug] Sample statistics: samples_with_correct={samples_with_correct}, "
                                                  f"samples_with_wrong={samples_with_wrong}, "
                                                  f"samples_all_correct={samples_all_correct}, "
                                                  f"samples_all_wrong={samples_all_wrong}")
                        # Update old_rewards and save samples for teacher_student curriculum learning
                        if curri_method == "teacher_student":
                            self._update_old_rewards(batch, reward_tensor)
                            self._save_rollout_samples(batch, reward_tensor, epoch)
                            self._update_student_answer_history(batch, reward_tensor)
                            # Compute teacher_student specific metrics
                            self._compute_teacher_student_metrics(batch, reward_tensor, metrics)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        
                        # token_level_scores should already be set above
                        if "token_level_scores" not in batch.batch:
                            batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                        
                        # For teacher_student v8/v7: optional token-level advantage re-assignment.
                        curri_method = self.config.data.get("curri_method", None)
                        ts_version = self.config.data.get("ts_version", None)
                        if curri_method == "teacher_student" and ts_version == "v8":
                            # v8 two-stage assignment:
                            # Stage-1: for t4 mixed samples, compute Dr.GRPO advantages independently for each local part
                            #          using per-part binary correctness, then write to that part's token span.
                            # Stage-2: compute sample-level Dr.GRPO on (t4 samples + original samples), and only assign
                            #          this stage-2 advantage to original samples' valid response tokens.
                            if "advantages" in batch.batch and "teacher_student_mixed" in batch.non_tensor_batch:
                                teacher_student_mixed_flags = batch.non_tensor_batch["teacher_student_mixed"]
                                if isinstance(teacher_student_mixed_flags, np.ndarray):
                                    advantages = batch.batch["advantages"]
                                    modified_advantages = advantages.clone()
                                    device = advantages.device
                                    v8_adv_group_mode = str(self.config.data.get("v8_adv_group_mode", "together")).lower()
                                    v7_t_mix_mode_cfg = str(self.config.data.get("v7_t_mix_mode", "")).lower()
                                    v7_curri_level_cfg = self.config.data.get("v7_curri_level", 4)
                                    try:
                                        v7_curri_level = int(v7_curri_level_cfg)
                                    except (TypeError, ValueError):
                                        v7_curri_level = 4
                                    v7_curri_level = max(1, min(4, v7_curri_level))
                                    mix44_curriculum_t = (
                                        v7_curri_level if (v7_t_mix_mode_cfg == "mix44" and not self.v8_use_adaptive) else 4
                                    )
                                    use_k_as_subproblem_reward_cfg = self.config.data.get("use_k_as_subproblem_reward", False)
                                    if isinstance(use_k_as_subproblem_reward_cfg, str):
                                        use_k_as_subproblem_reward = use_k_as_subproblem_reward_cfg.strip().lower() in (
                                            "1", "true", "yes", "y", "on"
                                        )
                                    else:
                                        use_k_as_subproblem_reward = bool(use_k_as_subproblem_reward_cfg)
                                    adv_shape_mode_cfg = self.config.data.get("adv_shape_mode", 0)
                                    try:
                                        adv_shape_mode = int(adv_shape_mode_cfg)
                                    except (TypeError, ValueError):
                                        adv_shape_mode = 0
                                    if adv_shape_mode == 1:
                                        adv_part_scales = [0.5, 0.8, 1.2, 2.0]
                                    else:
                                        adv_part_scales = [1.0, 1.0, 1.0, 1.0]

                                    if "response_mask" in batch.batch:
                                        response_mask = batch.batch["response_mask"].bool()
                                    else:
                                        response_mask = torch.ones_like(advantages, dtype=torch.bool)

                                    v7_t_arr = batch.non_tensor_batch.get("v7_t", None)
                                    if isinstance(v7_t_arr, np.ndarray):
                                        v7_t_list = v7_t_arr.tolist()
                                    elif v7_t_arr is None:
                                        v7_t_list = None
                                    else:
                                        v7_t_list = list(v7_t_arr)

                                    part_correct = batch.non_tensor_batch.get("subproblem_part_correctness_local", None)
                                    part_token_mask = batch.batch.get("subproblem_part_token_mask", None)
                                    subproblem_correct_count = batch.non_tensor_batch.get("subproblem_correct_count", None)
                                    token_level_scores = batch.batch.get("token_level_scores", None)

                                    def _dr_grpo_adv(sample_rewards: torch.Tensor) -> torch.Tensor:
                                        # Match GRPO-style sample normalization used by this run.
                                        centered = sample_rewards - sample_rewards.mean()
                                        if norm_adv_by_std_in_grpo:
                                            std = centered.std(unbiased=False)
                                            if std.item() > 1e-6:
                                                centered = centered / (std + 1e-6)
                                        return centered

                                    def _k_to_reward(k_val: int) -> float:
                                        if k_val <= 0:
                                            return 0.0
                                        if k_val == 1:
                                            return 0.1
                                        if k_val == 2:
                                            return 0.2
                                        if k_val == 3:
                                            return 0.5
                                        return 1.0

                                    def _get_local_part_reward(part_correct_np_local: np.ndarray, sample_i: int, p_idx: int):
                                        # Return binary reward for local part p_idx in {0,1}.
                                        # For unavailable parts (e.g. v8 mix44 with v7_curri_level < 4), return 0.
                                        try:
                                            corr_val = int(part_correct_np_local[sample_i, p_idx])
                                        except Exception:
                                            return 0
                                        if corr_val not in (0, 1):
                                            return 0
                                        if not use_k_as_subproblem_reward:
                                            return corr_val
                                        # k-aware local rule: only contiguous prefix-correct parts are rewarded as 1.
                                        try:
                                            prefix = np.asarray(part_correct_np_local[sample_i, : p_idx + 1]).astype(np.int64)
                                            return 1 if np.all(prefix == 1) else 0
                                        except Exception:
                                            return corr_val

                                    mixed_indices = np.where(teacher_student_mixed_flags)[0].tolist()
                                    original_indices = np.where(~teacher_student_mixed_flags)[0].tolist()

                                    # t4 subset among mixed samples.
                                    t4_indices = []
                                    for i in mixed_indices:
                                        if v7_t_list is not None and i < len(v7_t_list):
                                            try:
                                                if int(v7_t_list[i]) == mix44_curriculum_t:
                                                    t4_indices.append(int(i))
                                            except Exception:
                                                pass

                                    stage1_assigned = 0
                                    stage2_assigned = 0

                                    if v8_adv_group_mode == "separate":
                                        # v8 separate:
                                        # - t4 samples: per-part local Dr.GRPO in each uid-group.
                                        # - original samples: Dr.GRPO only among originals in each uid-group.
                                        uid_arr = batch.non_tensor_batch.get("uid", None)
                                        if uid_arr is None:
                                            uid_to_indices = {"__all__": list(range(advantages.size(0)))}
                                        else:
                                            uid_to_indices = defaultdict(list)
                                            uid_list = uid_arr.tolist() if isinstance(uid_arr, np.ndarray) else list(uid_arr)
                                            for ii, uid_val in enumerate(uid_list):
                                                uid_to_indices[str(uid_val)].append(ii)

                                        part_correct_np = (
                                            part_correct
                                            if isinstance(part_correct, np.ndarray)
                                            else (np.asarray(part_correct) if part_correct is not None else None)
                                        )
                                        subproblem_k_arr = (
                                            subproblem_correct_count
                                            if isinstance(subproblem_correct_count, np.ndarray)
                                            else (np.asarray(subproblem_correct_count, dtype=object) if subproblem_correct_count is not None else None)
                                        )
                                        v8_mix_group_arr = batch.non_tensor_batch.get("v8_mix_group", None)
                                        if v8_mix_group_arr is not None and not isinstance(v8_mix_group_arr, np.ndarray):
                                            v8_mix_group_arr = np.asarray(v8_mix_group_arr, dtype=np.int32)
                                        is_v8_mix44444_adaptive = bool(
                                            self.ts_version == "v8"
                                            and self.v8_mix44444_adaptive
                                            and str(self.config.data.get("v7_t_mix_mode", "")).lower() == "mix44444"
                                        )

                                        for _, grp_indices in uid_to_indices.items():
                                            grp_t4 = []
                                            grp_orig = []
                                            grp_q = {1: [], 2: [], 3: [], 4: []}
                                            for gi in grp_indices:
                                                is_mixed = bool(teacher_student_mixed_flags[gi])
                                                t_val = None
                                                if v7_t_list is not None and gi < len(v7_t_list):
                                                    try:
                                                        t_val = int(v7_t_list[gi])
                                                    except Exception:
                                                        t_val = None
                                                if is_mixed and t_val == mix44_curriculum_t:
                                                    grp_t4.append(int(gi))
                                                elif not is_mixed:
                                                    grp_orig.append(int(gi))
                                                    if v8_mix_group_arr is not None and gi < len(v8_mix_group_arr):
                                                        try:
                                                            gid = int(v8_mix_group_arr[gi])
                                                            if gid in (1, 2, 3, 4):
                                                                grp_q[gid].append(int(gi))
                                                        except Exception:
                                                            pass

                                            is_mix44_like = len(grp_t4) == 4 and len(grp_orig) == 4
                                            is_mix44444_like = len(grp_t4) == 4 and all(len(grp_q[qid]) == 4 for qid in [1, 2, 3, 4])
                                            total_q = sum(len(grp_q[qid]) for qid in [1, 2, 3, 4])
                                            is_mix44444_adaptive_like = (
                                                is_v8_mix44444_adaptive
                                                and len(grp_t4) >= 4
                                                and len(grp_q[4]) >= 4
                                                and all(len(grp_q[qid]) in (0, 4) for qid in [1, 2, 3])
                                                and total_q == len(grp_orig)
                                                and (len(grp_t4) + total_q) == len(grp_indices)
                                            )

                                            # Apply separate to mix44 and mix44444 layouts.
                                            if (
                                                is_mix44_like
                                                or is_mix44444_like
                                                or is_mix44444_adaptive_like
                                            ) and part_correct_np is not None and part_token_mask is not None:
                                                parse_fail_indices = set()
                                                if subproblem_k_arr is not None:
                                                    for si in grp_t4:
                                                        try:
                                                            if int(subproblem_k_arr[si]) == -1:
                                                                parse_fail_indices.add(si)
                                                        except Exception:
                                                            pass

                                                # Stage-1 on t4: per-part reward in {0,1}, parse-fail treated as [0,0,0,0].
                                                per_part_adv = {}
                                                for p_idx in range(4):
                                                    rewards_local = []
                                                    for sample_i in grp_t4:
                                                        if sample_i in parse_fail_indices:
                                                            rewards_local.append(0.0)
                                                            continue
                                                        corr_val = _get_local_part_reward(part_correct_np, sample_i, p_idx)
                                                        rewards_local.append(1.0 if int(corr_val) == 1 else 0.0)

                                                    reward_t = torch.tensor(rewards_local, device=device, dtype=torch.float32)
                                                    adv_t = _dr_grpo_adv(reward_t)
                                                    adv_t = adv_t * float(adv_part_scales[p_idx])
                                                    per_part_adv[p_idx] = adv_t

                                                    for loc, sample_i in enumerate(grp_t4):
                                                        if sample_i in parse_fail_indices:
                                                            continue
                                                        part_mask = (part_token_mask[sample_i, p_idx] > 0.5) & response_mask[sample_i]
                                                        modified_advantages[sample_i, part_mask] = adv_t[loc]
                                                        stage1_assigned += int(part_mask.sum().item())

                                                # Parse-fail t4 sample: use min over 4 part advantages for this sample.
                                                for sample_i in parse_fail_indices:
                                                    # sample_i's local position in grp_t4
                                                    loc = grp_t4.index(sample_i)
                                                    local_min_adv = torch.stack(
                                                        [per_part_adv[p_idx][loc] for p_idx in range(4)],
                                                        dim=0,
                                                    ).min()
                                                    valid_mask = response_mask[sample_i]
                                                    modified_advantages[sample_i, valid_mask] = local_min_adv
                                                    stage1_assigned += int(valid_mask.sum().item())

                                                # For mix44/mix44444-like paths: any t4 response token not covered
                                                # by part masks is explicitly neutralized to 0 (avoid fallback to
                                                # initial GRPO advantage on leaked tokens).
                                                if is_mix44_like or is_mix44444_like or is_mix44444_adaptive_like:
                                                    for sample_i in grp_t4:
                                                        if sample_i in parse_fail_indices:
                                                            continue
                                                        part_union_mask = (
                                                            (part_token_mask[sample_i, :4] > 0.5).any(dim=0)
                                                            & response_mask[sample_i]
                                                        )
                                                        leak_mask = response_mask[sample_i] & (~part_union_mask)
                                                        if leak_mask.any():
                                                            modified_advantages[sample_i, leak_mask] = 0.0
                                                            stage1_assigned += int(leak_mask.sum().item())

                                                # Stage-2 on original-template branches (separate from t4).
                                                if token_level_scores is not None:
                                                    if is_mix44444_like or is_mix44444_adaptive_like:
                                                        # Four independent GRPO groups: q1/q2/q3/q4, each with 4 samples.
                                                        for qid in [1, 2, 3, 4]:
                                                            grp_qi = grp_q[qid]
                                                            if len(grp_qi) <= 1:
                                                                continue
                                                            q_rewards = []
                                                            for sample_i in grp_qi:
                                                                reward_scalar = (token_level_scores[sample_i] * response_mask[sample_i].float()).sum().item()
                                                                q_rewards.append(float(reward_scalar))
                                                            q_reward_t = torch.tensor(q_rewards, device=device, dtype=torch.float32)
                                                            q_adv_t = _dr_grpo_adv(q_reward_t)
                                                            for j, sample_i in enumerate(grp_qi):
                                                                valid_mask = response_mask[sample_i]
                                                                modified_advantages[sample_i, valid_mask] = q_adv_t[j]
                                                                stage2_assigned += int(valid_mask.sum().item())
                                                    else:
                                                        # mix44 behavior: one GRPO group over original branch.
                                                        if len(grp_orig) > 1:
                                                            orig_rewards = []
                                                            for sample_i in grp_orig:
                                                                reward_scalar = (token_level_scores[sample_i] * response_mask[sample_i].float()).sum().item()
                                                                orig_rewards.append(float(reward_scalar))
                                                            orig_reward_t = torch.tensor(orig_rewards, device=device, dtype=torch.float32)
                                                            orig_adv_t = _dr_grpo_adv(orig_reward_t)
                                                            for j, sample_i in enumerate(grp_orig):
                                                                valid_mask = response_mask[sample_i]
                                                                modified_advantages[sample_i, valid_mask] = orig_adv_t[j]
                                                                stage2_assigned += int(valid_mask.sum().item())
                                            else:
                                                # Fallback to together-mode behavior when group is not mix44-like.
                                                if len(grp_t4) > 0 and part_correct_np is not None and part_token_mask is not None:
                                                    for p_idx in range(4):
                                                        valid_local = []
                                                        rewards_local = []
                                                        for sample_i in grp_t4:
                                                            corr_val = _get_local_part_reward(part_correct_np, sample_i, p_idx)
                                                            if corr_val in (0, 1):
                                                                valid_local.append(sample_i)
                                                                rewards_local.append(float(corr_val))
                                                        if len(valid_local) == 0:
                                                            continue
                                                        reward_t = torch.tensor(rewards_local, device=device, dtype=torch.float32)
                                                        adv_t = _dr_grpo_adv(reward_t)
                                                        adv_t = adv_t * float(adv_part_scales[p_idx])
                                                        for loc, sample_i in enumerate(valid_local):
                                                            part_mask = (part_token_mask[sample_i, p_idx] > 0.5) & response_mask[sample_i]
                                                            modified_advantages[sample_i, part_mask] = adv_t[loc]
                                                            stage1_assigned += int(part_mask.sum().item())

                                                if len(grp_t4) > 0 and len(grp_orig) > 0 and subproblem_k_arr is not None:
                                                    group_indices = grp_t4 + grp_orig
                                                    group_rewards = []
                                                    for sample_i in group_indices:
                                                        if sample_i in grp_t4:
                                                            try:
                                                                k_val = int(subproblem_k_arr[sample_i])
                                                            except Exception:
                                                                k_val = 0
                                                            group_rewards.append(_k_to_reward(k_val))
                                                        else:
                                                            if token_level_scores is not None:
                                                                reward_scalar = (token_level_scores[sample_i] * response_mask[sample_i].float()).sum().item()
                                                            else:
                                                                reward_scalar = 0.0
                                                            group_rewards.append(float(reward_scalar))
                                                    group_reward_t = torch.tensor(group_rewards, device=device, dtype=torch.float32)
                                                    group_adv_t = _dr_grpo_adv(group_reward_t)
                                                    for j, sample_i in enumerate(group_indices):
                                                        if sample_i in grp_orig:
                                                            valid_mask = response_mask[sample_i]
                                                            modified_advantages[sample_i, valid_mask] = group_adv_t[j]
                                                            stage2_assigned += int(valid_mask.sum().item())
                                    else:
                                        # together (default): current v8 behavior.
                                        # Stage-1: per-part local Dr.GRPO on t4 samples.
                                        if (
                                            len(t4_indices) > 0
                                            and part_correct is not None
                                            and part_token_mask is not None
                                        ):
                                            part_correct_np = (
                                                part_correct
                                                if isinstance(part_correct, np.ndarray)
                                                else np.asarray(part_correct)
                                            )
                                            for p_idx in range(4):
                                                valid_local = []
                                                rewards_local = []
                                                for sample_i in t4_indices:
                                                    corr_val = _get_local_part_reward(part_correct_np, sample_i, p_idx)
                                                    if corr_val in (0, 1):
                                                        valid_local.append(sample_i)
                                                        rewards_local.append(float(corr_val))
                                                if len(valid_local) == 0:
                                                    continue
                                                reward_t = torch.tensor(rewards_local, device=device, dtype=torch.float32)
                                                adv_t = _dr_grpo_adv(reward_t)
                                                adv_t = adv_t * float(adv_part_scales[p_idx])
                                                for loc, sample_i in enumerate(valid_local):
                                                    part_mask = (part_token_mask[sample_i, p_idx] > 0.5) & response_mask[sample_i]
                                                    modified_advantages[sample_i, part_mask] = adv_t[loc]
                                                    stage1_assigned += int(part_mask.sum().item())

                                        # Stage-2: sample-level Dr.GRPO on (t4 + original) for original samples only.
                                        if len(t4_indices) > 0 and len(original_indices) > 0 and subproblem_correct_count is not None:
                                            group_indices = t4_indices + original_indices
                                            group_rewards = []
                                            for sample_i in group_indices:
                                                if sample_i in t4_indices:
                                                    try:
                                                        k_val = int(subproblem_correct_count[sample_i])
                                                    except Exception:
                                                        k_val = 0
                                                    group_rewards.append(_k_to_reward(k_val))
                                                else:
                                                    if token_level_scores is not None:
                                                        reward_scalar = (token_level_scores[sample_i] * response_mask[sample_i].float()).sum().item()
                                                    else:
                                                        reward_scalar = 0.0
                                                    group_rewards.append(float(reward_scalar))
                                            group_reward_t = torch.tensor(group_rewards, device=device, dtype=torch.float32)
                                            group_adv_t = _dr_grpo_adv(group_reward_t)
                                            for j, sample_i in enumerate(group_indices):
                                                if sample_i in original_indices:
                                                    valid_mask = response_mask[sample_i]
                                                    modified_advantages[sample_i, valid_mask] = group_adv_t[j]
                                                    stage2_assigned += int(valid_mask.sum().item())

                                    batch.batch["advantages"] = modified_advantages
                                    batch.batch["returns"] = modified_advantages.clone()
                                    print(
                                        f"[v8_advantage] mode={v8_adv_group_mode}, "
                                        f"use_k_as_subproblem_reward={use_k_as_subproblem_reward}, "
                                        f"adv_shape_mode={adv_shape_mode}, "
                                        f"stage1_tokens={stage1_assigned}, "
                                        f"stage2_tokens={stage2_assigned}, t4_samples={len(t4_indices)}, "
                                        f"original_samples={len(original_indices)}"
                                    )

                        elif curri_method == "teacher_student" and ts_version == "v7":
                            if "token_correctness_mask" in batch.batch and "advantages" in batch.batch:
                                # Only modify advantages for teacher_student_mixed=True samples
                                if "teacher_student_mixed" in batch.non_tensor_batch:
                                    teacher_student_mixed_flags = batch.non_tensor_batch["teacher_student_mixed"]
                                    if isinstance(teacher_student_mixed_flags, np.ndarray):
                                        mixed_indices = np.where(teacher_student_mixed_flags)[0]
                                        
                                        if len(mixed_indices) > 0:
                                            token_correctness_mask = batch.batch["token_correctness_mask"]  # (batch_size, response_length)
                                            advantages = batch.batch["advantages"]  # (batch_size, response_length)
                                            
                                            # Start with original advantages (non-mixed samples will keep original)
                                            modified_advantages = advantages.clone()
                                            v7_adv_group_mode = str(self.config.data.get("v7_adv_group_mode", "together")).lower()
                                            v7_t_mix_mode_cfg = str(self.config.data.get("v7_t_mix_mode", "legacy_sub8")).lower()
                                            v7_reward_map_mode = str(self.config.data.get("v7_reward_map_mode", "legacy_k")).lower()
                                            v7_separate_enabled = v7_adv_group_mode == "separate" and v7_t_mix_mode_cfg == "mix44"
                                            v7_separate_groups_applied = 0

                                            # v7 mix44 separate:
                                            # For each problem group of 8 samples, split into:
                                            # - 4 mixed t4 samples: subgroup GRPO on v7 reward mapping
                                            # - 4 original samples: subgroup GRPO on reward_fn reward
                                            if v7_separate_enabled:
                                                response_mask_all = batch.batch.get("response_mask", None)
                                                token_level_scores_all = batch.batch.get("token_level_scores", None)
                                                problem_ids_arr = batch.non_tensor_batch.get("problem_id", None)
                                                v7_t_arr = batch.non_tensor_batch.get("v7_t", None)
                                                reward_models = batch.non_tensor_batch.get("reward_model", None)
                                                subproblem_correct_count = batch.non_tensor_batch.get("subproblem_correct_count", None)
                                                if (
                                                    response_mask_all is not None
                                                    and token_level_scores_all is not None
                                                    and problem_ids_arr is not None
                                                    and v7_t_arr is not None
                                                    and subproblem_correct_count is not None
                                                ):
                                                    if isinstance(problem_ids_arr, np.ndarray):
                                                        problem_ids_list = [str(x) for x in problem_ids_arr.tolist()]
                                                    else:
                                                        problem_ids_list = [str(x) for x in list(problem_ids_arr)]
                                                    if isinstance(v7_t_arr, np.ndarray):
                                                        v7_t_list = v7_t_arr.tolist()
                                                    else:
                                                        v7_t_list = list(v7_t_arr)

                                                    mixed_indices_set = set(int(x) for x in mixed_indices.tolist())
                                                    pid_to_indices = defaultdict(list)
                                                    for i, pid in enumerate(problem_ids_list):
                                                        pid_to_indices[pid].append(i)

                                                    def _dr_grpo_adv_v7(sample_rewards: torch.Tensor) -> torch.Tensor:
                                                        centered = sample_rewards - sample_rewards.mean()
                                                        if norm_adv_by_std_in_grpo:
                                                            std = centered.std(unbiased=False)
                                                            if std.item() > 1e-6:
                                                                centered = centered / (std + 1e-6)
                                                        return centered

                                                    def _k_to_reward_v7(sample_i: int, k_val: int) -> float:
                                                        if k_val <= 0:
                                                            return 0.0
                                                        if v7_reward_map_mode == "absolute_difficulty":
                                                            q_start = 1
                                                            if reward_models is not None and sample_i < len(reward_models):
                                                                rm = reward_models[sample_i]
                                                                if not isinstance(rm, dict):
                                                                    rm = rm.__dict__ if hasattr(rm, "__dict__") else {}
                                                                try:
                                                                    q_start = int(rm.get("v7_q_start", 1))
                                                                except Exception:
                                                                    q_start = 1
                                                            abs_q_idx = max(1, min(4, q_start + int(k_val) - 1))
                                                            if abs_q_idx == 1:
                                                                return 0.1
                                                            if abs_q_idx == 2:
                                                                return 0.2
                                                            if abs_q_idx == 3:
                                                                return 0.5
                                                            return 1.0
                                                        if k_val == 1:
                                                            return 0.1
                                                        if k_val == 2:
                                                            return 0.2
                                                        if k_val == 3:
                                                            return 0.5
                                                        return 1.0

                                                    for _pid, idxs in pid_to_indices.items():
                                                        if len(idxs) != 8:
                                                            continue
                                                        grp_mixed = [i for i in idxs if i in mixed_indices_set]
                                                        grp_orig = [i for i in idxs if i not in mixed_indices_set]
                                                        if len(grp_mixed) != 4 or len(grp_orig) != 4:
                                                            continue

                                                        is_t4_group = True
                                                        for i in grp_mixed:
                                                            try:
                                                                if int(v7_t_list[i]) != 4:
                                                                    is_t4_group = False
                                                                    break
                                                            except Exception:
                                                                is_t4_group = False
                                                                break
                                                        if not is_t4_group:
                                                            continue

                                                        mixed_rewards = []
                                                        for i in grp_mixed:
                                                            try:
                                                                k_val = int(subproblem_correct_count[i])
                                                            except Exception:
                                                                k_val = 0
                                                            mixed_rewards.append(_k_to_reward_v7(i, k_val))
                                                        r_m = torch.tensor(mixed_rewards, device=advantages.device, dtype=torch.float32)
                                                        a_m = _dr_grpo_adv_v7(r_m)
                                                        for j, sample_i in enumerate(grp_mixed):
                                                            valid_mask = response_mask_all[sample_i].bool()
                                                            modified_advantages[sample_i, valid_mask] = a_m[j]

                                                        orig_rewards = []
                                                        for i in grp_orig:
                                                            valid_mask = response_mask_all[i].bool()
                                                            reward_scalar = (token_level_scores_all[i] * valid_mask.float()).sum().item()
                                                            orig_rewards.append(float(reward_scalar))
                                                        r_o = torch.tensor(orig_rewards, device=advantages.device, dtype=torch.float32)
                                                        a_o = _dr_grpo_adv_v7(r_o)
                                                        for j, sample_i in enumerate(grp_orig):
                                                            valid_mask = response_mask_all[sample_i].bool()
                                                            modified_advantages[sample_i, valid_mask] = a_o[j]
                                                        v7_separate_groups_applied += 1
                                            
                                            # Convert mixed_indices to tensor for indexing
                                            mixed_indices_tensor = torch.tensor(mixed_indices, device=advantages.device, dtype=torch.long)
                                            
                                            # For each mixed sample, compute the absolute value of correct token advantage
                                            # Since GRPO is sequence-level, all tokens in a sample have the same advantage
                                            correctness_mask_bool = token_correctness_mask.bool()  # (batch_size, response_length)
                                            
                                            # Get advantages for mixed samples only
                                            mixed_advantages = modified_advantages[mixed_indices_tensor]  # (num_mixed, response_length)
                                            mixed_correctness_mask = correctness_mask_bool[mixed_indices_tensor]  # (num_mixed, response_length)
                                            
                                            # For each mixed sample, get the advantage value (same for all tokens due to GRPO)
                                            # We'll use the first token's advantage as representative, or find first correct token
                                            mixed_sample_advantages = mixed_advantages[:, 0]  # (num_mixed,) - use first token as default
                                            
                                            # For samples with correct tokens, use first correct token's advantage
                                            # Find first correct token index for each mixed sample
                                            has_correct = mixed_correctness_mask.any(dim=1)  # (num_mixed,) - whether sample has correct tokens
                                            if has_correct.any():
                                                # For samples with correct tokens, find first correct token index
                                                # Find first True in each row
                                                first_correct_indices = mixed_correctness_mask.long().argmax(dim=1)  # (num_mixed,)
                                                # Get advantages from first correct token using advanced indexing
                                                batch_indices_local = torch.arange(mixed_advantages.shape[0], device=advantages.device)  # (num_mixed,)
                                                correct_sample_advantages = mixed_advantages[batch_indices_local, first_correct_indices]  # (num_mixed,)
                                                # Use correct token advantage where available, otherwise use first token
                                                mixed_sample_advantages = torch.where(
                                                    has_correct,
                                                    correct_sample_advantages,
                                                    mixed_sample_advantages
                                                )
                                            
                                            # Get credit assignment mode from config (default to 1)
                                            credit_assignment_mode = self.config.data.get("CREDIT_ASSIGNMENT_MODE", 1)
                                            # Optional clip threshold for token-level advantages after reassignment.
                                            # Disabled by default; enabled when data.adv_clip > 0.
                                            adv_clip = self.config.data.get("adv_clip", None)
                                            try:
                                                adv_clip = float(adv_clip) if adv_clip is not None else None
                                            except (TypeError, ValueError):
                                                adv_clip = None
                                            
                                            # Get response mask for valid tokens (if available)
                                            if "response_mask" in batch.batch:
                                                response_mask = batch.batch["response_mask"]
                                                mixed_response_mask = response_mask[mixed_indices_tensor]  # (num_mixed, response_length)
                                                mixed_valid_mask = mixed_response_mask.bool()  # (num_mixed, response_length)
                                            else:
                                                mixed_valid_mask = torch.ones_like(mixed_correctness_mask, dtype=torch.bool)
                                            
                                            # Compute n_c and n_w for each sample (only count valid tokens)
                                            # n_c: number of correct tokens, n_w: number of wrong tokens
                                            mixed_valid_correct_mask = mixed_correctness_mask & mixed_valid_mask  # (num_mixed, response_length)
                                            mixed_valid_wrong_mask = (~mixed_correctness_mask) & mixed_valid_mask  # (num_mixed, response_length)
                                            
                                            n_c = mixed_valid_correct_mask.sum(dim=1).float()  # (num_mixed,) - number of correct tokens per sample
                                            n_w = mixed_valid_wrong_mask.sum(dim=1).float()  # (num_mixed,) - number of wrong tokens per sample
                                            
                                            # Get original advantage A for each sample
                                            A = mixed_sample_advantages  # (num_mixed,) - original advantage
                                            abs_A = torch.abs(A)  # (num_mixed,)
                                            
                                            # Initialize modified advantages
                                            modified_mixed_advantages = mixed_advantages.clone()
                                            
                                            # Apply credit assignment based on mode
                                            # Only process samples with both correct and wrong tokens (n_c > 0 and n_w > 0)
                                            # For boundary cases (n_c=0 or n_w=0), keep original advantages unchanged
                                            if credit_assignment_mode == 0:
                                                # Mode 0: pure GRPO token-level behavior for mixed samples.
                                                # Do not re-assign correct/wrong token advantages:
                                                # correct_advantage = A, wrong_advantage = A
                                                # Keep modified_mixed_advantages as original mixed_advantages.
                                                pass

                                            elif credit_assignment_mode == 1:
                                                # Mode 1: wrong token = -abs(A), correct token = A + (abs(A) + A) * n_w / n_c
                                                # Only apply to samples with both correct and wrong tokens
                                                has_both = (n_c > 0) & (n_w > 0)  # (num_mixed,)
                                                
                                                if has_both.any():
                                                    # Compute correct token advantage: A + (abs(A) + A) * n_w / n_c
                                                    correct_advantage = A + (abs_A + A) * n_w / n_c  # (num_mixed,)
                                                    
                                                    # Compute wrong token advantage: -abs(A)
                                                    wrong_advantage = -abs_A  # (num_mixed,)
                                                    
                                                    # Expand to (num_mixed, response_length)
                                                    correct_advantage_expanded = correct_advantage.unsqueeze(-1).expand_as(mixed_advantages)
                                                    wrong_advantage_expanded = wrong_advantage.unsqueeze(-1).expand_as(mixed_advantages)
                                                    
                                                    # Apply assignment only to samples with both correct and wrong tokens
                                                    modified_mixed_advantages = torch.where(
                                                        has_both.unsqueeze(-1),
                                                        torch.where(
                                                            mixed_valid_correct_mask,
                                                            correct_advantage_expanded,
                                                            torch.where(
                                                                mixed_valid_wrong_mask,
                                                                wrong_advantage_expanded,
                                                                modified_mixed_advantages
                                                            )
                                                        ),
                                                        modified_mixed_advantages  # Keep original for boundary cases
                                                    )
                                                
                                            elif credit_assignment_mode == 2:
                                                # Mode 2: correct token = abs(A), wrong token = A + (A - abs(A)) * n_c / n_w
                                                # Only apply to samples with both correct and wrong tokens
                                                has_both = (n_c > 0) & (n_w > 0)  # (num_mixed,)
                                                
                                                if has_both.any():
                                                    # Compute correct token advantage: abs(A)
                                                    correct_advantage = abs_A  # (num_mixed,)
                                                    
                                                    # Compute wrong token advantage: A + (A - abs(A)) * n_c / n_w
                                                    wrong_advantage = A + (A - abs_A) * n_c / n_w  # (num_mixed,)
                                                    
                                                    # Expand to (num_mixed, response_length)
                                                    correct_advantage_expanded = correct_advantage.unsqueeze(-1).expand_as(mixed_advantages)
                                                    wrong_advantage_expanded = wrong_advantage.unsqueeze(-1).expand_as(mixed_advantages)
                                                    
                                                    # Apply assignment only to samples with both correct and wrong tokens
                                                    modified_mixed_advantages = torch.where(
                                                        has_both.unsqueeze(-1),
                                                        torch.where(
                                                            mixed_valid_correct_mask,
                                                            correct_advantage_expanded,
                                                            torch.where(
                                                                mixed_valid_wrong_mask,
                                                                wrong_advantage_expanded,
                                                                modified_mixed_advantages
                                                            )
                                                        ),
                                                        modified_mixed_advantages  # Keep original for boundary cases
                                                    )
                                            
                                            elif credit_assignment_mode == 3:
                                                # Mode 3: Adaptive strategy based on A's sign
                                                # If A > 0: wrong token = -abs(A), correct token = A + (abs(A) + A) * n_w / n_c
                                                # If A <= 0: correct token = abs(A), wrong token = A + (A - abs(A)) * n_c / n_w
                                                # Only apply to samples with both correct and wrong tokens
                                                has_both = (n_c > 0) & (n_w > 0)  # (num_mixed,)
                                                
                                                if has_both.any():
                                                    # Determine which samples have A > 0
                                                    A_positive = A > 0  # (num_mixed,)
                                                    
                                                    # For A > 0: use mode 1-like strategy
                                                    # wrong token = -abs(A)
                                                    wrong_advantage_positive = -abs_A  # (num_mixed,)
                                                    # correct token = A + (abs(A) + A) * n_w / n_c
                                                    correct_advantage_positive = A + (abs_A + A) * n_w / n_c  # (num_mixed,)
                                                    
                                                    # For A <= 0: use mode 2-like strategy
                                                    # correct token = abs(A)
                                                    correct_advantage_negative = abs_A  # (num_mixed,)
                                                    # wrong token = A + (A - abs(A)) * n_c / n_w
                                                    wrong_advantage_negative = A + (A - abs_A) * n_c / n_w  # (num_mixed,)
                                                    
                                                    # Select advantages based on A's sign
                                                    correct_advantage = torch.where(
                                                        A_positive,
                                                        correct_advantage_positive,
                                                        correct_advantage_negative
                                                    )  # (num_mixed,)
                                                    wrong_advantage = torch.where(
                                                        A_positive,
                                                        wrong_advantage_positive,
                                                        wrong_advantage_negative
                                                    )  # (num_mixed,)
                                                    
                                                    # Expand to (num_mixed, response_length)
                                                    correct_advantage_expanded = correct_advantage.unsqueeze(-1).expand_as(mixed_advantages)
                                                    wrong_advantage_expanded = wrong_advantage.unsqueeze(-1).expand_as(mixed_advantages)
                                                    
                                                    # Apply assignment only to samples with both correct and wrong tokens
                                                    modified_mixed_advantages = torch.where(
                                                        has_both.unsqueeze(-1),
                                                        torch.where(
                                                            mixed_valid_correct_mask,
                                                            correct_advantage_expanded,
                                                            torch.where(
                                                                mixed_valid_wrong_mask,
                                                                wrong_advantage_expanded,
                                                                modified_mixed_advantages
                                                            )
                                                        ),
                                                        modified_mixed_advantages  # Keep original for boundary cases
                                                    )
                                            
                                            elif credit_assignment_mode == 4:
                                                # Mode 4: Adaptive strategy based on A's sign with zero assignment
                                                # If A > 0: wrong token = 0, correct token = A + A * n_w / n_c
                                                # If A < 0: correct token = 0, wrong token = A + A * n_c / n_w
                                                # Only apply to samples with both correct and wrong tokens
                                                has_both = (n_c > 0) & (n_w > 0)  # (num_mixed,)
                                                
                                                if has_both.any():
                                                    # Determine which samples have A > 0 and A < 0
                                                    A_positive = A > 0  # (num_mixed,)
                                                    A_negative = A < 0  # (num_mixed,)
                                                    
                                                    # For A > 0: wrong token = 0, correct token = A + A * n_w / n_c
                                                    wrong_advantage_positive = torch.zeros_like(A)  # (num_mixed,)
                                                    correct_advantage_positive = A + A * n_w / n_c  # (num_mixed,)
                                                    
                                                    # For A < 0: correct token = 0, wrong token = A + A * n_c / n_w
                                                    correct_advantage_negative = torch.zeros_like(A)  # (num_mixed,)
                                                    wrong_advantage_negative = A + A * n_c / n_w  # (num_mixed,)
                                                    
                                                    # For A == 0: keep original (both set to 0)
                                                    # Select advantages based on A's sign
                                                    correct_advantage = torch.where(
                                                        A_positive,
                                                        correct_advantage_positive,
                                                        torch.where(
                                                            A_negative,
                                                            correct_advantage_negative,
                                                            torch.zeros_like(A)  # A == 0 case
                                                        )
                                                    )  # (num_mixed,)
                                                    wrong_advantage = torch.where(
                                                        A_positive,
                                                        wrong_advantage_positive,
                                                        torch.where(
                                                            A_negative,
                                                            wrong_advantage_negative,
                                                            torch.zeros_like(A)  # A == 0 case
                                                        )
                                                    )  # (num_mixed,)
                                                    
                                                    # Expand to (num_mixed, response_length)
                                                    correct_advantage_expanded = correct_advantage.unsqueeze(-1).expand_as(mixed_advantages)
                                                    wrong_advantage_expanded = wrong_advantage.unsqueeze(-1).expand_as(mixed_advantages)
                                                    
                                                    # Apply assignment only to samples with both correct and wrong tokens
                                                    modified_mixed_advantages = torch.where(
                                                        has_both.unsqueeze(-1),
                                                        torch.where(
                                                            mixed_valid_correct_mask,
                                                            correct_advantage_expanded,
                                                            torch.where(
                                                                mixed_valid_wrong_mask,
                                                                wrong_advantage_expanded,
                                                                modified_mixed_advantages
                                                            )
                                                        ),
                                                        modified_mixed_advantages  # Keep original for boundary cases
                                                    )
                                            
                                            elif credit_assignment_mode == 5:
                                                # Mode 5: correct token = 3 * abs(A), wrong token = A + (A - 3 * abs_A) * n_c / n_w
                                                # Only apply to samples with both correct and wrong tokens
                                                has_both = (n_c > 0) & (n_w > 0)  # (num_mixed,)
                                                
                                                if has_both.any():
                                                    # Compute correct token advantage: 3 * abs(A)
                                                    correct_advantage = 3 * abs_A  # (num_mixed,)
                                                    
                                                    # Compute wrong token advantage: A + (A - 3 * abs_A) * n_c / n_w
                                                    wrong_advantage = A + (A - 3 * abs_A) * n_c / n_w  # (num_mixed,)
                                                    
                                                    # Expand to (num_mixed, response_length)
                                                    correct_advantage_expanded = correct_advantage.unsqueeze(-1).expand_as(mixed_advantages)
                                                    wrong_advantage_expanded = wrong_advantage.unsqueeze(-1).expand_as(mixed_advantages)
                                                    
                                                    # Apply assignment only to samples with both correct and wrong tokens
                                                    modified_mixed_advantages = torch.where(
                                                        has_both.unsqueeze(-1),
                                                        torch.where(
                                                            mixed_valid_correct_mask,
                                                            correct_advantage_expanded,
                                                            torch.where(
                                                                mixed_valid_wrong_mask,
                                                                wrong_advantage_expanded,
                                                                modified_mixed_advantages
                                                            )
                                                        ),
                                                        modified_mixed_advantages  # Keep original for boundary cases
                                                    )
                                            
                                            else:
                                                # Default behavior (original implementation)
                                                abs_correct_advantages_expanded = abs_A.unsqueeze(-1).expand_as(mixed_advantages)
                                                modified_mixed_advantages = torch.where(
                                                    mixed_correctness_mask,  # If correct token
                                                    mixed_advantages,  # Keep original advantage for each token position
                                                    -abs_correct_advantages_expanded  # If wrong token, set to -abs(correct_advantage)
                                                )

                                            # Optional clipping on valid response tokens only.
                                            if adv_clip is not None and adv_clip > 0:
                                                modified_mixed_advantages = torch.where(
                                                    mixed_valid_mask,
                                                    torch.clamp(modified_mixed_advantages, min=-adv_clip, max=adv_clip),
                                                    modified_mixed_advantages,
                                                )
                                            
                                            # Apply response_mask to ensure we only modify valid tokens
                                            if "response_mask" in batch.batch:
                                                response_mask = batch.batch["response_mask"]
                                                mixed_response_mask = response_mask[mixed_indices_tensor]
                                                modified_mixed_advantages = modified_mixed_advantages * mixed_response_mask
                                            
                                            # Update advantages only for mixed samples
                                            modified_advantages[mixed_indices_tensor] = modified_mixed_advantages
                                            
                                            # Update advantages
                                            batch.batch["advantages"] = modified_advantages
                                            
                                            # Also update returns to match (for consistency)
                                            batch.batch["returns"] = modified_advantages.clone()
                                            
                                            # Print statistics (only count valid tokens, exclude padding, only for mixed samples)
                                            if "response_mask" in batch.batch:
                                                response_mask = batch.batch["response_mask"]
                                                mixed_response_mask = response_mask[mixed_indices_tensor]
                                                mixed_valid_mask = mixed_response_mask.bool()  # (num_mixed, response_length)
                                                
                                                # Count only valid tokens for mixed samples (exclude padding)
                                                mixed_valid_correct_count = ((mixed_correctness_mask > 0.5) & mixed_valid_mask).sum().item()
                                                mixed_valid_wrong_count = ((mixed_correctness_mask <= 0.5) & mixed_valid_mask).sum().item()
                                                mixed_valid_total_count = mixed_valid_mask.sum().item()
                                                
                                                # Compute average correct and wrong advantage for logging (only valid tokens, only mixed samples)
                                                mixed_valid_correct_mask = (mixed_correctness_mask > 0.5) & mixed_valid_mask
                                                mixed_valid_wrong_mask = (mixed_correctness_mask <= 0.5) & mixed_valid_mask
                                                mixed_correct_advantages = mixed_advantages[mixed_valid_correct_mask]
                                                mixed_wrong_advantages = modified_mixed_advantages[mixed_valid_wrong_mask]
                                                avg_correct_adv = mixed_correct_advantages.mean().item() if len(mixed_correct_advantages) > 0 else 0.0
                                                avg_wrong_adv = mixed_wrong_advantages.mean().item() if len(mixed_wrong_advantages) > 0 else 0.0
                                                
                                                # Debug: Print advantage modification statistics (only for mixed samples)
                                                mixed_original_avg_adv = mixed_advantages[mixed_valid_mask].mean().item()
                                                mixed_original_std_adv = mixed_advantages[mixed_valid_mask].std().item()
                                                mixed_modified_avg_adv = modified_mixed_advantages[mixed_valid_mask].mean().item()
                                                mixed_modified_std_adv = modified_mixed_advantages[mixed_valid_mask].std().item()
                                                
                                                # Debug: Print per-sample statistics (only for mixed samples)
                                                mixed_sample_has_correct = mixed_correctness_mask.any(dim=1)  # (num_mixed,)
                                                mixed_samples_with_correct = mixed_sample_has_correct.sum().item()
                                                mixed_samples_with_wrong = (~mixed_sample_has_correct).sum().item()
                                                
                                                # Debug: Print advantage range (only for mixed samples)
                                                mixed_original_min = mixed_advantages[mixed_valid_mask].min().item()
                                                mixed_original_max = mixed_advantages[mixed_valid_mask].max().item()
                                                mixed_modified_min = modified_mixed_advantages[mixed_valid_mask].min().item()
                                                mixed_modified_max = modified_mixed_advantages[mixed_valid_mask].max().item()
                                                
                                                print(f"[v7_advantage] Modified advantages for {len(mixed_indices)} mixed samples: "
                                                      f"correct_tokens keep original (avg={avg_correct_adv:.4f}), "
                                                      f"wrong_tokens set to -abs(correct) (avg={avg_wrong_adv:.4f}), "
                                                      f"valid_correct_tokens={mixed_valid_correct_count}, valid_wrong_tokens={mixed_valid_wrong_count}, "
                                                      f"valid_total={mixed_valid_total_count}")
                                                print(f"[v7_debug] Mixed samples advantage statistics: original (mean={mixed_original_avg_adv:.4f}, std={mixed_original_std_adv:.4f}, "
                                                      f"min={mixed_original_min:.4f}, max={mixed_original_max:.4f}), "
                                                      f"modified (mean={mixed_modified_avg_adv:.4f}, std={mixed_modified_std_adv:.4f}, "
                                                      f"min={mixed_modified_min:.4f}, max={mixed_modified_max:.4f})")
                                                print(f"[v7_debug] Mixed samples distribution: samples_with_correct={mixed_samples_with_correct}, "
                                                      f"samples_with_wrong={mixed_samples_with_wrong}, "
                                                      f"correct_token_ratio={mixed_valid_correct_count/mixed_valid_total_count*100:.1f}%")
                                            else:
                                                # Fallback: count all tokens if response_mask not available (only for mixed samples)
                                                mixed_correct_count = (mixed_correctness_mask > 0.5).sum().item()
                                                mixed_wrong_count = (mixed_correctness_mask <= 0.5).sum().item()
                                                mixed_total_count = mixed_correctness_mask.numel()
                                                
                                                mixed_correct_advantages = mixed_advantages[mixed_correctness_mask > 0.5]
                                                mixed_wrong_advantages = modified_mixed_advantages[mixed_correctness_mask <= 0.5]
                                                avg_correct_adv = mixed_correct_advantages.mean().item() if len(mixed_correct_advantages) > 0 else 0.0
                                                avg_wrong_adv = mixed_wrong_advantages.mean().item() if len(mixed_wrong_advantages) > 0 else 0.0
                                                
                                                print(f"[v7_advantage] Modified advantages for {len(mixed_indices)} mixed samples: "
                                                      f"correct_tokens keep original (avg={avg_correct_adv:.4f}), "
                                                      f"wrong_tokens set to -abs(correct) (avg={avg_wrong_adv:.4f}), "
                                                      f"correct_tokens={mixed_correct_count}, wrong_tokens={mixed_wrong_count}, total={mixed_total_count} "
                                                      f"(NOTE: includes padding tokens, response_mask not available)")
                                            if v7_separate_enabled:
                                                print(
                                                    f"[v7_advantage] mode={v7_adv_group_mode}, mix_mode={v7_t_mix_mode_cfg}, "
                                                    f"separate_groups_applied={v7_separate_groups_applied}"
                                                )
                                        else:
                                            # No mixed samples, no modification needed
                                            pass
                                    else:
                                        # teacher_student_mixed_flags is not numpy array, skip
                                        pass
                                else:
                                    # No teacher_student_mixed flag, skip modification
                                    pass

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)
                
                # Generate hints for teacher_student curriculum learning (after batch processing)
                if curri_method == "teacher_student" and self.hint_generator is not None:
                    self._generate_teacher_hints()

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
