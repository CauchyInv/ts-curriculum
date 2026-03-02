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

import copy
import json
import logging
import os
import re
import traceback
from collections import defaultdict
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import pad_sequence_to_length

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \\*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_samples = max_samples
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.image_patch_size = config.get("image_patch_size", 14)
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        self.tool_config_path = config.get("tool_config_path", None)
        self.tool_schemas = None
        if self.tool_config_path:
            try:
                from verl.tools.utils.tool_registry import initialize_tools_from_config

                tool_list = initialize_tools_from_config(self.tool_config_path)
                # match ToolAgentLoop behaviour: model_dump to plain dicts
                self.tool_schemas = [
                    tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list
                ]
            except Exception as e:
                logger.warning("Failed to initialize tools from %s: %s", self.tool_config_path, e)
                self.tool_schemas = None

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count()) if self.num_workers is not None else None
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")

        # Teacher-student version related parameters (optional)
        self.ts_version = config.get("ts_version", None)
        self.teacher_hint_dict = config.get("teacher_hint_dict", None)
        self.crafted_wrong_answer = config.get("crafted_wrong_answer", None)
        self.teacher_lemmas = config.get("teacher_lemmas", None)
        self.subproblems_dict = config.get("subproblems_dict", None)
        self.subproblems_jsonl_path = "/hyk/algorithm_new/qinghua/yueyang/verl/data/subproblems.jsonl"
        
        # Load subproblems_dict if jsonl_path is provided
        if self.subproblems_jsonl_path is not None and self.subproblems_dict is None:
            self.subproblems_dict = {}
            try:
                with open(self.subproblems_jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            problem_id = str(data.get('problem_id', ''))
                            if problem_id:
                                self.subproblems_dict[problem_id] = data
            except Exception as e:
                logger.warning(f"Failed to load subproblems from {self.subproblems_jsonl_path}: {e}")
                self.subproblems_dict = {}
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    try:
                        messages = self._build_messages(doc)
                        # pass tool schemas if available so the processor can format prompts
                        apply_kwargs = dict(**self.apply_chat_template_kwargs)
                        if self.tool_schemas is not None:
                            apply_kwargs["tools"] = self.tool_schemas

                        raw_prompt = self.processor.apply_chat_template(
                            messages, add_generation_prompt=True, tokenize=False, **apply_kwargs
                        )
                        if image_key in doc and doc[image_key]:
                            images = [
                                process_image(image, image_patch_size=self.image_patch_size) for image in doc[image_key]
                            ]
                        else:
                            images = None

                        if video_key in doc and doc[video_key]:
                            videos, video_metadata = zip(
                                *[
                                    process_video(
                                        video, image_patch_size=self.image_patch_size, return_video_metadata=True
                                    )
                                    for video in doc[video_key]
                                ],
                                strict=True,
                            )
                            videos = list(videos)
                            video_metadata = list(video_metadata)
                            videos_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False}
                        else:
                            videos = None
                            videos_kwargs = {}

                        return len(
                            processor(text=[raw_prompt], images=images, videos=videos, videos_kwargs=videos_kwargs)[
                                "input_ids"
                            ][0]
                        )
                    except Exception:
                        print("Error processing one of the samples, skipping...")
                        traceback.print_exc()
                        return self.max_prompt_length + 1

            else:

                def doc2len(doc) -> int:
                    try:
                        apply_kwargs = dict(**self.apply_chat_template_kwargs)
                        if self.tool_schemas is not None:
                            apply_kwargs["tools"] = self.tool_schemas

                        return len(
                            tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True, **apply_kwargs)
                        )
                    except Exception:
                        print("Error processing one of the samples, skipping...")
                        traceback.print_exc()
                        return self.max_prompt_length + 1

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            multi_modal_data = {}

            images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                images = [process_image(image, image_patch_size=self.image_patch_size) for image in row_dict_images]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            videos_kwargs = {}
            row_dict_videos = row_dict.pop(self.video_key, None)
            if row_dict_videos:
                videos, video_metadata = zip(
                    *[
                        process_video(video, image_patch_size=self.image_patch_size, return_video_metadata=True)
                        for video in row_dict_videos
                    ],
                    strict=True,
                )
                videos = list(videos)
                video_metadata = list(video_metadata)
                videos_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False}

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [
                    (video.numpy(), metadata) for video, metadata in zip(videos, video_metadata, strict=True)
                ]

            model_inputs = self.processor(
                text=[raw_prompt], images=images, videos=videos, videos_kwargs=videos_kwargs, return_tensors="pt"
            )

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            if self.apply_chat_template_kwargs.get("chat_template") is None:
                assert hasattr(self.tokenizer, "chat_template"), (
                    "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "
                    "models like GLM can copy chat_template.jinja from instruct models"
                )
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        elif self.processor is not None and "Glm4vImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.glm4v import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        
        # Process teacher-student versions if enabled
        if self.ts_version is not None:
            self._process_ts_version(row_dict, messages, raw_prompt)
        
        return row_dict

    def _process_ts_version(self, row_dict: dict, messages: list, raw_prompt: str):
        """
        Process teacher-student version specific modifications to the prompt.
        
        Args:
            row_dict: The row dictionary to modify
            messages: The original messages list
            raw_prompt: The raw prompt string after chat template
        """
        if self.ts_version == 'v1':
            self._process_ts_v1(row_dict, raw_prompt)
        elif self.ts_version == 'v2':
            self._process_ts_v2(row_dict, messages, raw_prompt)
        elif self.ts_version == 'v3':
            self._process_ts_v3(row_dict, messages, raw_prompt)
        elif self.ts_version == 'v4':
            self._process_ts_v4(row_dict, messages, raw_prompt)
        elif self.ts_version == 'v5' or self.ts_version == 'v7':
            self._process_ts_v5(row_dict, messages, raw_prompt)
    
    def _process_ts_v1(self, row_dict: dict, raw_prompt: str):
        """Process ts_version v1: Add teacher hints to prompts."""
        if self.teacher_hint_dict is None:
            return
        
        problem_id = row_dict.get('problem_id')
        if problem_id is None:
            return
        
        problem_id_str = str(problem_id)
        # Get original messages
        original_messages = row_dict.get('raw_prompt', [])
        if not original_messages:
            return
        
        if problem_id_str not in self.teacher_hint_dict:
            row_dict['raw_prompt_hint'] = original_messages
            return
        
        hint_str = self.teacher_hint_dict[problem_id_str]
        if hint_str == "":
            row_dict['raw_prompt_hint'] = original_messages
            return
        
        try:
            hint_data = json.loads(hint_str)
            correct_prefix = hint_data.get("correct_prefix", "")
            teacher_hint = hint_data.get("hint", "")
            
            if len(correct_prefix) > 4000 or len(teacher_hint) > 500:
                logger.warning(
                    f"problem_id {problem_id} has too long correct_prefix or teacher_hint, skip hint addition."
                )
                row_dict['raw_prompt_hint'] = original_messages
                return
            
            # Get original system and user prompts
            sys_prompt = original_messages[0]['content'] if len(original_messages) > 0 and original_messages[0].get('role') == 'system' else ""
            user_prompt = original_messages[1]['content'] if len(original_messages) > 1 and original_messages[1].get('role') == 'user' else original_messages[0]['content'] if len(original_messages) > 0 else ""
            
            # Build new user prompt with hint
            hint_prompt_before = "**Important Hint**: "
            hint_prompt_after = "\nPlease continue your response based on the above hint.\n"
            teacher_hint_text = hint_prompt_before + teacher_hint + hint_prompt_after
            new_user_prompt = user_prompt + "\n\n" + teacher_hint_text
            
            # Build new messages
            new_messages = []
            if sys_prompt:
                new_messages.append({"role": "system", "content": sys_prompt})
            new_messages.append({"role": "user", "content": new_user_prompt})
            
            row_dict['raw_prompt_hint'] = new_messages
        except Exception as e:
            logger.warning(f"Error processing ts_v1 for problem_id {problem_id}: {e}")
            row_dict['raw_prompt_hint'] = original_messages
    
    def _process_ts_v2(self, row_dict: dict, messages: list, raw_prompt: str):
        """Process ts_version v2: Add crafted wrong answers to prompts."""
        if self.crafted_wrong_answer is None:
            return
        
        problem_id = row_dict.get('problem_id')
        if problem_id is None:
            return
        
        problem_id_str = str(problem_id)
        original_messages = messages
        
        if problem_id_str not in self.crafted_wrong_answer:
            row_dict['raw_prompt_hint'] = original_messages
            return
        
        wrong_answer_dict = self.crafted_wrong_answer[problem_id_str]
        raw_problem = messages[1]['content'] if len(messages) > 1 else ""
        crafted_wrong_answer = wrong_answer_dict.get('wrong_answer', "")
        error_type = wrong_answer_dict.get('error_type', "")
        
        if crafted_wrong_answer == "" or error_type == "parsing_failed":
            row_dict['raw_prompt_hint'] = original_messages
            return
        
        if len(crafted_wrong_answer) > 4000:
            logger.warning(
                f"problem_id {problem_id} has too long crafted_wrong_answer, skip hint addition."
            )
            row_dict['raw_prompt_hint'] = original_messages
            return
        
        system_prompt = """You are given a problem and a solution that contains a subtle error and wrong final answer.
Your task is to carefully analyze the provided solution, identify what is wrong or questionable, explain why it is incorrect, and provide the correct reasoning and final answer. Structure your response into two sections: Thought and Solution. In the Thought section, present your reasoning using the format: "<think>\n thoughts </think>\n". Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas. After "</think>\n," in the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section. If applicable, include the answer in \\boxed{}.
"""
        user_prompt = f"""**Problem:** {raw_problem}\n**Provided Solution (Contains ONE Error):**{crafted_wrong_answer}\n**Your Task:** Examine the solution above critically in your <think> section. Then, after "</think>\n", in your Solution section, provide the correct solution with final answer in \\boxed{{}}"""
        
        # Build new messages
        new_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        row_dict['raw_prompt_hint'] = new_messages
    
    def _process_ts_v3(self, row_dict: dict, messages: list, raw_prompt: str):
        """Process ts_version v3: Add crafted wrong answers and lemmas to prompts."""
        if self.crafted_wrong_answer is None or self.teacher_lemmas is None:
            return
        
        problem_id = row_dict.get('problem_id')
        if problem_id is None:
            return
        
        problem_id_str = str(problem_id)
        original_messages = messages
        raw_problem = messages[1]['content'] if len(messages) > 1 else ""
        sys_prompt = messages[0]['content'] if len(messages) > 0 and messages[0].get('role') == 'system' else ""
        
        # Process crafted wrong answer
        if problem_id_str in self.crafted_wrong_answer:
            wrong_answer_dict = self.crafted_wrong_answer[problem_id_str]
            crafted_wrong_answer = wrong_answer_dict.get('wrong_answer', "")
            error_type = wrong_answer_dict.get('error_type', "")
            
            if crafted_wrong_answer == "" or error_type == "parsing_failed":
                row_dict['raw_prompt_crafted'] = original_messages
            elif len(crafted_wrong_answer) > 4000:
                logger.warning(
                    f"problem_id {problem_id} has too long crafted_wrong_answer, skip hint addition."
                )
                row_dict['raw_prompt_crafted'] = original_messages
            else:
                system_prompt = """You are given a problem and a solution that contains a subtle error and wrong final answer.Your task is to carefully analyze the provided solution, identify what is wrong or questionable, explain why it is incorrect, and provide the correct reasoning and final answer. Structure your response into two sections: Thought and Solution. In the Thought section, present your reasoning using the format: "<think>\n thoughts </think>\n". Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas. After "</think>\n," in the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section. If applicable, include the answer in \\boxed{}."""
                user_prompt = f"""**Problem:** {raw_problem}\n**Provided Solution (Contains ONE Error and the final answer is wrong):**{crafted_wrong_answer}\n**Your Task:** Examine the solution above critically in your <think> section. Then, after "</think>\n", in your Solution section, provide the correct solution with final answer in \\boxed{{}}"""
                
                new_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                row_dict['raw_prompt_crafted'] = new_messages
        else:
            row_dict['raw_prompt_crafted'] = original_messages
        
        # Process lemmas
        if problem_id_str in self.teacher_lemmas:
            teacher_lemma_dict = self.teacher_lemmas[problem_id_str]
            for i in range(1, 4):
                lemma_key = f"lemma{i}"
                prompt_key = f"raw_prompt_lemma{i}"
                teacher_lemma = teacher_lemma_dict.get(lemma_key, {"lemma": "", "insight": ""})
                
                if teacher_lemma.get("lemma", "") == "":
                    row_dict[prompt_key] = original_messages
                    continue
                
                lemma = teacher_lemma.get("lemma", "")
                insight = teacher_lemma.get("insight", "")
                
                lemma_prompt = f"Problem: {raw_problem}\n\nUseful lemma: {lemma}\n\nWhy this matters: {insight}\n\nUsing this hint, solve the problem step by step and provide your answer in \\boxed{{}}."
                user_prompt = lemma_prompt
                
                new_messages = []
                if sys_prompt:
                    new_messages.append({"role": "system", "content": sys_prompt})
                new_messages.append({"role": "user", "content": user_prompt})
                
                row_dict[prompt_key] = new_messages
        else:
            for i in range(1, 4):
                row_dict[f'raw_prompt_lemma{i}'] = original_messages
    
    def _process_ts_v4(self, row_dict: dict, messages: list, raw_prompt: str):
        """Process ts_version v4: Add progressive subproblems to prompts."""
        if self.subproblems_dict is None:
            return
        
        problem_id = row_dict.get('problem_id')
        if problem_id is None:
            return
        
        problem_id_str = str(problem_id)
        if problem_id_str not in self.subproblems_dict:
            return
        
        subproblem_data = self.subproblems_dict[problem_id_str]
        problem_statement = subproblem_data.get('problem_statement', "")
        q_1 = subproblem_data.get('question_1', {}).get('statement', "")
        q_2 = subproblem_data.get('question_2', {}).get('statement', "")
        q_3 = subproblem_data.get('question_3', {}).get('statement', "")
        q_4 = subproblem_data.get('question_4', {}).get('statement', "")
        a_1 = subproblem_data.get('question_1', {}).get('ground_truth', "")
        a_2 = subproblem_data.get('question_2', {}).get('ground_truth', "")
        a_3 = subproblem_data.get('question_3', {}).get('ground_truth', "")
        a_4 = subproblem_data.get('question_4', {}).get('ground_truth', "")
        
        sys_prompt = """You are given a problem statement with progressive subproblems. Solve each subproblem sequentially, understanding how each builds upon the previous one, and provide the solution to **the last subproblem only**. Structure your response into two sections: In the Thought section, use "<think>\n{your thoughts}\n</think>" format with detailed analysis of each subproblem, step-by-step reasoning, and connections between subproblems. In the Solution section after "</think>", provide the final answer to the last subproblem clearly derived from your thought process, enclosed in \\boxed{}.\n"""
        statement_prompt = f"Problem Statement: {problem_statement}\n"
        subproblem1_prompt = f"Subproblem 1: {q_1}\n"
        subproblem2_prompt = f"Subproblem 2: {q_2}\n"
        subproblem3_prompt = f"Subproblem 3: {q_3}\n"
        subproblem4_prompt = f"Subproblem 4: {q_4}\n"
        user_prompt_base = """Please solve each subproblem carefully in the given order, use insights from earlier subproblems to inform later ones. Make sure your final answer to **the last subproblem only** is written as \\boxed{your_answer_here}."""
        
        # Build messages for each subproblem variant
        user_prompts = [
            statement_prompt + subproblem1_prompt + user_prompt_base,
            statement_prompt + subproblem1_prompt + subproblem2_prompt + user_prompt_base,
            statement_prompt + subproblem1_prompt + subproblem2_prompt + subproblem3_prompt + user_prompt_base,
            statement_prompt + subproblem1_prompt + subproblem2_prompt + subproblem3_prompt + subproblem4_prompt + user_prompt_base
        ]
        
        for i in range(1, 5):
            prompt_key = f"raw_prompt_subproblem{i}"
            user_prompt = user_prompts[i - 1]
            
            new_messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
            row_dict[prompt_key] = new_messages
        
        # Add ground truth to reward_model if it exists
        if 'reward_model' not in row_dict:
            row_dict['reward_model'] = {}
        row_dict['reward_model']['ground_truth_sub1'] = a_1
        row_dict['reward_model']['ground_truth_sub2'] = a_2
        row_dict['reward_model']['ground_truth_sub3'] = a_3
        row_dict['reward_model']['ground_truth_sub4'] = a_4
    
    def _process_ts_v5(self, row_dict: dict, messages: list, raw_prompt: str):
        """Process ts_version v5: Add all subproblems in one prompt."""
        if self.subproblems_dict is None:
            return
        
        problem_id = row_dict.get('problem_id')
        if problem_id is None:
            return
        
        problem_id_str = str(problem_id)
        if problem_id_str not in self.subproblems_dict:
            return
        
        subproblem_data = self.subproblems_dict[problem_id_str]
        problem_statement = subproblem_data.get('problem_statement', "")
        q_1 = subproblem_data.get('question_1', {}).get('statement', "")
        q_2 = subproblem_data.get('question_2', {}).get('statement', "")
        q_3 = subproblem_data.get('question_3', {}).get('statement', "")
        q_4 = subproblem_data.get('question_4', {}).get('statement', "")
        a_1 = subproblem_data.get('question_1', {}).get('ground_truth', "")
        a_2 = subproblem_data.get('question_2', {}).get('ground_truth', "")
        a_3 = subproblem_data.get('question_3', {}).get('ground_truth', "")
        a_4 = subproblem_data.get('question_4', {}).get('ground_truth', "")
        
        # sys_prompt = """You are a helpful math problem solver. Solve the following math problems efficiently and clearly. Please reason step by step, and put your final answer within \\boxed{answer}."""
        sys_prompt = messages[0]['content']
        statement_prompt = f"Problem Statement: {problem_statement}\n\n"
        subproblem1_prompt = f"Subproblem 1: {q_1}\n"
        subproblem2_prompt = f"Subproblem 2: {q_2}\n"
        subproblem3_prompt = f"Subproblem 3: {q_3}\n"
        subproblem4_prompt = f"Subproblem 4: {q_4}\n"

        user_prompt = f"""{statement_prompt}{subproblem1_prompt}{subproblem2_prompt}{subproblem3_prompt}{subproblem4_prompt}Please solve all 4 subproblems in order. For each subproblem, start your response with **Subproblem k** (where k is the subproblem number), show your reasoning, and provide your final answer in \\boxed{{answer}}."""
        # user_prompt = f"""{statement_prompt}{subproblem1_prompt}{subproblem2_prompt}{subproblem3_prompt}{subproblem4_prompt}INSTRUCTIONS:
        # Solve all 4 subproblems in order following this EXACT format:
        # **Subproblem 1**:
        # [Your reasoning here]
        # Final answer: \\boxed{{answer1}}
        # **Subproblem 2**:
        # [Your reasoning here]
        # Final answer: \\boxed{{answer2}}
        # **Subproblem 3**:
        # [Your reasoning here]
        # Final answer: \\boxed{{answer3}}
        # **Subproblem 4**:
        # [Your reasoning here]
        # Final answer: \\boxed{{answer4}}. CRITICAL: Your response MUST end immediately after \\boxed{{answer4}}. Do NOT write anything else after that - no summaries, no repeated answers, no extra comments. Just stop."""
        # Build new messages
        new_messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        row_dict['raw_prompt_subproblems'] = new_messages
        
        # Add ground truth to reward_model if it exists
        if 'reward_model' not in row_dict:
            row_dict['reward_model'] = {}
        row_dict['reward_model']['ground_truth_sub1'] = a_1
        row_dict['reward_model']['ground_truth_sub2'] = a_2
        row_dict['reward_model']['ground_truth_sub3'] = a_3
        row_dict['reward_model']['ground_truth_sub4'] = a_4

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
