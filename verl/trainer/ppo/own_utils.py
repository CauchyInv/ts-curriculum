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
"""
Utility functions for teacher-student curriculum learning.
"""

import os
import json
import torch
import numpy as np
from typing import List, Any, Dict, Tuple, Optional


def decode_tensor(tokenizer, tensor, skip_specials=True):
    """Remove padding and decode tensor token IDs."""
    tensor = tensor.to('cpu')
    pad_id = tokenizer.pad_token_id
    active_tokens_list = tensor.tolist()
    
    if pad_id is not None:
        active_tokens_list = [token for token in active_tokens_list if token != pad_id]

    active_tokens = torch.tensor(active_tokens_list)
    decoded_text = tokenizer.decode(
        active_tokens,
        skip_special_tokens=skip_specials
    ).strip()
    
    return active_tokens.shape[0], decoded_text


def save_problem_rollouts(
    output_dir: str, 
    current_problem_id: str, 
    uid_rewards: torch.Tensor, 
    problem_ids: List[str], 
    success_value: float, 
    batch: Any,
    tokenizer: Any
):
    """
    Save all rollout samples for a specific problem ID to a JSONL file.

    Args:
        output_dir (str): Root output directory, e.g., '/path/to/epoch_X'.
        current_problem_id (str): Current problem ID (string).
        uid_rewards (torch.Tensor): Reward tensor for all rollouts of current problem ID.
        problem_ids (List[str]): List of all problem_ids in the batch.
        success_value (float): Reward value indicating success.
        batch (Any): Batch object containing 'batch' attribute with tensor data.
        tokenizer (Any): Tokenizer for decoding token IDs.
    """
    # Create problem-specific output directory
    problem_output_dir = os.path.join(output_dir, f"P_{current_problem_id}")
    os.makedirs(problem_output_dir, exist_ok=True)

    jsonl_file_name = os.path.join(problem_output_dir, f"rollouts.jsonl")

    # Prepare index data
    problem_ids_int_list = [int(pid) for pid in problem_ids]
    problem_ids_tensor = torch.tensor(problem_ids_int_list, dtype=torch.long)
    current_problem_id_int = int(current_problem_id)
    
    # Get indices for this problem_id in the batch
    problem_indices = torch.where(problem_ids_tensor == current_problem_id_int)[0]

    if len(problem_indices) == 0:
        print(f"Warning: Could not find indices for problem ID {current_problem_id} in the batch index list. Skipping save.")
        return 0

    # Extract data and write to JSONL file
    tensor_batch = batch.batch
    
    with open(jsonl_file_name, 'a', encoding='utf-8') as f:
        for j, sample_idx in enumerate(problem_indices.tolist()):
            current_reward = uid_rewards[j].item()
            solved_status = "SOLVED" if current_reward == success_value else "FAILED"

            sample_data = {
                "problem_id": current_problem_id,
                "rollout_index": j + 1,
                "status": solved_status,
                "reward": current_reward,
                "data": {}
            }

            # Extract prompts
            if 'prompts' in tensor_batch:
                length, decoded_text = decode_tensor(tokenizer, tensor_batch['prompts'][sample_idx], skip_specials=False)
                sample_data["data"]["prompts"] = {
                    "length": length,
                    "text": decoded_text
                }

            # Extract responses
            if 'responses' in tensor_batch:
                length, decoded_text = decode_tensor(tokenizer, tensor_batch['responses'][sample_idx], skip_specials=False)
                sample_data["data"]["responses"] = {
                    "length": length,
                    "text": decoded_text
                }

            # Extract ground_truth
            try:
                ground_truth = batch[sample_idx].non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                if ground_truth is not None:
                    sample_data["data"]["ground_truth"] = {
                        "text": ground_truth
                    }
            except (KeyError, IndexError, AttributeError):
                # If ground_truth is not available, skip it
                pass

            # Extract teacher_student_mixed flag (False means in_state_0/original problem)
            try:
                teacher_student_mixed = batch[sample_idx].non_tensor_batch.get("teacher_student_mixed", False)
                sample_data["teacher_student_mixed"] = bool(teacher_student_mixed)
            except (KeyError, IndexError, AttributeError):
                # If teacher_student_mixed is not available, default to False (not mixed)
                sample_data["teacher_student_mixed"] = False

            # Extract truncated flag (True means response was successfully truncated in v5)
            try:
                # Try to get from batch-level non_tensor_batch first (if it's an array)
                if "truncated" in batch.non_tensor_batch:
                    truncated_arr = batch.non_tensor_batch["truncated"]
                    if isinstance(truncated_arr, (np.ndarray, list)):
                        truncated = truncated_arr[sample_idx] if sample_idx < len(truncated_arr) else False
                    else:
                        truncated = truncated_arr
                else:
                    # Fallback to sample-level non_tensor_batch
                    truncated = batch[sample_idx].non_tensor_batch.get("truncated", False)
                sample_data["truncated"] = bool(truncated)
            except (KeyError, IndexError, AttributeError):
                # If truncated is not available, default to False (not truncated)
                sample_data["truncated"] = False

            # Extract subproblem_correct_count (k value: number of correct subproblems)
            try:
                # Try to get from batch-level non_tensor_batch first (if it's an array)
                if "subproblem_correct_count" in batch.non_tensor_batch:
                    subproblem_correct_count_arr = batch.non_tensor_batch["subproblem_correct_count"]
                    if isinstance(subproblem_correct_count_arr, (np.ndarray, list)):
                        subproblem_correct_count = subproblem_correct_count_arr[sample_idx] if sample_idx < len(subproblem_correct_count_arr) else None
                    else:
                        subproblem_correct_count = subproblem_correct_count_arr
                else:
                    # Fallback to sample-level non_tensor_batch
                    subproblem_correct_count = batch[sample_idx].non_tensor_batch.get("subproblem_correct_count", None)
                
                # Convert to int if found, -1 means extraction failed, None means original problem (non-mixed)
                if subproblem_correct_count is not None:
                    sample_data["subproblem_correct_count"] = int(subproblem_correct_count)
                else:
                    sample_data["subproblem_correct_count"] = None
            except (KeyError, IndexError, AttributeError):
                # If subproblem_correct_count is not available, set to None
                sample_data["subproblem_correct_count"] = None

            # Extract token_level_scores
            if 'token_level_scores' in tensor_batch:
                score_tensor = tensor_batch['token_level_scores'][sample_idx]
                score_value = score_tensor.sum().item()
                sample_data["data"]["token_level_score_sum"] = score_value

            # Write to file
            f.write(json.dumps(sample_data, ensure_ascii=False) + '\n')
    
    return len(problem_indices)


def current_student_answer(
    current_problem_id: str, 
    uid_rewards: torch.Tensor, 
    problem_ids: List[str], 
    batch: Any,
    tokenizer: Any
):
    """
    Extract current student answer history for a specific problem ID.

    Args:
        current_problem_id (str): Current problem ID.
        uid_rewards (torch.Tensor): Reward tensor for all rollouts of current problem ID.
        problem_ids (List[str]): List of all problem_ids in the batch.
        batch (Any): Batch object containing 'batch' attribute with tensor data.
        tokenizer (Any): Tokenizer for decoding token IDs.
    
    Returns:
        dict: Student answer history with rollout data.
    """
    problem_ids_int_list = [int(pid) for pid in problem_ids]
    problem_ids_tensor = torch.tensor(problem_ids_int_list, dtype=torch.long)
    current_problem_id_int = int(current_problem_id)
    problem_indices = torch.where(problem_ids_tensor == current_problem_id_int)[0]
    
    if len(problem_indices) == 0:
        print(f"Warning: Could not find indices for problem ID {current_problem_id} in the batch index list. Skipping.")
        return {}
    
    tensor_batch = batch.batch
    student_answer = {}
    
    for j, sample_idx in enumerate(problem_indices.tolist()):
        current_reward = uid_rewards[j].item()
        sample_data = {
            "problem_id": current_problem_id,
            "reward": current_reward,
            "data": {}
        }
        
        if 'prompts' in tensor_batch:
            length, decoded_text = decode_tensor(tokenizer, tensor_batch['prompts'][sample_idx], skip_specials=False)
            sample_data["data"]["prompts"] = {
                "length": length,
                "text": decoded_text
            }
        
        if 'responses' in tensor_batch:
            length, decoded_text = decode_tensor(tokenizer, tensor_batch['responses'][sample_idx], skip_specials=False)
            sample_data["data"]["responses"] = {
                "length": length,
                "text": decoded_text
            }
        
        if 'raw_input_ids' in tensor_batch:
            length, decoded_text = decode_tensor(tokenizer, tensor_batch['raw_input_ids'][sample_idx], skip_specials=True)
            sample_data["data"]["raw_input_ids"] = {
                "length": length,
                "text": decoded_text
            }
        
        if 'tgt_input_ids' in tensor_batch:
            length, decoded_text = decode_tensor(tokenizer, tensor_batch['tgt_input_ids'][sample_idx], skip_specials=True)
            sample_data["data"]["tgt_input_ids"] = {
                "length": length,
                "text": decoded_text
            }
        
        student_answer[f"rollout_{j+1}"] = sample_data
    
    return student_answer


class BatchHintGenerator:
    """Batch hint generator using API."""
    
    def __init__(self, api_key: str, max_workers: int = 8):
        self.api_key = api_key
        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    def generate_hints_batch(self, student_answer_history: Dict, num_problems: int = 128) -> Dict[str, str]:
        """Batch generate hints for all problems."""
        hint_dict = {str(i): "" for i in range(num_problems)}
        
        # Filter problems that need hints
        problems_need_hint = []
        for problem_id, answer_history in student_answer_history.items():
            if not answer_history:
                continue
            
            needs_hint = all(
                rollout_data.get('reward', 0) != 1.0 
                for rollout_data in answer_history.values()
            )
            
            if needs_hint:
                problems_need_hint.append((problem_id, answer_history))
        
        if not problems_need_hint:
            return hint_dict
        
        # Concurrent API calls
        futures = []
        for problem_id, answer_history in problems_need_hint:
            future = self.executor.submit(
                self._generate_single_hint, 
                problem_id, 
                answer_history
            )
            futures.append((problem_id, future))
        
        # Collect results
        from tqdm import tqdm
        for problem_id, future in tqdm(futures, desc="Generating hints"):
            try:
                hint = future.result()
                hint_dict[str(problem_id)] = hint
            except Exception as e:
                print(f"Error generating hint for problem {problem_id}: {e}")
                hint_dict[str(problem_id)] = ""
        
        return hint_dict
    
    def _generate_single_hint(self, problem_id, answer_history):
        """Generate a single hint via API."""
        import time
        import re
        from openai import OpenAI

        def robust_json_parse(raw_text):
            text = raw_text.strip()
            if text.startswith("```"):
                first_newline = text.find('\n')
                if first_newline != -1:
                    text = text[first_newline+1:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            try:
                fixed_text = re.sub(r'\\(?![/u"\\bfnrt])', r'\\\\', text)
                return json.loads(fixed_text)
            except json.JSONDecodeError:
                pass

            data = {}
            rollout_match = re.search(r'"selected_rollout"\s*:\s*"([^"]+)"', text)
            if rollout_match:
                data["selected_rollout"] = rollout_match.group(1)
            
            hint_match = re.search(r'"hint"\s*:\s*"(.*?)"\s*\}', text, re.DOTALL)
            if hint_match:
                data["hint"] = hint_match.group(1)
                
            prefix_pattern = r'"correct_prefix"\s*:\s*"(.*?)"\s*,\s*"hint"'
            prefix_match = re.search(prefix_pattern, text, re.DOTALL)
            
            if prefix_match:
                raw_prefix = prefix_match.group(1)
                try:
                    decoded_prefix = raw_prefix.encode('utf-8').decode('unicode_escape')
                    data["correct_prefix"] = decoded_prefix
                except Exception:
                    data["correct_prefix"] = raw_prefix
            
            if all(k in data for k in ["selected_rollout", "correct_prefix", "hint"]):
                return data
                
            return None

        # Get recent 3 rollouts
        sorted_rollouts = sorted(
            answer_history.keys(), 
            key=lambda x: int(x.split('_')[1]) if x.startswith('rollout_') and x.split('_')[1].isdigit() else float('inf')
        )[-3:]
        
        interaction_details = []
        tgt_text = "No Target Text"
        
        for rollout_key in sorted_rollouts:
            sample = answer_history[rollout_key]
            reward = sample.get('reward', 'N/A')
            prompt_text = sample.get('data', {}).get('prompts', {}).get('text', 'No Prompt Text')
            response_text = sample.get('data', {}).get('responses', {}).get('text', 'No Student Response')
            
            current_tgt = sample.get('data', {}).get('tgt_input_ids', {}).get('text', None)
            if current_tgt:
                tgt_text = current_tgt
            
            interaction_details.append(
                f"--- {rollout_key} (Reward: {reward}) ---\n"
                f"  Problem: {prompt_text}\n"
                f"  Student Response: {response_text}\n"
            )
        
        interaction_details.append(
            f"\n{'='*60}\n"
            f"REFERENCE SOLUTION (Ground Truth):\n"
            f"{'='*60}\n"
            f"{tgt_text}\n"
            f"{'='*60}"
        )
        
        history_str = "\n".join(interaction_details)
        
        system_prompt = """You are an expert Teacher Model specializing in response quality assessment and precise error detection. Your task is to select the best student attempt from multiple rollouts, then identify the FIRST error and provide targeted feedback.

    **CRITICAL REQUIREMENTS:**

    Your response MUST be a valid JSON object with exactly three fields:

    {
    "selected_rollout": "string - the rollout identifier you selected (e.g., 'rollout_0', 'rollout_1', 'rollout_2')",
    "correct_prefix": "string - the exact substring from the START of the selected response up to (but NOT including) the first error",
    "hint": "string - a concise hint (max 100 tokens) explaining what the correct approach should be at this point, based on the reference solution"
    }

    **STEP-BY-STEP INSTRUCTIONS:**

    **PHASE 1: Select the Best Rollout**

    Evaluate all student rollouts based on these criteria (in priority order):
    1. **Format Validity**: Does it follow proper mathematical/logical formatting? No garbled text or nonsense?
    2. **Coherence**: Is the reasoning coherent and readable? Not random gibberish?
    3. **Alignment with Reference**: Which attempt's approach is closest to the reference solution's methodology?
    4. **Completeness**: Does it attempt to solve the problem systematically?

    Select the ONE rollout that scores highest on these criteria. Even if all attempts are poor, pick the "least bad" one.

    **PHASE 2: Analyze the Selected Response**

    1. **Study the Reference Solution**: Carefully read the ground truth solution to understand:
    - The correct solution approach and reasoning path
    - Key steps and intermediate results
    - The proper methodology and techniques used

    2. **Trace the Selected Student's Response**: Read their response from the beginning, comparing it line-by-line with the reference solution's approach.

    3. **Identify the First Divergence**: Find the exact point where the student FIRST deviates from the correct path. This could be:
    - Wrong reasoning or approach
    - Incorrect calculation or algebraic manipulation
    - Invalid assumption or setup
    - Misunderstanding of the problem requirement
    - Correct reasoning but computational error

    4. **Extract Correct Prefix**: Copy the exact text from the start of the selected student's response up to (but NOT including) the first error point. This must be:
    - A substring that starts from the very beginning of their response (the "Student Response:" field shown above)
    - An EXACT copy (preserve all formatting, spacing, LaTeX, etc.)
    - Everything the student did correctly before making their first mistake
    - **DO NOT include any prefixes like "Student Response:", "Assistant:", "<think>", etc. - only the actual response content**
    - **IMPORTANT**: If the correct prefix would be longer than 1500 characters, truncate it to approximately 1500 characters at a natural breaking point (end of a sentence or step)

    5. **Generate Hint Based on Reference Solution**: Write a brief hint (under 100 tokens) that:
    - Guides the student toward the correct approach shown in the reference solution
    - Explains what key concept, method, or step they should apply next (from the reference)
    - Does NOT directly copy from or reveal the reference solution's answer
    - Provides just enough direction to help them self-correct
    - Is specific and actionable

    **IMPORTANT NOTES:**
    - You MUST select exactly one rollout - use your best judgment even if all are poor quality
    - Use the reference solution as your source of truth for what is "correct"
    - If the selected response is gibberish from the start, set "correct_prefix" to "" and provide a hint to restart with the correct approach
    - If the selected response is correct so far but incomplete, set "correct_prefix" to their entire response
    - The "correct_prefix" must be an EXACT substring from the selected response - do not modify or paraphrase
    - Your hint should reflect the reasoning path of the reference solution
    - Your entire response must be valid JSON - no additional text before or after

    **OUTPUT FORMAT:**
    Return ONLY the JSON object with three fields: "selected_rollout", "correct_prefix", and "hint"."""

        user_prompt = f"""You are given multiple student attempts (rollouts) for the same problem. The REFERENCE SOLUTION at the end shows the correct approach.

    {history_str}

    Your task (TWO PHASES):

    **PHASE 1 - Select Best Rollout:**
    Evaluate all student rollouts and select the ONE with the highest quality based on:
    - Format validity (no garbled text)
    - Coherence (logical reasoning, not gibberish)
    - Alignment with reference solution's approach
    - Completeness of attempt

    **PHASE 2 - Error Analysis:**
    For the selected rollout:
    1. Compare with the reference solution's approach
    2. Identify where the student FIRST deviates from the correct path
    3. Extract the exact correct prefix (text before first error)
    4. Generate a hint based on what the reference solution does at this step (max 100 tokens)

    **CRITICAL: Your response must be ONLY a valid JSON object with THREE fields:**
    - "selected_rollout": the rollout key you chose (e.g., "rollout_0")
    - "correct_prefix": exact text from selected response before first error
    - "hint": guidance based on reference solution (max 100 tokens)

    Do not include any explanatory text, markdown formatting, or additional commentary. Output the raw JSON object directly.

    JSON Output:"""

        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                client = OpenAI(
                    api_key=self.api_key, 
                    base_url="https://api.deepseek.com",
                    timeout=180.0
                )
                
                response = client.chat.completions.create(
                    model="deepseek-reasoner", 
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=16384,
                    stream=False,
                )
                
                raw_response = response.choices[0].message.content
                hint_data = robust_json_parse(raw_response)
                
                if hint_data is None:
                    if attempt < max_retries - 1:
                        continue
                    else:
                        return ""

                if "selected_rollout" not in hint_data or "correct_prefix" not in hint_data or "hint" not in hint_data:
                    if attempt < max_retries - 1:
                        continue
                    else:
                        return ""
                
                if not isinstance(hint_data["selected_rollout"], str) or not isinstance(hint_data["correct_prefix"], str) or not isinstance(hint_data["hint"], str):
                    if attempt < max_retries - 1:
                        continue
                    else:
                        return ""
                
                hint = json.dumps(hint_data, ensure_ascii=False)
                return hint
                
            except Exception as e:
                error_msg = str(e).lower()
                is_timeout = "timeout" in error_msg or "timed out" in error_msg
                is_connection_error = "connection" in error_msg or "connection error" in error_msg
                is_rate_limit = "rate limit" in error_msg or "429" in error_msg
                
                should_retry = is_timeout or is_connection_error or is_rate_limit
                
                if should_retry and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                else:
                    return ""
        
        return ""
    
    def generate_subproblems_batch(
        self,
        problems: Dict[str, str],
        reference_answers: Dict[str, str],
        output_file: str = "/hyk/algorithm_new/qinghua/yueyang/verl/data/omni/subproblems.jsonl"
    ) -> Dict[str, Dict]:
        """
        批量生成递进式4个子问题，并保存到JSONL文件
        
        Args:
            problems: {problem_id: original_problem_text}
            reference_answers: {problem_id: reference_solution}
            output_file: 输出的JSONL文件路径
        
        Returns:
            {problem_id: subproblem_dict}
        """
        from pathlib import Path
        
        subproblems_dict = {}
        
        # 准备所有需要生成的问题
        problem_ids_to_process = []
        for problem_id in problems.keys():
            if problem_id not in reference_answers:
                print(f"⚠️ Skipping problem {problem_id}: no reference answer")
                continue
            problem_ids_to_process.append(problem_id)
        
        if not problem_ids_to_process:
            print(f"✅ No problems to generate subproblems for.")
            return subproblems_dict
        
        print(f"\n{'='*60}")
        print(f"🎯 Generating progressive subproblems for {len(problem_ids_to_process)} problems")
        print(f"📁 Output file: {output_file}")
        print(f"{'='*60}\n")
        
        # 确保输出目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 打开文件用于追加写入
        with open(output_file, 'w', encoding='utf-8') as f:
            # 并发调用 API
            futures = []
            for problem_id in problem_ids_to_process:
                future = self.executor.submit(
                    self._generate_single_subproblem,
                    problem_id,
                    problems[problem_id],
                    reference_answers[problem_id]
                )
                futures.append((problem_id, future))
            
            # 收集结果并实时写入文件
            from tqdm import tqdm
            success_count = 0
            fail_count = 0
            
            for problem_id, future in tqdm(futures, desc="Generating subproblems"):
                try:
                    subproblem_data = future.result()
                    subproblems_dict[str(problem_id)] = subproblem_data
                    
                    # 添加problem_id到数据中
                    subproblem_data_with_id = {"problem_id": str(problem_id), **subproblem_data}
                    
                    # 写入JSONL文件
                    f.write(json.dumps(subproblem_data_with_id, ensure_ascii=False) + '\n')
                    f.flush()  # 立即写入磁盘
                    
                    # 检查是否成功生成（非空的problem_statement表示成功）
                    if subproblem_data.get("problem_statement", ""):
                        success_count += 1
                        print(f"✅ Problem {problem_id}: Successfully generated subproblems")
                    else:
                        fail_count += 1
                        print(f"❌ Problem {problem_id}: Failed to generate valid subproblems")
                        
                except Exception as e:
                    print(f"❌ Unexpected error for problem {problem_id}: {e}")
                    # 保存空结构
                    empty_structure = self._create_empty_subproblem_structure(str(problem_id))
                    subproblems_dict[str(problem_id)] = empty_structure
                    f.write(json.dumps(empty_structure, ensure_ascii=False) + '\n')
                    f.flush()
                    fail_count += 1
        
        # 统计信息
        print(f"\n{'='*60}")
        print(f"✅ Successfully generated subproblems: {success_count}/{len(problem_ids_to_process)}")
        print(f"❌ Failed to generate subproblems: {fail_count}/{len(problem_ids_to_process)}")
        print(f"📁 Results saved to: {output_file}")
        print(f"{'='*60}\n")
        
        return subproblems_dict
    
    def _generate_single_subproblem(
        self,
        problem_id: str,
        original_problem: str,
        reference_answer: str,
        max_retries: int = 10
    ) -> Dict:
        """
        为单个问题生成递进式子问题，支持最多10次重试
        
        Args:
            problem_id: 问题ID
            original_problem: 原始问题文本
            reference_answer: 参考答案
            max_retries: 最大重试次数（默认10次）
        
        Returns:
            包含子问题的字典
        """
        import time
        import re
        from openai import OpenAI
        
        def robust_json_parse_subproblem(raw_text):
            """
            专门用于解析子问题JSON的鲁棒解析器
            """
            text = raw_text.strip()
            
            # 移除markdown代码块标记
            if text.startswith("```"):
                first_newline = text.find('\n')
                if first_newline != -1:
                    text = text[first_newline+1:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            # 尝试直接解析
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
            
            # 尝试修复转义字符
            try:
                fixed_text = re.sub(r'\\(?![/u"\\bfnrt])', r'\\\\', text)
                return json.loads(fixed_text)
            except json.JSONDecodeError:
                pass
            
            return None
        
        system_prompt = """You are an expert at breaking down complex mathematical problems into progressive sub-problems. Your task is to analyze a given mathematical problem and its reference solution, then create 4 progressive sub-questions where each builds upon the previous one.

**CRITICAL REQUIREMENTS:**

Your response MUST be a valid JSON object with the following structure:

{
  "problem_statement": "<The original problem WITHOUT the final question>",
  "question_1": {
    "statement": "<First sub-question statement>",
    "ground_truth": "<Final answer only, extractable content from \\boxed{}>"
  },
  "question_2": {
    "statement": "<Second sub-question statement>",
    "ground_truth": "<Final answer only, extractable content from \\boxed{}>"
  },
  "question_3": {
    "statement": "<Third sub-question statement>",
    "ground_truth": "<Final answer only, extractable content from \\boxed{}>"
  },
  "question_4": {
    "statement": "<Fourth sub-question statement - MUST be the original question>",
    "ground_truth": "<Final answer only, extractable content from \\boxed{}>"
  }
}

**Design Principles:**
1. **Progressive Difficulty**: Each question should be easier than the next, building foundational understanding
2. **Interconnected**: Later questions should build on concepts/results from earlier questions
3. **Question 4 Requirement**: Must be the exact final question from the original problem
4. **Ground Truth Format**: 
   - Should be the FINAL ANSWER ONLY (not the reasoning process)
   - Should be what would appear inside \\boxed{} in the solution
   - Should be directly comparable to student responses for correctness checking
   - Examples: a number like "88327", an expression like "2", a tuple like "(4, 9, 10)", etc.
5. **Problem Statement**: Extract the problem setup/context WITHOUT the final question

**Example Structure:**
- Question 1: Verify basic understanding or compute initial simple cases
- Question 2: Compute intermediate results or recognize patterns  
- Question 3: Discover key insights, formulas, or critical observations
- Question 4: Apply everything to solve the original problem

**Guidelines:**
- Keep question statements clear and concise
- Ensure each ground_truth is objective and verifiable
- Make sure Questions 1-3 naturally lead to Question 4
- The difficulty gap between consecutive questions should be manageable
- Ground truths should be concrete values/expressions, not explanations

**OUTPUT FORMAT:**
Return ONLY the JSON object. Do not include any explanatory text, markdown formatting, or additional commentary before or after the JSON."""

        user_prompt = f"""Given the following Original Problem and Reference Answer, generate the JSON output with 4 progressive sub-questions:

**Original Problem:**
{original_problem}

**Reference Answer:**
{reference_answer}

Output the JSON only, without any additional text or markdown formatting.

JSON Output:"""

        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.deepseek.com",
                    timeout=300.0  # 更长的超时时间，因为生成子问题可能需要更多时间
                )
                
                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=16384,
                    stream=False,
                )
                
                raw_response = response.choices[0].message.content
                
                # 解析JSON
                parsed_data = robust_json_parse_subproblem(raw_response)
                
                if parsed_data is None:
                    print(f"⚠️ Problem {problem_id}: JSON parsing failed (attempt {attempt+1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(base_delay)
                        continue
                    else:
                        print(f"❌ Problem {problem_id}: All parsing attempts failed")
                        return self._create_empty_subproblem_structure(problem_id)
                
                # 验证必需字段
                required_fields = ["problem_statement", "question_1", "question_2", "question_3", "question_4"]
                if not all(field in parsed_data for field in required_fields):
                    missing = [f for f in required_fields if f not in parsed_data]
                    print(f"⚠️ Problem {problem_id}: Missing fields {missing} (attempt {attempt+1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(base_delay)
                        continue
                    else:
                        return self._create_empty_subproblem_structure(problem_id)
                
                # 验证每个question的结构
                valid = True
                for i in range(1, 5):
                    q_key = f"question_{i}"
                    if not isinstance(parsed_data[q_key], dict):
                        print(f"⚠️ Problem {problem_id}: {q_key} is not a dict (attempt {attempt+1}/{max_retries})")
                        valid = False
                        break
                    if "statement" not in parsed_data[q_key] or "ground_truth" not in parsed_data[q_key]:
                        print(f"⚠️ Problem {problem_id}: {q_key} missing statement or ground_truth (attempt {attempt+1}/{max_retries})")
                        valid = False
                        break
                
                if not valid:
                    if attempt < max_retries - 1:
                        time.sleep(base_delay)
                        continue
                    else:
                        return self._create_empty_subproblem_structure(problem_id)
                
                # 验证通过，返回结果
                print(f"✅ Problem {problem_id}: Successfully generated subproblems (attempt {attempt+1})")
                sub_pbs = {
                    "problem_statement": parsed_data["problem_statement"],
                    "question_1": parsed_data["question_1"],
                    "question_2": parsed_data["question_2"],
                    "question_3": parsed_data["question_3"],
                    "question_4": parsed_data["question_4"]
                }
                print(f"✅ Problem {problem_id} subproblems: {sub_pbs}")
                return {
                    "problem_statement": parsed_data["problem_statement"],
                    "question_1": parsed_data["question_1"],
                    "question_2": parsed_data["question_2"],
                    "question_3": parsed_data["question_3"],
                    "question_4": parsed_data["question_4"]
                }
                
            except Exception as e:
                error_msg = str(e).lower()
                is_retryable = any(keyword in error_msg for keyword in [
                    "timeout", "timed out", "connection", "rate limit", "429", "500", "502", "503"
                ])
                
                if is_retryable and attempt < max_retries - 1:
                    delay = base_delay * (2 ** min(attempt, 5))  # 指数退避，最多32秒
                    print(f"⚠️ Problem {problem_id} failed (attempt {attempt+1}/{max_retries}): {e}")
                    print(f"   Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ Problem {problem_id}: Failed after {attempt+1} attempts: {e}")
                    return self._create_empty_subproblem_structure(problem_id)
        
        # 如果所有重试都失败，返回空结构
        return self._create_empty_subproblem_structure(problem_id)
    
    def _create_empty_subproblem_structure(self, problem_id: str) -> Dict:
        """
        创建空的子问题结构（当生成失败时使用）
        """
        return {
            "problem_id": str(problem_id),
            "problem_statement": "",
            "question_1": {
                "statement": "",
                "ground_truth": ""
            },
            "question_2": {
                "statement": "",
                "ground_truth": ""
            },
            "question_3": {
                "statement": "",
                "ground_truth": ""
            },
            "question_4": {
                "statement": "",
                "ground_truth": ""
            }
        }


import ray
from vllm import LLM, SamplingParams
from typing import Dict

@ray.remote(num_gpus=4)
class DynamicVLLMTeacherGenerator:
    """Dynamic VLLM teacher generator for generating hints, wrong answers, and lemmas."""
    
    def __init__(
        self, 
        model_path: str = "/hyk/algorithm_new/qinghua/yueyang/LUFFY/QWQ_32B",
        tensor_parallel_size: int = 4,
        gpu_ids: List[int] = [4, 5, 6, 7],
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_ids = gpu_ids
        self.llm = None
        
        self.sampling_params_wrong_answer = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=1500,
        )
        
        self.sampling_params_lemma = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1500,
        )
        
        self.sampling_params_subproblem = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=8192,  # 子问题生成需要更多token
        )
        
        self._load_vllm()
    
    def _load_vllm(self):
        """Load vLLM model."""
        if self.llm is not None:
            return
        import time
        import os
        
        # 设置 CUDA_VISIBLE_DEVICES 环境变量以指定使用的 GPU
        # vLLM 使用这个环境变量来控制 GPU 选择
        if self.gpu_ids:
            # 将 GPU IDs 列表转换为逗号分隔的字符串
            gpu_ids_str = ",".join(str(gpu_id) for gpu_id in self.gpu_ids)
            # 设置环境变量（只在当前进程中有效）
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
            print(f"Setting CUDA_VISIBLE_DEVICES={gpu_ids_str}")
            # 注意：vLLM 内部会将可见的 GPU 重新编号为 0, 1, 2, ...
            # 所以 tensor_parallel_size 应该等于 len(gpu_ids)
            if self.tensor_parallel_size != len(self.gpu_ids):
                print(f"⚠️  Warning: tensor_parallel_size ({self.tensor_parallel_size}) != len(gpu_ids) ({len(self.gpu_ids)})")
                print(f"   vLLM will use {len(self.gpu_ids)} GPUs (visible GPUs will be renumbered as 0, 1, 2, ...)")
        
        print(f"Loading vLLM on GPUs {self.gpu_ids} (visible as 0-{len(self.gpu_ids)-1})...")
        start_time = time.time()
        
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=0.70,
            max_model_len=8192,
            trust_remote_code=True,
        )
        
        elapsed = time.time() - start_time
        print(f"vLLM loaded in {elapsed:.2f}s")
    
    def generate_crafted_wrong_answers_batch(
        self, 
        problems: Dict[str, str],
        ground_truth_answers: Dict[str, str],
        error_types: List[str] = ["calculation", "logic", "assumption", "sign", "boundary"]
    ) -> Dict[str, Dict]:
        """Generate crafted wrong answers for a batch of problems."""
        wrong_answer_dict = {}
        
        prompts = []
        problem_ids = []
        
        for problem_id in problems.keys():
            if problem_id not in ground_truth_answers:
                continue
            
            prompt = self._build_crafted_wrong_answer_prompt(
                problems[problem_id],
                ground_truth_answers[problem_id],
                error_types
            )
            prompts.append(prompt)
            problem_ids.append(problem_id)
        
        if not prompts:
            return wrong_answer_dict
        
        try:
            outputs = self.llm.generate(prompts, self.sampling_params_wrong_answer)
            
            for problem_id, output in zip(problem_ids, outputs):
                raw_text = output.outputs[0].text.strip()
                parsed_result = self._parse_crafted_wrong_answer(raw_text)
                wrong_answer_dict[str(problem_id)] = parsed_result
            
        except Exception as e:
            print(f"vLLM generation failed: {e}")
            import traceback
            traceback.print_exc()
        
        return wrong_answer_dict
    
    def _build_crafted_wrong_answer_prompt(
        self, 
        problem_text: str,
        correct_answer: str,
        error_types: List[str]
    ) -> str:
        error_types_str = ", ".join(error_types)
        prompt = f"""You are an Expert Teacher. Your task is to create a "wrong answer" for training purposes.

**CRITICAL INSTRUCTION**: Do NOT use step-by-step reasoning or "<think>" tags. Generate a DIRECT, complete answer.

**Problem:**
{problem_text}

**Correct Answer:**
{correct_answer}

**Your Task:**
Write a complete wrong answer (2-3 paragraphs) that:
1. Looks convincing and professional
2. Contains ONE subtle error leading to wrong conclusion
3. Is written in a confident, direct style (NOT step-by-step)
4. The FINAL ANSWER must be WRONG (different from the ground truth)
5. Under 600 tokens

**Error Types to choose from:** {error_types_str}

**Output Format (JSON only, no markdown):**
{{
"error_type": "one of: {error_types_str}",
"wrong_answer": "A natural-sounding wrong answer in 2-3 paragraphs. Write as if directly presenting a solution."
}}

Generate JSON now:"""
        return prompt
    
    def _parse_crafted_wrong_answer(self, raw_text: str) -> Dict:
        def robust_json_parse(text):
            text = text.strip()
            if text.startswith("```"):
                first_newline = text.find('\n')
                if first_newline != -1:
                    text = text[first_newline+1:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            try:
                import re
                fixed_text = re.sub(r'\\(?![/u"\\bfnrt])', r'\\\\', text)
                return json.loads(fixed_text)
            except json.JSONDecodeError:
                pass

            data = {}
            import re
            error_type_match = re.search(r'"error_type"\s*:\s*"([^"]+)"', text)
            if error_type_match:
                data["error_type"] = error_type_match.group(1)
            
            wrong_answer_pattern = r'"wrong_answer"\s*:\s*"(.*?)"\s*\}'
            wrong_answer_match = re.search(wrong_answer_pattern, text, re.DOTALL)
            
            if wrong_answer_match:
                raw_answer = wrong_answer_match.group(1)
                try:
                    decoded_answer = raw_answer.replace(r'\"', '"').replace(r'\\', '\\').replace(r'\n', '\n')
                    data["wrong_answer"] = decoded_answer
                except Exception:
                    data["wrong_answer"] = raw_answer
            
            if all(k in data for k in ["error_type", "wrong_answer"]):
                return data
                
            return None
        
        try:
            parsed = robust_json_parse(raw_text)
            
            if parsed is not None:
                return {
                    "error_type": parsed.get("error_type", "unknown"),
                    "wrong_answer": parsed.get("wrong_answer", "")
                }
            else:
                return {
                    "error_type": "parsing_failed",
                    "wrong_answer": raw_text
                }
                
        except Exception as e:
            return {
                "error_type": "parsing_error",
                "wrong_answer": raw_text
            }
    
    def generate_guided_lemmas_batch(
        self, 
        problems: Dict[str, str],
        ground_truth_answers: Dict[str, str],
        num_lemmas: int = 3
    ) -> Dict[str, Dict]:
        """Generate guided lemmas for a batch of problems."""
        lemma_dict = {}
        
        prompts = []
        problem_ids = []
        
        for problem_id in problems.keys():
            if problem_id not in ground_truth_answers:
                continue
            
            prompt = self._build_lemma_generation_prompt(
                problems[problem_id],
                ground_truth_answers[problem_id],
                num_lemmas
            )
            prompts.append(prompt)
            problem_ids.append(problem_id)
        
        if not prompts:
            return lemma_dict
        
        try:
            outputs = self.llm.generate(prompts, self.sampling_params_lemma)
            
            for problem_id, output in zip(problem_ids, outputs):
                raw_text = output.outputs[0].text.strip()
                parsed_result, _ = self._parse_lemmas(raw_text, num_lemmas)
                lemma_dict[str(problem_id)] = parsed_result
            
        except Exception as e:
            print(f"vLLM generation failed: {e}")
            import traceback
            traceback.print_exc()
        
        return lemma_dict
    
    def _build_lemma_generation_prompt(
        self, 
        problem_text: str,
        correct_answer: str,
        num_lemmas: int
    ) -> str:
        prompt = f"""You are an Expert Math Teacher. Your task is to create {num_lemmas} progressive hints (lemmas) to guide a struggling student.

**Problem:**
{problem_text}

**Correct Solution:**
{correct_answer}

**Your Task:**
Analyze the correct solution deeply and extract {num_lemmas} KEY LEMMAS that:
1. **Progressively guide** the student from understanding → insight → solution
2. Each lemma should be **precise and concise** (1-2 sentences)
3. Each lemma reveals a CRITICAL insight that bridges understanding gaps
4. Lemmas should be ordered from foundational → advanced
5. DO NOT give away the final answer directly

**Output Format (JSON only):**
{{
    "lemmas": [
        {{
            "order": 1,
            "hint": "First key insight (1-2 sentences)",
            "insight": "Why this matters (1-2 sentence)"
        }},
        {{
            "order": 2,
            "hint": "Second key insight (1-2 sentences)",
            "insight": "Why this matters (1-2 sentence)"
        }},
        {{
            "order": 3,
            "hint": "Third key insight (1-2 sentences)",
            "insight": "Why this matters (1-2 sentence)"
        }}
    ]
}}

Generate JSON now:"""
        return prompt
    
    def _parse_lemmas(self, raw_text: str, expected_count: int) -> tuple:
        """Parse lemmas from raw text."""
        def robust_json_parse(text):
            text = text.strip()
            if text.startswith("```"):
                first_newline = text.find('\n')
                if first_newline != -1:
                    text = text[first_newline+1:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            try:
                import re
                fixed_text = re.sub(r'\\(?![/u"\\bfnrt])', r'\\\\', text)
                return json.loads(fixed_text)
            except json.JSONDecodeError:
                pass

            try:
                import re
                lemmas = []
                lemma_pattern = r'\{\s*"order"\s*:\s*(\d+)\s*,\s*"hint"\s*:\s*"(.*?)"\s*,\s*"insight"\s*:\s*"(.*?)"\s*\}'
                matches = re.finditer(lemma_pattern, text, re.DOTALL)
                
                for match in matches:
                    order = int(match.group(1))
                    hint = match.group(2).replace(r'\"', '"').replace(r'\\', '\\')
                    insight = match.group(3).replace(r'\"', '"').replace(r'\\', '\\')
                    lemmas.append({
                        "order": order,
                        "hint": hint,
                        "insight": insight
                    })
                
                if lemmas:
                    return {"lemmas": lemmas}
            except Exception:
                pass
                
            return None
        
        result = {
            f"lemma{i+1}": {"lemma": "", "insight": ""} 
            for i in range(expected_count)
        }
        
        try:
            parsed = robust_json_parse(raw_text)
            
            if parsed is not None and "lemmas" in parsed:
                lemmas = parsed["lemmas"]
                non_empty_count = 0
                
                if isinstance(lemmas, list):
                    for i, lemma in enumerate(lemmas[:expected_count]):
                        if all(k in lemma for k in ["order", "hint", "insight"]):
                            hint_content = lemma['hint'].strip()
                            insight_content = lemma['insight'].strip()
                            
                            result[f"lemma{i+1}"] = {
                                "lemma": hint_content,
                                "insight": insight_content
                            }
                            
                            if hint_content and insight_content:
                                non_empty_count += 1
                        else:
                            result[f"lemma{i+1}"] = {"lemma": "", "insight": ""}
                    
                    if non_empty_count == expected_count:
                        return result, "success"
                    elif non_empty_count > 0:
                        return result, "partial"
                    else:
                        return result, "failed"
            
            return result, "failed"
                
        except Exception as e:
            return result, "failed"
    
    def generate_subproblems_batch(
        self,
        problems: Dict[str, str],
        reference_answers: Dict[str, str],
        output_file: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        Generate progressive subproblems for a batch of problems using local vLLM model.
        
        Args:
            problems: {problem_id: original_problem_text}
            reference_answers: {problem_id: reference_solution}
            output_file: Optional output JSONL file path. If None, results are only returned.
        
        Returns:
            {problem_id: subproblem_dict}
        """
        from pathlib import Path
        
        subproblems_dict = {}
        
        # 准备所有需要生成的问题
        problem_ids_to_process = []
        prompts = []
        
        for problem_id in problems.keys():
            if problem_id not in reference_answers:
                print(f"⚠️ Skipping problem {problem_id}: no reference answer")
                continue
            
            problem_ids_to_process.append(problem_id)
            prompt = self._build_subproblem_prompt(
                problems[problem_id],
                reference_answers[problem_id]
            )
            prompts.append(prompt)
        
        if not problem_ids_to_process:
            print(f"✅ No problems to generate subproblems for.")
            return subproblems_dict
        
        print(f"\n{'='*60}")
        print(f"🎯 Generating progressive subproblems for {len(problem_ids_to_process)} problems using local model")
        if output_file:
            print(f"📁 Output file: {output_file}")
        print(f"{'='*60}\n")
        
        # 确保输出目录存在
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # 使用 vLLM 批量生成
            outputs = self.llm.generate(prompts, self.sampling_params_subproblem)
            
            success_count = 0
            fail_count = 0
            
            # 处理结果
            for problem_id, output in zip(problem_ids_to_process, outputs):
                raw_text = output.outputs[0].text.strip()
                parsed_result = self._parse_subproblem_response(raw_text, problem_id)
                
                if parsed_result and parsed_result.get("problem_statement", ""):
                    subproblems_dict[str(problem_id)] = parsed_result
                    success_count += 1
                    print(f"✅ Problem {problem_id}: Successfully generated subproblems")
                else:
                    # 创建空结构
                    empty_structure = self._create_empty_subproblem_structure_vllm(str(problem_id))
                    subproblems_dict[str(problem_id)] = empty_structure
                    fail_count += 1
                    print(f"❌ Problem {problem_id}: Failed to generate valid subproblems")
            
            # 写入文件（如果指定）
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    for problem_id in problem_ids_to_process:
                        subproblem_data = subproblems_dict.get(str(problem_id), {})
                        subproblem_data_with_id = {"problem_id": str(problem_id), **subproblem_data}
                        f.write(json.dumps(subproblem_data_with_id, ensure_ascii=False) + '\n')
                        f.flush()
            
            # 统计信息
            print(f"\n{'='*60}")
            print(f"✅ Successfully generated subproblems: {success_count}/{len(problem_ids_to_process)}")
            print(f"❌ Failed to generate subproblems: {fail_count}/{len(problem_ids_to_process)}")
            if output_file:
                print(f"📁 Results saved to: {output_file}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"❌ vLLM generation failed: {e}")
            import traceback
            traceback.print_exc()
            # 为所有问题创建空结构
            for problem_id in problem_ids_to_process:
                empty_structure = self._create_empty_subproblem_structure_vllm(str(problem_id))
                subproblems_dict[str(problem_id)] = empty_structure
        
        return subproblems_dict
    
    def _build_subproblem_prompt(
        self,
        original_problem: str,
        reference_answer: str
    ) -> str:
        """Build prompt for subproblem generation."""
        prompt = f"""You are an expert at breaking down complex mathematical problems into progressive sub-problems. Your task is to analyze a given mathematical problem and its reference solution, then create 4 progressive sub-questions where each builds upon the previous one.

**CRITICAL REQUIREMENTS:**

Your response MUST be a valid JSON object with the following structure:

{{
  "problem_statement": "<The original problem WITHOUT the final question>",
  "question_1": {{
    "statement": "<First sub-question statement>",
    "ground_truth": "<Final answer only, extractable content from \\\\boxed{{}}>"
  }},
  "question_2": {{
    "statement": "<Second sub-question statement>",
    "ground_truth": "<Final answer only, extractable content from \\\\boxed{{}}>"
  }},
  "question_3": {{
    "statement": "<Third sub-question statement>",
    "ground_truth": "<Final answer only, extractable content from \\\\boxed{{}}>"
  }},
  "question_4": {{
    "statement": "<Fourth sub-question statement - MUST be the original question>",
    "ground_truth": "<Final answer only, extractable content from \\\\boxed{{}}>"
  }}
}}

**Design Principles:**
1. **Progressive Difficulty**: Each question should be easier than the next, building foundational understanding
2. **Interconnected**: Later questions should build on concepts/results from earlier questions
3. **Question 4 Requirement**: Must be the exact final question from the original problem
4. **Ground Truth Format**: 
   - Should be the FINAL ANSWER ONLY (not the reasoning process)
   - Should be what would appear inside \\boxed{{}} in the solution
   - Should be directly comparable to student responses for correctness checking
   - Examples: a number like "88327", an expression like "2", a tuple like "(4, 9, 10)", etc.
   - Should be a single numerical value, not a list, set, or other complex data structures.
5. **Problem Statement**: Extract the problem setup/context WITHOUT the final question

**Example Structure:**
- Question 1: Verify basic understanding or compute initial simple cases
- Question 2: Compute intermediate results or recognize patterns  
- Question 3: Discover key insights, formulas, or critical observations
- Question 4: Apply everything to solve the original problem

**Guidelines:**
- Keep question statements clear and concise
- Ensure each ground_truth is objective and verifiable
- Make sure Questions 1-3 naturally lead to Question 4
- The difficulty gap between consecutive questions should be manageable
- Ground truths should be concrete values/expressions, not explanations

**OUTPUT FORMAT:**
Return ONLY the JSON object. Do not include any explanatory text, markdown formatting, or additional commentary before or after the JSON.

Given the following Original Problem and Reference Answer, generate the JSON output with 4 progressive sub-questions:

**Original Problem:**
{original_problem}

**Reference Answer:**
{reference_answer}

Output the JSON only, without any additional text or markdown formatting.

JSON Output:"""
        return prompt
    
    def _parse_subproblem_response(self, raw_text: str, problem_id: str) -> Optional[Dict]:
        """Parse subproblem generation response from model."""
        import re
        
        def robust_json_parse_subproblem(text):
            """Robust JSON parser for subproblem responses."""
            text = text.strip()
            
            # 移除markdown代码块标记
            if text.startswith("```"):
                first_newline = text.find('\n')
                if first_newline != -1:
                    text = text[first_newline+1:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            # 尝试直接解析
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
            
            # 尝试修复转义字符
            try:
                fixed_text = re.sub(r'\\(?![/u"\\bfnrt])', r'\\\\', text)
                return json.loads(fixed_text)
            except json.JSONDecodeError:
                pass
            
            return None
        
        parsed_data = robust_json_parse_subproblem(raw_text)
        
        if parsed_data is None:
            print(f"⚠️ Problem {problem_id}: JSON parsing failed")
            return None
        
        # 验证必需字段
        required_fields = ["problem_statement", "question_1", "question_2", "question_3", "question_4"]
        if not all(field in parsed_data for field in required_fields):
            missing = [f for f in required_fields if f not in parsed_data]
            print(f"⚠️ Problem {problem_id}: Missing fields {missing}")
            return None
        
        # 验证每个question的结构
        for i in range(1, 5):
            q_key = f"question_{i}"
            if not isinstance(parsed_data[q_key], dict):
                print(f"⚠️ Problem {problem_id}: {q_key} is not a dict")
                return None
            if "statement" not in parsed_data[q_key] or "ground_truth" not in parsed_data[q_key]:
                print(f"⚠️ Problem {problem_id}: {q_key} missing statement or ground_truth")
                return None
        
        # 返回解析后的数据
        return {
            "problem_statement": parsed_data["problem_statement"],
            "question_1": parsed_data["question_1"],
            "question_2": parsed_data["question_2"],
            "question_3": parsed_data["question_3"],
            "question_4": parsed_data["question_4"]
        }
    
    def _create_empty_subproblem_structure_vllm(self, problem_id: str) -> Dict:
        """Create empty subproblem structure (used when generation fails)."""
        # 注意：不包含 problem_id，因为保存时会统一添加
        return {
            "problem_statement": "",
            "question_1": {
                "statement": "",
                "ground_truth": ""
            },
            "question_2": {
                "statement": "",
                "ground_truth": ""
            },
            "question_3": {
                "statement": "",
                "ground_truth": ""
            },
            "question_4": {
                "statement": "",
                "ground_truth": ""
            }
        }

def truncate_response_by_subproblem_correctness(
    gen_batch_output: Any,
    mixed_data: Any,
    tokenizer: Any,
    reward_fn: Any,
    use_subproblem_prompt: bool = False
) -> Any:
    """
    For teacher_student v5: Truncate responses for teacher_student_mixed=True samples
    based on subproblem correctness.
    
    Strategy:
    - Split response by **Subproblem k** markers (each part corresponds to a subproblem answer)
    - Check correctness of each subproblem (1-4) by comparing with ground_truth_sub1-4
    - Find the largest k where first k subproblems are correct but (k+1)th is wrong
    - If k is in [1, 3], truncate response to keep only first k parts
    - If k=0 (first wrong), k=4 (all correct), or cannot split into 4 parts, keep original response
    
    Args:
        gen_batch_output: DataProto containing generated responses
        mixed_data: DataProto containing mixed data with reward_model and teacher_student_mixed
        tokenizer: Tokenizer for encoding/decoding
        reward_fn: Reward function to check subproblem correctness (optional, for future use)
        use_subproblem_prompt: If True, the chat template adds "Assistant:**Subproblem 1**:\n",
            so the model response doesn't include "**Subproblem 1**:" marker. We need to add it
            before parsing, then remove it after truncation.
    
    Returns:
        Modified gen_batch_output with truncated responses for teacher_student_mixed=True samples
    """
    import torch
    import numpy as np
    from verl.protocol import DataProto
    if mixed_data is None:
        return gen_batch_output
    
    # Check if teacher_student_mixed flag exists
    if "teacher_student_mixed" not in mixed_data.non_tensor_batch:
        return gen_batch_output
    teacher_student_mixed_flags = mixed_data.non_tensor_batch["teacher_student_mixed"]
    if not isinstance(teacher_student_mixed_flags, np.ndarray):
        teacher_student_mixed_flags: Any = np.array(teacher_student_mixed_flags)
    
    # Only process samples where teacher_student_mixed=True
    mixed_indices = np.where(teacher_student_mixed_flags)[0]
    if len(mixed_indices) == 0:
        return gen_batch_output
    
    # Get responses and reward_models
    responses = gen_batch_output.batch["responses"]
    reward_models = mixed_data.non_tensor_batch.get("reward_model", None)
    
    if reward_models is None:
        return gen_batch_output
    
    # Process each mixed sample
    modified_responses = responses.clone()
    # Track which samples were successfully truncated
    truncated_flags = np.zeros(len(responses), dtype=bool)
    # Track the valid length (number of valid tokens including EOS) for each truncated sample
    # None: not truncated, int: valid length for truncated samples
    truncated_valid_lengths = np.full(len(responses), None, dtype=object)
    # Track how many subproblems were correct for each sample
    # None: original problem (non-mixed), -1: extraction failed, 0-4: number of correct subproblems
    subproblem_correct_count = np.full(len(responses), None, dtype=object)
    
    for idx in mixed_indices:
        idx = int(idx)
        # Decode response
        response_tensor = responses[idx]
        _, response_text_original = decode_tensor(tokenizer, response_tensor, skip_specials=False)
        
        # Get reward_model for this sample
        reward_model = reward_models[idx] if isinstance(reward_models, np.ndarray) else reward_models
        if isinstance(reward_model, dict):
            pass  # Already a dict
        elif hasattr(reward_model, '__dict__'):
            reward_model = reward_model.__dict__
        else:
            # Cannot process reward_model, mark k=-1 for invalid
            subproblem_correct_count[idx] = -1
            continue
        
        # Get ground truths for subproblems
        ground_truths = [
            reward_model.get('ground_truth_sub1', ''),
            reward_model.get('ground_truth_sub2', ''),
            reward_model.get('ground_truth_sub3', ''),
            reward_model.get('ground_truth_sub4', '')
        ]
        
        # Handle use_subproblem_prompt case: if chat template adds "Assistant:**Subproblem 1**:\n",
        # the model response won't include "**Subproblem 1**:" marker, so we need to add it first
        # But we save the original response_text for truncation
        response_text = response_text_original
        added_prefix = False
        prefix_length = 0
        if use_subproblem_prompt:
            import re
            subproblem_pattern = r'\*\*Subproblem\s+(\d+)\*\*'
            # Check if response already starts with **Subproblem 1**
            first_match = re.search(subproblem_pattern, response_text, re.IGNORECASE)
            if not first_match:
                # No subproblem marker found at all, add **Subproblem 1**: at the beginning
                prefix_text = "**Subproblem 1**:\n"
                response_text = prefix_text + response_text
                added_prefix = True
                prefix_length = len(prefix_text)
            elif first_match.start() > 0 or first_match.group(1) != '1':
                # Response doesn't start with **Subproblem 1** (either has prefix text or starts with different number)
                # Add **Subproblem 1**: at the beginning
                prefix_text = "**Subproblem 1**:\n"
                response_text = prefix_text + response_text
                added_prefix = True
                prefix_length = len(prefix_text)
        
        # Split response by **Subproblem k** markers
        # Pattern: **Subproblem 1**, **Subproblem 2**, **Subproblem 3**, **Subproblem 4**
        import re
        subproblem_pattern = r'\*\*Subproblem\s+(\d+)\*\*'
        
        # Find all subproblem markers and their positions
        matches = list(re.finditer(subproblem_pattern, response_text, re.IGNORECASE))
        if len(matches) < 4:
            # Cannot find 4 subproblem markers, keep original
            # Mark k=-1 for invalid parsing
            subproblem_correct_count[idx] = -1
            continue
        
        # Extract parts between markers (and after the last marker)
        subproblem_parts = []
        for i in range(len(matches)):
            start_pos = matches[i].end()  # Start after the marker
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()  # End before next marker
                part = response_text[start_pos:end_pos].strip()
            else:
                # Last part: from marker to end of response
                part = response_text[start_pos:].strip()
            subproblem_parts.append(part)
        
        
        # Check correctness of each subproblem using reward_fn
        # Extract answer from each part (the whole part is the answer)
        correct_flags = []
        for i, (part, gt) in enumerate(zip(subproblem_parts, ground_truths)):
            if not gt or str(gt).strip() == '':
                # No ground_truth for this subproblem, assume correct
                correct_flags.append(True)
                continue
            
            # The whole part is the answer (no need to extract after </think> anymore)
            answer_part = part.strip()
            
            if not answer_part:
                # Empty answer, assume wrong
                correct_flags.append(False)
                continue
            
            # Use prime_math_refine.compute_score directly to check correctness
            # This is simpler and more direct than creating DataProto and calling reward_fn
            try:
                from verl.utils.reward_score.prime_math_refine import compute_score as prime_math_refine_score
                # prime_math_refine.compute_score expects:
                # - model_output: str (the answer part, which already contains the full text)
                # - ground_truth: str (the ground truth answer)
                # Returns: (is_correct, format_correctness, extracted_answer)
                is_correct, format_correctness, extracted_answer = prime_math_refine_score(answer_part, str(gt))
            except Exception as e:
                # If prime_math_refine is not available or fails, fallback to simple string matching
                # But still only extract from \boxed{} to avoid matching numbers from calculation process
                print(f"error in prime_math_refine_score: {e}")
                import re
                boxed_pattern = r'\\boxed\{([^}]+)\}'
                boxed_matches = list(re.finditer(boxed_pattern, answer_part))
                if boxed_matches:
                    # Use the last \boxed{} answer (typically the final answer)
                    extracted_answer = boxed_matches[-1].group(1).strip()
                    answer_clean = extracted_answer.lower().strip()
                else:
                    # No \boxed{} found, use whole part (fallback)
                    answer_clean = answer_part.lower().strip()
                gt_clean = str(gt).strip().lower()
                is_correct = (gt_clean == answer_clean)  # Use exact match instead of substring match
            
            correct_flags.append(is_correct)
        
        # Calculate k: number of consecutive correct subproblems from the beginning
        # k=-1: extraction failed (already handled above, should not reach here)
        # k=0: extraction successful but no subproblem is correct
        # k=1,2,3,4: first 1,2,3,4 subproblems are correct
        k = 0  # Start with 0 (extraction successful but no correct subproblems yet)
        for i in range(4):
            if correct_flags[i]:
                k = i + 1  # k is the count of correct subproblems (1-indexed)
            else:
                break  # Found first wrong, stop counting
        
        # Record k value:
        # k=-1: extraction failed (set above when len(matches) < 4)
        # k=0: extraction successful but no subproblem is correct
        # k=1,2,3,4: first 1,2,3,4 subproblems are correct
        subproblem_correct_count[idx] = k
        
        # Only truncate if k is in [1, 3] (partial correct: 1-3 subproblems correct)
        if 1 <= k <= 3:
            # Find the truncation position in the original response_text
            # We want to keep everything up to (but not including) the (k+1)-th subproblem marker
            # matches[k] is the (k+1)-th subproblem marker (0-indexed, so matches[0] is Subproblem 1, matches[1] is Subproblem 2, etc.)
            # We already checked len(matches) >= 4, so matches[k] should exist (k max is 3)
            if k >= len(matches):
                # Safety check: if k >= len(matches), keep original (should not happen)
                subproblem_correct_count[idx] = -1
                continue
            truncate_pos_in_modified = matches[k].start()  # Position of (k+1)-th subproblem marker
            
            # Map this position back to original response_text
            if use_subproblem_prompt and added_prefix:
                # The modified response_text has a prefix, so we need to subtract prefix_length
                # matches[k] is the (k+1)-th subproblem in modified text
                # In original text, this would be at position (matches[k].start() - prefix_length)
                truncate_pos_in_original = matches[k].start() - prefix_length
            else:
                # No prefix added, positions are the same
                truncate_pos_in_original = truncate_pos_in_modified
            
            # Ensure truncate_pos_in_original is within bounds
            truncate_pos_in_original = max(0, min(truncate_pos_in_original, len(response_text_original)))
            
            # Directly truncate the original response_text (this ensures it's a substring from the start)
            truncated_response_text = response_text_original[:truncate_pos_in_original]
            
            # Re-encode to tensor
            truncated_tokens = tokenizer.encode(
                truncated_response_text,
                add_special_tokens=False,
                return_tensors='pt'
            )[0]
            
            # Add EOS token at the end if not already present
            eos_token_id = tokenizer.eos_token_id
            if eos_token_id is not None and (truncated_tokens.numel() == 0 or truncated_tokens[-1].item() != eos_token_id):
                truncated_tokens = torch.cat([truncated_tokens, torch.tensor([eos_token_id], dtype=truncated_tokens.dtype, device=truncated_tokens.device)])
            
            # Pad or truncate to match original length
            original_len = response_tensor.shape[0]
            # valid_length is the number of valid tokens (including EOS) before padding
            valid_length = truncated_tokens.shape[0]
            if truncated_tokens.shape[0] < original_len:
                # Pad with pad_token_id
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                padding = torch.full(
                    (original_len - truncated_tokens.shape[0],),
                    pad_token_id,
                    dtype=truncated_tokens.dtype,
                    device=truncated_tokens.device
                )
                truncated_tokens = torch.cat([truncated_tokens, padding])
            elif truncated_tokens.shape[0] > original_len:
                # Truncate (but keep EOS if possible)
                truncated_tokens = truncated_tokens[:original_len]
                # If we truncated and the last token is not EOS, replace it with EOS
                if eos_token_id is not None and truncated_tokens[-1].item() != eos_token_id:
                    truncated_tokens[-1] = eos_token_id
                valid_length = original_len
            
            # Update the response
            modified_responses[idx] = truncated_tokens.to(responses.device)
            
            # Update attention_mask for this sample
            # The attention_mask should be 1 for valid tokens (0 to valid_length-1) and 0 for padding
            if "attention_mask" in gen_batch_output.batch:
                attention_mask = gen_batch_output.batch["attention_mask"]
                # attention_mask shape: (batch_size, seq_len)
                # We need to update the response part of the attention_mask
                # The response part is typically the last response_length tokens
                response_length = responses.size(1)
                seq_len = attention_mask.size(1)
                response_start = seq_len - response_length
                
                # Create new attention mask for response part
                new_response_mask = torch.zeros(response_length, dtype=attention_mask.dtype, device=attention_mask.device)
                # Set valid tokens to 1 (up to valid_length)
                new_response_mask[:min(valid_length, response_length)] = 1
                
                # Update the response part of attention_mask
                attention_mask[idx, response_start:] = new_response_mask
                gen_batch_output.batch["attention_mask"] = attention_mask
            
            # Update input_ids if it exists (should match responses)
            if "input_ids" in gen_batch_output.batch:
                input_ids = gen_batch_output.batch["input_ids"]
                # input_ids typically contains prompt + response
                # We need to update the response part
                response_length = responses.size(1)
                seq_len = input_ids.size(1)
                response_start = seq_len - response_length
                
                # Update the response part of input_ids to match truncated response
                input_ids[idx, response_start:] = truncated_tokens
                gen_batch_output.batch["input_ids"] = input_ids
            
            # Update response_mask if it exists (must be updated after attention_mask is updated)
            # response_mask is computed from attention_mask[:, -response_length:] (same as compute_response_mask)
            if "response_mask" in gen_batch_output.batch and "attention_mask" in gen_batch_output.batch:
                # Directly extract from updated attention_mask to ensure consistency with compute_response_mask
                attention_mask = gen_batch_output.batch["attention_mask"]
                response_mask = gen_batch_output.batch["response_mask"]
                # Extract response part from attention_mask (same logic as compute_response_mask)
                updated_response_mask = attention_mask[idx, -response_length:]
                response_mask[idx] = updated_response_mask
                gen_batch_output.batch["response_mask"] = response_mask
            
            # Note: We don't update rm_scores here even if it exists, because:
            # 1. rm_scores is only used to determine reward_tensor in reward_fn
            # 2. The fit function will unconditionally zero out and reset reward_tensor to 1.0
            #    at the correct position for truncated samples (after reward computation)
            # 3. Therefore, updating rm_scores here would be redundant
            
            # Update rollout_log_probs if it exists
            # rollout_log_probs shape: (batch_size, response_length), log probs for each token
            # After truncation, we need to zero out log probs for truncated tokens
            if "rollout_log_probs" in gen_batch_output.batch:
                rollout_log_probs = gen_batch_output.batch["rollout_log_probs"]
                # Zero out log probs for tokens beyond valid_length
                rollout_log_probs[idx, valid_length:] = 0.0
                gen_batch_output.batch["rollout_log_probs"] = rollout_log_probs
            
            # Note: position_ids typically don't need to be updated because:
            # 1. The prompt part's position_ids remain unchanged
            # 2. The response part's position_ids are relative positions that don't change
            #    even if the content is truncated (the sequence length stays the same)
            # However, if position_ids need to be updated in the future, it should be done here
            
            # Mark this sample as successfully truncated
            truncated_flags[idx] = True
            # Save the valid length for this truncated sample
            truncated_valid_lengths[idx] = min(valid_length, response_length)
            
            # Debug logging: report successful truncations with problem_id if available
            problem_id = None
            if "problem_id" in mixed_data.non_tensor_batch:
                try:
                    pid_arr = mixed_data.non_tensor_batch["problem_id"]
                    problem_id = pid_arr[idx] if hasattr(pid_arr, "__len__") else pid_arr
                except Exception:
                    problem_id = None
            print(f"[truncate_v5] truncated response idx={idx}, problem_id={problem_id}, kept_subproblems={k}")
        else:
            # Not truncated: log reason
            if k == 0:
                reason = "no_subproblem_correct"
            elif k == 4:
                reason = "all_correct"
            else:
                reason = f"not_in_truncation_range_k={k}"
            problem_id = None
            if "problem_id" in mixed_data.non_tensor_batch:
                try:
                    pid_arr = mixed_data.non_tensor_batch["problem_id"]
                    problem_id = pid_arr[idx] if hasattr(pid_arr, "__len__") else pid_arr
                except Exception:
                    problem_id = None
            # print(f"[truncate_v5] no_truncate idx={idx}, problem_id={problem_id}, reason={reason}")
    
    # Update gen_batch_output with modified responses
    gen_batch_output.batch["responses"] = modified_responses
    
    # Add truncated flags to non_tensor_batch so we can set reward=1.0 later
    if truncated_flags.any():
        gen_batch_output.non_tensor_batch["truncated"] = truncated_flags
        # Save truncated_valid_lengths for efficient reward setting
        gen_batch_output.non_tensor_batch["truncated_valid_lengths"] = truncated_valid_lengths
    
    # Add subproblem correct count (k value) to non_tensor_batch
    # None: original problem (non-mixed), -1: extraction failed, 0-4: number of correct subproblems
    gen_batch_output.non_tensor_batch["subproblem_correct_count"] = subproblem_correct_count
    
    return gen_batch_output

def mark_response_tokens_by_subproblem_correctness(
    gen_batch_output: Any,
    mixed_data: Any,
    tokenizer: Any,
    reward_fn: Any,
    use_subproblem_prompt: bool = False,
    format_mode: str = "subproblem",
    num_problems: int = 4,
    require_strict_eos: bool = True,
    parse_fail_policy: str = "hard",
) -> Any:
    """
    For teacher_student v7: Mark tokens in responses for teacher_student_mixed=True samples
    based on subproblem correctness, WITHOUT truncating the response.
    
    Strategy:
    - Split response by **Subproblem k** markers (each part corresponds to a subproblem answer)
    - Check correctness of each subproblem (1-4) by comparing with ground_truth_sub1-4
    - Find the largest k where first k subproblems are correct but (k+1)th is wrong
    - Mark each token: 1 if it belongs to correct subproblems (1-k), 0 if it belongs to wrong subproblem (k+1)
    - Keep the full response intact (no truncation)
    
    Args:
        gen_batch_output: DataProto containing generated responses
        mixed_data: DataProto containing mixed data with reward_model and teacher_student_mixed
        tokenizer: Tokenizer for encoding/decoding
        reward_fn: Reward function to check subproblem correctness (optional, for future use)
        use_subproblem_prompt: If True, the chat template adds "Assistant:**Subproblem 1**:\n",
            so the model response doesn't include "**Subproblem 1**:" marker. We need to add it
            before parsing.
        format_mode: "subproblem" (legacy **Subproblem k**) or "pn" (<pN></pN> protocol).
        num_problems: default number of problems for "pn" mode.
        require_strict_eos: If True, enforce strict boxed->EOS check when all parts are correct.
        parse_fail_policy: "hard" or "soft" for "pn" parser strictness.
    
    Returns:
        Modified gen_batch_output with token_correctness_mask in batch
        - token_correctness_mask: (batch_size, response_length) tensor, 1 for correct tokens, 0 for wrong tokens
        - subproblem_correct_count: (batch_size,) array, k values for each sample
    """
    import torch
    import numpy as np
    from verl.protocol import DataProto
    
    if mixed_data is None:
        return gen_batch_output
    
    # Check if teacher_student_mixed flag exists
    if "teacher_student_mixed" not in mixed_data.non_tensor_batch:
        return gen_batch_output
    teacher_student_mixed_flags = mixed_data.non_tensor_batch["teacher_student_mixed"]
    if not isinstance(teacher_student_mixed_flags, np.ndarray):
        teacher_student_mixed_flags: Any = np.array(teacher_student_mixed_flags)
    
    # Only process samples where teacher_student_mixed=True
    mixed_indices = np.where(teacher_student_mixed_flags)[0]
    if len(mixed_indices) == 0:
        return gen_batch_output
    
    # Get responses and reward_models
    responses = gen_batch_output.batch["responses"]
    reward_models = mixed_data.non_tensor_batch.get("reward_model", None)
    
    if reward_models is None:
        return gen_batch_output
    
    format_mode = str(format_mode).lower()
    parse_fail_policy = str(parse_fail_policy).lower()

    # Initialize token_correctness_mask: 1 for correct tokens, 0 for wrong tokens
    # For non-mixed samples, we'll set all tokens to 1 (default: treat as correct)
    response_length = responses.size(1)
    batch_size = responses.size(0)
    token_correctness_mask = torch.ones(batch_size, response_length, dtype=torch.float32, device=responses.device)
    
    # Track how many subproblems were correct for each sample
    # None: original problem (non-mixed), -1: extraction failed, 0-4: number of correct subproblems
    subproblem_correct_count = np.full(batch_size, None, dtype=object)

    # v8 helper fields:
    # - subproblem_part_correctness_local: per-sample local part correctness (1/0, -1 parse-fail, -2 unused)
    # - subproblem_part_token_mask: per-sample/local-part token span mask in response tokens
    subproblem_part_correctness_local = np.full((batch_size, 4), -2, dtype=np.int32)
    subproblem_part_token_mask = torch.zeros(
        batch_size, 4, response_length, dtype=torch.float32, device=responses.device
    )
    
    for idx in mixed_indices:
        idx = int(idx)
        
        # Get problem_id for this sample (for debug logging)
        problem_id = None
        if "problem_id" in mixed_data.non_tensor_batch:
            try:
                pid_arr = mixed_data.non_tensor_batch["problem_id"]
                problem_id = pid_arr[idx] if hasattr(pid_arr, "__len__") else pid_arr
            except Exception:
                problem_id = None
        
        # Decode response
        response_tensor = responses[idx]
        _, response_text_original = decode_tensor(tokenizer, response_tensor, skip_specials=False)
        
        # Get reward_model for this sample
        reward_model = reward_models[idx] if isinstance(reward_models, np.ndarray) else reward_models
        if isinstance(reward_model, dict):
            pass  # Already a dict
        elif hasattr(reward_model, '__dict__'):
            reward_model = reward_model.__dict__
        else:
            # Cannot process reward_model, mark k=-1 for invalid
            subproblem_correct_count[idx] = -1
            # Mark all tokens as wrong (0) for invalid samples
            token_correctness_mask[idx, :] = 0.0
            continue
        
        # Get ground truths for subproblems
        ground_truths = [
            reward_model.get('ground_truth_sub1', ''),
            reward_model.get('ground_truth_sub2', ''),
            reward_model.get('ground_truth_sub3', ''),
            reward_model.get('ground_truth_sub4', '')
        ]

        # New mode: generic <pN>...</pN> format for variable-K problems.
        # Keep legacy "subproblem" logic untouched below.
        if format_mode == "pn":
            import re
            response_text = response_text_original

            sample_num_problems = num_problems
            rm_k = reward_model.get("num_problems", None)
            if rm_k is not None:
                try:
                    sample_num_problems = int(rm_k)
                except Exception:
                    pass
            sample_num_problems = max(1, min(4, int(sample_num_problems)))
            sample_ground_truths = ground_truths[:sample_num_problems]

            parse_failed = False
            parse_fail_reason = ""
            parts = []
            segment_start_chars = []
            cursor = 0

            for p_idx in range(1, sample_num_problems + 1):
                open_pat = re.compile(rf"<p{p_idx}>", re.IGNORECASE)
                close_pat = re.compile(rf"</p{p_idx}>", re.IGNORECASE)
                m_open = open_pat.search(response_text, cursor)
                if m_open is None:
                    parse_failed = True
                    parse_fail_reason = f"missing_open_p{p_idx}"
                    break
                m_close = close_pat.search(response_text, m_open.end())
                if m_close is None:
                    parse_failed = True
                    parse_fail_reason = f"missing_close_p{p_idx}"
                    break

                # Strict mode: forbid non-whitespace prefix before <p1>.
                if p_idx == 1 and parse_fail_policy == "hard":
                    if response_text[:m_open.start()].strip():
                        parse_failed = True
                        parse_fail_reason = "non_whitespace_before_p1"
                        break

                segment_start_chars.append(m_open.start())
                parts.append(response_text[m_open.end():m_close.start()].strip())
                cursor = m_close.end()

            if (not parse_failed) and parse_fail_policy == "hard":
                # In hard mode, disallow additional <pN> tags after expected K parts.
                if re.search(r"<\s*/?\s*p\d+\s*>", response_text[cursor:], re.IGNORECASE):
                    parse_failed = True
                    parse_fail_reason = "extra_p_tags_after_expected_k"

            if parse_failed:
                subproblem_correct_count[idx] = -1
                token_correctness_mask[idx, :] = 0.0
                subproblem_part_correctness_local[idx, :sample_num_problems] = -1
                print(f"[mark_v7_pn] idx={idx}, problem_id={problem_id}, k=-1, parse_failed={parse_fail_reason}")
                continue

            correct_flags = []
            for part, gt in zip(parts, sample_ground_truths):
                if not gt or str(gt).strip() == '':
                    correct_flags.append(True)
                    continue
                answer_part = part.strip()
                if not answer_part:
                    correct_flags.append(False)
                    continue
                try:
                    from verl.utils.reward_score.prime_math_refine import compute_score as prime_math_refine_score
                    is_correct, _, _ = prime_math_refine_score(answer_part, str(gt))
                except Exception:
                    boxed_pattern = r'\\boxed\{([^}]+)\}'
                    boxed_matches = list(re.finditer(boxed_pattern, answer_part))
                    if boxed_matches:
                        extracted_answer = boxed_matches[-1].group(1).strip()
                        answer_clean = extracted_answer.lower().strip()
                    else:
                        answer_clean = answer_part.lower().strip()
                    gt_clean = str(gt).strip().lower()
                    is_correct = (gt_clean == answer_clean)
                correct_flags.append(is_correct)

            # Save per-part correctness for v8 (local indices 1..K).
            for p_local in range(sample_num_problems):
                subproblem_part_correctness_local[idx, p_local] = 1 if correct_flags[p_local] else 0

            k = 0
            for i in range(sample_num_problems):
                if correct_flags[i]:
                    k = i + 1
                else:
                    break
            subproblem_correct_count[idx] = k

            token_boundaries = []
            for char_boundary in segment_start_chars + [len(response_text_original)]:
                prefix_tokens = tokenizer.encode(
                    response_text_original[:char_boundary],
                    add_special_tokens=False,
                    return_tensors='pt'
                )[0]
                token_boundaries.append(min(prefix_tokens.shape[0], response_length))

            # Per-local-part token spans for v8: [start_of_<pN>, start_of_<pN+1>) with last to end.
            for p_local in range(sample_num_problems):
                start_tok = token_boundaries[p_local]
                end_tok = token_boundaries[p_local + 1]
                if 0 <= start_tok < end_tok <= response_length:
                    subproblem_part_token_mask[idx, p_local, start_tok:end_tok] = 1.0

            if k == 0:
                token_correctness_mask[idx, :] = 0.0
            elif 1 <= k < sample_num_problems:
                wrong_start_token = token_boundaries[k]
                token_correctness_mask[idx, wrong_start_token:] = 0.0
            elif k == sample_num_problems and require_strict_eos:
                # Strictly require last boxed in final part to be followed by EOS.
                # Use extracted final part directly.
                last_part_text = parts[-1]
                boxed_start_idx = last_part_text.rfind("\\boxed")
                if boxed_start_idx < 0:
                    boxed_start_idx = last_part_text.rfind("\\fbox")
                boxed_end_in_original = None
                if boxed_start_idx >= 0:
                    i = boxed_start_idx
                    left_brace_idx = None
                    right_brace_idx = None
                    num_left_braces_open = 0
                    while i < len(last_part_text):
                        if last_part_text[i] == "{":
                            num_left_braces_open += 1
                            if left_brace_idx is None:
                                left_brace_idx = i
                        elif last_part_text[i] == "}":
                            num_left_braces_open -= 1
                            if num_left_braces_open == 0:
                                right_brace_idx = i
                                break
                        i += 1
                    if left_brace_idx is not None and right_brace_idx is not None:
                        # Find final part in original response; use rfind to avoid ambiguity.
                        final_part_start = response_text_original.rfind(last_part_text)
                        if final_part_start >= 0:
                            boxed_end_in_original = final_part_start + right_brace_idx + 1

                if boxed_end_in_original is None:
                    token_correctness_mask[idx, :] = 0.0
                    subproblem_correct_count[idx] = -1
                    subproblem_part_correctness_local[idx, :sample_num_problems] = -1
                    k = -1
                else:
                    prefix_tokens = tokenizer.encode(
                        response_text_original[:boxed_end_in_original],
                        add_special_tokens=False,
                        return_tensors='pt'
                    )[0]
                    boxed_end_token_pos = min(prefix_tokens.shape[0], response_length)
                    eos_token_id = tokenizer.eos_token_id
                    if eos_token_id is None or boxed_end_token_pos >= response_length:
                        token_correctness_mask[idx, :] = 0.0
                        subproblem_correct_count[idx] = -1
                        subproblem_part_correctness_local[idx, :sample_num_problems] = -1
                        k = -1
                    else:
                        next_token_id = response_tensor[boxed_end_token_pos].item()
                        if next_token_id != eos_token_id:
                            token_correctness_mask[idx, :] = 0.0
                            subproblem_correct_count[idx] = -1
                            subproblem_part_correctness_local[idx, :sample_num_problems] = -1
                            k = -1

            # Mark padding after first EOS as wrong to avoid leaking pad tokens as "correct".
            eos_token_id = tokenizer.eos_token_id
            if eos_token_id is not None:
                eos_positions = (response_tensor == eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    padding_start_token = eos_positions[0].item() + 1
                    if padding_start_token < response_length:
                        token_correctness_mask[idx, padding_start_token:] = 0.0

            correct_tokens = token_correctness_mask[idx].sum().item()
            total_tokens = token_correctness_mask[idx].numel()
            print(f"[mark_v7_pn] idx={idx}, problem_id={problem_id}, k={subproblem_correct_count[idx]}, correct_tokens={correct_tokens}/{total_tokens}, K={sample_num_problems}")
            continue
        
        # Handle use_subproblem_prompt case: if chat template adds "Assistant:**Subproblem 1**:\n",
        # the model response won't include "**Subproblem 1**:" marker, so we need to add it first
        # But we save the original response_text for token mapping
        response_text = response_text_original
        added_prefix = False
        prefix_length = 0
        if use_subproblem_prompt:
            import re
            subproblem_pattern = r'\*\*Subproblem\s+(\d+)\*\*'
            # Check if response already starts with **Subproblem 1**
            first_match = re.search(subproblem_pattern, response_text, re.IGNORECASE)
            if not first_match:
                # No subproblem marker found at all, add **Subproblem 1**: at the beginning
                prefix_text = "**Subproblem 1**:\n"
                response_text = prefix_text + response_text
                added_prefix = True
                prefix_length = len(prefix_text)
            elif first_match.start() > 0 or first_match.group(1) != '1':
                # Response doesn't start with **Subproblem 1** (either has prefix text or starts with different number)
                # Add **Subproblem 1**: at the beginning
                prefix_text = "**Subproblem 1**:\n"
                response_text = prefix_text + response_text
                added_prefix = True
                prefix_length = len(prefix_text)
        
        # Split response by **Subproblem k** markers
        # Pattern: **Subproblem 1**, **Subproblem 2**, **Subproblem 3**, **Subproblem 4**
        import re
        subproblem_pattern = r'\*\*Subproblem\s+(\d+)\*\*'
        
        # Find all subproblem markers and their positions
        matches = list(re.finditer(subproblem_pattern, response_text, re.IGNORECASE))
        if len(matches) != 4:
            # Cannot find 4 subproblem markers, mark k=-1 for invalid parsing
            subproblem_correct_count[idx] = -1
            # Mark all tokens as wrong (0) for invalid parsing
            token_correctness_mask[idx, :] = 0.0
            continue
        
        # Extract parts between markers (and after the last marker)
        subproblem_parts = []
        for i in range(len(matches)):
            start_pos = matches[i].end()  # Start after the marker
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()  # End before next marker
                part = response_text[start_pos:end_pos].strip()
            else:
                # Last part: from marker to end of response
                part = response_text[start_pos:].strip()
            subproblem_parts.append(part)
        
        # Check correctness of each subproblem using reward_fn
        # Extract answer from each part (the whole part is the answer)
        correct_flags = []
        for i, (part, gt) in enumerate(zip(subproblem_parts, ground_truths)):
            if not gt or str(gt).strip() == '':
                # No ground_truth for this subproblem, assume correct
                correct_flags.append(True)
                continue
            
            # The whole part is the answer (no need to extract after </think> anymore)
            answer_part = part.strip()
            
            if not answer_part:
                # Empty answer, assume wrong
                correct_flags.append(False)
                continue
            
            # Use prime_math_refine.compute_score directly to check correctness
            try:
                from verl.utils.reward_score.prime_math_refine import compute_score as prime_math_refine_score
                is_correct, format_correctness, extracted_answer = prime_math_refine_score(answer_part, str(gt))
            except Exception as e:
                # If prime_math_refine is not available or fails, fallback to simple string matching
                print(f"error in prime_math_refine_score: {e}")
                import re
                boxed_pattern = r'\\boxed\{([^}]+)\}'
                boxed_matches = list(re.finditer(boxed_pattern, answer_part))
                if boxed_matches:
                    extracted_answer = boxed_matches[-1].group(1).strip()
                    answer_clean = extracted_answer.lower().strip()
                else:
                    answer_clean = answer_part.lower().strip()
                gt_clean = str(gt).strip().lower()
                is_correct = (gt_clean == answer_clean)
            
            correct_flags.append(is_correct)
        
        # Calculate k: number of consecutive correct subproblems from the beginning
        k = 0  # Start with 0 (extraction successful but no correct subproblems yet)
        for i in range(4):
            if correct_flags[i]:
                k = i + 1  # k is the count of correct subproblems (1-indexed)
            else:
                break  # Found first wrong, stop counting
        
        # Record k value
        subproblem_correct_count[idx] = k
        
        # Map text positions to token positions
        # Strategy: encode the full original response, then find token positions for each subproblem
        # by encoding prefixes up to each subproblem boundary
        
        # Find character positions for each subproblem boundary in original text
        # We need to find where each subproblem starts and ends in the original text
        subproblem_char_boundaries = []
        
        # Find the start of first subproblem
        if use_subproblem_prompt and added_prefix:
            # First subproblem starts at position 0 in original text
            subproblem_char_boundaries.append(0)
        else:
            # First subproblem starts after first marker
            # matches[0] is in modified text, need to map to original
            if matches[0].start() >= prefix_length:
                subproblem_char_boundaries.append(matches[0].end() - prefix_length)
            else:
                subproblem_char_boundaries.append(0)
        
        # Find boundaries for subsequent subproblems
        for i in range(1, len(matches)):
            # matches[i] is the i-th subproblem marker (0-indexed, so matches[1] is Subproblem 2)
            if use_subproblem_prompt and added_prefix:
                # Map from modified text to original text
                char_pos = matches[i].start() - prefix_length
            else:
                char_pos = matches[i].start()
            subproblem_char_boundaries.append(max(0, char_pos))
        
        # Add end boundary
        subproblem_char_boundaries.append(len(response_text_original))
        
        # Map character boundaries to token boundaries by encoding prefixes
        subproblem_token_boundaries = []
        for char_boundary in subproblem_char_boundaries:
            # Encode prefix up to this character position
            prefix_text = response_text_original[:char_boundary]
            prefix_tokens = tokenizer.encode(
                prefix_text,
                add_special_tokens=False,
                return_tensors='pt'
            )[0]
            token_pos = prefix_tokens.shape[0]
            # Ensure token position is within response_length
            token_pos = min(token_pos, response_length)
            subproblem_token_boundaries.append(token_pos)
        
        # Mark tokens based on correctness
        # For tokens in subproblems 1 to k: mark as correct (1)
        # For tokens in subproblem k+1 and beyond: mark as wrong (0)
        if k == 0:
            # No correct subproblems, mark all as wrong
            token_correctness_mask[idx, :] = 0.0
        elif k == 4:
            # All 4 subproblems are correct, but need to check if subproblem 4 ends properly
            # Check if the last \boxed{} in subproblem 4 is followed by EOS token
            # Find the last \boxed{} in subproblem 4 part of response_text
            control_para = 1  # 1: mark all tokens as wrong, 2: mark tokens after \boxed{} as wrong
            import re
            
            # Find the start position of subproblem 4 in response_text
            subproblem_4_start_in_response_text = matches[3].end()  # matches[3] is Subproblem 4 marker
            # Subproblem 4 part is from matches[3].end() to the end of response_text
            subproblem_4_text = response_text[subproblem_4_start_in_response_text:]
            
            # Find the last \boxed{} in subproblem 4 part, handling nested braces
            # Use the same method as prime_math_refine to handle nested braces like \boxed{\frac{3}{5}}
            boxed_start_idx = subproblem_4_text.rfind("\\boxed")
            if boxed_start_idx < 0:
                boxed_start_idx = subproblem_4_text.rfind("\\fbox")
            
            boxed_end_in_response_text = None
            if boxed_start_idx >= 0:
                # Found \boxed or \fbox, now find the matching closing brace
                i = boxed_start_idx
                left_brace_idx = None
                right_brace_idx = None
                num_left_braces_open = 0
                while i < len(subproblem_4_text):
                    if subproblem_4_text[i] == "{":
                        num_left_braces_open += 1
                        if left_brace_idx is None:
                            left_brace_idx = i
                    elif subproblem_4_text[i] == "}":
                        num_left_braces_open -= 1
                        if num_left_braces_open == 0:
                            right_brace_idx = i
                            break
                    i += 1
                
                if left_brace_idx is not None and right_brace_idx is not None:
                    # Found matching closing brace, right_brace_idx is the position of '}'
                    # boxed_end_in_response_text should be the position after '}'
                    boxed_end_in_response_text = subproblem_4_start_in_response_text + right_brace_idx + 1
            
            if boxed_end_in_response_text is not None:
                
                # Map this position to response_text_original
                if use_subproblem_prompt and added_prefix:
                    # Need to subtract prefix_length to map to original text
                    boxed_end_in_original = boxed_end_in_response_text - prefix_length
                else:
                    boxed_end_in_original = boxed_end_in_response_text
                
                # Ensure position is within bounds
                boxed_end_in_original = max(0, min(boxed_end_in_original, len(response_text_original)))
                
                # Map character position to token position
                # boxed_end_in_original is the position after \boxed{}'s closing brace '}'
                # We want to include the '}' token in correct tokens, so we encode up to boxed_end_in_original
                # which includes the '}' character
                prefix_text = response_text_original[:boxed_end_in_original]
                prefix_tokens = tokenizer.encode(
                    prefix_text,
                    add_special_tokens=False,
                    return_tensors='pt'
                )[0]
                # boxed_end_token_pos is the position after the last token of \boxed{} (including '}')
                # So tokens from 0 to boxed_end_token_pos-1 are correct (including \boxed{} and '}')
                # Tokens from boxed_end_token_pos onwards should be wrong
                boxed_end_token_pos = prefix_tokens.shape[0]
                # Ensure boxed_end_token_pos doesn't exceed response_length
                boxed_end_token_pos = min(boxed_end_token_pos, response_length)
                
                # Check if \boxed{} is followed by EOS (<|im_end|>)
                # If not, mark all tokens as wrong
                # If yes, mark tokens after EOS (padding tokens) as wrong
                eos_token_id = tokenizer.eos_token_id
                if eos_token_id is None:
                    # No EOS token defined, mark all tokens as wrong
                    token_correctness_mask[idx, :] = 0.0
                    subproblem_correct_count[idx] = -1
                    k = -1
                    print(f"[mark_v7] idx={idx}, problem_id={problem_id}, k={k}, no_eos_token_defined, marking all tokens as wrong")
                elif boxed_end_token_pos < response_length:
                    # Check if next token after \boxed{} is EOS
                    next_token_id = response_tensor[boxed_end_token_pos].item()
                    if next_token_id != eos_token_id:
                        # Last \boxed{} is not followed by EOS (<|im_end|>)
                        if control_para == 1:
                            # Mark all tokens as wrong (0.0)
                            token_correctness_mask[idx, :] = 0.0
                            subproblem_correct_count[idx] = -1
                            k = -1
                            print(f"[mark_v7] idx={idx}, problem_id={problem_id}, k={k}, last_boxed_not_followed_by_eos, control_para=1, marking all tokens as wrong, boxed_end_token_pos={boxed_end_token_pos}, next_token_id={next_token_id}, eos_token_id={eos_token_id}, response_length={response_length}")
                        elif control_para == 2:
                            # Mark tokens from \boxed{} end position onwards as wrong (0.0)
                            # Tokens before \boxed{} end (i.e., \boxed{} and all previous tokens) remain correct (already 1.0)
                            # boxed_end_token_pos is the position after the last token of \boxed{} (including '}')
                            # So we mark from boxed_end_token_pos onwards as wrong
                            wrong_start_token = boxed_end_token_pos
                            if wrong_start_token < response_length:
                                token_correctness_mask[idx, wrong_start_token:] = 0.0
                            # Debug: check the token at boxed_end_token_pos-1 to verify it's the closing brace
                            if boxed_end_token_pos > 0:
                                prev_token_id = response_tensor[boxed_end_token_pos - 1].item()
                                prev_token_text = tokenizer.decode([prev_token_id], skip_special_tokens=False)
                                print(f"[mark_v7] idx={idx}, problem_id={problem_id}, k={k}, last_boxed_not_followed_by_eos, control_para=2, marking tokens from {wrong_start_token} as wrong, boxed_end_token_pos={boxed_end_token_pos}, prev_token_id={prev_token_id}, prev_token_text='{prev_token_text}', next_token_id={next_token_id}, eos_token_id={eos_token_id}, response_length={response_length}")
                            else:
                                print(f"[mark_v7] idx={idx}, problem_id={problem_id}, k={k}, last_boxed_not_followed_by_eos, control_para=2, marking tokens from {wrong_start_token} as wrong, boxed_end_token_pos={boxed_end_token_pos}, next_token_id={next_token_id}, eos_token_id={eos_token_id}, response_length={response_length}")
                    else:
                        # Last \boxed{} is followed by EOS (<|im_end|>)
                        # Find the first EOS token position to mark padding tokens as wrong
                        eos_positions = (response_tensor == eos_token_id).nonzero(as_tuple=True)[0]
                        if len(eos_positions) > 0:
                            actual_response_end_pos = eos_positions[0].item()
                            # Mark tokens after EOS (padding tokens) as wrong
                            # Tokens up to and including EOS remain correct (already 1.0)
                            padding_start_token = actual_response_end_pos + 1
                            if padding_start_token < response_length:
                                token_correctness_mask[idx, padding_start_token:] = 0.0
                        # If no EOS found (shouldn't happen if next_token_id == eos_token_id), all tokens stay correct
                else:
                    # boxed_end_token_pos >= response_length (shouldn't happen, but handle it)
                    # Mark all tokens as wrong
                    token_correctness_mask[idx, :] = 0.0
                    subproblem_correct_count[idx] = -1
                    k = -1
                    print(f"[mark_v7] idx={idx}, problem_id={problem_id}, k={k}, boxed_end_exceeds_response_length, marking all tokens as wrong, boxed_end_token_pos={boxed_end_token_pos}, response_length={response_length}")
            else:
                # No \boxed{} found in subproblem 4, mark as invalid
                subproblem_correct_count[idx] = -1
                # Mark all tokens as wrong (0) for invalid parsing
                token_correctness_mask[idx, :] = 0.0
                print(f"[mark_v7] idx={idx}, problem_id={problem_id}, k={k}, no_boxed_in_subproblem_4")
        elif 1 <= k <= 3:
            # First k subproblems are correct, (k+1)th is wrong
            # Mark tokens in subproblems 1 to k as correct (already 1.0)
            # Mark tokens in subproblem k+1 and beyond as wrong (0.0)
            if k < len(subproblem_token_boundaries) - 1:
                wrong_start_token = subproblem_token_boundaries[k]
                # Mark all tokens from wrong_start_token to end as wrong
                token_correctness_mask[idx, wrong_start_token:] = 0.0
        
        # Debug logging
        # problem_id is already defined at the start of the loop
        correct_tokens = token_correctness_mask[idx].sum().item()
        total_tokens = token_correctness_mask[idx].numel()
        
        # Debug: Find the last correct token and decode it along with previous 10 tokens
        correctness_mask_bool = token_correctness_mask[idx].bool()
        correct_indices = torch.where(correctness_mask_bool)[0]
        last_correct_token_info = "N/A"
        if len(correct_indices) > 0:
            last_correct_idx = correct_indices[-1].item()
            # Ensure index is within bounds
            if last_correct_idx < len(response_tensor):
                # Decode the last correct token
                last_token_id = response_tensor[last_correct_idx].item()
                last_token_text = tokenizer.decode([last_token_id], skip_special_tokens=False)
                
                # Check if it's EOS token
                eos_token_id = tokenizer.eos_token_id
                is_eos = (eos_token_id is not None and last_token_id == eos_token_id)
                eos_marker = " [EOS]" if is_eos else ""
                
                # Decode previous 10 tokens (if available)
                start_idx = max(0, last_correct_idx - 9)  # Include last_correct_idx, so 9 previous + 1 current = 10 tokens
                prev_token_ids = response_tensor[start_idx:last_correct_idx + 1].tolist()
                prev_tokens_text = tokenizer.decode(prev_token_ids, skip_special_tokens=False)
                
                last_correct_token_info = f"last_correct_token_idx={last_correct_idx}, last_token_id={last_token_id}{eos_marker}, last_token='{last_token_text}', prev_10_tokens='{prev_tokens_text}'"
            else:
                last_correct_token_info = f"last_correct_token_idx={last_correct_idx} (out_of_bounds, response_len={len(response_tensor)})"
        else:
            last_correct_token_info = "no_correct_tokens"
        
        print(f"[mark_v7] idx={idx}, problem_id={problem_id}, k={k}, correct_tokens={correct_tokens}/{total_tokens}, {last_correct_token_info}")
    
    # Add token_correctness_mask to batch
    gen_batch_output.batch["token_correctness_mask"] = token_correctness_mask
    
    # Add subproblem_correct_count to non_tensor_batch
    gen_batch_output.non_tensor_batch["subproblem_correct_count"] = subproblem_correct_count

    # Add v8 helper fields. These are no-ops for old branches unless consumed by v8 logic.
    gen_batch_output.non_tensor_batch["subproblem_part_correctness_local"] = subproblem_part_correctness_local
    gen_batch_output.batch["subproblem_part_token_mask"] = subproblem_part_token_mask

    return gen_batch_output
