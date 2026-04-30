#!/usr/bin/env python3
import argparse
import ast
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from vllm import LLM, SamplingParams
except Exception:
    from verl.third_party.vllm import LLM  # type: ignore
    from vllm import SamplingParams  # type: ignore

from transformers import AutoTokenizer

from verl.utils.reward_score import default_compute_score


def _try_parse_obj(x: Any) -> Any:
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return x
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                pass
            try:
                return ast.literal_eval(s)
            except Exception:
                pass
    return x


def load_parquet_rows(path: str) -> List[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        return table.to_pylist()
    except Exception:
        pass

    try:
        import pandas as pd

        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    except Exception as e:
        raise RuntimeError(
            "Cannot read parquet. Install either pyarrow or pandas in your env (e.g. curri)."
        ) from e


def normalize_messages(prompt_field: Any) -> Optional[List[Dict[str, str]]]:
    prompt_field = _try_parse_obj(prompt_field)
    if not isinstance(prompt_field, list):
        return None
    messages: List[Dict[str, str]] = []
    for item in prompt_field:
        item = _try_parse_obj(item)
        if not isinstance(item, dict):
            return None
        role = item.get("role")
        content = item.get("content")
        if role is None or content is None:
            return None
        messages.append({"role": str(role), "content": str(content)})
    return messages


def render_prompt(prompt_field: Any, tokenizer: Any) -> str:
    messages = normalize_messages(prompt_field)
    if messages is not None:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
        text = []
        for m in messages:
            text.append(f"{m['role']}\n{m['content']}")
        text.append("assistant\n")
        return "\n".join(text)

    prompt_field = _try_parse_obj(prompt_field)
    if isinstance(prompt_field, str):
        return prompt_field
    return str(prompt_field)


def extract_ground_truth(row: Dict[str, Any]) -> Any:
    if "ground_truth" in row and row["ground_truth"] is not None:
        return row["ground_truth"]

    reward_model = _try_parse_obj(row.get("reward_model"))
    if isinstance(reward_model, dict):
        if "ground_truth" in reward_model:
            return reward_model["ground_truth"]
        if "answer" in reward_model:
            return reward_model["answer"]

    extra_info = _try_parse_obj(row.get("extra_info"))
    if isinstance(extra_info, dict):
        if "ground_truth" in extra_info:
            return extra_info["ground_truth"]
        if "answer" in extra_info:
            return extra_info["answer"]

    return row.get("answer", "")


def power2_ks(max_k: int) -> List[int]:
    base = [1, 2, 4, 8, 16, 32, 64]
    return [k for k in base if k <= max_k]


def safe_name(s: str) -> str:
    return s.replace("/", "__").replace(" ", "_")


def append_jsonl_record(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def mean(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def passk_combinatorial(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator:
    1 - C(n-c, k) / C(n, k)
    """
    if k <= 0 or n <= 0:
        return 0.0
    if c <= 0:
        return 0.0
    if k > n:
        k = n
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def passk_verl_bootstrap(binary_scores: Sequence[float], k: int, n_bootstrap: int = 1000, seed: int = 42) -> float:
    """Replicate verl metric_utils.bootstrap_metric + reduce_fn=np.max behavior.

    For a single question with n sampled responses, estimate best@k/mean via
    bootstrap with replacement.
    """
    if len(binary_scores) == 0 or k <= 0:
        return 0.0
    np.random.seed(seed)
    arr = np.asarray(binary_scores, dtype=np.float32)
    vals = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(arr), size=k, replace=True)
        vals.append(float(np.max(arr[idx])))
    return float(np.mean(vals))


def evaluate_one_ckpt(
    ckpt: str,
    grouped_rows: Dict[str, List[Dict[str, Any]]],
    n: int,
    temperature: float,
    top_k: int,
    top_p: float,
    max_tokens: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    dtype: str,
    batch_size: int,
    passk_mode: str,
    verl_bootstrap_samples: int,
    verl_bootstrap_seed: int,
    save_rollout: bool,
    rollout_dir: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    llm = LLM(
        model=ckpt,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
        trust_remote_code=True,
    )
    sp = SamplingParams(
        n=n,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    ks = power2_ks(n)
    ckpt_results: Dict[str, Dict[str, Any]] = {}

    for ds, rows in grouped_rows.items():
        empirical_prefix: Dict[int, List[float]] = {k: [] for k in ks}
        combin_prefix: Dict[int, List[float]] = {k: [] for k in ks}
        verl_prefix: Dict[int, List[float]] = {k: [] for k in ks}
        total = len(rows)
        rollout_fp = None
        if save_rollout and rollout_dir is not None:
            os.makedirs(rollout_dir, exist_ok=True)
            rollout_path = os.path.join(rollout_dir, f"{safe_name(ds)}.jsonl")
            rollout_fp = open(rollout_path, "a", encoding="utf-8")
        for st in range(0, total, batch_size):
            batch_rows = rows[st : st + batch_size]
            prompts = [render_prompt(r.get("prompt"), tokenizer) for r in batch_rows]
            outputs = llm.generate(prompts, sp, use_tqdm=False)

            for local_i, (r, out) in enumerate(zip(batch_rows, outputs)):
                gt = extract_ground_truth(r)
                ds_name = str(r.get("data_source", ds))
                sample_scores: List[float] = []
                for cand_idx, cand in enumerate(out.outputs):
                    resp = cand.text
                    score = default_compute_score(
                        data_source=ds_name,
                        solution_str=resp,
                        ground_truth=gt,
                        extra_info=r.get("extra_info"),
                    )
                    score_f = float(score if not isinstance(score, dict) else score.get("score", 0.0))
                    sample_scores.append(score_f)
                    if rollout_fp is not None:
                        decoded_with_specials = tokenizer.decode(cand.token_ids, skip_special_tokens=False)
                        record = {
                            "ckpt": ckpt,
                            "data_source": ds_name,
                            "batch_start": st,
                            "sample_index_in_batch": local_i,
                            "candidate_index": cand_idx,
                            "problem_id": r.get("problem_id"),
                            "uid": r.get("uid"),
                            "prompt": prompts[local_i],
                            "response": decoded_with_specials,
                            "response_raw": resp,
                            "finish_reason": getattr(cand, "finish_reason", None),
                            "ground_truth": gt,
                            "score": score_f,
                            "is_correct": 1 if score_f > 0.5 else 0,
                        }
                        rollout_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                if len(sample_scores) < n:
                    sample_scores.extend([0.0] * (n - len(sample_scores)))
                is_correct = [1.0 if x > 0.5 else 0.0 for x in sample_scores[:n]]
                c = int(sum(is_correct))
                for k in ks:
                    if passk_mode in ("empirical", "both", "both_all"):
                        empirical_prefix[k].append(1.0 if any(is_correct[:k]) else 0.0)
                    if passk_mode in ("combinatorial", "both", "both_all"):
                        combin_prefix[k].append(passk_combinatorial(n=n, c=c, k=k))
                    if passk_mode in ("verl_bootstrap", "both_all"):
                        verl_prefix[k].append(
                            passk_verl_bootstrap(
                                binary_scores=is_correct,
                                k=k,
                                n_bootstrap=verl_bootstrap_samples,
                                seed=verl_bootstrap_seed,
                            )
                        )

        one = {
            "ckpt": ckpt,
            "data_source": ds,
            "num_questions": total,
            "n": n,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        for k in ks:
            if passk_mode == "empirical":
                one[f"pass@{k}"] = mean(empirical_prefix[k])
            elif passk_mode == "combinatorial":
                one[f"pass@{k}"] = mean(combin_prefix[k])
            elif passk_mode == "verl_bootstrap":
                one[f"pass@{k}"] = mean(verl_prefix[k])
            else:
                if passk_mode == "both":
                    # both: keep explicit fields + set pass@k to combinatorial for primary reporting
                    one[f"pass_empirical@{k}"] = mean(empirical_prefix[k])
                    one[f"pass_combinatorial@{k}"] = mean(combin_prefix[k])
                    one[f"pass@{k}"] = one[f"pass_combinatorial@{k}"]
                else:
                    # both_all: include empirical/combinatorial/verl_bootstrap
                    one[f"pass_empirical@{k}"] = mean(empirical_prefix[k])
                    one[f"pass_combinatorial@{k}"] = mean(combin_prefix[k])
                    one[f"pass_verl_bootstrap@{k}"] = mean(verl_prefix[k])
                    one[f"pass@{k}"] = one[f"pass_verl_bootstrap@{k}"]
        ckpt_results[ds] = one
        if rollout_fp is not None:
            rollout_fp.close()

    del llm
    return ckpt_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HF checkpoints on parquet benchmark with pass@k via vLLM.")
    parser.add_argument("--ckpts", nargs="+", required=True, help="One or more HF checkpoint paths.")
    parser.add_argument("--parquet", required=True, help="Benchmark parquet path.")
    parser.add_argument(
        "--output_dir",
        default="/hyk/algorithm_new/qinghua/yueyang/verl/eval_models/results",
        help="Output directory.",
    )
    parser.add_argument("--n", type=int, default=64, help="Number of samples per prompt for pass@n.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Validation temperature.")
    parser.add_argument("--top_k", type=int, default=-1, help="Validation top_k (default matches rollout val default).")
    parser.add_argument("--top_p", type=float, default=1.0, help="Validation top_p (default matches rollout val default).")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Max response length.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="vLLM tensor parallel size.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="vLLM GPU memory utilization.")
    parser.add_argument("--max_model_len", type=int, default=16384, help="vLLM max model len.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16"], help="vLLM dtype.")
    parser.add_argument("--batch_size", type=int, default=32, help="Prompts per generate call.")
    parser.add_argument(
        "--passk_mode",
        default="combinatorial",
        choices=["empirical", "combinatorial", "verl_bootstrap", "both", "both_all"],
        help="pass@k metric mode: empirical hit@k, combinatorial estimator, or both.",
    )
    parser.add_argument(
        "--verl_bootstrap_samples",
        type=int,
        default=1000,
        help="Bootstrap iterations for passk_mode=verl_bootstrap (match verl default=1000).",
    )
    parser.add_argument(
        "--verl_bootstrap_seed",
        type=int,
        default=42,
        help="Bootstrap seed for passk_mode=verl_bootstrap (match verl default=42).",
    )
    parser.add_argument(
        "--save_rollout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save all rollout samples during evaluation.",
    )
    parser.add_argument(
        "--rollout_dir",
        default="/hyk/algorithm_new/qinghua/yueyang/verl/eval_models/eval_rollout_samples",
        help="Directory to save rollout samples.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_parquet_rows(args.parquet)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        ds = str(r.get("data_source", "unknown"))
        grouped[ds].append(r)

    print(f"[info] loaded rows={len(rows)}, data_sources={sorted(grouped.keys())}")
    print(
        f"[info] sampling n={args.n}, temperature={args.temperature}, top_k={args.top_k}, "
        f"top_p={args.top_p}, max_tokens={args.max_tokens}"
    )
    print(f"[info] passk_mode={args.passk_mode}")
    if args.passk_mode in ("verl_bootstrap", "both_all"):
        print(f"[info] verl_bootstrap_samples={args.verl_bootstrap_samples}, seed={args.verl_bootstrap_seed}")
    print(f"[info] save_rollout={args.save_rollout}, rollout_dir={args.rollout_dir}")

    run_rollout_root = None
    if args.save_rollout:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        parquet_stem = os.path.splitext(os.path.basename(args.parquet))[0]
        run_rollout_root = os.path.join(args.rollout_dir, f"{parquet_stem}_{run_tag}")
        os.makedirs(run_rollout_root, exist_ok=True)

    parquet_stem = os.path.splitext(os.path.basename(args.parquet))[0]
    for i, ckpt in enumerate(args.ckpts, start=1):
        print(f"[info] ({i}/{len(args.ckpts)}) evaluating ckpt: {ckpt}")
        ckpt_rollout_dir = None
        if run_rollout_root is not None:
            ckpt_rollout_dir = os.path.join(run_rollout_root, safe_name(os.path.basename(ckpt.rstrip("/"))))
        ckpt_res = evaluate_one_ckpt(
            ckpt=ckpt,
            grouped_rows=grouped,
            n=args.n,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            batch_size=args.batch_size,
            passk_mode=args.passk_mode,
            verl_bootstrap_samples=args.verl_bootstrap_samples,
            verl_bootstrap_seed=args.verl_bootstrap_seed,
            save_rollout=args.save_rollout,
            rollout_dir=ckpt_rollout_dir,
        )
        for ds, rec in ckpt_res.items():
            out_path = os.path.join(args.output_dir, f"{parquet_stem}__{safe_name(ds)}.jsonl")
            append_jsonl_record(out_path, rec)
            print(f"[done] appended 1 record -> {out_path}")

    if run_rollout_root is not None:
        print(f"[done] rollout samples saved under: {run_rollout_root}")

    print("[done] evaluation finished.")


if __name__ == "__main__":
    main()
