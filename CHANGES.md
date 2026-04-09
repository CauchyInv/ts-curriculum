# CHANGES (main -> dev1)

## Scope
This document summarizes code differences between local branches `main` (baseline verl) and `dev1` (curriculum/RL experiments).

- Comparison basis: `git diff main...dev1`
- Divergence commits on `dev1`: 1 (`57374d64 add curriculum learning code`)
- Files changed: 20
- Diff size: ~6997 insertions, 22 deletions

## High-level Summary
Changes are concentrated in math RL curriculum logic, reward scoring, and rollout instrumentation:

1. New teacher-student curriculum pipeline in PPO training (`ray_trainer.py` + new `own_utils.py`).
2. New `deepscaler` package for prompts, reward utilities, and optional LLM-based reward fallback.
3. New refined math scorer (`prime_math_refine`) and global reward routing changes.
4. Dataset support for subproblem JSONL injection and mixed prompt variants.
5. Added local subproblem data files for curriculum construction.

## Added Files
### Curriculum and training utilities
- `verl/trainer/ppo/own_utils.py`
  - Added rollout decoding/saving, student answer history, teacher hint generation, local vLLM teacher support, and subproblem-based response/token processing functions.

### Deepscaler package
- `verl/deepscaler/globals.py`
- `verl/deepscaler/system_prompts.py`
- `verl/deepscaler/utils.py`
- `verl/deepscaler/rewards/*`
  - Adds reward abstraction/types and math reward function with boxed-answer extraction + sympy/mathd equivalence checks.
  - Includes optional ORM path using Gemini/OpenAI reward model calls.

### Refined math scorer
- `verl/utils/reward_score/prime_math_refine/{__init__.py,grader.py,math_normalize.py,README.md}`
  - Adds stricter boxed-answer-first extraction and refined symbolic/format normalization pipeline.

### Data artifacts
- `data/subproblems.jsonl`
- `data/omni/subproblems.jsonl`

## Modified Files
### `verl/trainer/ppo/ray_trainer.py`
Major curriculum-learning integration:
- Adds `teacher_student` mode state (teacher model, answer history, hint dictionaries, subproblem tracking).
- Adds mixed prompt construction for multiple `ts_version` variants (`v1`-`v7`) and v4/v5/v7 subproblem handling paths.
- Adds rollout sample persistence hooks and teacher hint generation hooks.
- Adds reward remapping logic for subproblem correctness (notably v5/v7).
- Adds token-level advantage post-processing for `ts_version == v7`.
- Adds rich metrics for in-state-0 solve rate and subproblem solve distributions.

### `verl/utils/dataset/rl_dataset.py`
- Replaces hardcoded `subproblems.jsonl` path with configurable `subproblems_jsonl_path`.
- Loads subproblem dictionary from JSONL when configured.

### `verl/utils/reward_score/__init__.py`
- Behavioral change: for any truthy `data_source`, scoring now routes to `prime_math_refine.compute_score(...)`.
- Existing dataset-specific branches become effectively bypassed for non-empty `data_source` values.

### `verl/workers/reward_manager/naive.py`
- Adds debug print of `ground_truth` during reward computation.

## Behavioral Impact (pass@k-relevant)
Most likely to affect math `pass@k`:

1. Subproblem-driven rollout mixing and reward shaping (`ray_trainer.py`, `own_utils.py`).
2. Scoring strictness/extraction behavior change via `prime_math_refine` routing.
3. Token-level reward/advantage manipulation in `ts_version == v7`.

## Risks / Attention Points
1. **Reward router precedence risk**: `if data_source:` in `reward_score/__init__.py` may unintentionally override all dataset-specific scorers.
2. **Training overhead risk**: heavy logging/sample dumping in teacher-student loop can increase I/O cost.
3. **External dependency risk**: optional ORM path requires accessible LLM endpoints/credentials.

## Suggested Next Validation
1. Run A/B eval (`main` vs `dev1`) on the same math benchmark split and sampling config.
2. Verify scorer routing is intentional for every target dataset.
3. Profile rollout+reward step time with/without sample dumping and teacher hint generation.
