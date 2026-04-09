# Repository Guidelines

## Project Structure & Module Organization
This repository is a lightweight environment bootstrapper for `verl` workflows.

- Root scripts:
  - `install_fsdp_only.sh`: install dependencies for FSDP-only training.
  - `install_fsdp_vllm.sh`: install dependencies for FSDP training + vLLM inference.
- Docs:
  - `FSDP_vLLM_安装指南.md`: setup and troubleshooting reference.
- Model cache:
  - `models/`: local Hugging Face-style cache (`refs/`, `blobs/`). Treat as generated/runtime data, not hand-edited source.

## Build, Test, and Development Commands
Use these from repository root:

- `bash install_fsdp_only.sh`
  Installs base training dependencies and editable `verl` package.
- `bash install_fsdp_vllm.sh`
  Installs FSDP + vLLM environment.
- `python -c "import verl; print('ok')"`
  Verifies `verl` import after install.
- `python -c "import vllm; print('ok')"`
  Verifies vLLM import (when using vLLM flow).

## Coding Style & Naming Conventions
- Shell scripts should use `#!/bin/bash` and 4-space indentation in wrapped commands.
- Keep script names descriptive and task-oriented: `install_<stack>.sh`.
- Prefer uppercase env vars (`USE_MEGATRON`, `USE_SGLANG`, `MAX_JOBS`).
- Keep user-facing output explicit (`echo` steps and validation hints).

## Testing Guidelines
There is no automated test suite in this repository yet.

- For script changes, validate on a clean conda environment.
- Minimum verification:
  - Script exits non-zero on missing conda activation.
  - Key imports succeed (`verl`, optionally `vllm`).
- If adding tests later, place them under `tests/` and name files `test_<feature>.sh` or `test_<feature>.py`.

## Commit & Pull Request Guidelines
Git history is not available in this directory, so use these defaults:

- Commit messages: imperative, concise, scoped.
  - Example: `install: add fsdp-only dependency pin for numpy`
- PRs should include:
  - What changed and why.
  - Exact validation commands run.
  - Environment details (Python/CUDA versions).
  - Relevant logs or screenshots for install failures.

## Security & Configuration Tips
- Do not commit secrets, tokens, or private registry credentials.
- Avoid committing large model artifacts under `models/` unless explicitly required.
- Document any hard-coded absolute paths when introducing new scripts.
