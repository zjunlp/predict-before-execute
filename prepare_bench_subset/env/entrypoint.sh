#!/usr/bin/env bash
set -euo pipefail

# Initialize conda and activate 'agent'
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi
conda activate agent

# Hugging Face mirrors and caches (allow override by -e)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models}"

# Ensure both common env var names are set so huggingface_hub/transformers pick up the mirror
export HUGGINGFACE_HUB_ENDPOINT="${HUGGINGFACE_HUB_ENDPOINT:-$HF_ENDPOINT}"
export HF_HOME="${HF_HOME:-/app/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

# Increase timeouts/retries for downloads (reduce failures due to transient network/handshake slowness)
export HF_HUB_REQUEST_TIMEOUT="${HF_HUB_REQUEST_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_RETRIES="${HF_HUB_DOWNLOAD_RETRIES:-5}"

# Disable hf_transfer by default to avoid hf_transfer parallel-permits errors; can be overridden at runtime
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

# Ensure base dirs exist in container
mkdir -p /app/input /app/submission /app/workspaces "$HF_HOME" "$TRANSFORMERS_CACHE"

# Make both 'app' (package) and 'util' (top-level import) resolvable
# - '/': lets 'app' be found as a top-level package at /app when running -m app.run
# - '/app': lets absolute imports like 'from util.fs import ...' resolve to /app/util
export PYTHONPATH="/:/app:${PYTHONPATH:-}"

# Build argument list
if [ -n "${RUN_SCRIPT:-}" ]; then
  ARGS=("$RUN_SCRIPT")
else
  ARGS=("$@")
fi

# Prefer package execution inside the container when possible
if python - <<'PY' 2>/dev/null
import importlib, sys
try:
    importlib.import_module("app.run")
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
then
  exec python -u -m app.run "${ARGS[@]}"
else
  exec python -u /app/run.py "${ARGS[@]}"
fi

# If using RUN_SCRIPT: tee all output to exec_output.txt while preserving exit code
if [ -n "${RUN_SCRIPT:-}" ]; then
  mkdir -p /app/submission
  set +e
  {
    python -u - << 'PYBLOCK'
import os, sys, runpy
from pathlib import Path

# Utilities from our package
from util.fs import prepare_isolated_workspace, stage_input_from_data, export_submission_to_solution_dir
from util.task import infer_task_from_solution

script = Path(os.environ["RUN_SCRIPT"]).resolve()
data_dir = os.environ.get("DATA_DIR")
task_name = os.environ.get("TASK_NAME")

# Prepare workspace and input
ws = prepare_isolated_workspace(script)
if data_dir:
    t = infer_task_from_solution(script, task_name)
    if t:
        try:
            stage_input_from_data(data_dir, t)
        except Exception as e:
            print(f"[stage_input] {e}", file=sys.stderr)

# Run the solution
os.chdir(ws)
exit_code = 0
try:
    runpy.run_path(script.name, run_name="__main__")
except SystemExit as e:
    exit_code = int(e.code) if isinstance(e.code, int) else 1
except Exception as e:
    print(f"[run_error] {e}", file=sys.stderr)
    exit_code = 1

# Export submission back to host solution dir
try:
    export_submission_to_solution_dir(script)
except Exception as e:
    print(f"[export] {e}", file=sys.stderr)

sys.exit(exit_code)
PYBLOCK
  } 2>&1 | tee -a /app/submission/exec_output.txt
  ec=${PIPESTATUS[0]}
  exit $ec
else
  exec python -u /app/run.py "$@"
fi
