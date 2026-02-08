# Main CLI entrypoint for prepare_bench_subset.
# - Supports both batch mode (Docker-based multi-solution runner) and single-run mode.
# - Parses CLI/ENV configuration for tasks, data, GPUs, CPU scheduling, and tracing.
# - Handles GPU health detection and optional failover across devices.
# - Exposes a small API (run_batch, run_single) that can also be imported programmatically.

import sys
import os
import argparse
from pathlib import Path
from typing import List
import subprocess
import json
import re

# Ensure local packages are importable (util/*, env/*)
repo_root = Path(__file__).parent.resolve()
parent_root = repo_root.parent
for p in (str(repo_root), str(parent_root)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Support both "python -m prepare_bench_subset.run" and "python /path/run.py"
try:
    from .env.batch import run_batch  # noqa: E402
    from .env.single_run import run_single  # noqa: E402
except ImportError:
    from env.batch import run_batch  # noqa: E402
    from env.single_run import run_single  # noqa: E402

def parse_args(argv: List[str]):
    parser = argparse.ArgumentParser(add_help=False)
    # batch mode args
    parser.add_argument("--batch", action="store_true", help="Run batch mode: build image and dispatch solutions across GPUs")
    parser.add_argument("--summary", action="store_true", help="Print execution summary (total, completed, remaining) and exit")
    # summary-buggy: optional path. If provided without value -> True (print to stdout).
    # If provided with a path -> write report to that file and also print to stdout.
    parser.add_argument("--summary-buggy", nargs="?", const=True, default=False, metavar="OUT",
                        help="Report buggy solutions (have exec_output.txt but missing submission.csv). Optional OUT path to append the report.")
    
    parser.add_argument("--solutions-root", help="Root dir containing per-task solution folders (e.g., solutions_subset)")
    parser.add_argument("--task-file", help="Path to task_name.txt (defaults to repository task list)")
    parser.add_argument("--dockerfile", help="Path to Dockerfile (default: env/Dockerfile under this repo)")
    parser.add_argument("--build-context", help="Docker build context (default: this repo root)")
    parser.add_argument("--max-parallel", "-p", type=int, help="Max number of containers to run concurrently")
    # Global toggle in run.py: whether to enable "strict CPU scheduling" (SCHED_STRICT_CPU).
    parser.add_argument("--sched-strict-cpu", action="store_true",
                        help="Enable conservative CPU scheduling (cpu affinity, caps, etc.). "
                             "Default off = legacy preemptive behavior")
    # New: kw.json path; if provided, filter solutions using rank_1 keyword quotas.
    parser.add_argument("--kw-json", help="Path to keywords_all_tasks.json; if set, filter solutions by rank_1 keyword quotas")

    # single-run args (inside container)
    parser.add_argument("-s", "--solution", help="Path to solution.py to run")
    parser.add_argument("-i", "--input-src", help="Path to data to place into /app/input")
    parser.add_argument("-o", "--submission-dst", "--output-dst", dest="submission_dst",
                        help="Destination path to copy /app/submission after run")
    parser.add_argument("-d", "--data-dir", help="Root directory containing per-task prepared/public data")
    parser.add_argument("-t", "--task", help="Task name (task_n). If omitted, inferred from solution.py path using task_name.txt")
    # tracing toggle (mirrors TRACE_VARS env used in env/single_run.py)
    parser.add_argument("--trace-vars", action="store_true",
                        help="Enable per-line locals tracing via stdlib 'trace'; logs to /app/submission/trace.log")
    # cleanup toggles
    parser.add_argument("--clean-links", action="store_true", help="After run, remove local symlinks created for ./input and ./submission when possible")
    parser.add_argument("--clean-working", action="store_true", help="After run, move working/submission.csv into /app/submission and remove ./working")
    parser.add_argument("--clean-workspace", action="store_true", help="After run, remove the isolated /app/workspaces/<solution> directory")
    # GPU health/failover controls
    parser.add_argument("--gpus", help="Comma-separated GPU indices to consider; if omitted, auto-detect healthy GPUs")
    parser.add_argument("--gpu-failover", action="store_true", default=False,
                        help="Enable failover to next healthy GPU when the current GPU is unhealthy or errors occur")
    parser.add_argument("--gpu-try-limit", type=int, default=0,
                        help="Limit of GPUs to try per task when --gpu-failover is enabled (0 = try all healthy)")
    # Capture unknown args to forward to the solution (single-run path)
    ns, unknown = parser.parse_known_args(argv)

    # Write the CLI strict-CPU switch into ENV so batch / single_run can read it consistently.
    if getattr(ns, "sched_strict_cpu", False):
        os.environ["SCHED_STRICT_CPU"] = "1"
    else:
        os.environ.setdefault("SCHED_STRICT_CPU", "0")

    # Write kw.json path into ENV for env/batch.py to consume.
    kw_json = getattr(ns, "kw_json", None)
    if kw_json:
        os.environ["KW_JSON_PATH"] = str(Path(kw_json).resolve())
    else:
        # Do not overwrite an existing value, so external export KW_JSON_PATH still works.
        os.environ.setdefault("KW_JSON_PATH", "")

    return ns, unknown

# ---------------- GPU health helpers ----------------

def _parse_gpu_list(spec: str) -> List[int]:
    if not spec:
        return []
    return [int(x) for x in re.split(r"[,\s]+", spec.strip()) if x != ""]

def _gpustat_json() -> dict:
    """
    Return gpustat --json payload or {} if unavailable.
    """
    try:
        out = subprocess.check_output(["gpustat", "--json"], stderr=subprocess.STDOUT, text=True, timeout=5)
        return json.loads(out)
    except Exception:
        return {}

def _nvidia_smi_list() -> List[int]:
    """
    Return indices from `nvidia-smi -L` or [] if unavailable.
    """
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.STDOUT, text=True, timeout=5)
        # Lines like: "GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-xxxx)"
        idxs = []
        for line in out.splitlines():
            m = re.match(r"\s*GPU\s+(\d+):", line)
            if m:
                idxs.append(int(m.group(1)))
        return idxs
    except Exception:
        return []

def _healthy_gpu_indices(preferred: List[int] | None = None) -> List[int]:
    """
    Determine healthy GPUs using gpustat; mark GPUs with 'Unknown Error' as unhealthy.
    Order by ascending memory.used for better load distribution.
    Fallback to nvidia-smi list or CUDA_VISIBLE_DEVICES if gpustat is not available.
    """
    preferred = preferred or []
    payload = _gpustat_json()
    indices: List[int] = []

    if payload and "gpus" in payload and isinstance(payload["gpus"], list):
        gpus = payload["gpus"]
        # Build candidate list
        for g in gpus:
            idx = g.get("index")
            name = str(g.get("name", ""))
            # consider unhealthy if gpustat marks unknown error in name
            if "Unknown Error" in name:
                continue
            indices.append(idx)
        # Filter by preferred if provided
        if preferred:
            indices = [i for i in indices if i in preferred]
        # Order by memory.used (ascending)
        usage = {g.get("index"): int(g.get("memory", {}).get("used", 0)) for g in gpus}
        indices.sort(key=lambda i: usage.get(i, 0))
        return indices

    # Fallbacks
    if preferred:
        return preferred
    env_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_cvd:
        env_idxs = []
        for tok in env_cvd.split(","):
            tok = tok.strip()
            if tok.isdigit():
                env_idxs.append(int(tok))
        if env_idxs:
            return env_idxs
    # nvidia-smi -L
    return _nvidia_smi_list()

def _export_healthy_env(healthy: List[int]) -> None:
    """
    Export CUDA_VISIBLE_DEVICES and HEALTHY_GPUS for downstream tooling.
    """
    devices = ",".join(str(i) for i in healthy)
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    os.environ["HEALTHY_GPUS"] = devices

def _select_candidates(ns) -> List[int]:
    preferred = _parse_gpu_list(getattr(ns, "gpus", None))
    healthy = _healthy_gpu_indices(preferred=preferred)
    return healthy

def main():
    ns, unknown = parse_args(sys.argv[1:])
    # When batch mode also needs tracing, set TRACE_VARS here so batch can read and forward it.
    if getattr(ns, "trace_vars", False):
        os.environ["TRACE_VARS"] = "1"

    # In summary / summary-buggy mode: only compute statistics; no GPU processing and no single-run.
    sb = getattr(ns, "summary_buggy", False)
    if getattr(ns, "summary", False) or bool(sb):
        run_batch(ns)
        return

    # Compute healthy candidates once
    healthy_candidates = _select_candidates(ns)

    if ns.batch:
        # Batch mode: export only healthy GPUs for downstream scheduler
        if healthy_candidates:
            _export_healthy_env(healthy_candidates)
            print(f"[GPU] Batch mode: healthy GPUs -> {healthy_candidates} (exported to CUDA_VISIBLE_DEVICES/HEALTHY_GPUS)")
        else:
            print("[GPU] Batch mode: no healthy GPUs detected; leaving environment unchanged")
        run_batch(ns)
        return

    # Single run with optional GPU failover
    if not healthy_candidates:
        print("[GPU] Single-run: no healthy GPUs detected; running without GPU filtering")
        run_single(ns, unknown)
        return

    if not getattr(ns, "gpu_failover", False):
        # Use the first healthy GPU
        first = healthy_candidates[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(first)
        print(f"[GPU] Single-run: using GPU {first} (no failover)")
        run_single(ns, unknown)
        return

    # Failover path: try GPUs in order of lowest memory usage
    try_limit = int(getattr(ns, "gpu_try_limit", 0) or 0)
    if try_limit > 0:
        candidates = healthy_candidates[:try_limit]
    else:
        candidates = healthy_candidates[:]

    last_exc: Exception | None = None
    for idx in candidates:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        print(f"[GPU] Single-run: attempting on GPU {idx}")
        try:
            run_single(ns, unknown)
            print(f"[GPU] Single-run: succeeded on GPU {idx}")
            return
        except Exception as e:
            last_exc = e
            msg = str(e)
            print(f"[GPU] Single-run: GPU {idx} failed with error: {msg}")
            # Try next candidate regardless; many GPU driver errors are opaque
            continue

    # Exhausted candidates
    if last_exc:
        raise last_exc
    else:
        raise RuntimeError("Single-run failed: no GPU candidates succeeded")

if __name__ == "__main__":
    main()