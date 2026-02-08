# Batch orchestration utilities: build the Docker image, distribute solutions across GPUs/CPUs,
# enforce optional strict CPU scheduling, and summarize successes/failures for all tasks.

import os
import sys
import subprocess
import threading
import queue
import datetime
import time
from pathlib import Path
from typing import Optional, List, Tuple
import random
import json
import io
import contextlib

def _ts() -> str:
    # e.g., 2025-10-25 13:45:12.345
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def _run(cmd: List[str], check: bool = True, capture: bool = False, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=True, capture_output=capture, env=env)

def get_gpu_ids() -> List[Optional[int]]:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        ids = [s.strip() for s in cvd.split(",") if s.strip() != ""]
        try:
            return [int(i) for i in ids]
        except ValueError:
            pass
    try:
        out = _run(["nvidia-smi", "-L"], check=True, capture=True).stdout.strip().splitlines()
        n = len(out)
        return list(range(n)) if n > 0 else [None]
    except Exception:
        return [None]

def build_image(image_tag: str, dockerfile: Path, context_dir: Path):
    print(f"[build] Building image '{image_tag}' from {dockerfile} with context {context_dir}")
    _run(["docker", "build", "-f", str(dockerfile), "-t", image_tag, str(context_dir)])

def remove_image(image_tag: str):
    try:
        print(f"[cleanup] Removing image '{image_tag}'")
        _run(["docker", "rmi", "-f", image_tag], check=False)
    except Exception:
        pass

def read_tasks(task_file: Path) -> List[str]:
    if not task_file.exists():
        return []
    with open(task_file, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

def list_solutions_for_task(solutions_root: Path, task: str) -> List[Path]:
    # Previously used solutions_root / task; now each task has a 'code' subdirectory.
    task_dir = solutions_root / task / "code"
    if not task_dir.exists():
        return []
    sols: List[Path] = []
    for p in task_dir.rglob("*.py"):
        if p.name == "__init__.py":
            continue
        # Unified layout: code-side submission folder:
        #   <solutions_root>/<task>/code/submission_<stem>/submission.csv
        dest = p.parent / f"submission_{p.stem}"
        # Resume check: only skip if CSV exists inside the submission folder
        if (dest / "submission.csv").exists():
            continue
        sols.append(p.resolve())
    return sorted(sols)

# --- kw.json + annotations helpers ---

def _load_kw_rank1_limits() -> Optional[dict]:
    """
    Read kw.json from the KW_JSON_PATH environment variable and return
    rank_1 keyword quotas as {keyword: quota}. If not configured or any
    error occurs, return None (meaning: do not apply keyword filtering).
    """
    kw_path = os.environ.get("KW_JSON_PATH", "").strip()
    if not kw_path:
        return None
    p = Path(kw_path)
    if not p.exists():
        print(f"[kw] KW_JSON_PATH={kw_path} not found; skip keyword filtering", file=sys.stderr)
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rank1 = data.get("rank_1", {})
        if not isinstance(rank1, dict):
            print(f"[kw] rank_1 missing or not dict in {kw_path}; skip keyword filtering", file=sys.stderr)
            return None
        # Make a mutable copy of remaining quotas.
        limits: dict[str, int] = {}
        for k, v in rank1.items():
            try:
                limits[str(k)] = int(v)
            except Exception:
                # Ignore entries that cannot be converted to int.
                pass
        print(f"[kw] loaded rank_1 limits from {kw_path}: {len(limits)} keywords", file=sys.stderr)
        return limits
    except Exception as e:
        print(f"[kw] failed to load {kw_path}: {e}; skip keyword filtering", file=sys.stderr)
        return None

def _load_annotations_for_task(solutions_root: Path, task: str) -> Optional[dict]:
    """
    Load annotations_semantic.json for the given task if it exists.
    """
    ann_path = solutions_root / task / "annotation" / "annotations_semantic.json"
    if not ann_path.exists():
        return None
    try:
        with ann_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[kw] failed to load annotations for task={task}: {e}", file=sys.stderr)
        return None

def _infer_rank1_kw(annotations: dict, solution_path: Path) -> Optional[str]:
    """
    Infer the rank-1 keyword for a given solution from annotations_semantic.json.

    Key formats supported:
      - Exact key: the full solution ID.
      - Simple key: 'solution_<stem>'.
      - Fallback: any key that contains the stem.

    Values look like: [ [kw1, kw2, ...], ... ]; we take value[0][0] as rank-1 keyword.
    """
    if not annotations:
        return None
    stem = solution_path.stem

    candidate_keys: List[str] = []
    simple_key = f"solution_{stem}"

    # 1) Exact hit: solution_<stem>
    if simple_key in annotations:
        candidate_keys.append(simple_key)

    # 2) Keys starting with solution_<stem> (usually with UUID suffix)
    for k in annotations.keys():
        if k.startswith(simple_key):
            candidate_keys.append(k)

    # 3) Fallback: any key containing stem (e.g., filename or path-like keys)
    if not candidate_keys:
        for k in annotations.keys():
            if stem in k:
                candidate_keys.append(k)

    # Deduplicate while preserving order.
    seen = set()
    candidate_keys = [k for k in candidate_keys if not (k in seen or seen.add(k))]
    if not candidate_keys:
        return None

    v = annotations.get(candidate_keys[0])
    if not isinstance(v, list) or not v:
        return None
    first_list = v[0]
    if not isinstance(first_list, list) or not first_list:
        return None
    kw = first_list[0]
    return str(kw) if isinstance(kw, str) else None

def _read_tail(path: Path, max_lines: int = 20) -> List[str]:
    """
    Read last max_lines lines from a text file; return [] if not readable.
    """
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return lines[-max_lines:]
    except Exception:
        return []

def print_batch_summary(solutions_root: Path, task_file: Path) -> None:
    """
    Print only counts for each task: total solutions / finished / remaining.

    A solution is considered finished if:
      <task>/code/submission_<stem>/submission.csv exists.
    """
    tasks = read_tasks(task_file)
    total_all = 0
    done_all = 0

    header = f"{'Task':<50} | {'Total':<6} | {'Done':<6} | {'Left':<6}"
    print(header)
    print("-" * len(header))

    for t in tasks:
        task_dir = solutions_root / t / "code"
        if not task_dir.exists():
            continue

        task_total = 0
        task_done = 0

        for p in task_dir.rglob("*.py"):
            if p.name == "__init__.py":
                continue
            task_total += 1
            # Keep consistent with list_solutions_for_task:
            # check code/submission_<stem>/submission.csv
            dest = p.parent / f"submission_{p.stem}"
            if (dest / "submission.csv").exists():
                task_done += 1

        if task_total == 0:
            continue

        remaining = task_total - task_done
        total_all += task_total
        done_all += task_done
        print(f"{t:<50} | {task_total:<6} | {task_done:<6} | {remaining:<6}")

    print("-" * len(header))
    print(f"{'TOTAL':<50} | {total_all:<6} | {done_all:<6} | {total_all - done_all:<6}")

def print_buggy_summary(solutions_root: Path, task_file: Path, tail_lines: int = 20) -> None:
    """
    Print only “buggy solutions”:
      - code/submission_<stem>/exec_output.txt exists, AND
      - code/submission_<stem>/submission.csv does NOT exist.
    """
    tasks = read_tasks(task_file)
    total_buggy = 0

    print("[buggy-summary] Start scanning buggy solutions")
    for t in tasks:
        task_dir = solutions_root / t / "code"
        if not task_dir.exists():
            continue

        buggy_items: List[Tuple[Path, Path]] = []  # (solution.py, exec_output.txt)
        for p in task_dir.rglob("*.py"):
            if p.name == "__init__.py":
                continue
            sub_dir = p.parent / f"submission_{p.stem}"
            exec_log = sub_dir / "exec_output.txt"
            sub_csv = sub_dir / "submission.csv"
            if exec_log.exists() and not sub_csv.exists():
                buggy_items.append((p, exec_log))

        if not buggy_items:
            continue

        print(f"\n[buggy-summary] Task: {t}  buggy_solutions={len(buggy_items)}")
        for sol, log_path in sorted(buggy_items, key=lambda x: str(x[0])):
            total_buggy += 1
            print(f"  - Solution: {sol}")
            print(f"    Log: {log_path}")
            tail = _read_tail(log_path, max_lines=tail_lines)
            if tail:
                print("    --- exec_output tail ---")
                for ln in tail:
                    sys.stdout.write("    " + ln.rstrip("\n") + "\n")
                print("    --- end tail ---")
            else:
                print("    (no readable exec_output.txt or empty)")

    print(f"\n[buggy-summary] Total buggy solutions: {total_buggy}")

# --- Strict CPU scheduling toggle ---
def _strict_cpu_enabled() -> bool:
    return str(os.environ.get("SCHED_STRICT_CPU", "0")).lower() in ("1", "true", "yes")

# Helpers used only when strict CPU mode is enabled; they stay here but are gated by the switch.
def _detect_cpu_cores() -> int:
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1

def _compute_cpu_slots(user_max: Optional[int] = None) -> int:
    cores = _detect_cpu_cores()
    base_slots = max(1, cores // 2)
    env_raw = os.environ.get("CPU_MAX_CONCURRENCY")
    env_slots: Optional[int] = None
    if env_raw:
        try:
            env_slots = max(1, int(env_raw))
        except Exception:
            env_slots = None
    cap = base_slots
    if env_slots is not None:
        cap = min(cap, env_slots)
    if user_max is not None:
        cap = min(cap, max(1, user_max))
    return max(1, cap)

def _split_cpus_for_workers(total_workers: int) -> List[List[int]]:
    n_cores = _detect_cpu_cores()
    all_ids = list(range(n_cores))
    if total_workers <= 0:
        return []
    if total_workers == 1:
        return [all_ids]
    groups: List[List[int]] = [[] for _ in range(total_workers)]
    for idx, cid in enumerate(all_ids):
        groups[idx % total_workers].append(cid)
    return [g for g in groups if g]

def run_container_for_solution(image_tag: str, script_path: Path, task: str, data_dir: Path,
                               gpu_id: Optional[int],
                               cpu_affinity: Optional[List[int]] = None,
                               per_container_cpu_max: Optional[int] = None):
    # Check TRACE_VARS from the host process to decide whether tracing is enabled.
    trace_env_raw = str(os.environ.get("TRACE_VARS", ""))
    trace_enabled = trace_env_raw.lower() in ("1", "true", "yes")
    envs = [
        "-e", f"RUN_SCRIPT={str(script_path)}",
        "-e", f"DATA_DIR={str(data_dir)}",
        "-e", f"TASK_NAME={task}",
        "-e", "PYTHONUNBUFFERED=1",
        "-e", f"HOST_UID={os.getuid()}",
        "-e", f"HOST_GID={os.getgid()}",
    ]

    # Only in strict CPU mode do we pass CPU-related information into the container.
    if _strict_cpu_enabled():
        if cpu_affinity:
            envs += ["-e", f"CPU_AFFINITY={','.join(str(c) for c in cpu_affinity)}"]
        if per_container_cpu_max is not None:
            envs += ["-e", f"CPU_MAX_CONCURRENCY={per_container_cpu_max}"]

    if trace_enabled:
        envs += ["-e", "TRACE_VARS=1"]

    gpus = []
    if gpu_id is not None:
        gpus = ["--gpus", f"device={gpu_id}"]

    # Dynamically mount required host paths (no hard-coded prefixes)
    mounts: List[str] = []
    host_paths = set()
    try:
        repo_root = Path(__file__).resolve().parents[1]
        host_paths.add(str(repo_root))
    except Exception:
        pass
    try:
        host_paths.add(str(script_path.resolve().parent))
    except Exception:
        pass
    try:
        host_paths.add(str(Path(data_dir).resolve()))
    except Exception:
        pass
    for hp in sorted(host_paths):
        mounts += ["-v", f"{hp}:{hp}"]

    cmd = ["docker", "run", "--rm", "--ipc", "host", *gpus, *envs, *mounts, image_tag]

    # Only in strict CPU mode do we wrap the entire container with taskset.
    if _strict_cpu_enabled() and cpu_affinity:
        cpu_list = ",".join(str(c) for c in cpu_affinity)
        cmd = ["taskset", "-c", cpu_list, *cmd]

    print(f"[run] GPU={gpu_id} task={task} solution={script_path.name}")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(f"{_ts()} [{script_path.stem}] {line}")
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"Container failed for {script_path} with code {returncode}")

def batch_orchestrate(solutions_root: Path, data_dir: Path, task_file: Path,
                      dockerfile: Path, context_dir: Path, image_tag: str = "mlagent",
                      max_parallel: Optional[int] = None,
                      summary_only: bool = False,
                      buggy_only: bool = False,
                      buggy_out_path: Optional[Path] = None):
    # Pure summary mode: do not build image, do not start containers.
    if summary_only and not buggy_only:
        print_batch_summary(solutions_root, task_file)
        return
    if buggy_only:
        # If an output path is given, mirror print_buggy_summary output into that file as well.
        if buggy_out_path is not None:
            buggy_out_path.parent.mkdir(parents=True, exist_ok=True)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                print_buggy_summary(solutions_root, task_file)
            content = buf.getvalue()
            # Print to current stdout.
            sys.stdout.write(content)
            sys.stdout.flush()
            # Truncate the target file before writing this run’s content.
            try:
                with buggy_out_path.open("w", encoding="utf-8"):
                    pass
            except Exception:
                # Even if truncation fails, still try to append.
                pass
            # Append current run’s content.
            with buggy_out_path.open("a", encoding="utf-8") as f:
                f.write(content)
        else:
            print_buggy_summary(solutions_root, task_file)
        return

    build_image(image_tag, dockerfile, context_dir)

    # If kw.json is configured, prepare rank-1 limits and a per-task annotations cache.
    rank1_limits = _load_kw_rank1_limits()
    annotations_cache: dict[str, Optional[dict]] = {}

    jobs: List[Tuple[str, Path]] = []
    tasks = read_tasks(task_file)

    # Stats for debugging “why are there no jobs”.
    total_solutions = 0
    matched_with_kw = 0
    filtered_no_ann = 0
    filtered_no_kw = 0
    filtered_not_in_rank1 = 0
    filtered_quota_exhausted = 0

    for t in tasks:
        sols = list_solutions_for_task(solutions_root, t)
        if not sols:
            continue
        total_solutions += len(sols)

        ann = None
        if rank1_limits is not None:
            ann = annotations_cache.get(t)
            if ann is None and t not in annotations_cache:
                ann = _load_annotations_for_task(solutions_root, t)
                annotations_cache[t] = ann

        for s in sols:
            if rank1_limits is not None:
                if not ann:
                    filtered_no_ann += 1
                    continue
                kw = _infer_rank1_kw(ann or {}, s)
                if not kw:
                    filtered_no_kw += 1
                    continue
                if kw not in rank1_limits:
                    filtered_not_in_rank1 += 1
                    continue
                if rank1_limits[kw] <= 0:
                    filtered_quota_exhausted += 1
                    continue
                # Check passed: decrement quota and add job.
                rank1_limits[kw] -= 1
                matched_with_kw += 1
                jobs.append((t, s))
            else:
                jobs.append((t, s))

    if rank1_limits is not None:
        print(f"[kw] stats: total_solutions={total_solutions} "
              f"with_kw_and_accepted={matched_with_kw} "
              f"no_annotations={filtered_no_ann} "
              f"no_keyword_match={filtered_no_kw} "
              f"kw_not_in_rank1={filtered_not_in_rank1} "
              f"quota_exhausted={filtered_quota_exhausted}",
              file=sys.stderr)

    # --- Shuffle jobs if any ---
    if jobs:
        # Support environment variable JOB_SHUFFLE_SEED to fix the random seed for reproducibility; otherwise use non-deterministic randomness.
        seed_env = os.environ.get("JOB_SHUFFLE_SEED")
        if seed_env:
            try:
                seed_val = int(seed_env)
                rng = random.Random(seed_val)
                rng.shuffle(jobs)
                print(f"[batch] Shuffled jobs with seed={seed_val}")
            except ValueError:
                random.shuffle(jobs)
                print("[batch] Shuffled jobs (JOB_SHUFFLE_SEED invalid, using default RNG)")
        else:
            random.shuffle(jobs)
            print("[batch] Shuffled jobs (no seed)")

    if not jobs:
        print("[batch] No jobs to run (all done or no solutions).")
        remove_image(image_tag)
        return

    print(f"[batch] Total jobs: {len(jobs)} across tasks: {len(tasks)}")
    gpu_ids = get_gpu_ids()
    print(f"[batch] GPUs detected: {gpu_ids}")

    if max_parallel is not None and max_parallel < 1:
        max_parallel = 1
    has_gpu = not (len(gpu_ids) == 1 and gpu_ids[0] is None)

    # Determine desired concurrency:
    # - if user provided max_parallel -> respect it (at least 1)
    # - else default to number of GPUs (one worker per GPU) or 1 for CPU-only
    n_gpus = len(gpu_ids) if has_gpu else 0
    if max_parallel is None:
        desired = n_gpus if has_gpu else 1
    else:
        desired = max(1, int(max_parallel))
    # Never exceed number of jobs
    desired = min(desired, len(jobs))
    print(f"[batch] concurrency desired={desired} (max_parallel={max_parallel}, GPUs={n_gpus})")

    strict_cpu = _strict_cpu_enabled()
    cpu_groups: List[List[int]] = []
    per_container_cpu_max: Optional[int] = None

    if strict_cpu:
        # In strict CPU mode, further cap desired concurrency by CPU slot count.
        cpu_slots_cap = _compute_cpu_slots(user_max=max_parallel)
        print(f"[batch] CPU-based concurrency cap: {cpu_slots_cap}")
        desired = min(desired, cpu_slots_cap)
        if desired <= 0:
            desired = 1

    if has_gpu:
        n = len(gpu_ids)
        base_slots = desired // n
        rem = desired % n
        workers: List[Optional[int]] = []
        for i, gid in enumerate(gpu_ids):
            slots = base_slots + (1 if i < rem else 0)
            workers.extend([gid] * slots)
        print(f"[batch] Using {len(workers)} workers mapped per GPU: {workers}")
    else:
        workers = [None] * desired
        print(f"[batch] Using {desired} CPU workers")

    if strict_cpu:
        # Assign a CPU set to each worker and compute a per-container CPU_MAX_CONCURRENCY hint.
        cpu_groups = _split_cpus_for_workers(len(workers))
        while len(cpu_groups) < len(workers):
            cpu_groups.append([])
        total_cores = _detect_cpu_cores()
        approx_per_container = max(1, total_cores // max(1, len(workers)))
        # This is only a “hint” for scripts inside the container; they may ignore it.
        per_container_cpu_max = approx_per_container

    q: "queue.Queue[tuple[str, Path]]" = queue.Queue()
    for job in jobs:
        q.put(job)

    failures: List[tuple[str, Path, str]] = []
    lock = threading.Lock()

    def worker(worker_idx: int, gid: Optional[int]):
        local_cpu_affinity: Optional[List[int]] = None
        if strict_cpu and 0 <= worker_idx < len(cpu_groups):
            local_cpu_affinity = cpu_groups[worker_idx]
        while True:
            try:
                t, s = q.get_nowait()
            except queue.Empty:
                return
            try:
                run_container_for_solution(
                    image_tag,
                    s,
                    t,
                    data_dir,
                    gid,
                    cpu_affinity=local_cpu_affinity,
                    per_container_cpu_max=per_container_cpu_max,
                )
            except Exception as e:
                with lock:
                    failures.append((t, s, str(e)))
                print(f"[error] task={t} solution={s.name}: {e}")
            finally:
                q.task_done()
                try:
                    time.sleep(60)
                except Exception:
                    pass

    threads = [threading.Thread(target=worker, args=(idx, gid), daemon=True)
               for idx, gid in enumerate(workers)]
    for th in threads:
        th.start()
    q.join()
    for th in threads:
        th.join(timeout=1)

    if failures:
        print(f"[batch] {len(failures)} job(s) failed:")
        for t, s, msg in failures:
            print(f" - {t} :: {s} :: {msg}")
    # Add English summary: successes, failures, total
    total_jobs = len(jobs)
    success_jobs = total_jobs - len(failures)
    print(f"[batch-summary] Success={success_jobs} Failures={len(failures)} Total={total_jobs}")

    remove_image(image_tag)

def run_batch(ns) -> None:
    # Resolve repo root relative to this file (prepare_bench_subset/)
    repo_root = Path(ns.build_context or os.environ.get("BUILD_CONTEXT") or Path(__file__).resolve().parents[1]).resolve()

    # solutions_root: CLI > ENV > sibling 'solutions_subset' > local 'solutions_subset'
    sr = getattr(ns, "solutions_root", None) or os.environ.get("SOLUTIONS_ROOT")
    if sr:
        solutions_root = Path(sr).resolve()
    else:
        sibling = repo_root.parent / "solutions_subset"
        local = repo_root / "solutions_subset"
        solutions_root = (sibling if sibling.exists() else local).resolve()

    # data_dir: CLI > ENV > sibling 'data' > local 'data'
    dd = getattr(ns, "data_dir", None) or os.environ.get("DATA_DIR")
    if dd:
        data_dir = Path(dd).resolve()
    else:
        sibling = repo_root.parent / "data"
        local = repo_root / "data"
        data_dir = (sibling if sibling.exists() else local).resolve()

    # task_file: CLI > ENV > repo_root/task_name.txt
    tf = getattr(ns, "task_file", None) or os.environ.get("TASK_FILE") or str(repo_root / "task_name.txt")
    task_file = Path(tf).resolve()

    # dockerfile: CLI > ENV > repo_root/env/Dockerfile
    df = getattr(ns, "dockerfile", None) or os.environ.get("DOCKERFILE") or str(repo_root / "env" / "Dockerfile")
    dockerfile = Path(df).resolve()

    max_par = ns.max_parallel
    if max_par is None:
        mp_env = os.environ.get("MAX_PARALLEL")
        if mp_env:
            try:
                max_par = int(mp_env)
            except ValueError:
                max_par = None

    # Unified check for enabling tracing (either via CLI or TRACE_VARS environment).
    trace_enabled = bool(getattr(ns, "trace_vars", False) or str(os.environ.get("TRACE_VARS", "")).lower() in ("1", "true", "yes"))
    if trace_enabled:
        os.environ["TRACE_VARS"] = "1"  # Set in this process so worker threads can see it.
        print("[batch-trace] tracing enabled; will pass -e TRACE_VARS=1 into container")

    summary = getattr(ns, "summary", False)
    sb = getattr(ns, "summary_buggy", False)
    buggy_only = bool(sb)
    # summary_buggy may be True / False / a path string.
    buggy_out: Optional[Path] = None
    if isinstance(sb, str):
        buggy_out = Path(sb).resolve()

    # For summary/summary-buggy we only print to stdout; file output is handled in batch_orchestrate.
    batch_orchestrate(
        solutions_root,
        data_dir,
        task_file,
        dockerfile,
        repo_root,
        image_tag="mlagent",
        max_parallel=max_par,
        summary_only=summary,
        buggy_only=buggy_only,
        buggy_out_path=buggy_out,
    )
