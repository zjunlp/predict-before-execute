# Batch GPU rewrite pipeline that rewrites selected solutions to GPU based on keyword annotations.
import json
import os
import sys
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, TextIO

from .debug_runner import gpu_rewrite_code_with_llm


def _log_verbose_gpu_rewrite(
    verbose_fh: Optional[TextIO],
    changed: bool,
    solution_py: str,
    raw_response: Optional[str],
    new_code: Optional[str],
) -> None:
    """Write a structured log entry for a GPU rewrite operation if verbose_fh is set."""
    if verbose_fh is None:
        return

    try:
        verbose_fh.write("=== GPU_REWRITE_ENTRY ===\n")
        verbose_fh.write(f"CHANGED={1 if changed else 0}\n")
        verbose_fh.write(f"PATH={solution_py}\n")
        if raw_response is not None:
            verbose_fh.write("\n=== RAW_RESPONSE_BEGIN ===\n")
            verbose_fh.write(raw_response)
            verbose_fh.write("\n=== RAW_RESPONSE_END ===\n")
        try:
            with open(solution_py, "r", encoding="utf-8") as rf:
                final = rf.read()
            final_bytes = final.encode("utf-8")
            final_sha = hashlib.sha256(final_bytes).hexdigest()
            final_size = len(final_bytes)
            verbose_fh.write("\n=== FINAL_FILE_BEGIN ===\n")
            verbose_fh.write(final)
            verbose_fh.write("\n=== FINAL_FILE_END ===\n")
            verbose_fh.write(f"FINAL_SIZE={final_size}\n")
            verbose_fh.write(f"FINAL_SHA256={final_sha}\n")
        except Exception as e:
            verbose_fh.write(f"FINAL_FILE_READ_ERROR: {e}\n")

        verbose_fh.write("=== GPU_REWRITE_ENTRY_END ===\n\n")
        verbose_fh.flush()
    except Exception as e:
        print(f"[GPU-REWRITE][VERBOSE] Failed to write verbose log for {solution_py}: {e}", file=sys.stderr)


def _gpu_rewrite_one_solution(
    task_name: str,
    solution_key: str,
    first_kw: str,
    solution_py: str,
    verbose_fh: Optional[TextIO],
) -> Tuple[str, bool]:
    """Rewrite a single solution file on GPU. Returns (solution_py, changed?)."""
    print(f"[GPU-REWRITE] Rewriting solution: task={task_name}, key={solution_key}, kw={first_kw}")
    try:
        with open(solution_py, "r", encoding="utf-8") as sf:
            code = sf.read()
    except Exception as e:
        print(f"[GPU-REWRITE] Failed to read {solution_py}: {e}", file=sys.stderr)
        _log_verbose_gpu_rewrite(verbose_fh, False, solution_py, None, None)
        return (solution_py, False)

    try:
        new_code, raw_resp = gpu_rewrite_code_with_llm(
            code=code,
            solution_context=f"{task_name}:{solution_key}:{first_kw}",
            return_raw_response=True,
        )
    except Exception as e:
        print(f"[GPU-REWRITE] LLM error on {solution_py}: {e}", file=sys.stderr)
        _log_verbose_gpu_rewrite(verbose_fh, False, solution_py, None, None)
        return (solution_py, False)

    if not new_code or new_code == code:
        print(f"[GPU-REWRITE] No changes for {solution_py}")
        _log_verbose_gpu_rewrite(verbose_fh, False, solution_py, raw_resp, new_code)
        return (solution_py, False)

    try:
        with open(solution_py, "w", encoding="utf-8") as sf:
            sf.write(new_code)
            sf.flush()
            try:
                os.fsync(sf.fileno())
            except Exception:
                pass
        print(f"[GPU-REWRITE] Updated solution file: {solution_py}")
        _log_verbose_gpu_rewrite(verbose_fh, True, solution_py, raw_resp, new_code)
        return (solution_py, True)
    except Exception as e:
        print(f"[GPU-REWRITE] Failed to write {solution_py}: {e}", file=sys.stderr)
        _log_verbose_gpu_rewrite(verbose_fh, False, solution_py, raw_resp, new_code)
        return (solution_py, False)


def run_gpu_rewrite_phase(
    root: str,
    kw_path: str,
    verbose_fh: Optional[TextIO],
) -> None:
    """
    Pre-pass: given a keyword file, run a GPU rewrite phase on solution files
    that are annotated with those keywords (LGBM/XGBoost related), then return.

    This only rewrites LGBM/XGBoost training/inference code to GPU.
    """
    print(f"[GPU-REWRITE] run_gpu_rewrite_phase entered. root={root}, kw_file={kw_path}")

    if not os.path.isfile(kw_path):
        print(f"[GPU-REWRITE] kw file not found, skipping GPU rewrite phase: {kw_path}", file=sys.stderr)
        return

    print(f"[GPU-REWRITE] Starting GPU rewrite phase using kw file: {kw_path}")
    try:
        with open(kw_path, "r", encoding="utf-8") as f:
            keywords = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"[GPU-REWRITE] Failed to read kw file {kw_path}: {e}", file=sys.stderr)
        return

    if not keywords:
        print("[GPU-REWRITE] kw file is empty, skipping GPU rewrite phase.")
        return

    print(f"[GPU-REWRITE] Loaded keywords for LGBM/XGBoost GPU rewrite from file: {keywords}")

    root_abs = os.path.abspath(root)
    solutions_root = root_abs
    print(f"[GPU-REWRITE] Using solutions_root: {solutions_root}")

    kw_set = set(keywords)
    tasks_processed = 0
    to_rewrite: List[Tuple[str, str, str, str]] = []

    try:
        task_names = [
            d for d in os.listdir(root_abs)
            if os.path.isdir(os.path.join(root_abs, d))
        ]
    except Exception as e:
        print(f"[GPU-REWRITE] Failed to list tasks under {root_abs}: {e}", file=sys.stderr)
        return

    if not task_names:
        print(f"[GPU-REWRITE] No task directories found under: {root_abs}")
        print(f"[GPU-REWRITE] GPU rewrite phase finished. Tasks processed: 0, solutions rewritten: 0")
        return

    print(f"[GPU-REWRITE] Discovered task directories under root: {task_names}")

    for task_name in task_names:
        ann_path = os.path.join(root_abs, task_name, "annotation", "annotations_semantic.json")
        if not os.path.isfile(ann_path):
            print(f"[GPU-REWRITE] annotations_semantic.json not found for task {task_name}, skip.")
            continue

        tasks_processed += 1
        print(f"[GPU-REWRITE] Processing task: {task_name}, annotations: {ann_path}")

        try:
            with open(ann_path, "r", encoding="utf-8") as jf:
                ann = json.load(jf)
        except Exception as e:
            print(f"[GPU-REWRITE] Failed to load {ann_path}: {e}", file=sys.stderr)
            continue

        matched_in_task = 0

        for solution_key, kw_lists in ann.items():
            if not kw_lists:
                continue
            kw_list = kw_lists[0] or []
            if not kw_list:
                continue
            first_kw = kw_list[0]
            if first_kw not in kw_set:
                continue

            solution_py = os.path.join(
                solutions_root,
                task_name,
                "code",
                f"{solution_key}.py",
            )
            if not os.path.isfile(solution_py):
                print(f"[GPU-REWRITE] Solution file not found, skip: {solution_py}", file=sys.stderr)
                continue

            matched_in_task += 1
            to_rewrite.append((task_name, solution_key, first_kw, solution_py))

        print(f"[GPU-REWRITE] Task {task_name}: matched {matched_in_task} solutions by kw file")

    if not to_rewrite:
        print(f"[GPU-REWRITE] No solutions matched kw file under root: {root_abs}")
        print(f"[GPU-REWRITE] GPU rewrite phase finished. Tasks processed: {tasks_processed}, solutions rewritten: 0")
        return

    print(f"[GPU-REWRITE] Total solutions to rewrite: {len(to_rewrite)}")
    for tname, skey, fkw, spy in to_rewrite:
        print(f"[GPU-REWRITE] TO_REWRITE: task={tname}, solution_key={skey}, kw={fkw}, path={spy}")

    solutions_touched = 0
    max_workers = min(32, (os.cpu_count() or 4) * 2)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_gpu_rewrite_one_solution, task_name, solution_key, first_kw, solution_py, verbose_fh)
            for (task_name, solution_key, first_kw, solution_py) in to_rewrite
        ]
        for fut in futures:
            try:
                _, changed = fut.result()
                if changed:
                    solutions_touched += 1
            except Exception as e:
                print(f"[GPU-REWRITE] Worker exception: {e}", file=sys.stderr)

    print(f"[GPU-REWRITE] GPU rewrite phase finished. Tasks processed: {tasks_processed}, solutions rewritten: {solutions_touched}")
