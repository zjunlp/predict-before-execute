# Legacy GPU rewrite phase for LightGBM/XGBoost solutions using keyword annotations and LLM diffs.
import os
import sys
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List, TextIO

from . import edit_code_with_llm, gpu_rewrite_code_with_llm  # re-exported in __init__.py


VERBOSE_LOG_FH: Optional[TextIO] = None


def set_verbose_log_fh(fh: Optional[TextIO]) -> None:
    """Setter used by clean.py to share the verbose log file handle."""
    global VERBOSE_LOG_FH
    VERBOSE_LOG_FH = fh


def _log_verbose_gpu_rewrite(
    changed: bool,
    solution_py: str,
    raw_response: Optional[str],
) -> None:
    """
    If VERBOSE_LOG_FH is set, write a structured log entry for a GPU rewrite
    operation on LightGBM / XGBoost gradient boosting models.
    """
    global VERBOSE_LOG_FH
    if VERBOSE_LOG_FH is None:
        return

    try:
        VERBOSE_LOG_FH.write("=== GPU_REWRITE_ENTRY ===\n")
        VERBOSE_LOG_FH.write(f"CHANGED={1 if changed else 0}\n")
        VERBOSE_LOG_FH.write(f"PATH={solution_py}\n")
        if raw_response is not None:
            VERBOSE_LOG_FH.write("\n=== RAW_RESPONSE_BEGIN ===\n")
            VERBOSE_LOG_FH.write(raw_response)
            VERBOSE_LOG_FH.write("\n=== RAW_RESPONSE_END ===\n")
        try:
            with open(solution_py, "r", encoding="utf-8") as rf:
                final = rf.read()
            final_bytes = final.encode("utf-8")
            final_sha = hashlib.sha256(final_bytes).hexdigest()
            final_size = len(final_bytes)
            VERBOSE_LOG_FH.write("\n=== FINAL_FILE_BEGIN ===\n")
            VERBOSE_LOG_FH.write(final)
            VERBOSE_LOG_FH.write("\n=== FINAL_FILE_END ===\n")
            VERBOSE_LOG_FH.write(f"FINAL_SIZE={final_size}\n")
            VERBOSE_LOG_FH.write(f"FINAL_SHA256={final_sha}\n")
        except Exception as e:
            VERBOSE_LOG_FH.write(f"FINAL_FILE_READ_ERROR: {e}\n")

        VERBOSE_LOG_FH.write("=== GPU_REWRITE_ENTRY_END ===\n\n")
        VERBOSE_LOG_FH.flush()
    except Exception as e:
        print(f"[GPU-REWRITE][VERBOSE] Failed to write verbose log for {solution_py}: {e}", file=sys.stderr)


def _gpu_rewrite_one_solution(task_name: str, solution_key: str, first_kw: str, solution_py: str) -> Tuple[str, bool]:
    """
    Rewrite a single solution file to use GPU for LightGBM / XGBoost
    gradient boosting models. Returns (solution_py, changed?).
    Designed to be used from a ThreadPoolExecutor.
    """
    print(f"[GPU-REWRITE] Rewriting boosting solution: task={task_name}, key={solution_key}, kw={first_kw}")
    try:
        with open(solution_py, "r", encoding="utf-8") as sf:
            code = sf.read()
    except Exception as e:
        print(f"[GPU-REWRITE] Failed to read {solution_py}: {e}", file=sys.stderr)
        _log_verbose_gpu_rewrite(False, solution_py, None)
        return (solution_py, False)

    try:
        new_code, raw_resp = gpu_rewrite_code_with_llm(
            code=code,
            solution_context=f"{task_name}:{solution_key}:{first_kw}",
            return_raw_response=True,
        )
    except Exception as e:
        print(f"[GPU-REWRITE] LLM error on {solution_py}: {e}", file=sys.stderr)
        _log_verbose_gpu_rewrite(False, solution_py, None)
        return (solution_py, False)

    if not new_code or new_code == code:
        print(f"[GPU-REWRITE] No changes for {solution_py}")
        _log_verbose_gpu_rewrite(False, solution_py, raw_resp)
        return (solution_py, False)

    try:
        with open(solution_py, "w", encoding="utf-8") as sf:
            sf.write(new_code)
            sf.flush()
            try:
                os.fsync(sf.fileno())
            except Exception:
                pass
        print(f"[GPU-REWRITE] Updated boosting solution file: {solution_py}")
        _log_verbose_gpu_rewrite(True, solution_py, raw_resp)
        return (solution_py, True)
    except Exception as e:
        print(f"[GPU-REWRITE] Failed to write {solution_py}: {e}", file=sys.stderr)
        _log_verbose_gpu_rewrite(False, solution_py, raw_resp)
        return (solution_py, False)


def run_gpu_rewrite_phase(root: str, boosting_kw_file: str) -> None:
    """
    Optional pre-pass: if a boosting_kw_file is provided, run a GPU rewrite
    phase over solution files that are annotated with keywords contained in
    that file. This phase only touches LightGBM / XGBoost gradient boosting
    solutions.
    """
    print(f"[GPU-REWRITE] run_gpu_rewrite_phase entered. root={root}, boosting_kw_file={boosting_kw_file}")

    if not os.path.isfile(boosting_kw_file):
        print(f"[GPU-REWRITE] boosting kw file not found, skipping GPU rewrite phase: {boosting_kw_file}", file=sys.stderr)
        return

    print(f"[GPU-REWRITE] Starting GPU rewrite phase using boosting kw file: {boosting_kw_file}")
    try:
        with open(boosting_kw_file, "r", encoding="utf-8") as f:
            keywords = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"[GPU-REWRITE] Failed to read boosting kw file {boosting_kw_file}: {e}", file=sys.stderr)
        return

    if not keywords:
        print("[GPU-REWRITE] boosting kw file is empty, skipping GPU rewrite phase.")
        return

    print(f"[GPU-REWRITE] Loaded boosting keywords (for LightGBM/XGBoost) from file: {keywords}")

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

        print(f"[GPU-REWRITE] Task {task_name}: matched {matched_in_task} boosting solutions by keyword file")

    if not to_rewrite:
        print(f"[GPU-REWRITE] No solutions matched boosting kw file under root: {root_abs}")
        print(f"[GPU-REWRITE] GPU rewrite phase finished. Tasks processed: {tasks_processed}, solutions rewritten: 0")
        return

    print(f"[GPU-REWRITE] Total boosting solutions to rewrite (LightGBM/XGBoost): {len(to_rewrite)}")
    for tname, skey, fkw, spy in to_rewrite:
        print(f"[GPU-REWRITE] TO_REWRITE: task={tname}, solution_key={skey}, kw={fkw}, path={spy}")

    solutions_touched = 0
    max_workers = min(32, (os.cpu_count() or 4) * 2)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_gpu_rewrite_one_solution, task_name, solution_key, first_kw, solution_py)
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