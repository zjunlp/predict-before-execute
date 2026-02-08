# Fix evaluation-time submission errors by editing solution files with an LLM using grading JSON context.
import json
import os
import sys
import shutil
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, Optional, TextIO, List

from .debug_runner import fix_eval_error_with_llm

logger = logging.getLogger("debug_eval")

# module-global verbose handle (can be set via setter or passed to run function)
VERBOSE_LOG_FH: Optional[TextIO] = None


def set_eval_verbose_log_fh(fh: Optional[TextIO]) -> None:
    """Setter used by clean.py to share the verbose log file handle (optional)."""
    global VERBOSE_LOG_FH
    VERBOSE_LOG_FH = fh


def _parse_grading_json_block(error_text: str) -> Tuple[Optional[Dict], Optional[str], Optional[str]]:
    """
    Extract the JSON object after 'full grading report JSON:' using brace matching.
    Returns (parsed_json, submission_path, grader_error).
    """
    submission_path: Optional[str] = None
    grader_error: Optional[str] = None

    marker = "full grading report JSON"
    idx = error_text.find(marker)
    if idx == -1:
        # try to find a JSON object anywhere
        try:
            obj = json.loads(error_text)
            submission_path = obj.get("submission_path")
            grader_error = obj.get("grader_error")
            return (obj, submission_path, grader_error)
        except Exception:
            return (None, None, None)

    # find first '{' after marker
    brace_idx = error_text.find("{", idx)
    if brace_idx == -1:
        return (None, None, None)

    depth = 0
    end_idx = None
    for i in range(brace_idx, len(error_text)):
        ch = error_text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break
    if end_idx is None:
        return (None, None, None)

    json_str = error_text[brace_idx:end_idx]
    try:
        obj = json.loads(json_str)
    except Exception:
        return (None, None, None)

    submission_path = obj.get("submission_path")
    grader_error = obj.get("grader_error")
    return (obj, submission_path, grader_error)


def _solution_py_from_error_key(error_key: str) -> Optional[str]:
    """
    Map an error key like:
      /.../code/submission_solution_<SOLUTION_STEM>[_run_xxx]
    to the solution .py:
      /.../code/solution_<SOLUTION_STEM>[_run_xxx].py

    If mapping fails or file not present, return None.
    """
    err_key = error_key.rstrip(os.sep)
    parent = os.path.dirname(err_key)
    leaf = os.path.basename(err_key)
    if leaf.startswith("submission_"):
        solution_stem = leaf[len("submission_") :]
        solution_file = f"{solution_stem}.py"
        return os.path.join(parent, solution_file)
    # fallback heuristics: try replacing 'submission_' with ''
    if "submission_" in leaf:
        solution_stem = leaf.replace("submission_", "")
        return os.path.join(parent, f"{solution_stem}.py")
    return None


def _eval_verbose_log_entry(
    solution_py: str,
    error_text: str,
    raw_response: Optional[str],
) -> None:
    """Write structured verbose entry (if VERBOSE_LOG_FH set)."""
    global VERBOSE_LOG_FH
    if VERBOSE_LOG_FH is None:
        return
    try:
        fh = VERBOSE_LOG_FH
        fh.write("=== EVAL_FIX_ENTRY ===\n")
        fh.write(f"PATH={solution_py}\n")

        fh.write("\n--- EVAL_ERROR_TEXT_BEGIN ---\n")
        fh.write(error_text)
        if not error_text.endswith("\n"):
            fh.write("\n")
        fh.write("--- EVAL_ERROR_TEXT_END ---\n")

        if raw_response is not None:
            fh.write("\n--- RAW_RESPONSE_BEGIN ---\n")
            fh.write(raw_response)
            if not raw_response.endswith("\n"):
                fh.write("\n")
            fh.write("--- RAW_RESPONSE_END ---\n")

        try:
            with open(solution_py, "r", encoding="utf-8") as rf:
                final_code = rf.read()
            final_bytes = final_code.encode("utf-8")
            final_sha = hashlib.sha256(final_bytes).hexdigest()
            final_size = len(final_bytes)
            fh.write("\n--- FINAL_FILE_BEGIN ---\n")
            fh.write(final_code)
            if not final_code.endswith("\n"):
                fh.write("\n")
            fh.write("--- FINAL_FILE_END ---\n")
            fh.write(f"FINAL_SIZE={final_size}\n")
            fh.write(f"FINAL_SHA256={final_sha}\n")
        except Exception as e:
            fh.write(f"\nFINAL_FILE_READ_ERROR: {e}\n")

        fh.write("=== EVAL_FIX_ENTRY_END ===\n\n")
        fh.flush()
    except Exception as e:
        print(f"[EVAL-FIX][VERBOSE] failed to write verbose log for {solution_py}: {e}", file=sys.stderr)


def _fix_one_entry(err_key: str, err_text: str, validate_syntax: bool, data_root: str) -> Tuple[str, bool, str]:
    """
    Worker: attempt to fix a single eval-error entry.
    Returns (err_key, changed?, message)
    """
    solution_py = _solution_py_from_error_key(err_key)
    if not solution_py or not os.path.isfile(solution_py):
        msg = f"Solution file not found for key {err_key}: {solution_py}"
        print(f"[EVAL-FIX] {msg}", file=sys.stderr)
        _eval_verbose_log_entry(solution_py or err_key, err_text, raw_response=None)
        return (err_key, False, msg)

    grading_json, submission_path, grader_error = _parse_grading_json_block(err_text)
    competition_id = ""
    score_val = None
    grading_json_str = None
    if grading_json is not None:
        competition_id = str(grading_json.get("competition_id", "") or "")
        score = grading_json.get("score", None)
        if isinstance(score, (int, float)):
            score_val = float(score)
        grading_json_str = json.dumps(grading_json, indent=2, ensure_ascii=False)

    try:
        with open(solution_py, "r", encoding="utf-8") as sf:
            code = sf.read()
    except Exception as e:
        msg = f"Failed to read {solution_py}: {e}"
        print(f"[EVAL-FIX] {msg}", file=sys.stderr)
        _eval_verbose_log_entry(solution_py, err_text, raw_response=None)
        return (err_key, False, msg)

    try:
        new_code, raw_resp = fix_eval_error_with_llm(
            solution_path=solution_py,
            code=code,
            competition_id=competition_id,
            grader_error=grader_error,
            score=score_val,
            submission_path=submission_path,
            full_grading_json=grading_json_str,
            validate_syntax=validate_syntax,
            return_raw_response=True,
            data_root=data_root,
        )
    except Exception as e:
        msg = f"LLM error while fixing {solution_py}: {e}"
        print(f"[EVAL-FIX] {msg}", file=sys.stderr)
        _eval_verbose_log_entry(solution_py, err_text, raw_response=None)
        return (err_key, False, msg)

    if not new_code or new_code == code:
        print(f"[EVAL-FIX] No changes for {solution_py}")
        _eval_verbose_log_entry(solution_py, err_text, raw_response=raw_resp)
        changed = False
    else:
        try:
            with open(solution_py, "w", encoding="utf-8") as wf:
                wf.write(new_code)
            print(f"[EVAL-FIX] Updated solution file: {solution_py}")
            _eval_verbose_log_entry(solution_py, err_text, raw_response=raw_resp)
            changed = True
        except Exception as e:
            msg = f"Failed to write {solution_py}: {e}"
            print(f"[EVAL-FIX] {msg}", file=sys.stderr)
            _eval_verbose_log_entry(solution_py, err_text, raw_response=raw_resp)
            return (err_key, False, msg)

    # remove prior submission directory so next run regenerates it
    try:
        if os.path.isdir(err_key):
            shutil.rmtree(err_key, ignore_errors=False)
            print(f"[EVAL-FIX] Removed submission directory: {err_key}")
    except Exception as e:
        print(f"[EVAL-FIX] Failed to remove submission dir {err_key}: {e}", file=sys.stderr)

    return (err_key, changed, "ok")


def run_eval_fix_from_json(
    json_path: str,
    data_root: str,
    verbose_fh: Optional[TextIO] = None,
    validate_syntax: bool = True,
    max_workers: Optional[int] = None,
) -> None:
    """
    High-level entrypoint: read an error_eval_*.json file, and for each entry in
    errors{}, try to fix the corresponding solution code in parallel.
    """
    global VERBOSE_LOG_FH
    if verbose_fh is not None:
        set_eval_verbose_log_fh(verbose_fh)

    if not os.path.isfile(json_path):
        print(f"[EVAL-FIX] JSON file not found: {json_path}", file=sys.stderr)
        return

    print(f"[EVAL-FIX] Loading eval error JSON: {json_path}")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        print(f"[EVAL-FIX] Failed to parse JSON: {e}", file=sys.stderr)
        return

    errors: Dict[str, str] = payload.get("errors") or {}
    if not errors:
        print("[EVAL-FIX] No errors found in JSON.")
        return

    entries = list(errors.items())
    print(f"[EVAL-FIX] Found {len(entries)} error entries to process.")

    # determine workers
    if max_workers is None or max_workers <= 0:
        max_workers = min(32, (os.cpu_count() or 4) * 2)
    max_workers = min(max_workers, len(entries))

    fixed = 0
    total = len(entries)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_fix_one_entry, key, text, validate_syntax, data_root): key
            for key, text in entries
        }
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                _k, changed, msg = fut.result()
                if changed:
                    fixed += 1
            except Exception as e:
                print(f"[EVAL-FIX] Worker exception for {key}: {e}", file=sys.stderr)

    print(f"[EVAL-FIX] Finished. Solutions attempted: {total}, solutions changed: {fixed}")
