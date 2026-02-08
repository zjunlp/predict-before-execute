# Compile all Python files under a root directory and use an LLM editor to auto-fix syntax errors based on py_compile output.

import os
import sys
import py_compile
import warnings
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Tuple, List

from . import edit_code_with_llm


MAX_DEBUG_DEPTH = 3  # default; clean.py will override via set_max_debug_depth


def set_max_debug_depth(depth: int) -> None:
    global MAX_DEBUG_DEPTH
    MAX_DEBUG_DEPTH = depth


def find_py_files(root: str) -> List[str]:
    files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ('.git', '.hg', '.svn', '__pycache__', 'venv', '.venv', 'env')]
        for fn in filenames:
            if fn.endswith(".py"):
                files.append(os.path.join(dirpath, fn))
    return files


def compile_one(path: str) -> Tuple[str, bool, str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            py_compile.compile(path, doraise=True)
        return (path, True, "")
    except Exception as e:
        return (path, False, f"{type(e).__name__}: {e}")


def _extract_error_line(error_msg: str) -> int:
    m = re.search(r"line\s+(\d+)", error_msg)
    return int(m.group(1)) if m else -1


def _attempt_fix(path: str, error_msg: str) -> Tuple[bool, str]:
    """
    Attempt to fix a single file using ONLY the LLM editor based on the compile error.
    Returns (ok, new_error_msg_if_any).
    """
    if edit_code_with_llm is None:
        return (False, error_msg)

    try:
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        return (False, f"IOError: {e}")

    lines = code.splitlines(keepends=True)
    err_line_no = _extract_error_line(error_msg)
    start = max(0, (err_line_no - 1) - 5) if err_line_no > 0 else 0
    end = min(len(lines), (err_line_no - 1) + 6) if err_line_no > 0 else min(len(lines), 20)
    context = "".join(lines[start:end])

    description = (
        "Fix this Python file so that it compiles without syntax errors.\n"
        f"File: {path}\n"
        f"py_compile error:\n{error_msg}\n"
        f"Error line: {err_line_no if err_line_no > 0 else 'unknown'}\n\n"
        "Guidance:\n"
        "- Remove any accidentally pasted console log lines (e.g., lines starting with a timestamp like "
        "[YYYY-MM-DD HH:MM:SS,mmm] [run.py:...] [Container]).\n"
        "- Fix unterminated string literals or incomplete statements around the error line.\n"
        "- Make the minimal changes necessary and preserve behavior.\n\n"
        "Respond ONLY with SEARCH/REPLACE diff blocks in the exact format, or a single fenced code block containing the full corrected code.\n\n"
        "Error-near code context:\n"
        f"{context}"
    )

    print(f"[LLM] Invoking on {path} ...")
    try:
        new_code = edit_code_with_llm(
            code=code,
            description=description,
            validate_syntax=True,
        )
    except Exception as e:
        print(f"[LLM ERROR] API exception invoking LLM for {path}: {e}", file=sys.stderr)
        os._exit(2)

    if not new_code:
        print(f"[LLM] No output for {path}")
        return (False, error_msg)
    if new_code == code:
        print(f"[LLM] No changes for {path}")
        return (False, error_msg)

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_code)
    except Exception as e:
        return (False, f"IOError(on write): {e}")

    _, ok, msg = compile_one(path)
    if ok:
        print(f"[LLM FIXED] {path}")
        return (True, "")
    else:
        print(f"[LLM STILL FAIL] {path}: {msg.splitlines()[0] if msg else ''}")
        return (False, msg)


def _process_one(path: str) -> Tuple[str, bool, str]:
    """
    Pipeline for a single file:
      1) compile; if ok => done
      2) if fail => attempt LLM fixes up to MAX_DEBUG_DEPTH, re-compiling after each fix
    """
    _, ok, msg = compile_one(path)
    if ok:
        return (path, True, "")
    last_msg = msg
    for depth in range(1, MAX_DEBUG_DEPTH + 1):
        print(f"[DEBUG] {path}: fix round {depth}/{MAX_DEBUG_DEPTH}")
        ok, new_msg = _attempt_fix(path, last_msg)
        if ok:
            return (path, True, "")
        last_msg = new_msg or last_msg
    return (path, False, last_msg)


def run_compile_fix_pipeline(root: str, workers: int) -> Tuple[int, int, int]:
    """
    Run the compile + LLM-fix pipeline for all .py files under root.
    Returns (checked, failed, ok).
    """
    py_files = find_py_files(root)
    if not py_files:
        print("[INFO] No .py files found.")
        return (0, 0, 0)

    results: List[Tuple[str, bool, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for path, ok, msg in ex.map(_process_one, py_files):
            results.append((path, ok, msg))
            if not ok and msg:
                print(f"[ERROR] {path}")
                print(msg)

    checked = len(results)
    failed = sum(1 for _, ok, _ in results if not ok)
    ok_count = checked - failed
    print(f"[DONE] Checked: {checked}, Failed: {failed}, OK: {ok_count}")
    return (checked, failed, ok_count)
