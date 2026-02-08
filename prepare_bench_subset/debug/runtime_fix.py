# Parse run logs and use an LLM to fix solutions that crashed at runtime based on exec_output tails.
import os
import sys
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Iterable, Optional

from .debug_runner import fix_runtime_error_with_llm

RUNTIME_VERBOSE_LOG_FH: Optional[object] = None


def set_runtime_verbose_log_fh(fh: Optional[object]) -> None:
    """Configure verbose log file handle for runtime-fix phase (shared with clean.py)."""
    global RUNTIME_VERBOSE_LOG_FH
    RUNTIME_VERBOSE_LOG_FH = fh


def _runtime_verbose_log_entry(
    solution_py: str,
    exec_tail: Optional[str] = None,
    raw_response: Optional[str] = None,
) -> None:
    """
    Write a structured verbose log entry for a runtime fix attempt.

    - EXEC_OUTPUT_TAIL: the error tail from exec_output.txt
    - RAW_RESPONSE: raw LLM response (contains diffs or full code, i.e. 'different parts')
    - FINAL_FILE: final code after writing back to disk
    """
    global RUNTIME_VERBOSE_LOG_FH
    if RUNTIME_VERBOSE_LOG_FH is None:
        return

    try:
        fh = RUNTIME_VERBOSE_LOG_FH
        fh.write("=== RUNTIME_FIX_ENTRY ===\n")
        fh.write(f"PATH={solution_py}\n")

        if exec_tail is not None:
            fh.write("\n--- EXEC_OUTPUT_TAIL_BEGIN ---\n")
            fh.write(exec_tail)
            if not exec_tail.endswith("\n"):
                fh.write("\n")
            fh.write("--- EXEC_OUTPUT_TAIL_END ---\n")

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

        fh.write("=== RUNTIME_FIX_ENTRY_END ===\n\n")
        fh.flush()
    except Exception as e:
        print(f"[RUNTIME-FIX][VERBOSE] failed to write verbose log for {solution_py}: {e}", file=sys.stderr)


def _iter_buggy_entries(log_path: str) -> Iterable[Tuple[str, str]]:
    """
    Yield (solution_py_path, exec_output_tail) from a runs_log debug log.

    Each buggy solution block looks like:

        [buggy-summary] Task: ...
          - Solution: /path/to/solution.py
            Log: /path/to/exec_output.txt
            --- exec_output tail ---
            ... error lines ...
            --- end tail ---
    """
    current_solution: str = ""
    collecting = False
    tail_lines: List[str] = []

    with open(log_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if line.strip().startswith("- Solution:"):
                # Flush previous block if any
                if current_solution and tail_lines:
                    yield (current_solution, "\n".join(tail_lines))
                # Start new solution block
                current_solution = line.split("Solution:", 1)[1].strip()
                collecting = False
                tail_lines = []
                continue

            if "--- exec_output tail ---" in line:
                collecting = True
                tail_lines = []
                continue

            if "--- end tail ---" in line:
                collecting = False
                # we'll yield when we see the next Solution or at EOF
                continue

            if collecting:
                tail_lines.append(line)

    # Final flush
    if current_solution and tail_lines:
        yield (current_solution, "\n".join(tail_lines))


def _fix_one_from_log_entry(
    solution_py: str,
    exec_tail: str,
    validate_syntax: bool,
) -> Tuple[str, bool]:
    """Worker: fix a single solution file based on its exec_output tail."""
    print(f"[RUNTIME-FIX] Processing solution: {solution_py}")
    if not os.path.isfile(solution_py):
        print(f"[RUNTIME-FIX] Solution file not found, skip: {solution_py}", file=sys.stderr)
        return (solution_py, False)

    try:
        with open(solution_py, "r", encoding="utf-8") as sf:
            code = sf.read()
    except Exception as e:
        print(f"[RUNTIME-FIX] Failed to read {solution_py}: {e}", file=sys.stderr)
        return (solution_py, False)

    try:
        new_code, raw_resp = fix_runtime_error_with_llm(
            solution_path=solution_py,
            code=code,
            exec_output_tail=exec_tail,
            validate_syntax=validate_syntax,
            return_raw_response=True,
        )
    except Exception as e:
        print(f"[RUNTIME-FIX] LLM error while fixing {solution_py}: {e}", file=sys.stderr)
        _runtime_verbose_log_entry(solution_py, exec_tail=exec_tail, raw_response=None)
        return (solution_py, False)

    if not new_code or new_code == code:
        print(f"[RUNTIME-FIX] No changes for {solution_py}")
        _runtime_verbose_log_entry(solution_py, exec_tail=exec_tail, raw_response=raw_resp)
        return (solution_py, False)

    try:
        with open(solution_py, "w", encoding="utf-8") as sf:
            sf.write(new_code)
        print(f"[RUNTIME-FIX] Updated solution file: {solution_py}")
        _runtime_verbose_log_entry(solution_py, exec_tail=exec_tail, raw_response=raw_resp)
        return (solution_py, True)
    except Exception as e:
        print(f"[RUNTIME-FIX] Failed to write {solution_py}: {e}", file=sys.stderr)
        _runtime_verbose_log_entry(solution_py, exec_tail=exec_tail, raw_response=raw_resp)
        return (solution_py, False)


def run_runtime_fix_from_log(
    log_path: str,
    validate_syntax: bool = True,
    max_workers: int = 8,
) -> None:
    """
    High-level entrypoint: read a runs_log debug log, and for each buggy
    solution entry, apply LLM-based fixes in-place (in parallel).
    """
    if not os.path.isfile(log_path):
        print(f"[RUNTIME-FIX] Log file not found: {log_path}", file=sys.stderr)
        return

    print(f"[RUNTIME-FIX] Parsing buggy solutions from log: {log_path}")
    entries = list(_iter_buggy_entries(log_path))
    if not entries:
        print("[RUNTIME-FIX] No buggy solutions found in log.")
        return

    print(f"[RUNTIME-FIX] Found {len(entries)} buggy solution entries in log.")

    # Choose worker count: cap to len(entries) and CPU*2
    if max_workers is None or max_workers <= 0:
        max_workers = min(32, (os.cpu_count() or 4) * 2)
    max_workers = min(max_workers, len(entries))

    fixed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_path = {
            ex.submit(_fix_one_from_log_entry, solution_py, exec_tail, validate_syntax): solution_py
            for (solution_py, exec_tail) in entries
        }
        for fut in as_completed(future_to_path):
            solution_py = future_to_path[fut]
            try:
                _path, changed = fut.result()
                if changed:
                    fixed += 1
            except Exception as e:
                print(f"[RUNTIME-FIX] Worker exception for {solution_py}: {e}", file=sys.stderr)

    print(f"[RUNTIME-FIX] Finished. Buggy solutions fixed: {fixed}/{len(entries)}")
