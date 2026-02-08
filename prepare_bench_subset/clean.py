#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compile and optionally auto-fix Python solution files under a root directory using
LLM-based tools, including GPU rewrite for boosting models, runtime-error fixes, and
grading-time evaluation error fixes.
"""

import os
import sys
import argparse
from typing import Optional, TextIO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from .debug import edit_code_with_llm  # ensure LLM editor is available
from .debug.compile_fix import run_compile_fix_pipeline, set_max_debug_depth
from .debug.gpu_rewrite import run_gpu_rewrite_phase, set_verbose_log_fh
from .debug.runtime_fix import run_runtime_fix_from_log, set_runtime_verbose_log_fh
from .debug.eval_fix import run_eval_fix_from_json


VERBOSE_LOG_FH: Optional[TextIO] = None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compile and optionally auto-fix Python files under a root directory using the LLM editor. "
            "Optionally, pre-rewrite LightGBM/XGBoost gradient boosting solutions to use GPU, "
            "and/or fix buggy solutions based on runs_log runtime error logs."
        )
    )
    parser.add_argument(
        "-r",
        "--root",
        required=True,
        help="Root directory to search for .py files (default: %(default)s)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=min(32, (os.cpu_count() or 4) * 2),
        help="Number of worker threads to use for the compile+fix pipeline (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--max-depth",
        type=int,
        default=3,
        help="Maximum number of LLM fix attempts per file (default: %(default)s)",
    )
    parser.add_argument(
        "--eval-error-json",
        type=str,
        default=None,
        help=(
            "Optional path to an evaluation error JSON (e.g., runs_log/error_eval_xxx.json). "
            "If provided, run a pass that uses the LLM to fix solutions whose submissions "
            "were rejected during grading, then delete their previous submission directories."
        ),
    )
    parser.add_argument(
        "--gpu-boosting-kw-file",
        type=str,
        default=None,
        help=(
            "Optional keyword file for GPU rewrite phase. If provided, run a pre-pass that rewrites "
            "LightGBM/XGBoost (gradient boosting) solution files to use GPU based on keywords in this file."
        ),
    )
    parser.add_argument(
        "--verbose-log",
        type=str,
        default=None,
        help=(
            "Optional path to a verbose log file. If set, the GPU rewrite phase will log path, "
            "LLM diff/full-code response, and final file content for each LightGBM/XGBoost solution."
        ),
    )
    parser.add_argument(
        "--runtime-log",
        type=str,
        default=None,
        help=(
            "Optional path to a runs_log debug log (e.g., runs_log/debug_xxx.log). "
            "If provided, run a pass that parses buggy solutions and uses the LLM to "
            "fix their runtime errors based on the recorded exec_output tail."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help=(
            "Root directory of competition data (containing <competition_id>/prepared/public). "
            "Used by eval-error fix prompts to load description.md and sample_submission.csv "
            "(default: %(default)s)."
        ),
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    workers = args.workers
    set_max_debug_depth(args.max_depth)

    global VERBOSE_LOG_FH
    if args.verbose_log:
        try:
            os.makedirs(os.path.dirname(args.verbose_log), exist_ok=True)
            VERBOSE_LOG_FH = open(args.verbose_log, "a", encoding="utf-8")
            # GPU rewrite verbose
            set_verbose_log_fh(VERBOSE_LOG_FH)
            # runtime-fix verbose
            set_runtime_verbose_log_fh(VERBOSE_LOG_FH)
            print(f"[INFO] Verbose logging enabled: {args.verbose_log}")
        except Exception as e:
            print(f"[WARN] Failed to open verbose log file {args.verbose_log}: {e}", file=sys.stderr)
            VERBOSE_LOG_FH = None
            set_verbose_log_fh(None)
            set_runtime_verbose_log_fh(None)
    else:
        set_verbose_log_fh(None)
        set_runtime_verbose_log_fh(None)

    if not os.path.isdir(root):
        print(f"[ERROR] Root not found: {root}", file=sys.stderr)
        sys.exit(2)

    if edit_code_with_llm is None:
        print(
            "[ERROR] LLM editor unavailable: import failed. "
            "Try running with `python -m prepare_bench_subset.clean` or check imports.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Optional eval-error fix phase (grading-time invalid submissions)
    if args.eval_error_json:
        print(
            "[EVAL-FIX] eval-error-json provided, running LLM-based fixes for solutions "
            "whose submissions failed during grading..."
        )
        run_eval_fix_from_json(
            args.eval_error_json,
            verbose_fh=VERBOSE_LOG_FH,
            data_root=args.data_root,
        )
        print("[EVAL-FIX] Evaluation error fix phase completed. Continuing to other phases...")

    # Optional GPU rewrite phase for LightGBM/XGBoost boosting models
    if args.gpu_boosting_kw_file:
        print("[GPU-REWRITE] boosting kw-file provided, entering GPU rewrite phase for LightGBM/XGBoost solutions...")
        run_gpu_rewrite_phase(root, args.gpu_boosting_kw_file)
        print("[GPU-REWRITE] GPU rewrite phase completed. Continuing to compile/check phase...")

    # Optional: fix buggy solutions from runs_log before compile+fix
    if args.runtime_log:
        print(
            "[RUNTIME-FIX] runtime-log provided, running LLM-based runtime error fixes "
            "for buggy solutions recorded in the log..."
        )
        run_runtime_fix_from_log(args.runtime_log, validate_syntax=True, max_workers=workers)
        print("[RUNTIME-FIX] Runtime error fix phase completed. Continuing to compile/check phase...")

    checked, failed, ok_count = run_compile_fix_pipeline(root, workers)

    if VERBOSE_LOG_FH is not None:
        VERBOSE_LOG_FH.close()

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()