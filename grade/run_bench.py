# CLI entrypoint for skip_bench grading.
import argparse
from . import runner_main  # type: ignore


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file", default="/newdisk1/zjs/skip_bench/prepare_bench_subset/task_name.txt")
    ap.add_argument(
        "--solutions-dir",
        default=None,
        help=(
            "Root directory for solution files. Still needed even with --groundtruth-file when group 'paths' are "
            "relative. You may omit or pass empty ('') ONLY if your group JSON uses absolute paths."
        ),
    )
    ap.add_argument("--n", type=int, default=0, help="n in groups_n.json; 0 means process all")
    ap.add_argument("--model", default=None, help="LLM model to use (overrides env/default)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument(
        "--partial-json",
        action="store_true",
        default=True,
        help=(
            "If set, save per-task checkpoint JSON (grade_results_partial.json) "
            "under each task's report directory for resume. "
            "When disabled, no per-task partial JSON is written."
        ),
    )
    # Sources for task description and data analysis
    ap.add_argument("--tasks-root", default=None, help="Mode 1: root containing description/ and data_analysis/result/")
    ap.add_argument("--desc-dir", default=None, help="Mode 2: directory containing description_*.md files")
    ap.add_argument("--da-dir", default=None, help="Mode 2: directory containing da_result_*.txt files")
    ap.add_argument("--raw-data-sample", default=None, help="Mode 3: directory containing raw_data_sample_*.txt files (replaces --da-dir)")
    # Report configuration
    ap.add_argument("--report-out", default=None, help="path to a combined human-readable report; if omitted, per-task reports are written")
    # Retry when JSON parsing fails
    ap.add_argument("--max-retries", type=int, default=3, help="max times to re-query when JSON parsing fails")
    # New: parallelism
    ap.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Parallelism: 1 = serial; >1 = parallel; actual worker threads = min(--parallel, (os.cpu_count() or 4) * 2)",
    )
    # NEW: task-level parallelism
    ap.add_argument(
        "--task-parallel",
        type=int,
        default=1,
        help="Number of tasks to process in parallel (task-level concurrency). Default: 32.",
    )
    # New: allow chain-of-thought before final JSON
    ap.add_argument(
        "--cot",
        action="store_true",
        default=False,
        help="Allow the model to include brief reasoning before the final JSON (parser will extract the last JSON object).",
    )
    # New: prompt boost toggle
    ap.add_argument(
        "--prompt-boost",
        action="store_true",
        default=True,
        help="Emphasize equal consideration of task description/data analysis and provide trade-off guidance in the prompt.",
    )
    # New: allow manually specify a groundtruth (group) json file.
    ap.add_argument(
        "--groundtruth-file",
        default=None,
        help=(
            "(advanced) Path to a groundtruth/group JSON file to use instead of auto-discovery. "
            "May contain '{task}' to substitute the task name. Requires --n != 0. "
            "Note: If the group's 'paths' are relative, --solutions-dir is still required."
        ),
    )
    # New: append random task description/data analysis as noise
    ap.add_argument(
        "--concat-random-task-text",
        action="store_true",
        default=False,
        help="If set, and when task description/data analysis exist, append a random other task's description and data analysis respectively.",
    )
    # new: toggle token counting over all prompts sent to LLM
    ap.add_argument(
        "--count-tokens",
        action="store_true",
        default=False,
        help="If set, estimate total tokens sent to the LLM based on constructed prompts (uses tiktoken when available) and exit without calling the LLM.",
    )
    # mode: build all-tasks report purely from existing per-task reports
    ap.add_argument(
        "--from-existing-reports",
        action="store_true",
        default=False,
        help=(
            "If set, do not call the LLM. Instead, for each task in --task-file, "
            "locate its latest single-task grade_report_*.txt under <solutions_dir>/<task>/report, "
            "parse the header metrics, and synthesize an all-tasks metadata report."
        ),
    )
    # CHANGE: resume-from-partial is now a boolean flag (no explicit path)
    ap.add_argument(
        "--resume-from-partial",
        action="store_true",
        default=False,
        help=(
            "If set, scan each <solutions_dir>/<task>/report/grade_results_partial.json "
            "and resume only those tasks from their per-task checkpoint."
        ),
    )
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    # Dispatch to the actual execution logic
    runner_main.run_with_args(args)


if __name__ == "__main__":
    main()