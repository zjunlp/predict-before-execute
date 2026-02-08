# Adapted from mle-bench (https://github.com/openai/mle-bench)

#!/usr/bin/env python3
# Standalone grading CLI entrypoint: supports single submissions, JSONL batches, and auto-grading runs.

import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from .env.registry import SimpleRegistry
from .util.grading import grade_csv, grade_jsonl


def _do_grade(
    task_name: str,
    submission_csv: Path,
    competition_id: str,
    out_path: Path,
    data_dir: Path,
    competitions_dir: Path,
    treat_zero_score_as_bug: bool,
) -> tuple[str, Path, bool, str]:
    """
    Grade a single submission in a worker process and, when possible, bind the process to a specific CPU core.
    This must be a top-level function so that ProcessPoolExecutor can pickle it.
    """
    # --- Bind process to a CPU core: distribute PIDs across available CPUs ---
    try:
        available_cpus = sorted(os.sched_getaffinity(0))
        if available_cpus:
            cpu_index = os.getpid() % len(available_cpus)
            target_cpu = {available_cpus[cpu_index]}
            os.sched_setaffinity(0, target_cpu)
    except Exception:
        # Ignore silently if the platform does not support this or if setting fails.
        pass

    # Reconstruct registry and competition in the worker process
    try:
        sub_registry = SimpleRegistry(data_dir=data_dir, competitions_dir=competitions_dir)
        competition = sub_registry.get_competition(competition_id)
    except Exception as e:
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        error_eval_path = out_dir / "error_eval.txt"
        msg = f"Failed to init competition `{competition_id}` in worker: {e}"
        try:
            with open(error_eval_path, "w", encoding="utf-8") as ef:
                ef.write("[ERROR] Failed to initialize competition in worker process.\n")
                ef.write(f"[ERROR] {e}\n")
                ef.write(f"[INFO] submission_csv: {submission_csv}\n")
        except Exception:
            return task_name, out_path, False, msg
        return task_name, error_eval_path, False, msg

    try:
        report = grade_csv(submission_csv, competition, treat_zero_score_as_bug=treat_zero_score_as_bug)
        report_dict = report.to_dict()

        submission_exists = bool(report_dict.get("submission_exists"))
        valid_submission = bool(report_dict.get("valid_submission"))

        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        error_eval_path = out_dir / "error_eval.txt"

        # 2. Correct (non-buggy) solutions: write eval_output.json normally.
        if submission_exists and valid_submission:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report_dict, f, ensure_ascii=False, indent=2)
            if error_eval_path.exists():
                try:
                    error_eval_path.unlink()
                except Exception:
                    pass
            return task_name, out_path, True, ""

        # 3. Incorrect (buggy) solutions: write error_eval.txt.
        reason = []
        if not submission_exists:
            reason.append("submission_exists is False")
        if not valid_submission:
            reason.append("valid_submission is False")
        reason_msg = "; ".join(reason) or "Unknown invalid state"

        # Write both the reason it is marked as a bug and any grading error messages into error_eval.txt.
        # Note: if grade_helpers already logged similar errors, reason_msg and the report may contain the same info.
        with open(error_eval_path, "w", encoding="utf-8") as ef:
            ef.write(f"[BUG] Auto-grade marked this submission as invalid: {reason_msg}\n")
            # If the grading report contains a grader_error field (from Grader logs), include it as well.
            grader_error = report_dict.get("grader_error")
            if grader_error:
                ef.write(f"[ERROR] grading error: {grader_error}\n")
            ef.write(f"[INFO] submission_path: {report_dict.get('submission_path')}\n")
            ef.write("[INFO] full grading report JSON:\n")
            json.dump(report_dict, ef, ensure_ascii=False, indent=2)
            ef.write("\n")

        if out_path.exists():
            try:
                out_path.unlink()
            except Exception:
                pass

        return task_name, error_eval_path, False, reason_msg
    except Exception as e:
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        error_eval_path = out_dir / "error_eval.txt"
        msg = f"Exception during grading: {e}"
        try:
            with open(error_eval_path, "w", encoding="utf-8") as ef:
                # Write the same exception message as printed to the CLI, to keep logs aligned.
                ef.write("[ERROR] Auto-grade raised an exception.\n")
                ef.write(f"[ERROR] {e}\n")
                ef.write(f"[INFO] submission_csv: {submission_csv}\n")
        except Exception:
            return task_name, out_path, False, msg
        return task_name, error_eval_path, False, msg


def main(argv=None):
    parser = argparse.ArgumentParser(description="Standalone grader (split into env/util modules).")
    subparsers = parser.add_subparsers(dest="command")

    # grade-sample
    p_sample = subparsers.add_parser("grade-sample", help="Grade a single competition submission CSV.")
    p_sample.add_argument("submission", type=str, help="Path to the submission CSV file.")
    p_sample.add_argument("competition_id", type=str, help="Competition ID to grade.")
    p_sample.add_argument("--data-dir", type=str, required=True, help="Data directory.")
    p_sample.add_argument("--competitions-dir", type=str, required=True, help="Path to competitions directory.")
    # New flag: allow a score of 0 to be treated as valid (by default 0 is treated as a bug).
    p_sample.add_argument("--allow-zero-score", action="store_true", help="If set, a score of 0 is NOT treated as a bug.")

    # grade (JSONL)
    p_jsonl = subparsers.add_parser("grade", help="Grade multiple submissions from a JSONL file.")
    p_jsonl.add_argument("--submission", type=str, required=True, help="Path to JSONL of submissions.")
    p_jsonl.add_argument("--output-dir", type=str, required=True, help="Output dir for the aggregated report.")
    p_jsonl.add_argument("--data-dir", type=str, required=True, help="Data directory.")
    p_jsonl.add_argument("--competitions-dir", type=str, required=True, help="Path to competitions directory.")

    # auto-grade
    p_auto = subparsers.add_parser("auto-grade", help="Auto grade submissions discovered from solutions directory.")
    p_auto.add_argument("--task-list", type=str, required=True, help="Path to task_name.txt (one task per line).")
    p_auto.add_argument("--solutions-dir", type=str, default=None, help="Solution Mode 1: Root directory for the old layout: each task has submissions under <solutions-dir>/<task>/code/submission*/submission.csv.",
    )
    p_auto.add_argument("--runs-dir", type=str, default=None, help="Solution Mode 2: Root directory for agent runs: searches each <runs-dir>/<task>_*/ and its save_*/ subdirs for submission/submission.csv and logs/all_nodes/node_*/submission.csv."
    )
    p_auto.add_argument("--data-dir", type=str, required=True, help="Data directory.")
    p_auto.add_argument("--competitions-dir", type=str, required=True, help="Path to competitions directory.")
    
    # New flag: overwrite behavior. By default, skip already evaluated submissions (resume mode).
    p_auto.add_argument("--overwrite", action="store_true", help="Overwrite existing eval_output.json. By default, skip already evaluated submissions.")
    # New options: global error report and parallelism control.
    p_auto.add_argument("--error-report", type=str, default=None, help="Path to write a global error_report.json collecting all error_eval.txt. Optional.")
    p_auto.add_argument("--workers", type=int, default=64, help=f"Number of worker threads for auto-grade (default: 64).")
    p_auto.add_argument("--allow-zero-score", action="store_true", help="If set, a score of 0 is NOT treated as a bug in auto-grade.")

    # Backward compatibility: when no subcommand is provided, treat it as grade-sample.
    parser.add_argument("--as-sample", action="store_true", help=argparse.SUPPRESS)

    args, extras = parser.parse_known_args(argv)

    if args.command is None and args.as_sample is False:
        if len(extras) >= 2:
            # Compatibility: grade.py <submission> <competition_id> --data-dir ... [--competitions-dir ...]
            submission = extras[0]
            competition_id = extras[1]
            sample_argv = ["grade-sample", submission, competition_id] + extras[2:]
            return main(sample_argv)
        else:
            parser.error("Please provide a subcommand (grade-sample/grade) or at least: <submission> <competition_id> --data-dir ...")

    if args.command in ("grade-sample", "grade", "auto-grade"):
        competitions_dir = Path(args.competitions_dir)
        data_dir = Path(args.data_dir)
        registry = SimpleRegistry(data_dir=data_dir, competitions_dir=competitions_dir)

    if args.command == "grade-sample":
        competition = registry.get_competition(args.competition_id)
        submission = Path(args.submission)
        # Whether to treat a score of 0 as a bug depends on the flag.
        treat_zero = not args.allow_zero_score
        report = grade_csv(submission, competition, treat_zero_score_as_bug=treat_zero)
        print(json.dumps(report.to_dict(), indent=4))
        return

    if args.command == "grade":
        # Keep original behavior: in JSONL mode still use the old handling for score 0
        # (you can change it by passing treat_zero_score_as_bug=True).
        submission_jsonl = Path(args.submission)
        output_dir = Path(args.output_dir)
        grade_jsonl(submission_jsonl, output_dir, registry)
        return

    if args.command == "auto-grade":
        # --- 1. Argument validation and mode selection ---
        if args.solutions_dir and args.runs_dir:
            parser.error("Please specify EITHER --solutions-dir OR --runs-dir, not both.")
        if not args.solutions_dir and not args.runs_dir:
            parser.error("You must specify either --solutions-dir (old structure) or --runs-dir (new structure).")

        mode = "solutions" if args.solutions_dir else "runs"
        root_dir = Path(args.solutions_dir) if mode == "solutions" else Path(args.runs_dir)
        task_list_path = Path(args.task_list)

        # Parse the output path for error_report.
        error_report_base: Path | None = None
        if args.error_report:
            error_report_base = Path(args.error_report)
            error_report_base.parent.mkdir(parents=True, exist_ok=True)

        if not task_list_path.is_file():
            parser.error(f"--task-list does not exist: {task_list_path}")
        if not root_dir.is_dir():
            parser.error(f"Root directory does not exist: {root_dir}")

        with open(task_list_path, "r") as f:
            task_names = [line.strip() for line in f if line.strip()]

        # Initialize per-task statistics.
        task_stats: dict[str, dict[str, int]] = {
            tn: {"total": 0, "success": 0, "error": 0} for tn in task_names
        }

        # jobs: (task_name, submission_csv, competition, out_path)
        jobs: list[tuple[str, Path, str, Path]] = []

        # --- 2. Scan tasks (different modes) ---
        
        if mode == "solutions":
            # ================= OLD MODE: solutions-dir =================
            print(f"[INFO] Mode: Solutions (task/code/submission/...) in {root_dir}")
            
            for task in task_names:
                task_dir = root_dir / task / "code"
                if not task_dir.is_dir():
                    print(f"[WARN] Task directory not found, skipping: {task_dir}")
                    continue

                try:
                    competition = registry.get_competition(task)
                except Exception as e:
                    print(f"[WARN] Failed to load competition `{task}`, skipping. Error: {e}")
                    continue

                submission_dirs = sorted([p for p in task_dir.iterdir() if p.is_dir() and p.name.startswith("submission")])
                if not submission_dirs:
                    print(f"[WARN] No submission* directories found under: {task_dir}")
                    continue

                for subdir in submission_dirs:
                    csv_candidates = list(subdir.rglob("submission.csv"))
                    if len(csv_candidates) != 1:
                        print(f"[WARN] Expected exactly one submission.csv in {subdir}, found {len(csv_candidates)}. Skipping.")
                        continue

                    submission_csv = csv_candidates[0]
                    jobs.append((task, submission_csv, task, submission_csv.parent / "eval_output.json"))

        else:
            # ================= NEW MODE: runs-dir (Updated with save_*) =================
            print(f"[INFO] Mode: Runs (agent_runs/task_hash/[save_*]/...) in {root_dir}")
            print(f"[INFO] Scanning directory list...")
            all_run_dirs = [p for p in root_dir.iterdir() if p.is_dir()]
            print(f"[INFO] Found {len(all_run_dirs)} run directories.")

            for task in task_names:
                # Match directories: taskname_xxxx
                target_prefix = f"{task}_"
                matched_runs = [p for p in all_run_dirs if p.name.startswith(target_prefix)]

                if not matched_runs:
                    print(f"[WARN] No run directories found for task: {task} (prefix: {target_prefix})")
                    continue

                try:
                    registry.get_competition(task)
                except Exception as e:
                    print(f"[WARN] Failed to load competition `{task}`, skipping. Error: {e}")
                    continue

                for run_dir in matched_runs:
                    # --- Core change: define list of search root directories ---
                    # 1. run_dir itself (handles submission/logs directly under the run root).
                    # 2. run_dir/save_* (handles submission/logs under save_1, save_2, etc.).
                    search_roots = [run_dir]
                    save_dirs = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("save_")]
                    search_roots.extend(save_dirs)

                    # Search for CSVs in all search roots.
                    csv_candidates: list[Path] = []
                    
                    for root in search_roots:
                        # Path A: .../submission/submission.csv
                        main_sub = root / "submission" / "submission.csv"
                        if main_sub.exists():
                            csv_candidates.append(main_sub)

                        # Path B: .../logs/all_nodes/node_xxx/submission.csv
                        nodes_root = root / "logs" / "all_nodes"
                        if nodes_root.is_dir():
                            for node_dir in nodes_root.iterdir():
                                if node_dir.is_dir():
                                    node_sub = node_dir / "submission.csv"
                                    if node_sub.exists():
                                        csv_candidates.append(node_sub)

                    if not csv_candidates:
                        # Only warn when neither the root nor any save_* directories contain a CSV.
                        # (Log only the run_dir name to keep logs concise.)
                        print(f"[WARN] Run dir (and its saves) found but NO submission.csv: {run_dir.name}")
                        continue

                    for submission_csv in csv_candidates:
                        # Output next to the submission: e.g., save_1/submission/eval_output.json
                        # or save_1/logs/.../eval_output.json.
                        jobs.append((task, submission_csv, task, submission_csv.parent / "eval_output.json"))

        # --- 3. Normalize jobs (deduplicate, clean old files, enqueue) ---
        final_jobs = []
        for (task, submission_csv, comp_id, out_path) in jobs:
            task_stats[task]["total"] += 1

            if out_path.exists() and not args.overwrite:
                task_stats[task]["success"] += 1
                # print(f"[INFO] Skip existing: {out_path}")
                continue

            # Remove stale error_eval.txt if it exists.
            error_eval_path = out_path.parent / "error_eval.txt"
            if error_eval_path.exists():
                try:
                    error_eval_path.unlink()
                except Exception:
                    pass
            
            final_jobs.append((task, submission_csv, comp_id, out_path))

        jobs = final_jobs  # Update to the final list to be executed

        if not jobs:
            print("[INFO] No new submissions to evaluate. Exiting.")
            return

        workers = max(1, int(args.workers))
        print(f"[INFO] Will evaluate {len(jobs)} submissions in parallel (workers={workers}).")

        treat_zero = not args.allow_zero_score
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(_do_grade, t, s, comp_id, o, data_dir, competitions_dir, treat_zero)
                for (t, s, comp_id, o) in jobs
            ]
            # results: list of (task_name, out_path, ok, err)
            results: list[tuple[str, Path, bool, str]] = []
            for fut in as_completed(futures):
                task_name, out_path, ok, err = fut.result()
                results.append((task_name, out_path, ok, err))
                if ok:
                    print(f"[INFO] Saved evaluation result: {out_path}")
                else:
                    print(f"[ERROR] Grading or write failed: {out_path} Error: {err}")

            # --------- Task-level stats: aggregate Total / Success / Error by task ----------
            # results only updates success / error (total was already counted during collection).
            for task_name, _, ok, _ in results:
                s = task_stats.setdefault(task_name, {"total": 0, "success": 0, "error": 0})
                if ok:
                    s["success"] += 1
                else:
                    s["error"] += 1

            # Print table header.
            header = "Task                                               | Total  | Sucess   | Error  "
            sep = "-" * len(header)
            print(header)
            print(sep)

            grand_total = grand_success = grand_error = 0
            for task_name in sorted(task_stats.keys()):
                st = task_stats[task_name]
                grand_total += st["total"]
                grand_success += st["success"]
                grand_error += st["error"]
                print(f"{task_name:<51} | {st['total']:<6} | {st['success']:<7} | {st['error']:<5}")

            print(sep)
            print(f"{'TOTAL':<51} | {grand_total:<6} | {grand_success:<7} | {grand_error:<5}")

            # 5. If --error-report is specified, aggregate all error_eval.txt into a single JSON report.
            if error_report_base is not None:
                error_map: dict[str, str] = {}
                for task_name, out_path, ok, _ in results:
                    if ok:
                        continue
                    # out_path is error_eval_path when the worker returned an error.
                    try:
                        p = Path(out_path)
                        if p.name != "error_eval.txt" or not p.is_file():
                            continue
                        dir_key = str(p.parent.resolve())
                        with open(p, "r", encoding="utf-8") as ef:
                            error_map[dir_key] = ef.read()
                    except Exception:
                        # Ignore individual errors and try to collect as much information as possible.
                        continue

                try:
                    # Insert a summary section at the beginning of error_report.
                    report_obj = {
                        "summary": {
                            task: {
                                "total": st["total"],
                                "success": st["success"],
                                "error": st["error"],
                            }
                            for task, st in task_stats.items()
                        },
                        "errors": error_map,
                    }
                    # Use the user-provided path as the final error report file.
                    error_report_path = error_report_base
                    error_report_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(error_report_path, "w", encoding="utf-8") as rf:
                        json.dump(report_obj, rf, ensure_ascii=False, indent=2)
                    print(f"[INFO] Written global error report to: {error_report_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to write error_report.json: {e}")
        return


if __name__ == "__main__":
    main()
