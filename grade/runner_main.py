# Main runner for skip_bench grading workflows.
import argparse
import json
import os
import sys
import glob  # NEW
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from typing import Dict, List  # NEW

from .util import utils  # type: ignore
from .util import runner  # type: ignore
from .util import report as report_mod  # type: ignore
from .util import count_tokens as count_tokens_mod  # type: ignore


def _write_checkpoint(data: dict, path: str) -> None:
    """
    Atomically write a JSON checkpoint to the given path.
    Callers must pass a task-specific checkpoint path.
    """
    try:
        tmp = f"{path}.tmp"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception as e:
        print(f"warning: failed to write checkpoint {path}: {e}")


def _build_alltask_from_reports(args: argparse.Namespace) -> None:
    """
    Scan each task's report directory, sort by timestamp, and treat
    the same index across tasks as the same run. Generate an all-tasks
    summary report for each run.
    """
    import re
    from collections import defaultdict
    
    task_file = args.task_file
    solutions_dir = args.solutions_dir
    if not solutions_dir:
        print("error: --from-existing-reports requires --solutions-dir")
        sys.exit(2)

    task_names = utils.load_task_names(task_file)
    
    # Step 1: collect all reports per task, sorted by timestamp
    task_reports: Dict[str, List[str]] = {}  # task -> [sorted report paths]
    
    for task in task_names:
        task_dir = os.path.join(solutions_dir, task)
        report_dir = os.path.join(task_dir, "report")
        if not os.path.isdir(report_dir):
            print(f"[from-existing-reports] no report directory for task {task}, skipping")
            continue
        
        pattern = os.path.join(report_dir, "grade_report_*.txt")
        candidates = [p for p in glob.glob(pattern) if os.path.isfile(p)]
        if not candidates:
            continue
        
        # Extract timestamps and sort
        def extract_ts(path: str) -> str:
            fname = os.path.basename(path)
            m = re.search(r'_(\d{8}_\d{6})\.txt$', fname)
            return m.group(1) if m else ""
        
        candidates_with_ts = [(p, extract_ts(p)) for p in candidates if extract_ts(p)]
        if not candidates_with_ts:
            continue
        
        # Sort by timestamp (ascending)
        sorted_reports = [p for p, _ in sorted(candidates_with_ts, key=lambda x: x[1])]
        task_reports[task] = sorted_reports
        print(f"[from-existing-reports] task {task}: found {len(sorted_reports)} reports")
    
    if not task_reports:
        print("[from-existing-reports] no usable reports found; nothing to do.")
        return
    
    # Step 2: determine the maximum number of runs
    max_runs = max(len(reports) for reports in task_reports.values())
    print(f"[from-existing-reports] detected {max_runs} run(s) across all tasks")
    
    # Step 3: for each run index, generate an all-tasks summary report
    for run_idx in range(max_runs):
        print(f"\n[from-existing-reports] processing run #{run_idx}...")
        per_task_info = []
        
        for task in task_names:
            reports = task_reports.get(task, [])
            if run_idx >= len(reports):
                print(f"[from-existing-reports] task {task} has no report at index {run_idx}, skipping")
                continue
            
            rep_path = reports[run_idx]
            info = utils.parse_single_task_report(rep_path)
            if not info.get("task_name"):
                info["task_name"] = task
            per_task_info.append(info)
            print(f"[from-existing-reports] run #{run_idx}, task={task}: using {os.path.basename(rep_path)}")
        
        if not per_task_info:
            print(f"[from-existing-reports] run #{run_idx}: no usable reports found")
            continue
        
        # Extract base parameters from the first task's report
        base_run_params = per_task_info[0]["run_params"] or {}
        ts_obj = datetime.now()
        ts_str = ts_obj.isoformat(timespec="seconds")
        ts_tag = ts_obj.strftime("%Y%m%d_%H%M%S")
        
        # Aggregate metrics
        total_pairs = 0
        sum_acc_weighted = 0.0
        total_multi = 0
        sum_spearman_weighted = 0.0
        
        acc_list = []
        pairs_list = []
        multi_list = []
        spearman_list = []
        
        per_task_lines = []
        for info in per_task_info:
            tname = info["task_name"]
            m = info["metrics"] or {}
            pairs = int(m.get("pairs_count") or 0)
            acc = m.get("accuracy_avg")
            multi = int(m.get("multi_count") or 0)
            sp = m.get("spearman_avg")
            
            total_pairs += pairs
            total_multi += multi
            if isinstance(acc, (int, float)):
                sum_acc_weighted += float(acc) * pairs
                acc_list.append(float(acc))
            if isinstance(sp, (int, float)):
                sum_spearman_weighted += float(sp) * multi
                spearman_list.append(float(sp))
            pairs_list.append(pairs)
            multi_list.append(multi)
            
            per_task_lines.append(
                f"- {tname}: pairs_count={pairs}, accuracy_avg={acc}, multi_count={multi}, spearman_avg={sp}"
            )
        
        overall_record_accuracy = (sum_acc_weighted / total_pairs) if total_pairs > 0 and sum_acc_weighted else None
        overall_record_spearman = (sum_spearman_weighted / total_multi) if total_multi > 0 and sum_spearman_weighted else None
        
        overall_task_pairs_avg = (sum(pairs_list) / len(pairs_list)) if pairs_list else None
        overall_task_acc_avg = (sum(acc_list) / len(acc_list)) if acc_list else None
        overall_task_multi_avg = (sum(multi_list) / len(multi_list)) if multi_list else None
        overall_task_spearman_avg = (sum(spearman_list) / len(spearman_list)) if spearman_list else None
        
        # Build report content
        lines = []
        n_str = base_run_params.get("n") or str(args.n or 0)
        try:
            n_val = int(n_str)
        except Exception:
            n_val = int(args.n or 0)
        lines.append(f"Skip-bench grading metadata summary from existing reports (run #{run_idx}, n={n_val})")
        lines.append("Run parameters (from first task report; may vary slightly across tasks):")
        for k in [
            "task_file",
            "solutions_dir",
            "model",
            "temperature",
            "max_retries",
            "base_url",
            "out_json",
            "report_out",
            "tasks_root",
            "desc_dir",
            "da_dir",
            "parallel",
            "workers",
            "cot",
            "prompt_boost",
            "content_mode",
            "timestamp",
        ]:
            v = base_run_params.get(k, "")
            lines.append(f"- {k}: {v}")
        lines.append(f"- synthesized_at: {ts_str}")
        lines.append(f"- source: existing per-task grade_report_*.txt (run index: {run_idx})")
        lines.append("")
        
        total_tasks = len(per_task_info)
        lines.append(f"Total tasks: {total_tasks}")
        lines.append(f"Total groups (sum over tasks): {total_pairs}")
        lines.append("Overall metrics (record-level, weighted by groups per task):")
        lines.append(f"- pairs_count: {total_pairs}")
        lines.append(f"- accuracy_avg: {overall_record_accuracy}")
        lines.append(f"- multi_count: {total_multi}")
        lines.append(f"- spearman_avg: {overall_record_spearman}")
        lines.append("")
        lines.append("Overall metrics (task-level, averaging per-task summaries):")
        lines.append(f"- pairs_count_avg_over_tasks: {overall_task_pairs_avg}")
        lines.append(f"- accuracy_avg_over_tasks: {overall_task_acc_avg}")
        lines.append(f"- multi_count_avg_over_tasks: {overall_task_multi_avg}")
        lines.append(f"- spearman_avg_over_tasks: {overall_task_spearman_avg}")
        lines.append("")
        lines.append("Per-task metrics:")
        lines.extend(per_task_lines)
        lines.append("")
        
        # Write file
        root_report_dir = os.path.join(solutions_dir, "report")
        os.makedirs(root_report_dir, exist_ok=True)
        model_tag = utils.sanitize_for_filename(base_run_params.get("model") or (args.model or "default"))
        temp_tag = utils.format_temp_for_name(base_run_params.get("temperature") or args.temperature)
        cot_tag = "cot" if str(base_run_params.get("cot") or args.cot).lower() in ("true", "1") else "nocot"
        boost_tag = "pboost" if str(base_run_params.get("prompt_boost") or args.prompt_boost).lower() in ("true", "1") else "std"
        
        global_report_name = (
            f"grade_report_alltasks_from_reports_run{run_idx}_n{int(n_val)}_"
            f"{model_tag}_{temp_tag}_{boost_tag}_{cot_tag}_{ts_tag}.txt"
        )
        global_report_path = os.path.join(root_report_dir, global_report_name)
        with open(global_report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[from-existing-reports] Saved run #{run_idx} metadata report to {global_report_path}")


def _run_single_task(
    task: str,
    args: argparse.Namespace,
    params: dict,
    resumed_task_done_indices_for_task: dict[str, set[int]] | None,
) -> tuple[dict[str, list[dict]], list[dict]]:
    """
        Run a single task in a dedicated thread, including:
            - Resume support (skip completed groups within the task)
            - Serial/parallel processing within a group
            - Update per-task partial.json (if enabled)
            - Return (per_file_results, flat_list)
        Does not directly modify outer aggregated/global_flat_results; caller merges.
    """
    solutions_dir: str = args.solutions_dir
    is_resume_task = resumed_task_done_indices_for_task is not None
    done_indices_for_task = resumed_task_done_indices_for_task or {}

    # groundtruth files
    if args.groundtruth_file:
        if "{task}" in args.groundtruth_file:
            resolved = args.groundtruth_file.format(task=task)
        else:
            resolved = args.groundtruth_file
        if not os.path.isfile(resolved):
            print(f"no groundtruth file for task {task} at {resolved}, skipping")
            return {}, []
        group_files = [resolved]
    else:
        group_files = utils.find_group_files(solutions_dir, task, args.n)
    if not group_files:
        print(f"no group files for task {task} and n={args.n}, skipping")
        return {}, []

    per_file_results: dict[str, list[dict]] = {}
    # If resuming, preload existing partial.json results into per_file_results.
    if is_resume_task:
        task_dir = os.path.join(solutions_dir, task)
        report_dir = os.path.join(task_dir, "report")
        ckpt_path = os.path.join(report_dir, "grade_results_partial.json")
        try:
            with open(ckpt_path, "r", encoding="utf-8") as f:
                ckpt_data = json.load(f)
            if isinstance(ckpt_data, dict):
                for gf_base, res_list in ckpt_data.items():
                    if isinstance(res_list, list):
                        per_file_results[gf_base] = list(res_list)
        except Exception as e:
            print(f"[resume] failed to preload per-file results for task={task}: {e}")

    task_report_dir = os.path.join(solutions_dir, task, "report")
    task_checkpoint_path = os.path.join(task_report_dir, "grade_results_partial.json")

    requested_parallel = int(args.parallel or 1)
    # Group-level parallelism: keep the prior workers definition
    workers = 1 if requested_parallel <= 1 else min(requested_parallel, (os.cpu_count() or 4) * 2)

    ckpt_counter = 0
    CKPT_INTERVAL = 50
    slot_counter = 0  # client_slot counter

    if workers == 1:
        # Serial
        for gf in group_files:
            try:
                with open(gf, "r", encoding="utf-8") as f:
                    groups = json.load(f)
            except Exception as e:
                print(f"could not load {gf}: {e}")
                continue
            gf_base = os.path.basename(gf)
            existing_res_list = list(per_file_results.get(gf_base, []))
            results_this_file = list(existing_res_list)
            done_idx_set = done_indices_for_task.get(gf_base, set())
            for gi, group in enumerate(groups):
                if gi in done_idx_set:
                    continue
                client_slot = slot_counter
                slot_counter += 1
                res = runner.process_group(
                    task_name=task,
                    group_entry=group,
                    solutions_dir=solutions_dir,
                    model=args.model,
                    temperature=args.temperature,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    tasks_root=args.tasks_root,
                    desc_dir=args.desc_dir,
                    da_dir=args.da_dir,
                    raw_data_sample_dir=getattr(args, "raw_data_sample", None),
                    max_retries=args.max_retries,
                    allow_cot=args.cot,
                    prompt_boost=args.prompt_boost,
                    concat_random_task_text=args.concat_random_task_text,
                    client_slot=client_slot,
                )
                res_meta = {"group_file": gf, "group_index": gi, "paths": group.get("paths", [])}
                res_meta.update(res)
                results_this_file.append(res_meta)
                print(f"Processed task={task} group_file={os.path.basename(gf)} index={gi}")
                per_file_results[gf_base] = results_this_file
                if args.partial_json:
                    ckpt_counter += 1
                    if ckpt_counter % CKPT_INTERVAL == 0:
                        os.makedirs(task_report_dir, exist_ok=True)
                        _write_checkpoint(per_file_results, task_checkpoint_path)
            if gf_base not in per_file_results and results_this_file:
                per_file_results[gf_base] = results_this_file
        if args.partial_json and per_file_results:
            os.makedirs(task_report_dir, exist_ok=True)
            _write_checkpoint(per_file_results, task_checkpoint_path)
    else:
        # Parallel groups
        futures = {}
        file_to_groups: dict[str, list[dict]] = {}
        collected_by_file: dict[str, list[tuple[int, dict]]] = {}

        for gf in group_files:
            try:
                with open(gf, "r", encoding="utf-8") as f:
                    groups = json.load(f)
            except Exception as e:
                print(f"could not load {gf}: {e}")
                continue
            file_to_groups[gf] = groups
            gf_base = os.path.basename(gf)
            existing = per_file_results.get(gf_base, [])
            if existing:
                collected_by_file[gf] = [(r.get("group_index", 0), r) for r in existing]

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for gf, groups in file_to_groups.items():
                gf_base = os.path.basename(gf)
                done_idx_set = done_indices_for_task.get(gf_base, set())
                for gi, group in enumerate(groups):
                    if gi in done_idx_set:
                        continue
                    client_slot = slot_counter
                    slot_counter += 1
                    fut = ex.submit(
                        runner.process_group,
                        task_name=task,
                        group_entry=group,
                        solutions_dir=solutions_dir,
                        model=args.model,
                        temperature=args.temperature,
                        api_key=args.api_key,
                        base_url=args.base_url,
                        tasks_root=args.tasks_root,
                        desc_dir=args.desc_dir,
                        da_dir=args.da_dir,
                        raw_data_sample_dir=getattr(args, "raw_data_sample", None),
                        max_retries=args.max_retries,
                        allow_cot=args.cot,
                        prompt_boost=args.prompt_boost,
                        concat_random_task_text=args.concat_random_task_text,
                        client_slot=client_slot,
                    )
                    futures[fut] = (gf, gi, group)

            for fut in as_completed(futures):
                gf, gi, group = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    print(f"Failed processing task={task} group_file={os.path.basename(gf)} index={gi}: {e}")
                    res = {"parsed": {"error": str(e)}, "n": len(group.get("paths", []))}
                res_meta = {"group_file": gf, "group_index": gi, "paths": group.get("paths", [])}
                res_meta.update(res)
                collected_by_file.setdefault(gf, []).append((gi, res_meta))
                print(f"Processed task={task} group_file={os.path.basename(gf)} index={gi}")
                # Rebuild per_file_results for checkpointing
                temp_results: dict[str, list[dict]] = {}
                for gf_p, items in collected_by_file.items():
                    items_sorted = [rm for _, rm in sorted(items, key=lambda x: x[0])]
                    temp_results[os.path.basename(gf_p)] = items_sorted
                per_file_results.update(temp_results)
                if args.partial_json:
                    ckpt_counter += 1
                    if ckpt_counter % CKPT_INTERVAL == 0:
                        os.makedirs(task_report_dir, exist_ok=True)
                        _write_checkpoint(per_file_results, task_checkpoint_path)

        for gf, items in collected_by_file.items():
            items_sorted = [rm for _, rm in sorted(items, key=lambda x: x[0])]
            per_file_results[os.path.basename(gf)] = items_sorted

        if args.partial_json and per_file_results:
            os.makedirs(task_report_dir, exist_ok=True)
            _write_checkpoint(per_file_results, task_checkpoint_path)

    # Build flat_list
    flat_list: list[dict] = []
    for _gf_base, res_list in per_file_results.items():
        flat_list.extend(res_list)
    return per_file_results, flat_list


def run_with_args(args: argparse.Namespace) -> None:
    # from-existing-reports mode
    if args.from_existing_reports:
        _build_alltask_from_reports(args)
        return

    # groundtruth-file validation
    if args.groundtruth_file is not None and int(args.n or 0) == 0:
        print("error: --groundtruth-file requires --n to be non-zero")
        sys.exit(2)

    if args.groundtruth_file and (args.solutions_dir is None or str(args.solutions_dir).strip() == ""):
        print(
            "warning: --solutions-dir is empty while using --groundtruth-file. "
            "This is only safe if 'paths' in your group JSON are absolute."
        )

    # If max_retries is not explicitly provided, set a safe default (e.g., 2).
    # This value applies to:
    #   1) JSON-parse retries in runner.process_group
    #   2) backend.chat_complete retries on any LLM request error
    if getattr(args, "max_retries", None) is None:
        args.max_retries = 2

    task_names = utils.load_task_names(args.task_file)
    if not args.solutions_dir:
        print("error: --solutions-dir is required for normal grading / resume-from-partial modes")
        sys.exit(2)
    solutions_dir = args.solutions_dir

    # Resume: build resumed_task_done_indices (task -> gf_base -> set[gi])
    resume_tasks: set[str] = set()
    resumed_task_done_indices: dict[str, dict[str, set[int]]] = {}
    if args.resume_from_partial:
        for task in task_names:
            task_dir = os.path.join(solutions_dir, task)
            report_dir = os.path.join(task_dir, "report")
            ckpt_path = os.path.join(report_dir, "grade_results_partial.json")
            if not os.path.isfile(ckpt_path):
                continue
            try:
                with open(ckpt_path, "r", encoding="utf-8") as f:
                    ckpt_data = json.load(f)
                if not isinstance(ckpt_data, dict):
                    continue
                done_indices_per_file: dict[str, set[int]] = {}
                for gf_base, res_list in ckpt_data.items():
                    done = set()
                    if isinstance(res_list, list):
                        for r in res_list:
                            try:
                                gi = int(r.get("group_index", 0))
                                done.add(gi)
                            except Exception:
                                continue
                    if done:
                        done_indices_per_file[gf_base] = done
                if done_indices_per_file:
                    resume_tasks.add(task)
                    resumed_task_done_indices[task] = done_indices_per_file
                    print(f"[resume] found per-task checkpoint for task={task} at {ckpt_path}")
            except Exception as e:
                print(f"[resume] failed to load per-task checkpoint {ckpt_path}: {e}")

        if not resume_tasks:
            print("[resume] no per-task checkpoints found; nothing to resume. Exiting.")
            return
        # Only run tasks that have checkpoints
        task_names = [t for t in task_names if t in resume_tasks]
        if not task_names:
            print("[resume] no tasks in task-file match available per-task checkpoints; exiting.")
            return

    # Token count mode (mutually exclusive with resume/from-existing-reports)
    if args.count_tokens:
        total_prompt_tokens = 0
        total_calls = 0
        token_count_method: str = ""
        for task in task_names:
            # ...existing code for group_files and counting...
            if args.groundtruth_file:
                if "{task}" in args.groundtruth_file:
                    resolved = args.groundtruth_file.format(task=task)
                else:
                    resolved = args.groundtruth_file
                if not os.path.isfile(resolved):
                    print(f"no groundtruth file for task {task} at {resolved}, skipping")
                    continue
                group_files = [resolved]
            else:
                group_files = utils.find_group_files(solutions_dir, task, args.n)
            if not group_files:
                print(f"no group files for task {task} and n={args.n}, skipping")
                continue
            for gf in group_files:
                try:
                    with open(gf, "r", encoding="utf-8") as f:
                        groups = json.load(f)
                except Exception as e:
                    print(f"could not load {gf}: {e}")
                    continue
                for gi, group in enumerate(groups):
                    user_prompt_text = runner.prompt_module.build_user_prompt(  # type: ignore[attr-defined]
                        task_name=task,
                        group_entry=group,
                        solutions_dir=solutions_dir,
                        tasks_root=args.tasks_root,
                        desc_dir=args.desc_dir,
                        da_dir=args.da_dir,
                        raw_data_sample_dir=getattr(args, "raw_data_sample", None),
                        allow_cot=args.cot,
                        prompt_boost=args.prompt_boost,
                        concat_random_task_text=args.concat_random_task_text,
                    )
                    system_content = (
                        runner.prompt_module.BENCH_SYSTEM_PROMPT_COT  # type: ignore[attr-defined]
                        if args.cot
                        else runner.prompt_module.BENCH_SYSTEM_PROMPT   # type: ignore[attr-defined]
                    )
                    messages = [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_prompt_text},
                    ]
                    text = "\n".join((m.get("content", "") or "") for m in messages)
                    cnt, method = count_tokens_mod.count_tokens_in_text(
                        text,
                        model=(args.model or "gpt-4"),
                    )
                    total_prompt_tokens += cnt
                    total_calls += 1
                    if method:
                        token_count_method = method
                    print(f"Counted tokens for task={task} group_file={os.path.basename(gf)} index={gi}: {cnt}")
        print("Token usage summary (approximate, prompt side only; dry run, no LLM calls were made):")
        print(f"- total prompt calls: {total_calls}")
        print(f"- total prompt tokens: {total_prompt_tokens}")
        if token_count_method:
            print(f"- counting method: {token_count_method}")
        return

    # Normal grading / resume-from-partial
    requested_parallel = int(args.parallel or 1)
    requested_task_parallel = int(getattr(args, "task_parallel", 32) or 32)
    ts_obj = datetime.now()
    ts_str = ts_obj.isoformat(timespec="seconds")
    ts_tag = ts_obj.strftime("%Y%m%d_%H%M%S")
    params = {
        "task_file": args.task_file,
        "solutions_dir": solutions_dir,
        "n": args.n,
        "model": args.model or "default",
        "temperature": args.temperature,
        "max_retries": args.max_retries,   # Ensure an explicit integer is written
        "base_url": args.base_url,
        # out is only printed in reports; it is no longer a real file path
        "out": None,
        "report_out": args.report_out,
        "tasks_root": args.tasks_root,
        "desc_dir": args.desc_dir,
        "da_dir": args.da_dir,
        "raw_data_sample_dir": getattr(args, "raw_data_sample", None),
        "timestamp": ts_str,
        "timestamp_tag": ts_tag,
        "parallel": requested_parallel,
        "workers": None,  # Not used for report output; leave empty
        "cot": bool(args.cot),
        "groundtruth_file": args.groundtruth_file,
        "prompt_boost": bool(args.prompt_boost),
        "concat_random_task_text": bool(args.concat_random_task_text),
    }

    global_flat_results: dict[str, list[dict]] = {}

    # === Task-level parallelism: process per task ===
    task_workers = 1 if requested_task_parallel <= 1 else min(requested_task_parallel, len(task_names))
    if task_workers <= 1:
        # Serial tasks
        for task in task_names:
            per_file_results, flat_list = _run_single_task(
                task,
                args=args,
                params=params,
                resumed_task_done_indices_for_task=resumed_task_done_indices.get(task),
            )
            if not flat_list:
                continue
            # per-task JSON & report
            per_task_dir = os.path.join(solutions_dir, "per_task_results")
            os.makedirs(per_task_dir, exist_ok=True)
            per_task_json = os.path.join(
                per_task_dir,
                # Append a timestamp to the filename
                f"grade_results_{utils.sanitize_for_filename(task)}_n{int(args.n)}_{ts_tag}.json",
            )
            with open(per_task_json, "w", encoding="utf-8") as f:
                json.dump({task: per_file_results}, f, ensure_ascii=False, indent=2)
            print(f"Saved per-task results to {per_task_json}")

            aggregated_for_task = {task: flat_list}
            params_task = dict(params)
            params_task["single_task"] = True
            report_mod.write_reports(aggregated=aggregated_for_task, params=params_task)
            global_flat_results[task] = flat_list
    else:
        # Task-level multiprocessing parallelism
        with concurrent.futures.ProcessPoolExecutor(max_workers=task_workers) as ex:
            futures = {}
            for task in task_names:
                fut = ex.submit(
                    _run_single_task,
                    task,
                    args,
                    params,
                    resumed_task_done_indices.get(task),
                )
                futures[fut] = task

            for fut in concurrent.futures.as_completed(futures):
                task = futures[fut]
                try:
                    per_file_results, flat_list = fut.result()
                except Exception as e:
                    print(f"[task-level] task={task} failed: {e}")
                    per_file_results, flat_list = {}, []
                if not flat_list:
                    continue
                # Persist per-task JSON & report immediately after completion
                per_task_dir = os.path.join(solutions_dir, "per_task_results")
                os.makedirs(per_task_dir, exist_ok=True)
                per_task_json = os.path.join(
                    per_task_dir,
                    # Also append a timestamp
                    f"grade_results_{utils.sanitize_for_filename(task)}_n{int(args.n)}_{ts_tag}.json",
                )
                with open(per_task_json, "w", encoding="utf-8") as f:
                    json.dump({task: per_file_results}, f, ensure_ascii=False, indent=2)
                print(f"Saved per-task results to {per_task_json}")

                aggregated_for_task = {task: flat_list}
                params_task = dict(params)
                params_task["single_task"] = True
                report_mod.write_reports(aggregated=aggregated_for_task, params=params_task)
                global_flat_results[task] = flat_list

    # === Global all-tasks report (metadata-only) ===
    print("Per-task JSON files saved. Global meta report:")
    if global_flat_results:
        params_global = dict(params)
        params_global["single_task"] = False
        report_mod.write_reports(aggregated=global_flat_results, params=params_global)
