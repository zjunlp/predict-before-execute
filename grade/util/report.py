# Report builders and writers for skip_bench grading.
import os
from typing import Any, Dict, List
from . import utils
from . import prompt as prompt_module  # new: reuse loader to detect content mode
from . import print_detail  # new: generate alignment JSON right after single-task report

def aggregate_metrics(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate metrics across a list of per-group results.
    Returns a dict with overall counts and averaged metrics.
    """
    summary: Dict[str, Any] = {}
    # Only count numeric accuracy; filter out None/other types
    acc_vals = [
        r.get("accuracy")
        for r in all_results
        if r.get("n") == 2 and isinstance(r.get("accuracy"), (int, float))
    ]
    summary["pairs_count"] = len(acc_vals)
    summary["accuracy_avg"] = (sum(acc_vals) / len(acc_vals)) if acc_vals else None

    # Likewise, only count numeric spearman values
    spearman_vals = [
        r.get("spearman_rho")
        for r in all_results
        if (r.get("n", 0) or 0) > 2 and isinstance(r.get("spearman_rho"), (int, float))
    ]
    summary["multi_count"] = len(spearman_vals)
    summary["spearman_avg"] = (sum(spearman_vals) / len(spearman_vals)) if spearman_vals else None

    # Aggregate precision@k
    precision_accum: Dict[int, List[float]] = {}
    for r in all_results:
        prec = r.get("precision")
        if isinstance(prec, dict):
            for k_str, v in prec.items():
                try:
                    k = int(k_str)
                    precision_accum.setdefault(k, []).append(float(v))
                except Exception:
                    continue
    summary["precision_avg"] = {k: (sum(vs) / len(vs)) for k, vs in sorted(precision_accum.items())} if precision_accum else {}
    return summary

def build_report(aggregated: Dict[str, List[Dict[str, Any]]], n_arg: int, params: Dict[str, Any]) -> str:
    """
    Build a human-readable report:
      - Run parameters
      - Overall aggregated metrics
      - Per-task metrics
      - Full LLM interaction logs for all groups (all attempts)
    """
    lines: List[str] = []
    # Header + parameters
    lines.append(f"Skip-bench grading report (n={n_arg})")
    lines.append("Run parameters:")
    lines.append(f"- task_file: {params.get('task_file')}")
    lines.append(f"- solutions_dir: {params.get('solutions_dir')}")
    lines.append(f"- model: {params.get('model')}")
    lines.append(f"- temperature: {params.get('temperature')}")
    lines.append(f"- max_retries: {params.get('max_retries')}")
    lines.append(f"- base_url: {params.get('base_url')}")
    lines.append(f"- out_json: {params.get('out')}")
    lines.append(f"- report_out: {params.get('report_out')}")
    lines.append(f"- tasks_root: {params.get('tasks_root')}")
    lines.append(f"- desc_dir: {params.get('desc_dir')}")
    lines.append(f"- da_dir: {params.get('da_dir')}")
    lines.append(f"- raw_data_sample_dir: {params.get('raw_data_sample_dir')}")
    lines.append(f"- parallel: {params.get('parallel')}")
    lines.append(f"- workers: {params.get('workers')}")
    lines.append(f"- cot: {params.get('cot')}")
    lines.append(f"- prompt_boost: {params.get('prompt_boost')}")
    # For single-task reports, content_mode must be one of: data-both/data-desc-only/data-ana-only/none-task-data
    lines.append(f"- content_mode: {params.get('content_mode')}")
    lines.append(f"- timestamp: {params.get('timestamp')}")
    lines.append("")

    # Overall metrics
    all_results = [r for results in aggregated.values() for r in results]
    total_groups = len(all_results)
    lines.append(f"Total tasks: {len(aggregated)}")
    lines.append(f"Total groups: {total_groups}")
    overall = aggregate_metrics(all_results)
    lines.append("Overall metrics:")
    lines.append(f"- pairs_count: {overall.get('pairs_count')}")
    lines.append(f"- accuracy_avg: {overall.get('accuracy_avg')}")
    lines.append(f"- multi_count: {overall.get('multi_count')}")
    lines.append(f"- spearman_avg: {overall.get('spearman_avg')}")
    if overall.get("precision_avg"):
        prec_str = ", ".join([f"precision@{k}: {v:.4f}" for k, v in overall["precision_avg"].items()])
        lines.append(f"- {prec_str}")
    lines.append("")

    # Per-task summaries (aggregated only; no per-group summaries)
    lines.append("Per-task metrics:")
    for task, results in aggregated.items():
        summ = aggregate_metrics(results)
        lines.append(f"- {task}: pairs_count={summ.get('pairs_count')}, accuracy_avg={summ.get('accuracy_avg')}, multi_count={summ.get('multi_count')}, spearman_avg={summ.get('spearman_avg')}")
        if summ.get("precision_avg"):
            pks = ", ".join([f"precision@{k}: {v:.4f}" for k, v in summ["precision_avg"].items()])
            lines.append(f"  {pks}")
    lines.append("")

    # Full LLM interaction logs (ordered deterministically per task)
    lines.append("LLM interaction logs:")
    for task, results in aggregated.items():
        lines.append(f"[Task] {task}")
        sorted_results = sorted(
            results,
            key=lambda r: (os.path.basename(r.get("group_file", "")), int(r.get("group_index", 0) or 0)),
        )
        for r in sorted_results:
            gf = os.path.basename(r.get("group_file", ""))
            gi = r.get("group_index")
            n = r.get("n")
            lines.append(f"- group_file={gf} index={gi} n={n}")
            interactions = r.get("interaction_log", [])
            if interactions:
                for i, it in enumerate(interactions, 1):
                    lines.append(f"  [attempt {i}]")
                    msgs = it.get("messages", [])
                    for m in msgs:
                        role = m.get("role", "")
                        content = m.get("content", "") or ""
                        lines.append(f"    [{role}]")
                        lines.append(content)
                    lines.append("    [assistant]")
                    lines.append(it.get("response", "") or "")
                    lines.append("")
            else:
                raws = r.get("raw_responses") or ([r.get("raw_response", "")] if "raw_response" in r else [])
                for i, raw in enumerate(raws, 1):
                    lines.append(f"  [attempt {i}]")
                    lines.append("    [assistant]")
                    lines.append(raw if raw is not None else "")
                    lines.append("")
        lines.append("")
    return "\n".join(lines)

# new: metadata-only report without LLM interaction logs
def build_metadata_report(
    aggregated: Dict[str, List[Dict[str, Any]]],
    n_arg: int,
    params: Dict[str, Any],
) -> str:
    """
    Build a compact report containing:
      - Run parameters
      - Overall aggregated metrics (record-level and task-level)
      - Per-task aggregated metrics
      - Per-task content modes (if provided via params["task_modes"])
    Does NOT include per-group summaries or LLM interaction logs.
    """
    lines: List[str] = []
    lines.append(f"Skip-bench grading metadata summary (n={n_arg})")
    lines.append("Run parameters:")
    lines.append(f"- task_file: {params.get('task_file')}")
    lines.append(f"- solutions_dir: {params.get('solutions_dir')}")
    lines.append(f"- model: {params.get('model')}")
    lines.append(f"- temperature: {params.get('temperature')}")
    lines.append(f"- max_retries: {params.get('max_retries')}")
    lines.append(f"- base_url: {params.get('base_url')}")
    lines.append(f"- out_json: {params.get('out')}")
    lines.append(f"- report_out: {params.get('report_out')}")
    lines.append(f"- tasks_root: {params.get('tasks_root')}")
    lines.append(f"- desc_dir: {params.get('desc_dir')}")
    lines.append(f"- da_dir: {params.get('da_dir')}")
    lines.append(f"- raw_data_sample_dir: {params.get('raw_data_sample_dir')}")
    lines.append(f"- parallel: {params.get('parallel')}")
    lines.append(f"- workers: {params.get('workers')}")
    lines.append(f"- cot: {params.get('cot')}")
    lines.append(f"- prompt_boost: {params.get('prompt_boost')}")
    # Keep a high-level content_mode here (e.g., mixed); list per-task modes below
    lines.append(f"- content_mode: {params.get('content_mode')}")
    lines.append(f"- timestamp: {params.get('timestamp')}")
    lines.append("")

    # ===== Overall metrics: record-level =====
    all_results = [r for results in aggregated.values() for r in results]
    total_tasks = len(aggregated)
    total_groups = len(all_results)
    lines.append(f"Total tasks: {total_tasks}")
    lines.append(f"Total groups: {total_groups}")
    overall_record = aggregate_metrics(all_results)
    lines.append("Overall metrics (record-level, over all groups):")
    lines.append(f"- pairs_count: {overall_record.get('pairs_count')}")
    lines.append(f"- accuracy_avg: {overall_record.get('accuracy_avg')}")
    lines.append(f"- multi_count: {overall_record.get('multi_count')}")
    lines.append(f"- spearman_avg: {overall_record.get('spearman_avg')}")
    if overall_record.get("precision_avg"):
        prec_str = ", ".join([f"precision@{k}: {v:.4f}" for k, v in overall_record["precision_avg"].items()])
        lines.append(f"- {prec_str}")
    lines.append("")

    # ===== Overall metrics: task-level =====
    task_summaries: List[Dict[str, Any]] = []
    for _, results in aggregated.items():
        task_summaries.append(aggregate_metrics(results))

    def _avg_metric(key: str):
        vals = [ts[key] for ts in task_summaries if ts.get(key) is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    overall_task: Dict[str, Any] = {}
    overall_task["pairs_count_avg"] = _avg_metric("pairs_count")
    overall_task["accuracy_avg"] = _avg_metric("accuracy_avg")
    overall_task["multi_count_avg"] = _avg_metric("multi_count")
    overall_task["spearman_avg"] = _avg_metric("spearman_avg")

    prec_by_k: Dict[int, List[float]] = {}
    for ts in task_summaries:
        prec_map = ts.get("precision_avg") or {}
        for k, v in prec_map.items():
            try:
                k_int = int(k)
                prec_by_k.setdefault(k_int, []).append(float(v))
            except Exception:
                continue
    precision_task_avg = {
        k: (sum(vs) / len(vs)) for k, vs in sorted(prec_by_k.items())
    } if prec_by_k else {}

    lines.append("Overall metrics (task-level, averaging per-task summaries):")
    lines.append(f"- pairs_count_avg_over_tasks: {overall_task.get('pairs_count_avg')}")
    lines.append(f"- accuracy_avg_over_tasks: {overall_task.get('accuracy_avg')}")
    lines.append(f"- multi_count_avg_over_tasks: {overall_task.get('multi_count_avg')}")
    lines.append(f"- spearman_avg_over_tasks: {overall_task.get('spearman_avg')}")
    if precision_task_avg:
        prec_str = ", ".join([f"precision@{k}: {v:.4f}" for k, v in precision_task_avg.items()])
        lines.append(f"- {prec_str}")
    lines.append("")

    # ===== Per-task metrics =====
    lines.append("Per-task metrics:")
    for task, results in aggregated.items():
        summ = aggregate_metrics(results)
        lines.append(
            f"- {task}: pairs_count={summ.get('pairs_count')}, "
            f"accuracy_avg={summ.get('accuracy_avg')}, "
            f"multi_count={summ.get('multi_count')}, "
            f"spearman_avg={summ.get('spearman_avg')}"
        )
        if summ.get("precision_avg"):
            pks = ", ".join([f"precision@{k}: {v:.4f}" for k, v in summ["precision_avg"].items()])
            lines.append(f"  {pks}")
    lines.append("")

    # ===== Per-task content modes =====
    task_modes: Dict[str, str] = params.get("task_modes") or {}
    if task_modes:
        lines.append("Per-task content modes:")
        for task in sorted(task_modes.keys()):
            lines.append(f"- {task}: {task_modes[task]}")
        lines.append("")

    return "\n".join(lines)

def write_reports(aggregated: Dict[str, Any], params: Dict[str, Any]) -> None:
    """
    Write per-task reports (when aggregated is per-groundtruth or already flattened)
    and, optionally, a run-level metadata-only report under <solutions_dir>/report.

    aggregated can be either:
      - { task: { group_file_basename: [results...] } }  (per-groundtruth, multi-file)
      - { task: [results...] }                           (already flattened per-task)
    """
    if params.get("report_out"):
        # Build a single combined text with all tasks/files
        text = "Combined report is not supported in per-groundtruth mode."
        report_out = params["report_out"]
        base, ext = os.path.splitext(report_out)
        if not ext:
            report_out = f"{report_out}.txt"
        with open(report_out, "w", encoding="utf-8") as fout:
            fout.write(text)
        print(f"Saved combined report to {report_out}")
        return

    model_tag = utils.sanitize_for_filename(params.get("model") or "default")
    temp_tag = utils.format_temp_for_name(params.get("temperature"))
    cot_tag = "cot" if params.get("cot") else "nocot"
    boost_tag = "pboost" if params.get("prompt_boost") else "std"
    ts_tag = params.get("timestamp_tag") or "ts"

    global_flat: Dict[str, List[Dict[str, Any]]] = {}
    # new: record each task's content_mode for global metadata report
    task_modes: Dict[str, str] = {}

    for task, value in aggregated.items():
        if isinstance(value, list):
            # already flat for this task
            flat_results_for_task: List[Dict[str, Any]] = list(value)
            global_flat[task] = flat_results_for_task

            if flat_results_for_task:
                solutions_dir = params.get("solutions_dir") or "."
                task_dir = os.path.join(solutions_dir, task)
                report_dir = os.path.join(task_dir, "report")
                os.makedirs(report_dir, exist_ok=True)

                try:
                    n_local = int(params.get("n") or 0)
                except Exception:
                    n_local = 0

                # For flat single-task reports, infer content_mode from actual resources
                resources = prompt_module.load_task_resources(
                    task_name=task,
                    tasks_root=params.get("tasks_root"),
                    desc_dir=params.get("desc_dir"),
                    da_dir=params.get("da_dir"),
                    raw_data_sample_dir=params.get("raw_data_sample_dir"),
                )
                has_desc = bool((resources.get("task_desc") or "").strip())
                has_da = bool((resources.get("data_analysis") or "").strip())
                has_raw = bool((resources.get("raw_data_sample") or "").strip())
                
                if has_desc and (has_da or has_raw):
                    if resources.get("mode") == "raw_data_sample":
                        mode_str = "data-desc-rawsample"
                    else:
                        mode_str = "data-both"
                elif has_desc:
                    mode_str = "data-desc-only"
                elif has_da:
                    mode_str = "data-ana-only"
                elif has_raw:
                    mode_str = "data-rawsample-only"
                else:
                    mode_str = "none-task-data"
                mode_tag = utils.sanitize_for_filename(mode_str)
                task_modes[task] = mode_str  # keep raw string for all-tasks report

                report_name = (
                    f"grade_report_{task}_n{int(n_local)}_"
                    f"{mode_tag}_{model_tag}_{temp_tag}_{boost_tag}_{cot_tag}_{ts_tag}.txt"
                )
                report_path = os.path.join(report_dir, report_name)

                params_task = dict(params)
                params_task["content_mode"] = mode_str
                text = build_report({task: flat_results_for_task}, int(n_local), params_task)
                with open(report_path, "w", encoding="utf-8") as fout:
                    fout.write(text)
                print(f"Saved report to {report_path}")

                # new: immediately generate alignment JSON for this just-written report
                try:
                    # Force overwrite to keep alignment.json in sync with the new report
                    print_detail.generate_alignment_for_log(
                        log_path=report_path,
                        root_dir=solutions_dir,
                        overwrite=True,  # always regenerate to match new report
                    )
                except Exception:
                    # best-effort; do not break main grading/reporting pipeline
                    pass

            continue

        # value is { gf_base: [results...] }: write per-groundtruth reports and accumulate into global_flat
        flat_results_for_task: List[Dict[str, Any]] = []
        # Compute content_mode for the current task (shared by all groundtruth files)
        resources = prompt_module.load_task_resources(
            task_name=task,
            tasks_root=params.get("tasks_root"),
            desc_dir=params.get("desc_dir"),
            da_dir=params.get("da_dir"),
            raw_data_sample_dir=params.get("raw_data_sample_dir"),
        )
        has_desc = bool((resources.get("task_desc") or "").strip())
        has_da = bool((resources.get("data_analysis") or "").strip())
        has_raw = bool((resources.get("raw_data_sample") or "").strip())
        
        if has_desc and (has_da or has_raw):
            if resources.get("mode") == "raw_data_sample":
                mode_str = "data-desc-rawsample"
            else:
                mode_str = "data-both"
        elif has_desc:
            mode_str = "data-desc-only"
        elif has_da:
            mode_str = "data-ana-only"
        elif has_raw:
            mode_str = "data-rawsample-only"
        else:
            mode_str = "none-task-data"
        mode_tag = utils.sanitize_for_filename(mode_str)
        task_modes[task] = mode_str

        for gf_base, results in value.items():
            flat_results_for_task.extend(results)

            # derive n_local from filename suffix: _n{num}.json
            try:
                name = os.path.splitext(gf_base)[0]
                n_part = name.split("_n")[-1]
                n_local = int(n_part)
            except Exception:
                n_local = params.get("n") or 0

            solutions_dir = params.get("solutions_dir") or "."
            task_dir = os.path.join(solutions_dir, task)
            report_dir = os.path.join(task_dir, "report")
            os.makedirs(report_dir, exist_ok=True)

            report_name = (
                f"grade_report_{task}_n{int(n_local)}_"
                f"{mode_tag}_{model_tag}_{temp_tag}_{boost_tag}_{cot_tag}_{ts_tag}.txt"
            )
            report_path = os.path.join(report_dir, report_name)

            params_task = dict(params)
            params_task["content_mode"] = mode_str
            text = build_report({task: results}, int(n_local), params_task)
            with open(report_path, "w", encoding="utf-8") as fout:
                fout.write(text)
            print(f"Saved report to {report_path}")

        if flat_results_for_task:
            global_flat[task] = flat_results_for_task

    # If single-task mode is declared or global_flat has only one task, skip all-tasks report
    single_task_mode = bool(params.get("single_task"))
    if not global_flat or single_task_mode or len(global_flat) <= 1:
        return

    # Write global metadata-only report: solutions_dir/report/grade_report_alltasks_...
    solutions_dir = params.get("solutions_dir") or "."
    root_report_dir = os.path.join(solutions_dir, "report")
    os.makedirs(root_report_dir, exist_ok=True)

    try:
        n_arg = int(params.get("n") or 0)
    except Exception:
        n_arg = 0

    # new: derive global content_mode from per-task modes
    # If all task mode_str values are identical, use it; otherwise use "mixed"
    unique_modes = {m for m in task_modes.values() if m}
    if len(unique_modes) == 1:
        global_mode = next(iter(unique_modes))
    else:
        global_mode = "mixed"

    params_global = dict(params)
    params_global["content_mode"] = global_mode
    params_global["task_modes"] = task_modes

    global_report_name = (
        f"grade_report_alltasks_n{int(n_arg)}_{model_tag}_{temp_tag}_{boost_tag}_{cot_tag}_{ts_tag}.txt"
    )
    global_report_path = os.path.join(root_report_dir, global_report_name)
    text = build_metadata_report(global_flat, n_arg, params_global)
    with open(global_report_path, "w", encoding="utf-8") as fout:
        fout.write(text)
    print(f"Saved global metadata report to {global_report_path}")
