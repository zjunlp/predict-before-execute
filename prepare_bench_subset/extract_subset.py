"""Extract a subset of solutions for each task based on tasks.json and optionally
rebuild rich evaluation artifacts (ground truth groups, alignments, and reports)
for tasks that have them.
"""

import json
import os
import random
import shutil
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple
import argparse
import glob
import re
from datetime import datetime
import time

# Fixed random seed for reproducibility
random.seed(42)


def load_json(path: str) -> Any:
    """
    Load JSON file. If plain json.load fails (e.g., due to // comments),
    fall back to a simple JSON-with-comments loader that strips // comments.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: support // line comments (jsonc-like)
        cleaned_lines: List[str] = []
        for line in text.splitlines():
            stripped = line.lstrip()
            # skip full-line // comments
            if stripped.startswith("//"):
                continue
            # strip inline // comments, but keep content before //
            # this is simple and assumes // is only used for comments
            if "//" in line:
                line = line.split("//", 1)[0]
            cleaned_lines.append(line)
        cleaned_text = "\n".join(cleaned_lines)
        return json.loads(cleaned_text)


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_annotation_index(
    annotations: Dict[str, List[List[str]]]
) -> Dict:
    """
    Build a nested index to lookup solution_id lists by keyword path.
    Index structure example:
    {
      "__all__": {solution_ids...},  # root level
      "RoBERTa": {
        "__files__": {solution_ids...},   # files matching this keyword at this level
        "__all__": {solution_ids...},     # all files under this node
        ...
      },
      "MultiModal Transformer": {
        "__files__": {...},
        "__all__": {...},
        "MultiModal Model": {
          "__files__": {...},
          "__all__": {...},
          "Longformer-base Embeddings": {
            "__files__": {...},
            "__all__": {...}
          }
        }
      }
    }
    """
    index: Dict[str, Any] = {"__all__": set()}

    for sol_id, lists in annotations.items():
        for kw_list in lists:
            # kw_list is a keyword path
            node = index
            # root level __all__
            node.setdefault("__all__", set()).add(sol_id)
            for kw in kw_list:
                if kw not in node:
                    node[kw] = {"__files__": set(), "__all__": set()}
                node = node[kw]
                node.setdefault("__files__", set()).add(sol_id)
                node.setdefault("__all__", set()).add(sol_id)
    return index


def get_node_by_path(index: Dict, path: List[str]) -> Dict:
    node = index
    for kw in path:
        if kw not in node:
            return {}
        node = node[kw]
    return node


def sample_from_node(
    node: Dict,
    n: int,
    global_selected: Set[str],
    node_used: Set[str],
) -> List[str]:
    """
    Sample n solution_ids from a given index node that are not yet selected.
    Use node["__all__"] which is more permissive than "__files__".
    """
    all_files: Set[str] = set(node.get("__all__", []))
    # exclude globally selected and node-used files
    candidates = list(all_files - global_selected - node_used)
    if not candidates or n <= 0:
        return []
    random.shuffle(candidates)
    chosen = candidates[: min(n, len(candidates))]
    return chosen


def select_for_spec(
    index: Dict,
    spec: Any,
    path_keys: List[str],
    global_selected: Set[str],
    node_used_map: Dict[Tuple[str, ...], Set[str]],
) -> None:
    """
    Recursively select solution_ids according to spec (structure from tasks.json).
    path_keys indicates the current keyword path.
    node_used_map maps each path tuple to a set of files already used for that path.
    """
    if isinstance(spec, int):
        # Leaf node: sample spec items from the area indicated by path_keys
        n = spec
        if n <= 0:
            return

        # Handle "other": if last key is "other", use its parent node for sampling
        is_other = bool(path_keys and path_keys[-1] == "other")
        if is_other:
            real_path = path_keys[:-1]
        else:
            real_path = path_keys

        node = get_node_by_path(index, real_path)
        if not node:
            return

        path_tuple = tuple(real_path)
        node_used = node_used_map.setdefault(path_tuple, set())

        chosen = sample_from_node(node, n, global_selected, node_used)
        node_used.update(chosen)
        global_selected.update(chosen)
        return

    if isinstance(spec, dict):
        for key, sub_spec in spec.items():
            # recurse into next level
            select_for_spec(
                index=index,
                spec=sub_spec,
                path_keys=path_keys + [key],
                global_selected=global_selected,
                node_used_map=node_used_map,
            )


def _build_keywords_by_rank(annotations: Dict[str, List[List[str]]]) -> Dict[str, Dict[str, int]]:
    """
    Build a mapping rank_k -> { keyword -> count } from per-solution annotations.
    """
    rank_counts: Dict[int, Dict[str, int]] = {}
    for _, levels in annotations.items():
        if not isinstance(levels, list):
            continue
        flat: List[str] = []
        for level in levels:
            if isinstance(level, list):
                for phrase in level:
                    if isinstance(phrase, str):
                        flat.append(phrase)
        for idx, kw in enumerate(flat):
            rank = idx + 1
            bucket = rank_counts.setdefault(rank, {})
            bucket[kw] = bucket.get(kw, 0) + 1

    result: Dict[str, Dict[str, int]] = {}
    for rank, counts in sorted(rank_counts.items()):
        sorted_items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        rank_key = f"rank_{rank}"
        result[rank_key] = {kw: cnt for kw, cnt in sorted_items}
    return result


def copy_solution_files(
    task_name: str,
    solution_id: str,
    src_task_dir: str,
    dst_task_dir: str,
) -> None:
    # code
    src_code = os.path.join(src_task_dir, "code", f"{solution_id}.py")
    dst_code_dir = os.path.join(dst_task_dir, "code")
    ensure_dir(dst_code_dir)
    dst_code = os.path.join(dst_code_dir, f"{solution_id}.py")

    if os.path.exists(src_code):
        shutil.copy2(src_code, dst_code)
    else:
        print(f"[WARN] Code file missing: {src_code}")

    # output: solution_... -> output_... .txt
    if solution_id.startswith("solution_"):
        suffix = solution_id[len("solution_") :]
    else:
        suffix = solution_id
    output_filename = f"output_{suffix}.txt"

    src_output = os.path.join(src_task_dir, "output", output_filename)
    dst_output_dir = os.path.join(dst_task_dir, "output")
    ensure_dir(dst_output_dir)
    dst_output = os.path.join(dst_output_dir, output_filename)

    if os.path.exists(src_output):
        shutil.copy2(src_output, dst_output)
    else:
        print(f"[WARN] Output file missing: {src_output}")

    # NEW: copy submission directory if present: code/submission_<solution_id>/
    if solution_id.startswith("solution_"):
        sub_dir_name = f"submission_{solution_id}"
    else:
        sub_dir_name = f"submission_{solution_id}"
    src_sub_dir = os.path.join(src_task_dir, "code", sub_dir_name)
    dst_sub_dir = os.path.join(dst_code_dir, sub_dir_name)
    if os.path.isdir(src_sub_dir):
        # merge-style copy: recreate tree under dst_sub_dir
        for root, dirs, files in os.walk(src_sub_dir):
            rel = os.path.relpath(root, src_sub_dir)
            cur_dst = os.path.join(dst_sub_dir, rel) if rel != "." else dst_sub_dir
            ensure_dir(cur_dst)
            for fn in files:
                src_f = os.path.join(root, fn)
                dst_f = os.path.join(cur_dst, fn)
                shutil.copy2(src_f, dst_f)
    else:
        # There are annotations but no submission directory: warn but do not drop this solution.
        print(f"[WARN] Submission dir missing: {src_sub_dir}")


# ---------- Rich-directory related helpers ----------

def _short_tail_from_path(path: str) -> str:
    base = os.path.basename(path)
    m = re.search(r"([0-9a-fA-F]+)\.py$", base)
    if m:
        token = m.group(1)
        return f"{token[-4:]}.py" if len(token) >= 4 else f"{token}.py"
    return base


def _solution_id_from_path(path: str) -> str:
    """
    /.../code/solution_xxx_run_xxx.py -> solution_xxx_run_xxx
    """
    base = os.path.basename(path)
    if base.endswith(".py"):
        base = base[:-3]
    return base


def _find_rich_files_for_task(src_task_dir: str, task_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Collect all rich files for this task for different n values:

      {
        n_tag (str): {
          "groups": groups_path,
          "reports": [...],
          "alignments": [...]
        }
      }

    The groups subset is not guaranteed to be non-empty; that will be filtered later.
    """
    result: Dict[str, Dict[str, Any]] = {}
    gt_dir = os.path.join(src_task_dir, "ground_truth")
    rep_dir = os.path.join(src_task_dir, "report")
    if not (os.path.isdir(gt_dir) and os.path.isdir(rep_dir)):
        return {}

    # groups_<task>_n*.json
    for gp in glob.glob(os.path.join(gt_dir, f"groups_{task_name}_n*.json")):
        base = os.path.basename(gp)
        m = re.search(r"_n(\d+)\.json$", base)
        if not m:
            continue
        n_tag = f"n{m.group(1)}"
        item = result.setdefault(n_tag, {"groups": gp, "reports": [], "alignments": []})
        item["groups"] = gp

    # grade_report / alignment
    for rp in glob.glob(os.path.join(rep_dir, "grade_report_*.txt")):
        rp_base = os.path.basename(rp)
        # Reuse the rule from print_detail: the first nX segment after the task name.
        parts = rp_base
        if parts.startswith("grade_report_"):
            parts = parts[len("grade_report_") :]
        parts_list = parts.split("_")
        n_tag = None
        for p in parts_list:
            if len(p) >= 2 and p[0] == "n" and p[1].isdigit():
                n_tag = p
                break
        if not n_tag:
            continue
        align_name = "alignment_" + rp_base[len("grade_report_") :]
        align_name = re.sub(r"\.txt$", ".json", align_name)
        ap = os.path.join(rep_dir, align_name)
        if os.path.exists(ap):
            item = result.setdefault(n_tag, {"groups": None, "reports": [], "alignments": []})
            item["reports"].append(rp)
            item["alignments"].append(ap)
    # Remove n_tags that do not have a complete triple (groups + reports + alignments).
    cleaned: Dict[str, Dict[str, Any]] = {}
    for n_tag, info in result.items():
        if info.get("groups") and info.get("reports") and info.get("alignments"):
            cleaned[n_tag] = info
    return cleaned


def _build_groundtruth_subset(
    groups_path: str,
    selected_solutions: Set[str],
    dst_gt_dir: str,
    subset_code_dir: str,
) -> Tuple[str, Dict[frozenset, int]]:
    """
    Filter groups_path using selected_solutions and write the subset to dst_gt_dir.

    Returns:
      - subset_groups_path
      - mapping: tails_set -> new_group_index (used to align alignment data)
    """
    groups = load_json(groups_path)
    new_groups: List[Dict[str, Any]] = []
    tails_to_idx: Dict[frozenset, int] = {}

    for g in groups:
        paths = g.get("paths", [])
        sol_ids = [_solution_id_from_path(p) for p in paths]
        if not sol_ids:
            continue
        if not all(sid in selected_solutions for sid in sol_ids):
            continue
        # 构造 subset 路径
        new_paths = [
            os.path.join(subset_code_dir, os.path.basename(p)) for p in paths
        ]
        g2 = dict(g)
        g2["paths"] = new_paths
        idx = len(new_groups)
        new_groups.append(g2)
        tails = frozenset(_short_tail_from_path(p) for p in paths)
        tails_to_idx[tails] = idx

    os.makedirs(dst_gt_dir, exist_ok=True)
    dst_path = os.path.join(dst_gt_dir, os.path.basename(groups_path))
    save_json(dst_path, new_groups)
    return dst_path, tails_to_idx


def _build_alignment_subset(
    alignment_path: str,
    allowed_tails_to_idx: Dict[frozenset, int],
    dst_report_dir: str,
) -> Tuple[str, Dict[frozenset, Dict[str, Any]]]:
    """
    Trim alignment_path to a subset, keeping only items whose tails_set is in
    allowed_tails_to_idx.

    Returns:
      - subset_alignment_path
      - mapping: tails_set -> item (used later to construct flat_results)
    """
    data = load_json(alignment_path)
    results = data.get("results", []) or []
    kept: List[Dict[str, Any]] = []
    tails_to_item: Dict[frozenset, Dict[str, Any]] = {}
    for it in results:
        sols = it.get("solutions", []) or []
        tails = frozenset((s.get("path") for s in sols if isinstance(s, dict) and s.get("path")))
        if not tails:
            continue
        if tails not in allowed_tails_to_idx:
            continue
        kept.append(it)
        tails_to_item[tails] = it
    os.makedirs(dst_report_dir, exist_ok=True)
    dst_path = os.path.join(dst_report_dir, os.path.basename(alignment_path))
    save_json(dst_path, {"results": kept})
    return dst_path, tails_to_item


def _parse_run_params_from_report_header(report_path: str) -> Dict[str, Any]:
    """
    Parse the 'Run parameters' section from the header of a single-task report
    and return a dict mapping key -> value.
    """
    run_params: Dict[str, Any] = {}
    with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]
    in_params = False
    for ln in lines:
        if ln.strip().startswith("Run parameters:"):
            in_params = True
            continue
        if in_params:
            if not ln.startswith("- "):
                break
            m = re.match(r"-\s*([^:]+):\s*(.*)$", ln)
            if not m:
                continue
            key = (m.group(1) or "").strip()
            val_str = (m.group(2) or "").strip()
            run_params[key] = val_str
    return run_params


def _now_timestamp_tag() -> Tuple[str, str]:
    """
    Return (timestamp_iso, timestamp_tag).
    """
    now = datetime.now()
    ts_iso = now.isoformat(timespec="seconds")
    ts_tag = now.strftime("%Y%m%d_%H%M%S")
    return ts_iso, ts_tag


def _detect_content_mode(task_name: str, tasks_root: str, desc_dir: str, da_dir: str) -> str:
    """
    A lightweight reimplementation of grade.util.prompt.load_task_resources
    to decide the content mode (whether description / data-analysis resources exist).

    We only need a simplified check here: just see whether description / DA files exist.
    """
    # Simplified logic: prefer tasks_root, then fall back to desc_dir/da_dir.
    has_desc = False
    has_da = False
    if tasks_root:
        # Convention: tasks_root/description/<task>.md and
        #            tasks_root/data_analysis_best/report/da_result_<task>.txt
        desc_path = os.path.join(tasks_root, "description", f"{task_name}.md")
        da_path = os.path.join(tasks_root, "data_analysis_best", "report", f"da_result_{task_name}.txt")
        has_desc = os.path.exists(desc_path)
        has_da = os.path.exists(da_path)
    else:
        if desc_dir:
            desc_path = os.path.join(desc_dir, f"description_{task_name}.md")
            has_desc = os.path.exists(desc_path)
        if da_dir:
            da_path = os.path.join(da_dir, f"da_result_{task_name}.txt")
            has_da = os.path.exists(da_path)
    if has_desc and has_da:
        return "data-both"
    if has_desc:
        return "data-desc-only"
    if has_da:
        return "data-ana-only"
    return "none-task-data"


def _rebuild_single_report(
    task_name: str,
    n_tag: str,
    src_report_path: str,
    groups_subset_path: str,
    tails_to_idx: Dict[frozenset, int],
    tails_to_alignment_item: Dict[frozenset, Dict[str, Any]],
    subset_task_dir: str,
    solutions_root: str,
    tasks_json_path: str,
) -> None:
    """
    Using (groups_subset, alignment_subset) and the header parameters of one
    original report, reconstruct flat_results_for_task and then call
    grade.util.report.build_report to write the single-task report for the subset.
    """
    # Lazy import to avoid circular imports; only used when rich files are present.
    from skip_bench.grade.util import report as report_module  # type: ignore
    from skip_bench.grade.util import utils as grade_utils  # type: ignore

    src_report = src_report_path

    # Build flat_results_for_task.
    if not tails_to_idx:
        # No groups were kept in the subset; this report cannot be used.
        return

    flat_results: List[Dict[str, Any]] = []
    for tails, new_idx in tails_to_idx.items():
        align_item = tails_to_alignment_item.get(tails)
        if not align_item:
            # Alignment does not contain predictions for this group; skip it
            # instead of failing. Normally each group should have alignment.
            continue
        sols = align_item.get("solutions", []) or []
        n = len(sols)
        if n <= 0:
            continue
        # Only implement accuracy for n=2; for other n we keep accuracy as None.
        status = align_item.get("correct")
        if n == 2 and status == "correct":
            acc = 1.0
        elif n == 2 and status is not None:
            acc = 0.0
        else:
            acc = None  # For n>2, we currently do not compute accuracy.

        # Store a simplified version of the LLM output in raw_responses:
        # reasoning + best_index + confidence.
        pred = align_item.get("prediction", {}) or {}
        reasoning = pred.get("reasoning") or ""
        bi = pred.get("best_index")
        conf = pred.get("confidence")
        resp_lines: List[str] = []
        if reasoning:
            resp_lines.append(str(reasoning))
        summary = {"predicted_best_index": bi, "confidence": conf}
        try:
            resp_lines.append(json.dumps(summary, ensure_ascii=False))
        except Exception:
            resp_lines.append(str(summary))
        raw_resp = "\n".join(resp_lines)

        res_rec: Dict[str, Any] = {
            "task": task_name,
            "n": n,
            "group_file": groups_subset_path,
            "group_index": new_idx,
            "accuracy": acc,
            "raw_responses": [raw_resp],
        }
        flat_results.append(res_rec)

    if not flat_results:
        # This report's alignment has no usable groups in the subset.
        return

    # Parse header parameters from the original report.
    header_params = _parse_run_params_from_report_header(src_report)
    # Generate new timestamps.
    ts_iso, ts_tag = _now_timestamp_tag()

    # Assemble params_task.
    params_task: Dict[str, Any] = {}
    params_task["task_file"] = tasks_json_path  # Path to tasks.json for traceability.
    params_task["solutions_dir"] = subset_task_dir.rsplit(os.sep, 1)[0]  # subset_root
    params_task["model"] = header_params.get("model")
    params_task["temperature"] = header_params.get("temperature")
    params_task["max_retries"] = header_params.get("max_retries")
    params_task["base_url"] = header_params.get("base_url")
    params_task["out"] = None
    params_task["report_out"] = None
    params_task["tasks_root"] = None  # Do not infer tasks_root inside the subset.
    params_task["desc_dir"] = None
    params_task["da_dir"] = None
    params_task["parallel"] = header_params.get("parallel")
    params_task["workers"] = header_params.get("workers")
    params_task["cot"] = header_params.get("cot")
    params_task["prompt_boost"] = header_params.get("prompt_boost")
    params_task["content_mode"] = _detect_content_mode(
        task_name=task_name,
        tasks_root=None,
        desc_dir=None,
        da_dir=None,
    )
    params_task["timestamp"] = ts_iso
    params_task["timestamp_tag"] = ts_tag
    # To reuse report.write_reports naming logic we need model_tag/temp_tag/boost/cot.
    model_tag = grade_utils.sanitize_for_filename(params_task.get("model") or "default")
    temp_tag = grade_utils.format_temp_for_name(params_task.get("temperature") or 0)
    cot_tag = "cot" if str(params_task.get("cot")).lower() in ("true", "1") else "nocot"
    boost_tag = "pboost" if str(params_task.get("prompt_boost")).lower() in ("true", "1") else "std"
    mode_tag = grade_utils.sanitize_for_filename(params_task.get("content_mode") or "none-task-data")

    # n_arg: extract integer n from n_tag.
    try:
        n_arg = int(n_tag[1:])
    except Exception:
        n_arg = 0

    # Use build_report to generate the report text.
    text = report_module.build_report({task_name: flat_results}, n_arg, params_task)

    # Keep the filename logic consistent with write_reports.
    report_dir = os.path.join(subset_task_dir, "report")
    ensure_dir(report_dir)
    report_name = (
        f"grade_report_{task_name}_n{int(n_arg)}_"
        f"{mode_tag}_{model_tag}_{temp_tag}_{boost_tag}_{cot_tag}_{ts_tag}.txt"
    )
    report_path = os.path.join(report_dir, report_name)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[INFO] Rebuilt subset report: {report_path}")


def _process_rich_task(
    task_name: str,
    solutions_root: str,
    subset_root: str,
    selected_solutions: Set[str],
    tasks_json_path: str,
) -> None:
    """
    Entry point for processing rich directories:

      - Trim ground_truth based on selected_solutions
      - Trim alignment accordingly
      - Rebuild reports using the subset ground_truth + alignment

    If one set of rich files is unusable, try another set for the same task.
    """
    src_task_dir = os.path.join(solutions_root, task_name)
    dst_task_dir = os.path.join(subset_root, task_name)
    rich_map = _find_rich_files_for_task(src_task_dir, task_name)
    if not rich_map:
        return

    subset_code_dir = os.path.join(dst_task_dir, "code")
    dst_gt_dir = os.path.join(dst_task_dir, "ground_truth")
    dst_report_dir = os.path.join(dst_task_dir, "report")

    for n_tag, info in rich_map.items():
        groups_path = info["groups"]
        report_paths = sorted(info["reports"])
        align_paths = sorted(info["alignments"])
        # 1) Build a ground_truth subset.
        groups_subset_path, tails_to_idx = _build_groundtruth_subset(
            groups_path=groups_path,
            selected_solutions=selected_solutions,
            dst_gt_dir=dst_gt_dir,
            subset_code_dir=subset_code_dir,
        )
        if not tails_to_idx:
            print(f"[WARN] No groups kept in subset for task={task_name}, {n_tag}, skip this n_tag.")
            continue

        # Iterate over all runs (report + alignment pairs).
        for src_rep, src_align in zip(report_paths, align_paths):
            # 2) Build an alignment subset.
            subset_align_path, tails_to_alignment_item = _build_alignment_subset(
                alignment_path=src_align,
                allowed_tails_to_idx=tails_to_idx,
                dst_report_dir=dst_report_dir,
            )
            if not tails_to_alignment_item:
                continue

            # 3) Rebuild the report based on the subset groups + alignment
            #    and the header of the original report.
            _rebuild_single_report(
                task_name=task_name,
                n_tag=n_tag,
                src_report_path=src_rep,
                groups_subset_path=groups_subset_path,
                tails_to_idx=tails_to_idx,
                tails_to_alignment_item=tails_to_alignment_item,
                subset_task_dir=dst_task_dir,
                solutions_root=solutions_root,
                tasks_json_path=tasks_json_path,
            )
            # Sleep a bit to avoid timestamp collisions in generated filenames.
            time.sleep(1.1)


def process_task(
    task_name: str,
    spec: Any,
    solutions_root: str,
    subset_root: str,
    tasks_json_path: str,
) -> None:
    print(f"Processing task: {task_name}")
    src_task_dir = os.path.join(solutions_root, task_name)
    dst_task_dir = os.path.join(subset_root, task_name)

    # prepare destination directory structure
    ensure_dir(os.path.join(dst_task_dir, "annotation"))
    ensure_dir(os.path.join(dst_task_dir, "code"))
    ensure_dir(os.path.join(dst_task_dir, "output"))

    # read source annotations
    src_ann_path = os.path.join(src_task_dir, "annotation", "annotations_semantic.json")
    if not os.path.exists(src_ann_path):
        print(f"[WARN] annotation file not found for task {task_name}: {src_ann_path}")
        return
    annotations = load_json(src_ann_path)

    # build index
    index = build_annotation_index(annotations)

    # sample according to spec
    global_selected: Set[str] = set()
    node_used_map: Dict[Tuple[str, ...], Set[str]] = {}
    select_for_spec(
        index=index,
        spec=spec,
        path_keys=[],
        global_selected=global_selected,
        node_used_map=node_used_map,
    )

    print(f"  selected {len(global_selected)} solutions")

    # write subset annotations
    subset_annotations = {k: annotations[k] for k in global_selected if k in annotations}
    dst_ann_path = os.path.join(dst_task_dir, "annotation", "annotations_semantic.json")
    save_json(dst_ann_path, subset_annotations)

    # Build keywords_by_rank.json from the subset annotations.
    kw_by_rank = _build_keywords_by_rank(subset_annotations)
    dst_kw_path = os.path.join(dst_task_dir, "annotation", "keywords_by_rank.json")
    save_json(dst_kw_path, kw_by_rank)

    # copy files
    for sol_id in global_selected:
        copy_solution_files(task_name, sol_id, src_task_dir, dst_task_dir)

    # If this task uses the rich directory structure, further process
    # ground_truth / alignment / report subsets.
    _process_rich_task(
        task_name=task_name,
        solutions_root=solutions_root,
        subset_root=subset_root,
        selected_solutions=global_selected,
        tasks_json_path=tasks_json_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a subset of solutions according to tasks.json."
    )

    parser.add_argument(
        "--solutions-root",
        type=str,
        required=True,
        help="Root directory of full solutions",
    )
    parser.add_argument(
        "--subset-root",
        type=str,
        required=True,
        help="Output root directory for subset solutions",
    )
    parser.add_argument(
        "--tasks-json",
        type=str,
        required=True,
        help="Path to tasks.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    solutions_root = args.solutions_root
    subset_root = args.subset_root
    tasks_json = args.tasks_json

    tasks = load_json(tasks_json)
    for task_name, spec in tasks.items():
        process_task(task_name, spec, solutions_root, subset_root, tasks_json)


if __name__ == "__main__":
    main()
