# Utilities to parse grading logs and generate alignment JSON.
import argparse
import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures


def parse_log_for_items_meta(log_text: str, with_log_index: bool = True) -> List[Dict[str, Any]]:
    """
        Parse strictly by block:
            block = starts at the current "- group_file=... index=... n=..." line,
                            ends at the line before the next "- group_file=", inclusive,
                            or the end of file.

        For each block, return:
            {
                "log_index": Optional[int],
                "block_text": str,        # text of this single log block
            }
        Do not parse solutions / assistant / JSON here.
    """
    items: List[Dict[str, Any]] = []

    lines = log_text.splitlines()
    line_count = len(lines)

    # Precompute the start character offset for each line (only for locating block_text)
    line_starts: List[int] = [0] * (line_count + 1)
    acc = 0
    for i, ln in enumerate(lines):
        line_starts[i] = acc
        acc += len(ln) + 1  # assume '\n'
    line_starts[line_count] = acc

    # Find all group_file lines
    group_line_indices: List[int] = []
    group_log_indices: Dict[int, Optional[int]] = {}
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if stripped.startswith("- group_file="):
            idx_val: Optional[int] = None
            if "index=" in stripped:
                m = re.search(r"index=(\d+)", stripped)
                if m:
                    try:
                        idx_val = int(m.group(1))
                    except Exception:
                        idx_val = None
            group_line_indices.append(i)
            group_log_indices[i] = idx_val

    if not group_line_indices:
        return items

    # Build a block for each group_file line
    for k, start_line in enumerate(group_line_indices):
        end_line = group_line_indices[k + 1] - 1 if k + 1 < len(group_line_indices) else (line_count - 1)
        if end_line < start_line:
            continue

        # Line range [start_line, end_line]
        block_lines = lines[start_line : end_line + 1]
        block_text = "\n".join(block_lines)
        log_index = group_log_indices.get(start_line)

        # Always record log_index
        items.append(
            {
                "log_index": log_index,
                "block_text": block_text,
            }
        )

    return items


def parse_log_for_items(log_text: str, with_log_index: bool = True) -> List[Dict[str, Any]]:
    """
        Do all matching within each block:
            - Find Provided solutions / Solution 0/1/2/... in block_text
            - Find [assistant] in block_text
            - Find JSON and slice reasoning in block_text
        Avoid global log_text slicing to prevent out-of-bounds errors.
        Supports any n value (not limited to n=2).
    """
    meta_blocks = parse_log_for_items_meta(log_text, with_log_index=with_log_index)
    results: List[Dict[str, Any]] = []

    for meta in meta_blocks:
        block_text: str = meta["block_text"]
        log_index = meta["log_index"]

        # 1) Find Provided solutions inside the block
        lines = block_text.splitlines()
        line_count = len(lines)

        # 1.1 Find the Provided solutions line (optional)
        provided_line: Optional[int] = None
        for i, ln in enumerate(lines):
            if "Provided solutions" in ln:
                provided_line = i
                break
        search_start = provided_line if provided_line is not None else 0

        # 1.2 Find all Solution indices (not limited to 0/1)
        sols_map: Dict[int, str] = {}
        for i in range(search_start + 1, line_count):
            m = re.match(r'^###\s*Solution\s*(\d+):\s*path=(.*)$', lines[i])
            if not m:
                continue
            try:
                idx = int(m.group(1))
                path = (m.group(2) or "").strip()
                if path:
                    sols_map[idx] = path
            except Exception:
                continue
        
        if not sols_map:
            print(f"[WARN] Block skipped: no Solution paths found. block_id={log_index}")
            continue
        
        # Sort by index and build the solutions list
        max_idx = max(sols_map.keys())
        sols: List[str] = []
        for i in range(max_idx + 1):
            if i in sols_map:
                sols.append(sols_map[i])
            else:
                # If an index is missing, use an empty placeholder (should not happen)
                print(f"[WARN] Missing Solution {i} in block_id={log_index}, using empty placeholder")
                sols.append("")
        
        n = len(sols)

        # 2) Find the first [assistant] inside the block
        assistant_start_pos: Optional[int] = None
        # Line number -> character offset in block_text
        line_starts_block: List[int] = [0] * (line_count + 1)
        acc = 0
        for i, ln in enumerate(lines):
            line_starts_block[i] = acc
            acc += len(ln) + 1  # '\n'
        line_starts_block[line_count] = acc

        for i, ln in enumerate(lines):
            if re.match(r'^\s*\[assistant\]\s*$', ln.strip()):
                assistant_start_pos = line_starts_block[i]
                break
        if assistant_start_pos is None:
            print(f"[WARN] Block skipped: missing [assistant]. block_id={log_index}")
            continue

        # 3) Find JSON in block_text after assistant_start_pos
        prediction, json_start_rel, json_end_rel = _extract_first_json_after(block_text, assistant_start_pos)

        # If strict parsing fails, fall back to the loose "last JSON" mode
        if not isinstance(prediction, dict):
            loose_pred = _extract_last_json_loose(block_text)
            if loose_pred is None:
                print(f"[WARN] Block skipped: failed to extract JSON after [assistant]. block_id={log_index}")
                continue
            # Keep only required fields based on n
            if n == 2:
                prediction = {
                    "predicted_best_index": loose_pred.get("predicted_best_index"),
                    "confidence": loose_pred.get("confidence"),
                }
            else:
                prediction = {
                    "predicted_ranking": loose_pred.get("predicted_ranking"),
                    "confidence": loose_pred.get("confidence"),
                }
            json_start_rel = None
            json_end_rel = None

        # Always record log_index, but only write when with_log_index=True
        if log_index is not None:
            prediction["log_index"] = log_index if with_log_index else None

        # 5) Slice reasoning within block_text, from assistant_start_pos to json_start_rel
        if json_start_rel is not None:
            reasoning_seg = block_text[assistant_start_pos:json_start_rel].strip()
            if reasoning_seg:
                # Remove the leading [assistant]\n
                cleaned = re.sub(r'^\s*\[assistant\]\s*\n?', '', reasoning_seg, flags=re.MULTILINE).strip()
                cleaned = re.sub(r'^\s*```.*$', '', cleaned, flags=re.MULTILINE).strip()
                if cleaned and "reasoning" not in prediction:
                    prediction["reasoning"] = cleaned

        results.append(
            {
                "solutions": sols,
                "prediction": prediction,
                "n": n,  # Record n value
            }
        )

    return results


def _extract_first_json_after(s: str, start_pos: int) -> Tuple[Optional[Dict[str, Any]], Optional[int], Optional[int]]:
    """
    Within the given string s, find JSON starting from start_pos.
    Note: s is assumed to be a single block_text (no cross-block parsing).
    """
    n = len(s)
    segment = s[start_pos:]
    # Prefer only the last 5 lines
    last_lines = segment.splitlines()[-5:]
    for line in reversed(last_lines):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                obj = json.loads(stripped)
                rel_start = segment.rfind(line)
                if rel_start == -1:
                    continue
                abs_start = start_pos + rel_start
                abs_end = abs_start + len(line)
                return obj, abs_start, abs_end
            except Exception:
                continue
    # fallback: search backward for '{' within the last 500 bytes
    tail_len = min(500, len(segment))
    tail_segment = segment[-tail_len:]
    brace_positions = [m.start() for m in re.finditer(r'\{', tail_segment)]
    for i in reversed(brace_positions):
        abs_i = start_pos + len(segment) - tail_len + i
        depth = 0
        in_str = False
        esc = False
        end = None
        for j in range(abs_i, n):
            ch = s[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = j + 1
                        break
        if end is None:
            continue
        candidate = s[abs_i:end]
        try:
            obj = json.loads(candidate)
            return obj, abs_i, end
        except Exception:
            continue
    return None, None, None


def _extract_last_json_loose(s: str) -> Optional[Dict[str, Any]]:
    """
    Loose mode: search backwards for the last JSON-parsable fragment.
    It must start with '{' and end with '}', and be balanced by brace pairing.
    Return the parsed dict, or None on failure.
    """
    n = len(s)
    # Find all '{' positions from the end
    brace_positions = [m.start() for m in re.finditer(r'\{', s)]
    if not brace_positions:
        return None
    for abs_i in reversed(brace_positions):
        depth = 0
        in_str = False
        esc = False
        end = None
        for j in range(abs_i, n):
            ch = s[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = j + 1
                        break
        if end is None:
            continue
        candidate = s[abs_i:end]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def load_groundtruth(groups_path: str) -> List[Dict[str, Any]]:
    with open(groups_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_group_for_solutions(groups: List[Dict[str, Any]], sols: List[str]) -> Optional[Dict[str, Any]]:
    """
    Find the matching group by comparing short tails (e.g., 7e7e.py) of the provided solutions.
    Falls back to exact full-path matching if needed.
    Supports any n value.
    """
    target_tails = {_short_tail_from_path(p) for p in sols if p}  # filter empty strings
    for g in groups:
        g_paths = g.get("paths", [])
        if len(g_paths) != len(sols):
            continue
        g_tails = {_short_tail_from_path(p) for p in g_paths}
        if g_tails == target_tails:
            return g
    # Fallback to exact matching (in case tails are not unique)
    target_full = {p for p in sols if p}
    for g in groups:
        g_paths_set = set(g.get("paths", []))
        if len(g_paths_set) == len(target_full) and g_paths_set == target_full:
            return g
    return None


def load_annotations(annotations_path: str) -> Dict[str, Any]:
    """
    Load semantic annotations JSON like:
    {
      "solution_xxx_run_xxxxx.py": [
        [ "kw1", "kw2", ... ]
      ],
      ...
    }
    Returns the raw dict.
    """
    if not os.path.exists(annotations_path):
        return {}
    with open(annotations_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _short_tail_from_path(path: str) -> str:
    """
    Convert a full solution path to its short tail id, e.g.
    ..._run_9a25d253555a4739adc9151beb3b6ef5.py -> 6ef5.py
    Fallback to basename if pattern not matched.
    """
    base = os.path.basename(path)
    m = re.search(r'([0-9a-fA-F]+)\.py$', base)
    if m:
        token = m.group(1)
        return f"{token[-4:]}.py" if len(token) >= 4 else f"{token}.py"
    return base


def build_result_item(
    sols: List[str],
    prediction: Dict[str, Any],
    groups: List[Dict[str, Any]],
    annotations: Dict[str, Any],
) -> Dict[str, Any]:
    n = len(sols)
    group = find_group_for_solutions(groups, sols)
    
    # Select prediction fields based on n
    if n == 2:
        pred_idx_raw = prediction.get("predicted_best_index", None)
        pred_ranking = None
    else:
        pred_ranking = prediction.get("predicted_ranking", None)
        pred_idx_raw = pred_ranking[0] if (isinstance(pred_ranking, list) and pred_ranking) else None
    
    # Coerce pred index to int if it's a numeric string or float-like
    pred_idx = None
    if isinstance(pred_idx_raw, (int, float)):
        try:
            pred_idx = int(pred_idx_raw)
        except Exception:
            pred_idx = None
    elif isinstance(pred_idx_raw, str):
        try:
            pred_idx = int(float(pred_idx_raw.strip()))
        except Exception:
            pred_idx = None
    
    conf = prediction.get("confidence", None)
    try:
        # Normalize string confidence if needed
        if isinstance(conf, str):
            conf = float(conf.strip())
    except Exception:
        pass

    # For n > 2, use full_ranking instead of best_index
    if n == 2:
        gt_best = group.get("best_index") if group else None
        gt_full = None
    else:
        gt_full = group.get("full_ranking") if group else None
        gt_best = gt_full[0] if (isinstance(gt_full, list) and gt_full) else None

    mapped_idx = None
    status = None
    if group is not None and isinstance(pred_idx, int) and 0 <= pred_idx < len(sols):
        pred_path = sols[pred_idx]
        # Try exact path match first
        if pred_path in group["paths"]:
            mapped_idx = group["paths"].index(pred_path)
        else:
            # Fall back to short-tail match
            pred_tail = _short_tail_from_path(pred_path)
            group_tails = [_short_tail_from_path(p) for p in group["paths"]]
            if pred_tail in group_tails:
                mapped_idx = group_tails.index(pred_tail)
        
        # Determine correctness
        if n == 2:
            if mapped_idx is not None and gt_best is not None:
                status = "correct" if (mapped_idx == gt_best) else "false"
        else:
            # n > 2: only check whether top-1 is correct
            if mapped_idx is not None and gt_best is not None:
                status = "correct" if (mapped_idx == gt_best) else "false"

    def _recover_annotation_key(short_tail: str, group_paths: List[str]) -> Optional[str]:
        """
        Recover the annotations key from a short tail (e.g., 6ef5.py):
        - Iterate group_paths to find the full path matching the short tail
        - Extract solution_xxx_run_xxx from the full path
        """
        for full_path in group_paths:
            if short_tail == _short_tail_from_path(full_path):
                basename = os.path.basename(full_path)
                name_no_py = basename[:-3] if basename.endswith(".py") else basename
                m = re.match(r"(solution_.*?_run_[0-9a-fA-F]+)$", name_no_py)
                if m:
                    return m.group(1)
        return None

    solutions_info = []
    group_paths = group["paths"] if group else []
    scores = group.get("scores") if group else [None] * len(sols)  # default None padding
    for p, score in zip(sols, scores):
        if not p:  # skip empty placeholders
            continue
        short_tail = _short_tail_from_path(p)
        # Prefer recovering the annotations key from group_paths
        annotation_key = _recover_annotation_key(short_tail, group_paths)
        if not annotation_key:
            # If recovery fails, fall back to short-tail path (legacy compatibility)
            annotation_key = short_tail.removesuffix(".py")
        ann = annotations.get(annotation_key)
        kws = None
        if isinstance(ann, list) and ann:
            first = ann[0]
            if isinstance(first, list):
                kws = first
        solutions_info.append(
            {
                "path": short_tail,
                "kws": kws,
                "score": score,  # add score
            }
        )

    # Build prediction output based on n
    if n == 2:
        pred_out = {
            "best_index": pred_idx,
            "confidence": conf,
        }
    else:
        pred_out = {
            "ranking": pred_ranking if isinstance(pred_ranking, list) else None,
            "confidence": conf,
        }
    
    if prediction.get("reasoning"):
        pred_out["reasoning"] = prediction["reasoning"]

    # Extract log_index from prediction (if present)
    log_index = prediction.get("log_index")

    # Select groundtruth fields based on n
    if n == 2:
        groundtruth_info = {
            "best_index": gt_best,
            "is_lower_better": group.get("is_lower_better") if group else None,
        }
    else:
        groundtruth_info = {
            "full_ranking": gt_full,
            "is_lower_better": group.get("is_lower_better") if group else None,
        }

    # Keep "correct" first and add log_index
    result = {
        "correct": status,
        "solutions": solutions_info,
        "groundtruth": groundtruth_info,
        "prediction": pred_out,
    }
    if log_index is not None:
        result["log_index"] = log_index
    return result


def _infer_task_from_log_path(log_path: str) -> str:
    """
    Infer task name from log path:
    /.../solutions_subset_10/<task>/report/grade_report_<task>_nX_*.txt
    Simple rule: use the parent directory name as task.
    """
    # .../solutions_subset_10/<task>/report/xxx.txt
    task_dir = os.path.basename(os.path.dirname(os.path.dirname(log_path)))
    return task_dir


def _infer_n_from_log_filename(filename: str) -> Optional[str]:
    """
    Parse n2 / n3 / n45 from filenames like:
    grade_report_aptos2019-blindness-detection_n2_deepseek-...txt
    Convention: n tag is the segment between the task name and the base model name.
    """
    # Remove the "grade_report_" prefix
    name = filename
    if name.startswith("grade_report_"):
        name = name[len("grade_report_"):]
    parts = name.split("_")
    # Structure: <task>-maybe-with-dashes ... n2 <model> ...
    # Find the first segment that starts with 'n' followed by digits
    for p in parts:
        if len(p) >= 2 and p[0] == "n" and p[1].isdigit():
            return p
    return None


def _make_output_path_from_log(log_path: str) -> str:
    """
    grade_report_aptos2019-blindness-detection_n2_xxx.txt
    -> alignment_aptos2019-blindness-detection_n2_xxx.json
    Generate in the same directory.
    """
    dir_ = os.path.dirname(log_path)
    base = os.path.basename(log_path)
    if base.startswith("grade_report_"):
        rest = base[len("grade_report_"):]
    else:
        rest = base
    rest = re.sub(r"\.txt$", ".json", rest)
    out_name = f"alignment_{rest}"
    return os.path.join(dir_, out_name)


def _process_single_log(
    log_path: str,
    groundtruth_path: str,
    annotations_path: str,
    output_path: str,
    with_log_index: bool = True,
) -> None:
    """
    Read a single log, parse by group_file blocks, and write alignment JSON.
    (All parsing is confined within a block to avoid out-of-bounds.)
    """
    with open(log_path, "r", encoding="utf-8") as f:
        log_text = f.read()

    items = parse_log_for_items(log_text, with_log_index=with_log_index)
    groups = load_groundtruth(groundtruth_path)
    annotations = load_annotations(annotations_path)

    results: List[Dict[str, Any]] = []
    for it in items:
        res = build_result_item(it["solutions"], it["prediction"], groups, annotations)
        results.append(res)

    out = {"results": results}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def _iter_logs_under_root(root_dir: str):
    """
    Recursively find all grade_report_*.txt under root_dir,
    skipping files that contain '_n0_' (only annotate n!=0).
    Exclude the report directory under the root.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        # Exclude the report directory under the root
        if dirpath == os.path.join(root_dir, "report"):
            continue
        if os.path.basename(dirpath) != "report":
            continue
        for fn in filenames:
            if not fn.startswith("grade_report_") or not fn.endswith(".txt"):
                continue
            n_tag = _infer_n_from_log_filename(fn)
            if n_tag == "n0":
                continue
            yield os.path.join(dirpath, fn)


# new: reusable helper to derive groundtruth/annotations/output paths for a single log
def build_paths_for_log(
    log_path: str,
    root_dir: str,
) -> Optional[Tuple[str, str, str]]:
    """
    Given a single grade_report_*.txt path and the global root_dir
    (e.g. /.../solutions_subset_10), derive:
      - groundtruth JSON path
      - annotations JSON path
      - alignment output JSON path

    Returns (groundtruth_path, annotations_path, output_path) or None
    if any of the required inputs do not exist.
    """
    task = _infer_task_from_log_path(log_path)
    n_tag = _infer_n_from_log_filename(os.path.basename(log_path))
    if not n_tag:
        return None

    # task_dir: <root_dir>/<task>/
    task_dir = os.path.join(root_dir, task)
    if not os.path.isdir(task_dir):
        # fall back to inferring from log_path like the CLI batch mode
        task_dir = os.path.dirname(os.path.dirname(log_path))

    groundtruth_path = os.path.join(
        task_dir,
        "ground_truth",
        f"groups_{task}_{n_tag}.json",
    )
    if not os.path.exists(groundtruth_path):
        return None

    annotations_path = os.path.join(
        task_dir,
        "annotation",
        "annotations_semantic.json",
    )
    if not os.path.exists(annotations_path):
        return None

    output_path = _make_output_path_from_log(log_path)
    return groundtruth_path, annotations_path, output_path


# new: public function for external callers (default no-overwrite)
def generate_alignment_for_log(
    log_path: str,
    root_dir: str,
    overwrite: bool = False,
    with_log_index: bool = True,
) -> Optional[str]:
    """
    Generate alignment JSON for a single grade_report_*.txt.
    """
    # Default behavior: write log_index.
    # External callers can disable by passing with_log_index=False.
    paths = build_paths_for_log(log_path, root_dir)
    if paths is None:
        return None

    groundtruth_path, annotations_path, output_path = paths
    if os.path.exists(output_path) and not overwrite:
        return output_path

    _process_single_log(
        log_path=log_path,
        groundtruth_path=groundtruth_path,
        annotations_path=annotations_path,
        output_path=output_path,
        with_log_index=with_log_index,
    )
    return output_path


def process_one(
    log_path,
    overwrite,
    with_log_index,
):
    task = _infer_task_from_log_path(log_path)
    n_tag = _infer_n_from_log_filename(os.path.basename(log_path))
    if not n_tag:
        return None
    task_dir = os.path.dirname(os.path.dirname(log_path))
    gt_path = os.path.join(
        task_dir,
        "ground_truth",
        f"groups_{task}_{n_tag}.json",
    )
    if not os.path.exists(gt_path):
        return None
    annotations_path = os.path.join(
        task_dir,
        "annotation",
        "annotations_semantic.json",
    )
    if not os.path.exists(annotations_path):
        return None
    out_path = _make_output_path_from_log(log_path)
    if os.path.exists(out_path) and not overwrite:
        return None
    _process_single_log(
        log_path=log_path,
        groundtruth_path=gt_path,
        annotations_path=annotations_path,
        output_path=out_path,
        with_log_index=with_log_index,
    )
    return out_path


def main():
    parser = argparse.ArgumentParser()
    # Single-log mode (optional)
    parser.add_argument("--log", default=None)
    parser.add_argument("--groundtruth", default=None)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--output", default=None)

    # Batch mode root directory
    parser.add_argument(
        "--root_dir",
        default="/datadisk/zjs/skip_bench/solutions_subset_10",
        help="Root directory containing task subdirs like aptos2019-blindness-detection",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing alignment_*.json; otherwise skip (for resume).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of parallel workers for alignment generation in batch mode.",
    )
    # Default to writing log_index; disable via --no_log_index
    parser.add_argument(
        "--no_log_index",
        dest="with_log_index",
        action="store_false",
        help="If set, do NOT parse and attach log_index from '- group_file=... index=...' lines into alignment.json.",
    )
    # Default is True: log_index is written unless disabled
    parser.set_defaults(with_log_index=True)
    args = parser.parse_args()

    if args.log:
        if not (args.groundtruth and args.annotations and args.output):
            raise ValueError("Single-log mode requires --log, --groundtruth, --annotations, --output all set.")
        _process_single_log(
            log_path=args.log,
            groundtruth_path=args.groundtruth,
            annotations_path=args.annotations,
            output_path=args.output,
            with_log_index=args.with_log_index,
        )
        print(args.output)
        return

    root_dir = args.root_dir
    overwrite = args.overwrite
    workers = args.workers
    with_log_index = args.with_log_index

    log_paths = list(_iter_logs_under_root(root_dir))

    if workers <= 1:
        for log_path in log_paths:
            out_path = process_one(log_path, overwrite, with_log_index)
            if out_path:
                print(out_path)
    else:
        # Use a process pool for real multi-core parallelism
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(process_one, log_path, overwrite, with_log_index): log_path
                for log_path in log_paths
            }
            for fut in concurrent.futures.as_completed(futures):
                out_path = fut.result()
                if out_path:
                    print(out_path)


if __name__ == "__main__":
    main()