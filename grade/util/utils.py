# Utility helpers for file discovery and report parsing.
import glob
import os
import re
from typing import List, Any, Dict, Optional

__all__ = [
    "load_task_names",
    "find_group_files",
    "sanitize_for_filename",
    "format_temp_for_name",
    "find_latest_report_for_task",
    "parse_single_task_report",
]

def load_task_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def find_group_files(solutions_dir: str, task_name: str, n: int) -> List[str]:
    task_dir = os.path.join(solutions_dir, task_name)
    if not os.path.isdir(task_dir):
        return []
    # Prefer ground_truth subdir if present; fallback to task root for backward compatibility
    candidates: List[str] = []
    gt_dir = os.path.join(task_dir, "ground_truth")
    patterns = []
    if os.path.isdir(gt_dir):
        patterns.append(os.path.join(gt_dir, f"groups_{task_name}_n*.json"))
    patterns.append(os.path.join(task_dir, f"groups_{task_name}_n*.json"))
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    files = sorted(set(candidates))
    if n == 0:
        return files
    target_suffix = f"_n{n}.json"
    return [f for f in files if f.endswith(target_suffix)]

def sanitize_for_filename(s: str) -> str:
    """
    Sanitize a string for safe filenames: keep alnum, dash, underscore; replace others with '_'.
    """
    return re.sub(r"[^a-zA-Z0-9\-_]+", "_", s or "")

def format_temp_for_name(t: Any) -> str:
    """
    Format temperature for filename, e.g., 0.2 -> '0p2'
    """
    try:
        t_str = str(float(t))
    except Exception:
        t_str = str(t)
    t_str = t_str.replace(".", "p").replace("-", "m")
    return sanitize_for_filename(t_str)

def find_latest_report_for_task(solutions_dir: str, task_name: str) -> Optional[str]:
    """
    Locate the grade_report_*.txt for a given task with the latest timestamp in filename.
    Returns absolute path or None if no report dir/files are found.
    """
    import re
    if not solutions_dir:
        return None
    task_dir = os.path.join(solutions_dir, task_name)
    report_dir = os.path.join(task_dir, "report")
    if not os.path.isdir(report_dir):
        return None
    pattern = os.path.join(report_dir, "grade_report_*.txt")
    candidates = [p for p in glob.glob(pattern) if os.path.isfile(p)]
    if not candidates:
        return None

    # Extract timestamp from filename: _YYYYMMDD_HHMMSS before .txt
    def extract_ts(path: str) -> str:
        fname = os.path.basename(path)
        m = re.search(r'_(\d{8}_\d{6})\.txt$', fname)
        return m.group(1) if m else ""

    # Sort by timestamp string (lexicographically, which works for YYYYMMDD_HHMMSS)
    candidates_with_ts = [(p, extract_ts(p)) for p in candidates if extract_ts(p)]
    if not candidates_with_ts:
        return None
    # Pick the one with max timestamp
    latest_path = max(candidates_with_ts, key=lambda x: x[1])[0]
    return latest_path

def parse_single_task_report(path: str) -> Dict[str, Any]:
    """
    Parse a single-task textual report produced by build_report.
    Extract:
      - run_params: dict of key -> value (strings, basic casting for some fields)
      - task_name: from 'Per-task metrics:' section line
      - metrics: dict with keys: pairs_count, accuracy_avg, multi_count, spearman_avg
    Assumes the structure like:

      Skip-bench grading report (n=2)
      Run parameters:
      - key: value
      ...
      Total tasks: 1
      Total groups: 190
      Overall metrics:
      - pairs_count: 190
      - accuracy_avg: 0.73
      ...
      Per-task metrics:
      - task-name: pairs_count=..., accuracy_avg=..., multi_count=..., spearman_avg=...

    We prefer the per-task metrics line when aggregating across tasks.
    """
    run_params: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    task_name: Optional[str] = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]

    # parse Run parameters block
    in_params = False
    for ln in lines:
        if ln.strip().startswith("Run parameters:"):
            in_params = True
            continue
        if in_params:
            if not ln.startswith("- "):
                # end of params block
                break
            # format: "- key: value"
            m = re.match(r"-\s*([^:]+):\s*(.*)$", ln)
            if not m:
                continue
            key = (m.group(1) or "").strip()
            val_str = (m.group(2) or "").strip()
            run_params[key] = val_str

    # parse Per-task metrics section
    in_pt = False
    for ln in lines:
        if ln.strip().startswith("Per-task metrics:"):
            in_pt = True
            continue
        if in_pt:
            ln_stripped = ln.strip()
            if not ln_stripped:
                # blank line -> end of section
                break
            if not ln_stripped.startswith("- "):
                continue
            # line example:
            # - learning-agency-lab-automated-essay-scoring-2: pairs_count=190, accuracy_avg=0.73, multi_count=0, spearman_avg=None
            m = re.match(r"-\s*([^:]+):\s*(.*)$", ln_stripped)
            if not m:
                continue
            task_name = (m.group(1) or "").strip()
            tail = (m.group(2) or "").strip()
            # split by comma
            parts = [p.strip() for p in tail.split(",") if p.strip()]
            for part in parts:
                # each part like "pairs_count=190"
                kv = part.split("=", 1)
                if len(kv) != 2:
                    continue
                k = kv[0].strip()
                v_raw = kv[1].strip()
                # cast basic types
                if v_raw == "None":
                    v: Any = None
                else:
                    try:
                        if "." in v_raw:
                            v = float(v_raw)
                        else:
                            v = int(v_raw)
                    except Exception:
                        v = v_raw
                metrics[k] = v
            # only first line matters in single-task report
            break

    return {
        "run_params": run_params,
        "task_name": task_name or "",
        "metrics": metrics,
        "report_path": os.path.abspath(path),
    }
