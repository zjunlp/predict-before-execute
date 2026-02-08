# Utilities for loading task names and inferring the task from a solution path.

from pathlib import Path
from typing import Optional, List
import os

# Default: ENV > repo_root/task_name.txt
REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_FILE_DEFAULT = Path(os.environ.get("TASK_FILE", str(REPO_ROOT / "task_name.txt")))

def load_task_names(task_file: Optional[Path] = None) -> List[str]:
    task_file = task_file or TASK_FILE_DEFAULT
    try:
        with open(task_file, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    except Exception:
        return []

def infer_task_from_solution(solution_path: Path, explicit_task: Optional[str] = None) -> Optional[str]:
    if explicit_task:
        return explicit_task
    tasks = load_task_names()
    sp = str(solution_path)
    for t in tasks:
        if f"/{t}/" in sp or sp.endswith(f"/{t}") or sp.startswith(f"{t}/"):
            return t
    return None
