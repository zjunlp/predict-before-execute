import os
from pathlib import Path
from typing import List, Optional

from .consts import APP_DIR

def find_default_script() -> Path:
    candidates = [APP_DIR / "main.py", APP_DIR / "app.py", APP_DIR / "train.py"]
    for c in candidates:
        if c.exists():
            return c
    sol_root = APP_DIR / "solutions_subset"
    if sol_root.exists():
        for p in sol_root.rglob("*.py"):
            if p.name != "run.py":
                return p
    for p in APP_DIR.rglob("*.py"):
        if p.name not in {"run.py"}:
            return p
    raise FileNotFoundError("No target script found. Set RUN_SCRIPT env or pass a script path argument.")

def resolve_script(args: List[str], override: Optional[str] = None) -> Path:
    if override:
        raw = override
    else:
        env_path = (
            os.environ.get("SOLUTION_PATH")
            or os.environ.get("RUN_SCRIPT")
            or os.environ.get("SCRIPT_PATH")
        )
        raw = args[0] if args else env_path

    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = (APP_DIR / raw).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Script not found: {p}")
        return p
    return find_default_script()
