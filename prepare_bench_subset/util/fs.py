import os
import shutil
from pathlib import Path
from typing import Optional

from .consts import APP_DIR

def clean_dir(d: Path) -> None:
    if d.exists():
        for p in d.iterdir():
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
    else:
        d.mkdir(parents=True, exist_ok=True)

def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst)

def harvest_and_cleanup_working(script_dir: Path, remove: bool) -> None:
    working = script_dir / "working"
    if not working.exists() or not working.is_dir():
        return
    sub_csv = working / "submission.csv"
    dest_dir = APP_DIR / "submission"
    if sub_csv.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sub_csv, dest_dir / "submission.csv")
    if remove:
        shutil.rmtree(working, ignore_errors=True)

def _copytree_filtered(src: Path, dst: Path) -> None:
    ignore = shutil.ignore_patterns(
        "submission*", "working", "input", "submission",
        "__pycache__", ".git", ".venv", ".idea", "*.pyc", "*.pyo", ".DS_Store"
    )
    shutil.copytree(src, dst, ignore=ignore)

def prepare_isolated_workspace(script_path: Path) -> Path:
    ws_root = APP_DIR / "workspaces"
    ws_root.mkdir(parents=True, exist_ok=True)
    ws_dir = ws_root / f"{script_path.stem}"
    if ws_dir.exists():
        shutil.rmtree(ws_dir, ignore_errors=True)
    # Create workspace and copy only the target solution.py
    ws_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(script_path, ws_dir / script_path.name)
    # Link workspace ./input and ./submission to container-level dirs
    for name in ("input", "submission"):
        link = ws_dir / name
        if not link.exists():
            try:
                link.symlink_to(APP_DIR / name)
            except Exception:
                pass
    return ws_dir

def _get_host_ids() -> tuple[Optional[int], Optional[int]]:
    try:
        uid = int(os.environ.get("HOST_UID", ""))
        gid = int(os.environ.get("HOST_GID", ""))
        return uid, gid
    except Exception:
        return None, None

def _chown_recursive(path: Path, uid: Optional[int], gid: Optional[int]) -> None:
    if uid is None or gid is None:
        return
    try:
        for root, dirs, files in os.walk(path):
            for name in dirs:
                try:
                    os.chown(os.path.join(root, name), uid, gid)
                except Exception:
                    pass
            for name in files:
                try:
                    os.chown(os.path.join(root, name), uid, gid)
                except Exception:
                    pass
        try:
            os.chown(str(path), uid, gid)
        except Exception:
            pass
    except Exception:
        pass

def _copy_into(src: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        for root, dirs, files in os.walk(src):
            rel = Path(root).relative_to(src)
            cur_dst = dst_dir / rel
            cur_dst.mkdir(parents=True, exist_ok=True)
            for f in files:
                s = Path(root) / f
                d = cur_dst / f
                shutil.copy2(s, d)
    else:
        shutil.copy2(src, dst_dir / src.name)

def stage_input(input_src: Optional[str]) -> None:
    if not input_src:
        return
    p = Path(input_src)
    if not p.is_absolute():
        p = (APP_DIR / input_src).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Input source not found: {p}")
    _copy_into(p, APP_DIR / "input")

def stage_input_from_data(data_root: str, task: str) -> None:
    """
    Copy the contents of {data_root}/{task}/prepared/public into /app/input (not the 'public' dir itself).
    """
    src = Path(data_root) / task / "prepared" / "public"
    if not src.exists():
        raise FileNotFoundError(f"Expected data at: {src}")
    dest = APP_DIR / "input"
    dest.mkdir(parents=True, exist_ok=True)
    clean_dir(dest)
    for item in src.iterdir():
        if item.is_dir():
            shutil.copytree(item, dest / item.name)
        else:
            shutil.copy2(item, dest / item.name)

def export_submission(dst_path: Optional[str]) -> None:
    if not dst_path:
        return
    dst = Path(dst_path)
    if not dst.is_absolute():
        dst = (APP_DIR / dst_path).resolve()
    (APP_DIR / "submission").mkdir(parents=True, exist_ok=True)
    dst.mkdir(parents=True, exist_ok=True)
    _copy_into(APP_DIR / "submission", dst)
    uid, gid = _get_host_ids()
    _chown_recursive(dst, uid, gid)

def export_submission_to_solution_dir(script_path: Path) -> None:
    solution_name = script_path.stem
    in_app_named = APP_DIR / f"submission_{solution_name}"
    copy_tree(APP_DIR / "submission", in_app_named)
    uid, gid = _get_host_ids()
    _chown_recursive(in_app_named, uid, gid)
    host_dst = script_path.parent / f"submission_{solution_name}"
    if host_dst.exists():
        shutil.rmtree(host_dst, ignore_errors=True)
    shutil.copytree(in_app_named, host_dst)
    _chown_recursive(host_dst, uid, gid)

def ensure_dirs(base: Path):
    created = {"symlink": set(), "dir": set()}
    for name in ("input", "submission"):
        target = base / name
        if not target.exists():
            try:
                target.symlink_to((APP_DIR / name))
                created["symlink"].add(name)
            except Exception:
                target.mkdir(parents=True, exist_ok=True)
                created["dir"].add(name)
    return created
