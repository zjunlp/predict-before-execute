# Single-run entrypoint: prepare an isolated workspace, stage input, execute the solution script, and export submission artifacts.

import os
import sys
import shutil
import subprocess
import time
import datetime
from pathlib import Path
from typing import List

from ..util.consts import APP_DIR
from ..util.script import resolve_script
from ..util.task import infer_task_from_solution
from ..util.fs import (
    ensure_dirs,
    prepare_isolated_workspace,
    harvest_and_cleanup_working,
    stage_input,
    export_submission,
    export_submission_to_solution_dir,
)

# stage_input_from_data depends on APP_DIR and public data structure but is simple enough here
def stage_input_from_data(data_root: str, task: str) -> bool:
    from util.fs import clean_dir  # local import to avoid cycles
    root = Path(data_root)
    # Try common layouts in order (extended to be more permissive)
    candidates = [
        root / task / "prepared" / "public",
        root / task / "public",
        root / task / "prepared",
        root / task,
        # additional fallbacks that occur in some datasets
        root / "prepared" / task / "public",
        root / task / "prepared" / "private",
        root / "input" / task,
        root,  # last resort: root itself
    ]
    # debug: list candidates tried
    try:
        print(f"[stage_input] candidates: {[str(p) for p in candidates]}", file=sys.stderr)
    except Exception:
        pass

    src = next((p for p in candidates if p.exists()), None)
    if not src:
        print(f"[stage_input] data not found under {root}/{task}; skipping", file=sys.stderr)
        return False
    dest = APP_DIR / "input"
    dest.mkdir(parents=True, exist_ok=True)
    clean_dir(dest)
    copied = 0
    for item in src.iterdir():
        try:
            if item.is_dir():
                shutil.copytree(item, dest / item.name)
            else:
                shutil.copy2(item, dest / item.name)
            copied += 1
        except Exception:
            # best-effort copy; continue
            pass
    # report what was copied (helpful for diagnosing missing train.csv)
    try:
        listing = sorted([p.name for p in dest.iterdir()])
        print(f"[stage_input] staged from {src} -> {dest} (items_copied={copied}) listing={listing}", file=sys.stderr)
    except Exception:
        pass
    return True

def _strict_cpu_enabled() -> bool:
    return str(os.environ.get("SCHED_STRICT_CPU", "0")).lower() in ("1", "true", "yes")

def _parse_cpu_affinity_env() -> List[int]:
    raw = os.environ.get("CPU_AFFINITY", "").strip()
    if not raw:
        return []
    out: List[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            continue
    return out

def _apply_cpu_affinity() -> List[int]:
    cpus = _parse_cpu_affinity_env()
    if not cpus:
        return []
    try:
        os.sched_setaffinity(0, set(cpus))
        print(f"[cpu] set affinity to: {cpus}", file=sys.stderr)
    except Exception as e:
        print(f"[cpu] failed to set affinity {cpus}: {e}", file=sys.stderr)
    return cpus

def run_single(ns, unknown: List[str]) -> None:
    solution_raw = ns.solution or os.environ.get("SOLUTION_PATH") or os.environ.get("RUN_SCRIPT") or os.environ.get("SCRIPT_PATH")
    input_src = ns.input_src or os.environ.get("INPUT_SRC") or os.environ.get("DATA_PATH") or os.environ.get("INPUT_PATH")
    submission_dst = ns.submission_dst or os.environ.get("SUBMISSION_DST") or os.environ.get("OUTPUT_DST")
    data_dir = ns.data_dir or os.environ.get("DATA_DIR")
    task_name_arg = ns.task or os.environ.get("TASK_NAME")

    # Allow first positional as solution if not provided explicitly
    if not solution_raw and unknown:
        solution_raw = unknown[0]
        unknown = unknown[1:]

    # In strict CPU mode: first try to bind CPUs and propagate the suggested concurrency to the solution script.
    bound_cpus: List[int] = []
    max_conc = None
    if _strict_cpu_enabled():
        bound_cpus = _apply_cpu_affinity()
        max_conc_env = os.environ.get("CPU_MAX_CONCURRENCY")
        try:
            max_conc = int(max_conc_env) if max_conc_env else None
        except Exception:
            max_conc = None
        if max_conc is not None:
            print(f"[cpu] container-level CPU_MAX_CONCURRENCY={max_conc}", file=sys.stderr)
        if bound_cpus:
            os.environ["TASK_CPU_AFFINITY"] = ",".join(str(c) for c in bound_cpus)
        if max_conc is not None:
            os.environ["TASK_CPU_MAX_CONCURRENCY"] = str(max_conc)

    # Restore workspace preparation before building the command
    script_path = resolve_script([], override=solution_raw)
    script_dir = script_path.parent.resolve()  # noqa: F841

    workspace_dir = prepare_isolated_workspace(script_path)
    run_cwd = workspace_dir

    created_local = ensure_dirs(run_cwd)
    # Ensure ./working exists for solutions saving checkpoints
    try:
        (run_cwd / "working").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    inferred_task = infer_task_from_solution(script_path, task_name_arg)
    staged = False
    if data_dir and inferred_task:
        try:
            staged = stage_input_from_data(data_dir, inferred_task)
        except Exception as e:
            print(f"[stage_input] {e}", file=sys.stderr)
            staged = False
    if not staged and input_src:
        stage_input(input_src)

    # Diagnostic help when no input was staged
    if not staged and not input_src:
        try:
            print(f"[diag] staging failed. data_dir={data_dir} inferred_task={inferred_task}", file=sys.stderr)
            # list possible data_dir/task paths to help user spot where train.csv should be
            audit_paths = []
            if data_dir and inferred_task:
                base = Path(data_dir)
                audit_paths = [
                    base / inferred_task / "prepared" / "public",
                    base / inferred_task / "public",
                    base / inferred_task / "prepared",
                    base / inferred_task,
                    base,
                ]
            for p in audit_paths:
                try:
                    exists = p.exists()
                    contents = sorted([x.name for x in p.iterdir()]) if exists and p.is_dir() else []
                    print(f"[diag] {p} exists={exists} contents={contents}", file=sys.stderr)
                except Exception:
                    print(f"[diag] {p} exists=? (error listing)", file=sys.stderr)
            # also list container /app/input to show current state
            try:
                app_input = APP_DIR / "input"
                exists = app_input.exists()
                contents = sorted([x.name for x in app_input.iterdir()]) if exists and app_input.is_dir() else []
                print(f"[diag] {app_input} exists={exists} contents={contents}", file=sys.stderr)
            except Exception:
                pass
        except Exception:
            pass

    (APP_DIR / "submission").mkdir(parents=True, exist_ok=True)
    log_file = APP_DIR / "submission" / "exec_output.txt"

    os.chdir(run_cwd)

    try:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    except Exception:
        pass

    trace_vars_env = os.environ.get("TRACE_VARS", "")
    trace_vars = bool(getattr(ns, "trace_vars", False) or trace_vars_env.lower() in ("1", "true", "yes"))
    popen_env = os.environ.copy()
    # In strict CPU mode, propagate task-level CPU information to the child process.
    if _strict_cpu_enabled():
        if bound_cpus:
            popen_env["TASK_CPU_AFFINITY"] = ",".join(str(c) for c in bound_cpus)
        if max_conc is not None:
            popen_env["TASK_CPU_MAX_CONCURRENCY"] = str(max_conc)

    if trace_vars:
        # Create a lightweight bootstrap by copying the extracted tracer into the workspace.
        bootstrap_path = run_cwd / "__trace_bootstrap__.py"
        src_bootstrap = Path(__file__).parent / "trace" / "trace_bootstrap.py"
        try:
            # Prefer copying the real tracer file
            shutil.copy2(src_bootstrap, bootstrap_path)
        except Exception:
            # Fallback: write a tiny shim that imports the module via sys.path
            try:
                repo_root = Path(__file__).resolve().parents[1]
                shim = f"""# -*- coding: utf-8 -*-
import sys, runpy
sys.path.insert(0, {repr(str(repo_root))})
runpy.run_module("env.trace.trace_bootstrap", run_name="__main__")
"""
                with open(bootstrap_path, "w", encoding="utf-8") as f:
                    f.write(shim)
            except Exception:
                pass

        trace_log = APP_DIR / "submission" / "trace.log"
        try:
            trace_log.parent.mkdir(parents=True, exist_ok=True)
            with open(trace_log, "a", encoding="utf-8"):
                pass
        except Exception:
            pass
        popen_env["TRACE_LOG"] = str(trace_log)
        cmd = [sys.executable, "-u", bootstrap_path.name, script_path.name, *unknown]
    else:
        cmd = [sys.executable, "-u", script_path.name, *unknown]

    try:
        with open(log_file, "w", buffering=1) as lf:
            start_ts = time.time()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                # Only override env when we need to pass a custom env (trace or strict CPU mode).
                env=popen_env if trace_vars or _strict_cpu_enabled() else None,
            )

            assert proc.stdout is not None
            for line in proc.stdout:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                line_ts = f"{ts} {line}"
                sys.stdout.write(line_ts)
                lf.write(line_ts)
            returncode = proc.wait()
            elapsed = time.time() - start_ts
            runtime_msg = f"[runtime] wall_clock_seconds={elapsed:.3f}"
            try:
                sys.stdout.write(runtime_msg + "\n")
                lf.write(runtime_msg + "\n")
            except Exception:
                pass
            if returncode != 0:
                # Print English summary for single execution
                try:
                    sys.stdout.write("[run-summary] Success=0 Failures=1 Total=1\n")
                except Exception:
                    pass
                sys.exit(returncode)
            else:
                try:
                    sys.stdout.write("[run-summary] Success=1 Failures=0 Total=1\n")
                except Exception:
                    pass
    finally:
        try:
            # New: pass the "task root directory" via environment variable to util helpers,
            # so they can create directories under task/submission.
            try:
                # script_path: .../<task>/code/<solution>.py
                # task_root:   .../<task>
                task_root = script_path.parent.parent
                # Used by util.fs.export_submission_to_solution_dir, e.g.:
                #   <task_root>/submission/submission_<solution_name>/
                os.environ["SUBMISSION_TASK_ROOT"] = str(task_root)
            except Exception:
                pass
            # util.fs.export_submission_to_solution_dir should:
            #   - Prefer SUBMISSION_TASK_ROOT when it is set
            #   - Use <SUBMISSION_TASK_ROOT>/submission/submission_<stem>/ as the target directory
            export_submission_to_solution_dir(script_path)
        except Exception:
            pass

        clean_working_flag = ns.clean_working or os.environ.get("CLEAN_WORKING", "").lower() in ("1", "true", "yes")
        try:
            harvest_and_cleanup_working(run_cwd, remove=clean_working_flag)
        except Exception:
            pass

        clean_links_flag = ns.clean_links or os.environ.get("CLEAN_LINKS", "").lower() in ("1", "true", "yes")
        if clean_links_flag:
            for name in list(created_local.get("symlink", [])):
                p = run_cwd / name
                try:
                    if p.is_symlink():
                        p.unlink()
                except Exception:
                    pass
            for name in list(created_local.get("dir", [])):
                p = run_cwd / name
                try:
                    if p.exists() and p.is_dir() and not any(p.iterdir()):
                        p.rmdir()
                except Exception:
                    pass

        if ns.clean_workspace or os.environ.get("CLEAN_WORKSPACE", "").lower() in ("1", "true", "yes"):
            try:
                shutil.rmtree(workspace_dir, ignore_errors=True)
            except Exception:
                pass

        if submission_dst:
            export_submission(submission_dst)
            for name in list(created_local.get("symlink", [])):
                p = run_cwd / name
                try:
                    if p.is_symlink():
                        p.unlink()
                except Exception:
                    pass
            for name in list(created_local.get("dir", [])):
                p = run_cwd / name
                try:
                    if p.exists() and p.is_dir() and not any(p.iterdir()):
                        p.rmdir()
                except Exception:
                    pass

        if ns.clean_workspace or os.environ.get("CLEAN_WORKSPACE", "").lower() in ("1", "true", "yes"):
            try:
                shutil.rmtree(workspace_dir, ignore_errors=True)
            except Exception:
                pass

        if submission_dst:
            export_submission(submission_dst)

        # Post-run check for the trace log existence and size
        if trace_vars:
            try:
                tl = APP_DIR / "submission" / "trace.log"
                exists = tl.exists()
                size = tl.stat().st_size if exists else None
                print(f"[trace] Post-run: {tl} exists={exists} size={size}")
            except Exception as e:
                print(f"[trace] Post-run: failed to stat trace log: {e}", file=sys.stderr)
