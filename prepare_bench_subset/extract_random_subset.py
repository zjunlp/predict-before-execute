#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Randomly extract up to N Python solution files per task into a subset directory."""

import os
import sys
import shutil
import random
import argparse
from typing import List, Dict

def read_tasks(task_file: str) -> List[str]:
    tasks: List[str] = []
    with open(task_file, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                tasks.append(t)
    return tasks

def collect_all_py_files(root: str) -> List[str]:
    py_files: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".py"):
                py_files.append(os.path.join(dirpath, fn))
    return py_files

def map_task_to_files(tasks: List[str], all_files: List[str]) -> Dict[str, List[str]]:
    # Case-insensitive match: task substring contained anywhere in the file path
    files_lower = [(fp, fp.lower()) for fp in all_files]
    mapping: Dict[str, List[str]] = {t: [] for t in tasks}
    for task in tasks:
        task_l = task.lower()
        # Add files whose path contains the task name
        mapping[task] = [orig for orig, low in files_lower if task_l in low]
    return mapping

def copy_subset_for_task(task: str, files: List[str], solutions_root: str, out_root: str, k: int, dry_run: bool, rng: random.Random) -> List[str]:
    if not files:
        print(f"[WARN] Task '{task}': no matching .py files found.", file=sys.stderr)
        return []
    selected = files if len(files) <= k else rng.sample(files, k)

    copied: List[str] = []
    dest_dir = os.path.join(out_root, task)
    used_names = set()  # avoid collisions within current selection

    for src in selected:
        base_name = os.path.basename(src)
        candidate = base_name
        i = 1
        # ensure unique target filename considering both existing files and current batch
        while True:
            dest = os.path.join(dest_dir, candidate)
            if (candidate not in used_names) and (not os.path.exists(dest)):
                break
            name_no_ext, ext = os.path.splitext(base_name)
            candidate = f"{name_no_ext}_{i}{ext}"
            i += 1
        used_names.add(candidate)

        if not dry_run:
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(src, dest)
        copied.append(dest)

    print(f"[OK] Task '{task}': found={len(files)}, selected={len(selected)}" + (" (dry-run)" if dry_run else ""))
    return copied

def main():
    parser = argparse.ArgumentParser(description="Randomly extract up to N .py files per task from solutions.")
    parser.add_argument("--solutions_root", type=str, required=True, help="Root folder containing solution subfolders.")
    parser.add_argument("--tasks_file", type=str, required=True, help="Path to task_name.txt.")
    parser.add_argument("--out_root", type=str, required=True, help="Output root to place the subset per task.")
    parser.add_argument("--per_task", type=int, default=10, help="Number of .py files to sample per task.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--dry_run", action="store_true", help="Do not copy files; only print the plan.")
    args = parser.parse_args()

    solutions_root = os.path.abspath(args.solutions_root)
    tasks_file = os.path.abspath(args.tasks_file)
    out_root = os.path.abspath(args.out_root)

    if not os.path.isdir(solutions_root):
        print(f"[ERROR] solutions_root not found: {solutions_root}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(tasks_file):
        print(f"[ERROR] tasks_file not found: {tasks_file}", file=sys.stderr)
        sys.exit(1)
    if not args.dry_run:
        os.makedirs(out_root, exist_ok=True)

    tasks = read_tasks(tasks_file)
    if not tasks:
        print("[ERROR] No tasks loaded from tasks_file.", file=sys.stderr)
        sys.exit(1)

    all_py = collect_all_py_files(solutions_root)
    if not all_py:
        print("[ERROR] No .py files found under solutions_root.", file=sys.stderr)
        sys.exit(1)

    mapping = map_task_to_files(tasks, all_py)
    rng = random.Random(args.seed)

    total_selected = 0
    for task, files in mapping.items():
        copied = copy_subset_for_task(task, files, solutions_root, out_root, args.per_task, args.dry_run, rng)
        total_selected += len(copied) if not args.dry_run else min(len(files), args.per_task)

    print(f"[DONE] Tasks processed: {len(tasks)}, total selected: {total_selected}" + (" (dry-run)" if args.dry_run else ""))

if __name__ == "__main__":
    main()
