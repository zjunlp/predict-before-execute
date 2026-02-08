import argparse
import json
import logging
import math
import random  # added
from pathlib import Path
from typing import List, Dict, Any
from itertools import combinations
from collections import Counter

def load_tasks(task_file: Path) -> List[str]:
    tasks: List[str] = []
    with task_file.open("r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name or name.startswith("#"):
                logging.warning(f"Skipping invalid or commented task line: {line.strip()}")
                continue
            tasks.append(name)
    return tasks

def find_solutions(task_dir: Path) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if not task_dir.exists():
        logging.warning(f"Task dir not found: {task_dir}")
        return results

    # Prefer submissions under task_dir/code if present, otherwise fall back to task_dir
    code_dir = task_dir / "code"
    if code_dir.exists() and code_dir.is_dir():
        search_dir = code_dir
    else:
        search_dir = task_dir

    for sub in search_dir.iterdir():
        if not sub.is_dir():
            continue
        name = sub.name
        if not name.startswith("submission_"):
            continue
        eval_path = sub / "eval_output.json"
        if not eval_path.exists():
            continue
        try:
            with eval_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            logging.warning(f"Failed to read {eval_path}: {e}")
            continue

        # Filter to valid submissions
        if not meta.get("submission_exists", True):
            logging.warning(f"Skipping submission {name}: submission_exists=False")
            continue
        if not meta.get("valid_submission", True):
            logging.warning(f"Skipping submission {name}: valid_submission=False")
            continue
        if "score" not in meta or "is_lower_better" not in meta:
            logging.warning(f"Skipping submission {name}: missing score or is_lower_better in metadata")
            continue

        # Derive solution.py from submission_ prefix.
        # Prefer solution under the code dir (if used), otherwise check task_dir.
        sol_filename = name[len("submission_"):] + ".py"
        sol_path_code = (code_dir / sol_filename) if code_dir.exists() else None
        sol_path_task = task_dir / sol_filename
        sol_path = sol_path_code if (sol_path_code is not None and sol_path_code.exists()) else sol_path_task

        if not sol_path.exists():
            # Keep the derived path anyway, but warn (show both checked locations)
            checked = []
            if sol_path_code is not None:
                checked.append(str(sol_path_code))
            checked.append(str(sol_path_task))
            logging.warning(f"Solution file missing (checked {', '.join(checked)}). Using {sol_path} as placeholder.")

        results.append({
            "solution_path": str(sol_path),
            "score": float(meta["score"]),
            "is_lower_better": bool(meta["is_lower_better"]),
            "eval_path": str(eval_path),
        })
    return results

def build_groups(entries: List[Dict[str, Any]], group_size: int, balanced: bool = True) -> List[Dict[str, Any]]:
    """
    Build n-way groups without duplicate scores and compute the best solution index (best_index)
    and full ranking (full_ranking) within each group.

    If balanced=True:
      - For each candidate group, internally reorder the items so that the best solution
        appears at each position as evenly as possible across all groups (max-min <= 1).
      - During reordering, the corresponding paths/scores are swapped accordingly.
      - full_ranking and best_index are then recomputed based on the reordered scores.

    If balanced=False:
      - Return all candidate groups without reordering; original order is kept and the
        distribution of best_index positions may be imbalanced.
    """
    groups: List[Dict[str, Any]] = []
    if not entries or len(entries) < group_size:
        logging.warning(f"Not enough entries to form groups. Required: {group_size}, Found: {len(entries)}")
        return groups

    is_lower_better = entries[0]["is_lower_better"]
    filtered: List[Dict[str, Any]] = []
    for e in entries:
        if e["is_lower_better"] != is_lower_better:
            logging.warning(
                f"Inconsistent is_lower_better for {e['solution_path']}. Expected {is_lower_better}, got {e['is_lower_better']}. Skipping."
            )
            continue
        filtered.append(e)

    if len(filtered) < group_size:
        logging.warning(f"Not enough valid entries after filtering. Required: {group_size}, Found: {len(filtered)}")
        return groups

    # Enumerate all candidate combinations
    candidates: List[Dict[str, Any]] = []
    for combo in combinations(range(len(filtered)), group_size):
        scores = [float(filtered[i]["score"]) for i in combo]
        # Skip combinations with duplicate or near-duplicate scores
        if any(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for i, a in enumerate(scores) for b in scores[i+1:]):
            logging.warning(f"Skipping combination {combo}: contains duplicate or near-duplicate scores {scores}")
            continue
        paths = [filtered[i]["solution_path"] for i in combo]
        if is_lower_better:
            ranking = sorted(range(group_size), key=lambda i: scores[i])
        else:
            ranking = sorted(range(group_size), key=lambda i: scores[i], reverse=True)
        best_index = ranking[0]

        # No longer save indices
        candidates.append({
            "paths": paths,
            "scores": scores,
            "best_index": best_index,
            "full_ranking": ranking,
            "is_lower_better": is_lower_better,
        })

    if not balanced:
        return candidates  # Directly return all candidates (no indices)

    # balanced=True: For each candidate group, internally reorder to place the best solution at the target position,
    # balancing the best_index distribution
    counts = [0] * group_size
    groups = []

    # Ensure reproducible traversal order
    for cand in sorted(candidates, key=lambda c: tuple(c["paths"])):
        target_pos = min(range(group_size), key=lambda i: counts[i])

        paths = cand["paths"][:]
        scores = cand["scores"][:]
        curr_best = cand["best_index"]

        # Move the best solution to target_pos (sync paths/scores)
        if curr_best != target_pos:
            paths[curr_best], paths[target_pos] = paths[target_pos], paths[curr_best]
            scores[curr_best], scores[target_pos] = scores[target_pos], scores[curr_best]

        # Recompute full_ranking and best_index (based on reordered scores)
        if cand["is_lower_better"]:
            ranking = sorted(range(group_size), key=lambda i: scores[i])
        else:
            ranking = sorted(range(group_size), key=lambda i: scores[i], reverse=True)
        best_index = ranking[0]

        counts[best_index] += 1
        groups.append({
            "paths": paths,
            "scores": scores,
            "best_index": best_index,
            "full_ranking": ranking,
            "is_lower_better": cand["is_lower_better"],
        })

    return groups

def save_groups(task_dir: Path, groups: List[Dict[str, Any]], filename: str) -> None:
    out_path = task_dir / filename
    try:
        # Ensure parent directories exist (e.g., task_dir/ground_truth)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(groups, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Failed to write groups to {out_path}: {e}")

# added: verification helper
def verify_groups(groups: List[Dict[str, Any]]) -> bool:
    ok = True
    for gi, g in enumerate(groups):
        scores = g.get("scores", [])
        is_lower_better = g.get("is_lower_better", True)
        bi = g.get("best_index", None)
        if not scores or bi is None:
            logging.error(f"[group {gi}] invalid group structure.")
            ok = False
            continue
        best = min(range(len(scores)), key=lambda i: scores[i]) if is_lower_better \
               else max(range(len(scores)), key=lambda i: scores[i])
        if bi != best:
            logging.error(f"[group {gi}] best_index mismatch: got {bi}, expected {best}. scores={scores}")
            ok = False
        ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=not is_lower_better)
        if g.get("full_ranking") != ranking:
            logging.error(f"[group {gi}] full_ranking mismatch: got {g.get('full_ranking')}, expected {ranking}")
            ok = False
    dist = Counter(g["best_index"] for g in groups if "best_index" in g)
    logging.info(f"best_index distribution: {dict(dist)}")
    if dist:
        mx, mn = max(dist.values()), min(dist.values())
        logging.info(f"best_index balance (max-min): {mx - mn}")
    return ok

def main():
    parser = argparse.ArgumentParser(description="Build unordered n-way groups per task using eval_output.json scores.")
    parser.add_argument("-t", "--task-file", type=str, required=True, help="Path to task_name.txt")
    parser.add_argument("-s", "--solutions-root", type=str, required=True, help="Root dir of solutions_subset")
    parser.add_argument("-n", "--group-size", type=int, default=2, help="Group size n (default 2; n=2 covers A/B)")
    parser.add_argument("--overwrite", action="store_true", help="If set, overwrite existing group file by deleting it and rebuilding.")
    parser.add_argument(
        "--balanced",
        action="store_true",
        default=True,
        help="Balance best_index distribution across groups (each position appears equally or near equally)."
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Maximum number of groups to save. If specified, randomly sample from all generated groups."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42). Used when --max-groups is specified."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Set random seed for reproducibility
    random.seed(args.seed)

    task_file = Path(args.task_file)
    solutions_root = Path(args.solutions_root)

    tasks = load_tasks(task_file)
    if not tasks:
        logging.warning("No tasks found.")
        return

    for task in tasks:
        task_dir = solutions_root / task
        logging.info(f"Processing task: {task} -> {task_dir}")
        entries = find_solutions(task_dir)
        if not entries:
            logging.warning(f"No valid solutions found for task: {task}")
            continue
        groups = build_groups(entries, args.group_size, balanced=args.balanced)

        # Apply max_groups limit if specified
        if args.max_groups is not None and len(groups) > args.max_groups:
            logging.info(f"Sampling {args.max_groups} groups from {len(groups)} total groups (seed={args.seed})")
            groups = random.sample(groups, args.max_groups)

        # verify correctness and balance
        if not verify_groups(groups):
            logging.error("Verification failed. Skip saving for this task.")
            continue

        # Output to task_dir/ground_truth/groups_<task>_n{n}.json
        out_filename = f"groups_{task}_n{args.group_size}.json"
        out_path = task_dir / "ground_truth" / out_filename

        need_rebuild = True
        if out_path.exists():
            if args.overwrite:
                try:
                    out_path.unlink()
                    logging.info(f"--overwrite enabled: removed existing file {out_path}")
                except Exception as e:
                    logging.warning(f"Failed to remove existing file {out_path}: {e}. Will attempt to rebuild anyway.")
                need_rebuild = True
            else:
                try:
                    with out_path.open("r", encoding="utf-8") as f:
                        existing = json.load(f)
                    existing_len = len(existing) if isinstance(existing, list) else -1
                    if existing_len == len(groups):
                        logging.info(f"{out_path.name} exists with correct count ({existing_len}). Skipping rebuild.")
                        need_rebuild = False
                    else:
                        logging.info(f"{out_path.name} exists but count mismatch (found {existing_len}, expected {len(groups)}). Rebuilding.")
                except Exception as e:
                    logging.warning(f"Failed to read {out_path}: {e}. Rebuilding.")

        if need_rebuild:
            # Pass relative path including ground_truth so save_groups creates folders and writes there
            save_groups(task_dir, groups, filename=f"ground_truth/{out_filename}")
            logging.info(f"Saved {len(groups)} groups to {out_path}")

if __name__ == "__main__":
    main()
