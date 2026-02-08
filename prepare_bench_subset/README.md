# 🏗️ Corpus Construction Toolkit

`prepare_bench_subset` is a toolkit for sampling a **subset** of solutions from the original unfiltered pool (`solutions_all`) and then running a series of preparation steps (cleaning, execution, grading, and grouping) to construct the **Data-centric Solution Preference** corpus.

It exposes several CLI utilities that together form a pipeline:

1.  **🍒 Subset extraction**
    *   `extract_subset.py`: select solutions according to a semantic keyword specification in `tasks.json`.
    *   `extract_random_subset.py`: randomly sample `.py` solutions per task.

2.  **🧹 (Optional) LLM-based cleanup / fixing**
    *   `clean.py`: compile-check and optionally auto-fix Python solutions using LLM-based tools.

3.  **🏃 Run solutions**
    *   `run.py`: execute solutions (single run or batched in Docker) to produce groundtruth submission.

4.  **📝 Grade submissions**
    *   `grade.py`: grade generated `submission.csv` files, either for a single file, a JSONL batch, or auto-discovered runs/solutions.

5.  **⚖️ Build evaluation groups**
    *   `group.py`: read `eval_output.json` scores and build n-way comparison groups, with optional balancing of the best-index positions.

The typical workflow is:

> extract subset → (optionally) clean / fix solutions → run them → grade results (using a grader adapted from [mle-bench](https://github.com/openai/mle-bench)) → group solutions for evaluation.

---

## 📂 Expected Directory Layout

Structure is key! A minimal expected layout looks like:

```text
solutions_root/
├── <task_name>/
│   ├── annotation/
│   │   ├── annotations_semantic.json   # Semantic tags
│   │   └── keywords_by_rank.json       # Stats
│   ├── code/
│   │   ├── solution_*.py               # Source code
│   │   └── submission_solution_*/      # Execution artifacts
│   │       ├── submission.csv
│   │       ├── exec_output.txt
│   │       └── eval_output.json        # Scores
│   ├── ground_truth/
│   │   └── groups_<task_name>_n.json   # Comparison pairs (The output)
│   ├── output/output_*.txt
│   └── report/
│       ├── alignment_*.json
│       └── grade_report_*.txt
```

---

## 🔄 Pipeline Overview

This section explains what each stage in the pipeline does, which inputs it expects, and what artifacts it produces.

### 1. Extract a subset of solutions

`extract_subset.py` selects solutions **semantically** according to the specification in `tasks.json` and the per-solution annotations, then copies the selected code, outputs, and (optionally) rich evaluation artifacts.

**Semantic selection (driven by `tasks.json`):**

```bash
python -m prepare_bench_subset.extract_subset \
  --solutions-root /path/to/full_solutions \
  --subset-root /path/to/subset_solutions \
  --tasks-json /path/to/tasks.json
```

**Random selection (up to N `.py` files per task):**

```bash
python -m prepare_bench_subset.extract_random_subset \
  --solutions_root /path/to/full_solutions \
  --tasks_file /path/to/task_name.txt \
  --out_root /path/to/random_subset_solutions \
  --per_task 10 \
  --seed 42
```

### 2. (Optional) Clean / auto-fix solutions

`clean.py` runs a multi-phase LLM-assisted pipeline over all Python files. It includes:
1.  **Eval-time fixing (`eval_fix.py`)**: Fixes solutions rejected during grading.
2.  **GPU boosting rewrite (`gpu_rewrite.py`)**: Rewrites applicable algorithms (e.g., LightGBM) to use GPU.
3.  **Runtime-error fixing (`runtime_fix.py`)**: Fixes bugs based on execution logs.
4.  **Compile & syntax checking (`compile_fix.py`)**: Fixes syntax errors.

```bash
python -m prepare_bench_subset.clean \
  --root /path/to/subset_solutions \
  --workers 32 \
  --max-depth 3 \
  --eval-error-json /path/to/error_eval.json \
  --gpu-boosting-kw-file /path/to/boosting_kw_algo.txt \
  --runtime-log /path/to/runtime_logs.log \
  --data-root tasks/data \
  --verbose-log /path/to/verbose_logs.log
```

*(Note: Flags like `--runtime-log` are optional depending on which fixers you want to run.)*

### 3. Run solutions

`run.py` executes solutions in **Docker** to produce `submission.csv`.

*   **Batch mode (`--batch`)**: For mass execution. requires `docker pull johnsonzheng03/predict-before-execute`.
*   **Single-run mode**: For debugging one file.

```bash
python -m prepare_bench_subset.run \
  --batch \
  --solutions-root /path/to/subset_solutions \
  --task-file /path/to/task_name.txt \
  --dockerfile prepare_bench_subset/env/Dockerfile \
  --data-dir tasks/data \
  --max-parallel 8 \
  --clean-links \
  --clean-working \
  --clean-workspace
```

### 4. Grade submissions

`grade.py` evaluates `submission.csv` files and writes `eval_output.json`. It is adapted from MLE-bench.

*   **`auto-grade`**: Automatically finds submissions under a solution root or agent run directory.

```bash
python -m prepare_bench_subset.grade auto-grade \
  --task-list /path/to/task_name.txt \
  --solutions-dir /path/to/subset_solutions \
  --data-dir tasks/data \
  --competitions-dir tasks/competitions \
  --workers 64 \
  --error-report /path/to/error_report.json \
  --allow-zero-score
```

### 5. Build evaluation groups

`group.py` converts scores into **n-way comparison groups** (Ground Truth) for the preference task. It filters invalid scores and ensures balanced positions for the "winner".

**Example Group Format:**
```json
[
  {
    "paths": ["path/to/sol_A.py", "path/to/sol_B.py"],
    "scores": [0.85, 0.92],
    "best_index": 1,
    "full_ranking": [1, 0],
    "is_lower_better": false
  }
]
```

**Build Command:**

```bash
python -m prepare_bench_subset.group \
  --task-file /path/to/task_name.txt \
  --solutions-root /path/to/subset_solutions \
  --group-size 2 \
  --balanced \
  --seed 42
```

---

## 🚀 Example End-to-End Workflow

Below is a complete example of the pipeline:

```bash
# 1) Extract a semantic subset
python -m prepare_bench_subset.extract_subset \
  --solutions-root /path/to/source_solutions \
  --subset-root /path/to/subset_solutions \
  --tasks-json /path/to/tasks.json

# 2) (Optional) Clean and auto-fix subset solutions
python -m prepare_bench_subset.clean \
  --root /path/to/subset_solutions \
  --data-root tasks/data

# 3) Run solutions in batch mode
python -m prepare_bench_subset.run \
  --batch \
  --solutions-root /path/to/subset_solutions \
  --task-file /path/to/task_name.txt \
  --dockerfile prepare_bench_subset/env/Dockerfile \
  --data-dir tasks/data

# 4) Grade all submissions
python -m prepare_bench_subset.grade auto-grade \
  --task-list /path/to/task_name.txt \
  --solutions-dir /path/to/subset_solutions \
  --data-dir tasks/data \
  --competitions-dir tasks/competitions

# 5) Build A/B comparison groups (n=2)
python -m prepare_bench_subset.group \
  --task-file /path/to/task_name.txt \
  --solutions-root /path/to/subset_solutions \
  --group-size 2 \
  --balanced
```

---

## 🔧 Configuration Details

This repository relies on shared config files:

1.  **`/prepare_bench_subset/config/tasks.json`**:
    *   Defines semantic sampling quotas per task (e.g., "Sample 5 PyTorch solutions").
    *   Consumed by `extract_subset.py`.

2.  **`/prepare_bench_subset/config/task_name.txt`**:
    *   Plain-text list of tasks to process.

3.  **`/tasks/data/`**:
    *   Stores prepared competition data (input for execution and grading).

---

## ❓ Common Issues (Beginner Pitfalls)

1.  **Path issues**: Use absolute paths. Verify `solutions_root` and `tasks/data` exist.
2.  **Task name mismatch**: Ensure names in `task_name.txt` match directory names exactly.
3.  **Docker not prepared**: `run.py --batch` requires Docker. Pre-pull the image if needed.
4.  **No submission.csv**: Check `exec_output.txt` or use `run.py --summary-buggy` to debug failed runs.
5.  **Grading errors**: Verify `competitions-dir` points to the correct metadata location.

---

## ✅ Checklist

- [ ] I have a complete `solutions_root` (including `annotation/` and `code/`).
- [ ] My `tasks/data/<task>` directories exist.
- [ ] Task names in `task_name_subset.txt` match the directories.
- [ ] I can see `solution_*.py` under `out_root/<task>/code/`.
- [ ] `submission.csv` appears after running `run.py`.
- [ ] `eval_output.json` appears after running `grade.py`.
- [ ] `ground_truth/groups_<task>_n2.json` appears after running `group.py`.