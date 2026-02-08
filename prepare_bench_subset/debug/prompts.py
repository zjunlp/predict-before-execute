# Prompt builders for diff-based LLM editing, GPU rewrites, runtime fixes, and eval-error fixes.
from typing import List, Dict, Any, Optional
from .diff_utils import wrap_code
import os
import csv


def get_diff_instructions() -> List[str]:
    """Standard diff format instructions shown to the model."""
    return [
        "Use EXACT SEARCH/REPLACE format:",
        "<<<<<<< SEARCH",
        "# exact code to replace (must match exactly)",
        "=======",
        "# new code",
        ">>>>>>> REPLACE",
        "",
        "SEARCH block must match code exactly (including whitespace)",
        "Focus on targeted fixes, not full rewrites",
        "You can make multiple changes with multiple diff blocks",
        "Explain reasoning for each change before the diff blocks wrapped in <think> ... </think>",
    ]


def build_step_diff_prompt(
    current_step: Dict[str, Any],
    failed_step_code: str,
    error_output: str,
    prev_steps_code: Optional[str] = None,
    code_guidelines: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Prompt for step-level diff-based debugging."""
    prompt: Dict[str, Any] = {
        "Introduction": "You are debugging a failed ML code step. Use precise SEARCH/REPLACE format to fix errors.",
        "Current Step": str(current_step),
        "Failed Code": wrap_code(failed_step_code),
        "Error Output": error_output,
        "Instructions": {
            "IMPORTANT": [
                "You can ONLY modify the Failed Code shown above",
                "Do NOT search for code from previous steps",
                "Your SEARCH blocks must match code in the Failed Code section exactly",
                "Focus only on fixing the current step's implementation",
            ],
            "Diff Format": get_diff_instructions(),
        },
    }
    if code_guidelines:
        prompt["Instructions"].update(code_guidelines)
    if prev_steps_code:
        prompt["Previous Steps Code"] = f"Continue from (DO NOT MODIFY):\n\n{wrap_code(prev_steps_code)}\n\n"
    return prompt


def build_simple_diff_prompt(description: str, code: str) -> str:
    """
    Build a concise, single-message prompt for diff-based editing.
    Input: free-form description (bug/requirements) + full original code.
    Output expectation: one or more SEARCH/REPLACE diff blocks.
    """
    lines: List[str] = []
    lines.append("You are a precise code editing assistant.")
    lines.append("Modify the following code to satisfy the user's description.")
    lines.append("")
    lines.append("Description:")
    lines.append(description.strip())
    lines.append("")
    lines.append("Original Code:")
    lines.append(wrap_code(code))
    lines.append("")
    lines.append(
        "Respond ONLY with a single brief explanation wrapped in <think>...</think>, "
        "followed immediately by one or more SEARCH/REPLACE diff blocks in the exact format below. "
        "The <think> must come before any diff blocks."
    )
    lines.extend(get_diff_instructions())
    return "\n".join(lines)


def build_custom_diff_prompt(
    description: str,
    code: str,
    preface_lines: Optional[List[str]] = None,
) -> str:
    """
    Generic variant of build_simple_diff_prompt that lets callers inject
    task-specific instructions before the free-form description.
    """
    lines: List[str] = []
    lines.append("You are a precise code editing assistant.")
    if preface_lines:
        lines.extend(preface_lines)
        lines.append("")  # blank line after task-specific block
    lines.append("Modify the following code to satisfy the user's description.")
    lines.append("")
    lines.append("Description:")
    lines.append(description.strip())
    lines.append("")
    lines.append("Original Code:")
    lines.append(wrap_code(code))
    lines.append("")
    lines.append(
        "Respond ONLY with a single brief explanation wrapped in <think>...</think>, "
        "followed immediately by one or more SEARCH/REPLACE diff blocks in the exact format below. "
        "The <think> must come before any diff blocks."
    )
    lines.extend(get_diff_instructions())
    return "\n".join(lines)


def build_gpu_rewrite_diff_prompt(code: str) -> str:
    """
    Build a diff-style prompt for batch rewriting XGBoost/LightGBM
    training/inference sections to run on GPU, without any OpenCL usage.

    This is specifically targeting gradient boosting models implemented
    with LightGBM (LGBMClassifier / LGBMRegressor / Booster) and
    XGBoost (XGBClassifier / XGBRegressor / Booster).
    """
    gpu_preface: List[str] = [
        "You are given Python code that may use XGBoost and/or LightGBM "
        "for gradient boosting training and inference (e.g. XGBClassifier, "
        "XGBRegressor, LGBMClassifier, LGBMRegressor).",
        "",
        "Primary goal:",
        "- Move these gradient boosting models (LightGBM / XGBoost) onto GPU as much as reasonably possible,",
        "- While keeping external APIs and overall training / inference logic unchanged,",
        "- And completely removing any OpenCL-based configuration.",
        "",
        "You are allowed to use any **accurate and supported** GPU acceleration methods provided by XGBoost and LightGBM,",
        "as long as they are consistent with the libraries' public APIs and version assumptions.",
        "",
        "High-level objectives:",
        "- For **XGBoost** (XGBClassifier / XGBRegressor and similar):",
        "  * Ensure models run on NVIDIA GPU via CUDA.",
        "  * Prefer the modern XGBoost 2.x style with `device='cuda'` + `tree_method='gpu_hist'`.",
        "  * Use `xgboost.QuantileDMatrix` on GPU for both training and prediction where feasible.",
        "",
        "- For **LightGBM** (LGBMClassifier / LGBMRegressor and similar):",
        "  * Ensure models run on GPU using LightGBM's own GPU backend (e.g. `device='cuda'` or equivalent).",
        "  * Keep the high-level hyperparameters (learning rate, depth, num_leaves, n_estimators, seeds, metrics) unchanged.",
        "",
        "------------------------------------------------",
        "**Example (XGBoost CPU/OpenCL style -> CUDA + QuantileDMatrix)**",
        "",
        "**Before:**",
        "```python",
        "clf = xgb.XGBClassifier(",
        "        tree_method='gpu_hist',      # legacy GPU hint, may emit deprecation warning",
        "        gpu_id=0,",
        "        n_estimators=100,",
        "        seed=42)",
        "clf.fit(X_train, y_train)",
        "prob = clf.predict_proba(X_test)",
        "```",
        "",
        "**After:**",
        "```python",
        "// GPU-ized by LLM",
        "import xgboost as xgb",
        "",
        "clf = xgb.XGBClassifier(",
        "        device='cuda',               # explicit CUDA device",
        "        tree_method='gpu_hist',      # GPU tree method",
        "        n_estimators=100,",
        "        seed=42,",
        "        n_jobs=1)",
        "dtrain = xgb.QuantileDMatrix(X_train, y_train, device='cuda')",
        "clf.fit(dtrain, y_train)",
        "dpred = xgb.QuantileDMatrix(X_test, device='cuda')",
        "prob = clf.predict_proba(dpred)",
        "```",
        "------------------------------------------------",
        "**Example (LightGBM GPU backend)**",
        "```python",
        "import lightgbm as lgb",
        "",
        "clf = lgb.LGBMClassifier(",
        "        device='cuda',",
        "        n_estimators=100,",
        "        seed=42,",
        "        n_jobs=1)",
        "clf.fit(X_train, y_train)",
        "prob = clf.predict_proba(X_test)",
        "```",]

    description = (
        "Rewrite all LightGBM and XGBoost (gradient boosting) training/inference code in this file so that these "
        "boosting models run on GPU (CUDA for XGBoost, GPU backend for LightGBM) using accurate, supported GPU "
        "acceleration options, with no OpenCL-specific parameters remaining, while keeping all outer APIs and "
        "non-ML logic unchanged."
    )
    return build_custom_diff_prompt(description=description, code=code, preface_lines=gpu_preface)


def build_gpu_rewrite_diff_prompt_with_context(code: str, context: str) -> str:
    """
    Variant of build_gpu_rewrite_diff_prompt that prepends a short natural-language
    context (e.g., task name / solution key / first keyword from kw.txt) so the LLM
    knows which solution is being processed in this batch GPU rewrite.
    """
    base_prompt = build_gpu_rewrite_diff_prompt(code=code)
    ctx = context.strip()
    if not ctx:
        return base_prompt
    header = f"Target solution context: {ctx}\n\n"
    return header + base_prompt


def build_runtime_error_fix_prompt(
    solution_path: str,
    code: str,
    exec_output_tail: str,
) -> str:
    """
    Build a concise prompt for fixing runtime errors given a solution .py file
    and the tail of its exec_output (between --- exec_output tail --- and --- end tail ---).

    This is used to repair buggy benchmark solutions based on their recorded
    runtime stack traces / error messages.
    """
    description_lines: List[str] = [
        "You are fixing a Python solution file that failed at runtime in a benchmark run.",
        "",
        f"Solution path: {solution_path}",
        "",
        "The following is the tail of exec_output.txt captured during execution. ",
        "It typically includes the Python traceback and the final exception:",
        "",
        exec_output_tail.strip(),
        "",
        "Your task:",
        "- Modify ONLY this solution file so that it no longer raises this runtime error.",
        "- Prefer minimal and local changes that directly address the error.",
        "- Do NOT modify irrelevant code unless required to fix the bug.",
        "- Preserve existing behavior and interfaces as much as possible.",
        "",
        "Environment notes:",
        "- Code runs inside a Docker-based execution sandbox.",
        "- The filesystem is mostly read-only and common NLTK data paths are not writable.",
        "- Huggingface channel is accessible, but the intereactive downloading of NLTK data or installing new packages is not guaranteed.",
    ]

    description = "\n".join(description_lines)

    lines: List[str] = []
    lines.append("You are a precise code editing assistant.")
    lines.append("Use the runtime error log below to fix the Python code.")
    lines.append("")
    lines.append("Description and runtime error context:")
    lines.append(description.strip())
    lines.append("")
    lines.append("Original Code:")
    lines.append(wrap_code(code))
    lines.append("")
    lines.append(
        "Respond ONLY with a single brief explanation wrapped in <think>...</think>, "
        "followed immediately by one or more SEARCH/REPLACE diff blocks in the exact format below. "
        "The <think> must come before any diff blocks."
    )
    lines.extend(get_diff_instructions())
    return "\n".join(lines)


def _load_competition_public_dir(competition_id: str, data_root: str) -> Optional[str]:
    """
    Best-effort: return the public data dir for this competition, if it exists.
    Expected layout:
      <data_root>/<competition_id>/prepared/public
    """
    if not competition_id:
        return None
    base = data_root
    public_dir = os.path.join(base, competition_id, "prepared", "public")
    if os.path.isdir(public_dir):
        return public_dir
    return None


def _load_description_md(public_dir: str) -> Optional[str]:
    """Load description.md content if present."""
    path = os.path.join(public_dir, "description.md")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


def _load_sample_submission_preview(public_dir: str, max_rows: int = 5) -> Optional[str]:
    """
    Load sample_submission.csv and build a short text with:
      - basic meta (rows, cols, column names)
      - first max_rows raw lines (as CSV).
    """
    path = os.path.join(public_dir, "sample_submission.csv")
    if not os.path.isfile(path):
        return None

    try:
        lines: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            # we want both meta and raw first lines; easiest is readlines
            raw_lines = f.readlines()
        if not raw_lines:
            return None

        header = raw_lines[0].rstrip("\n")
        cols = next(csv.reader([header]))
        num_cols = len(cols)

        # estimate rows from file length (minus header)
        num_rows = max(0, len(raw_lines) - 1)

        lines.append("Sample submission metadata:")
        lines.append(f"- num_rows (excluding header): {num_rows}")
        lines.append(f"- num_columns: {num_cols}")
        lines.append(f"- columns: {', '.join(cols)}")
        lines.append("")
        lines.append(f"First {min(max_rows, len(raw_lines))} lines of sample_submission.csv:")
        for raw in raw_lines[:max_rows]:
            lines.append(raw.rstrip("\n"))

        return "\n".join(lines).strip()
    except Exception:
        return None


def build_eval_error_fix_prompt(
    solution_path: str,
    code: str,
    competition_id: str,
    grader_error: Optional[str],
    score: Optional[float],
    submission_path: Optional[str],
    full_grading_json: Optional[str],
    data_root: str,
) -> str:
    """
    Build a prompt for fixing *evaluation-time* errors (invalid submissions).
    These come from error_eval_*.json and include grader_error, score, etc.
    """
    desc_lines: List[str] = [
        "You are fixing a Kaggle-style benchmark solution whose generated submission",
        "file failed during *evaluation* (grading), not at runtime.",
        "",
        f"Solution path: {solution_path}",
        f"Competition id: {competition_id}",
        "",
        "The grading system reported the following problem:",
        f"- grader_error: {grader_error or 'None'}",
        f"- score: {score!r}",
    ]
    if submission_path:
        desc_lines.append(f"- submission_path: {submission_path}")
    if full_grading_json:
        desc_lines.extend(
            [
                "",
                "Full grading JSON (for context, do NOT try to modify it directly):",
                full_grading_json.strip(),
            ]
        )

    # Try to enrich context with competition description + sample submission preview
    public_dir = _load_competition_public_dir(competition_id, data_root=data_root)
    if public_dir:
        description_md = _load_description_md(public_dir)
        sample_preview = _load_sample_submission_preview(public_dir, max_rows=5)

        if description_md:
            desc_lines.extend(
                [
                    "",
                    "Competition description.md (public data):",
                    "----------------------------------------",
                    description_md,
                ]
            )
        if sample_preview:
            desc_lines.extend(
                [
                    "",
                    "Sample submission (metadata + first 5 lines):",
                    "---------------------------------------------",
                    sample_preview,
                ]
            )

    # Explicit guidance on typical reasons for metric/score == 0.0
    desc_lines.extend(
        [
            "",
            "When the metric/score is exactly 0.0 (even if grader_error is null),",
            "typical root causes include:",
            "- Every prediction is wrong (no position where y_pred equals y_true).",
            "- Label format mismatch: answers use strings while the submission uses integers (or vice versa),",
            "  or there are leading/trailing spaces or case differences that make element-wise equality fail.",
            "- Using a different label naming/encoding than the answers (e.g. answers use 'class_A'/'class_B',",
            "  but the submission uses 0/1 or other names).",
            "- Label ordering or alignment issues: even if ids are identical in both files,",
            "  if the meaning of each label index differs between answers and submission,",
            "  all comparisons will be counted as incorrect.",
            "- The submission predicts the same single wrong class (model collapse), so no row matches the true label.",
            "",
            "Your task:",
            "- Modify ONLY this solution Python file so that future runs will produce a *valid* submission.csv for this competition.",
            "- Focus on the root cause indicated by grader_error / score:",
            "  * Examples: missing required columns (e.g. Id, Comment), wrong number of rows, NaN/inf values,",
            "    probability rows not summing to 1, or any of the score==0.0 issues listed above.",
            "- Make minimal, local changes that preserve the original modeling logic as much as possible.",
            "- Do NOT hard-code paths to submission.csv; keep using the existing output location and naming convention.",
            "",
            "Environment notes:",
            "- Code runs in a Docker sandbox with standard Kaggle-like dataset layout.",
            "- The grader reads submission.csv from the path shown above.",
            "",
            "Shortcuts (CRITICAL — submission-format issues):",
            "- For any problems related to missing columns, column-name mismatches, wrong number of rows, or other",
            "  submission formatting errors, follow this single reliable approach:",
            "  1. At the **VERY BEGINNING** of the program, load './input/sample_submission.csv' into a dataframe or CSV object",
            "     and treat it as the official answer template (the submission sheet). **DON'T THE FUCK TRY TO LOAD IT WHEN NEEDED, REMEMBER TO LOAD IT ONCE AT THE START AND USE IT THROUGHOUT THE PROGRAM. YOU HAVE MADE MISTAKES MULTIPLE TIMES, FOOL!**",
            "  2. Do NOT recreate headers or row order manually; use the loaded template to guarantee correct columns,",
            "     header names, dtypes and row count (align by Id/index when filling predictions).",
            "  3. Fill the prediction columns in this template (ensuring alignment), then write the resulting file",
            "     to the original submission path used by the program.",
            "  4. Preserve header names exactly, avoid adding or dropping columns, and do not change row order.",
            "  5. **Do not try any fallback** to generate submission.csv manually; always use the sample_submission.csv as the template.",
            "- This template-based approach resolves most grader errors related to format/shape. The sample_submission",
            "  preview (if available) is included below in this prompt — use it as the template whenever present.",
        ]
    )

    description = "\n".join(desc_lines)

    lines: List[str] = []
    lines.append("You are a precise code editing assistant.")
    lines.append("Use the grading error information below to fix the Python code so that it generates a valid submission.")
    lines.append("")
    lines.append("Description and grading error context:")
    lines.append(description.strip())
    lines.append("")
    lines.append("Original Code:")
    lines.append(wrap_code(code))
    lines.append("")
    lines.append(
        "Respond ONLY with a single brief explanation wrapped in <think>...</think>, "
        "followed immediately by one or more SEARCH/REPLACE diff blocks in the exact format below. "
        "The <think> must come before any diff blocks."
    )
    lines.extend(get_diff_instructions())
    return "\n".join(lines)
