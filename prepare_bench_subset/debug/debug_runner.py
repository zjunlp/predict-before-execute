# Core LLM-based code editing entrypoints for diff-based GPU rewrites and error fixing (runtime and eval).
import ast
import logging
from typing import Optional, Tuple

from .diff_utils import extract_diff_blocks, apply_diff_to_code
from .prompts import (
    build_simple_diff_prompt,
    build_gpu_rewrite_diff_prompt_with_context,
    build_runtime_error_fix_prompt,
    build_eval_error_fix_prompt,
)
from ...backend import chat_complete

logger = logging.getLogger("debug")


def edit_code_with_llm(
    code: str,
    description: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    retries: int = 3,
    validate_syntax: bool = True,
    max_tokens: Optional[int] = None,
    system_prompt: str = "You are a precise code editing assistant. Only output SEARCH/REPLACE diffs.",
) -> str:
    """
    Edit code using an LLM via SEARCH/REPLACE diffs ONLY.
    Returns original code if all attempts fail. No execution is performed.
    """
    original_code = code
    temp = temperature

    for attempt in range(1, max(1, retries) + 1):
        prompt_text = build_simple_diff_prompt(description=description, code=code)

        logger.info(f"[LLM] Calling chat_complete (attempt {attempt}/{retries}, temp={temp}, max_tokens={max_tokens})")

        try:
            response = chat_complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                ],
                model=model,
                temperature=max(0.0, min(1.0, temp)),
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"[LLM] chat_complete failed: {e}")
            continue

        response = response or ""
        logger.info(f"[LLM] Response length: {len(response)}")

        diff_blocks = extract_diff_blocks(response)
        if diff_blocks:
            logger.info(f"[LLM] Extracted {len(diff_blocks)} diff blocks")
            applied = sum(1 for s, _ in diff_blocks if s in code)
            modified = apply_diff_to_code(code, diff_blocks)
            if not validate_syntax:
                logger.info(f"Applied {applied}/{len(diff_blocks)} diff blocks (syntax not validated).")
                return modified
            try:
                ast.parse(modified)
                logger.info(f"Applied {applied}/{len(diff_blocks)} diff blocks, syntax OK.")
                return modified
            except Exception as e:
                logger.info(f"Syntax check failed after diff (attempt {attempt}): {e}")

        logger.info(f"No valid SEARCH/REPLACE diff blocks. Retrying ({attempt}/{retries})...")
        temp = max(0.0, temp - 0.1)

    logger.error("LLM editing failed after retries. Returning original code.")
    return original_code


def gpu_rewrite_code_with_llm(
    code: str,
    solution_context: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    retries: int = 3,
    validate_syntax: bool = True,
    max_tokens: Optional[int] = None,
    system_prompt: str = "You are a precise code editing assistant. Only output SEARCH/REPLACE diffs.",
    return_raw_response: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Specialized entrypoint: rewrite XGBoost/LightGBM training & inference
    code in `code` to GPU, using the GPU rewrite prompt and SEARCH/REPLACE diffs ONLY.
    """
    original_code = code
    temp = temperature
    last_response: Optional[str] = None

    for attempt in range(1, max(1, retries) + 1):
        prompt_text = build_gpu_rewrite_diff_prompt_with_context(
            code=code,
            context=solution_context,
        )

        logger.info(
            f"[LLM][GPU-REWRITE] Calling chat_complete (attempt {attempt}/{retries}, "
            f"temp={temp}, max_tokens={max_tokens}, context={solution_context!r})"
        )

        try:
            response = chat_complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                ],
                model=model,
                temperature=max(0.0, min(1.0, temp)),
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"[LLM][GPU-REWRITE] chat_complete failed: {e}")
            continue

        last_response = response or ""
        logger.info(f"[LLM][GPU-REWRITE] Response length: {len(last_response)}")

        diff_blocks = extract_diff_blocks(last_response)
        if diff_blocks:
            logger.info(f"[LLM][GPU-REWRITE] Extracted {len(diff_blocks)} diff blocks")
            applied = sum(1 for s, _ in diff_blocks if s in code)
            modified = apply_diff_to_code(code, diff_blocks)
            if not validate_syntax:
                logger.info(f"[LLM][GPU-REWRITE] Applied {applied}/{len(diff_blocks)} diff blocks (no syntax validation).")
                return (modified, last_response if return_raw_response else None)
            try:
                ast.parse(modified)
                logger.info(f"[LLM][GPU-REWRITE] Applied {applied}/{len(diff_blocks)} diff blocks, syntax OK.")
                return (modified, last_response if return_raw_response else None)
            except Exception as e:
                logger.info(f"[LLM][GPU-REWRITE] Syntax check failed after diff (attempt {attempt}): {e}")

        logger.info(f"[LLM][GPU-REWRITE] No valid SEARCH/REPLACE diff blocks. Retrying ({attempt}/{retries})...")
        temp = max(0.0, temp - 0.1)

    logger.error("[LLM][GPU-REWRITE] LLM GPU rewrite failed after retries. Returning original code.")
    return (original_code, last_response if return_raw_response else None)


def fix_runtime_error_with_llm(
    solution_path: str,
    code: str,
    exec_output_tail: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    retries: int = 3,
    validate_syntax: bool = True,
    max_tokens: Optional[int] = None,
    system_prompt: str = "You are a precise code editing assistant. Only output SEARCH/REPLACE diffs.",
    return_raw_response: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Fix a solution file that failed at runtime, using the recorded exec_output tail
    (runtime traceback / error message) from runs_log, via SEARCH/REPLACE diffs ONLY.
    """
    original_code = code
    temp = temperature
    last_response: Optional[str] = None

    for attempt in range(1, max(1, retries) + 1):
        prompt_text = build_runtime_error_fix_prompt(
            solution_path=solution_path,
            code=code,
            exec_output_tail=exec_output_tail,
        )

        logger.info(
            f"[LLM][RUNTIME-FIX] Calling chat_complete (attempt {attempt}/{retries}, "
            f"temp={temp}, max_tokens={max_tokens}, solution={solution_path})"
        )

        try:
            response = chat_complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                ],
                model=model,
                temperature=max(0.0, min(1.0, temp)),
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"[LLM][RUNTIME-FIX] chat_complete failed: {e}")
            continue

        last_response = (response or "")
        logger.info(f"[LLM][RUNTIME-FIX] Response length: {len(last_response)}")

        diff_blocks = extract_diff_blocks(last_response)
        if diff_blocks:
            logger.info(f"[LLM][RUNTIME-FIX] Extracted {len(diff_blocks)} diff blocks")
            applied = sum(1 for s, _ in diff_blocks if s in code)
            modified = apply_diff_to_code(code, diff_blocks)
            if not validate_syntax:
                logger.info(
                    f"[LLM][RUNTIME-FIX] Applied {applied}/{len(diff_blocks)} diff blocks (no syntax validation)."
                )
                return (modified, last_response if return_raw_response else None)
            try:
                ast.parse(modified)
                logger.info(
                    f"[LLM][RUNTIME-FIX] Applied {applied}/{len(diff_blocks)} diff blocks, syntax OK."
                )
                return (modified, last_response if return_raw_response else None)
            except Exception as e:
                logger.info(f"[LLM][RUNTIME-FIX] Syntax check failed after diff (attempt {attempt}): {e}")

        logger.info(
            f"[LLM][RUNTIME-FIX] No valid SEARCH/REPLACE diff blocks. Retrying ({attempt}/{retries})..."
        )
        temp = max(0.0, temp - 0.1)

    logger.error("[LLM][RUNTIME-FIX] LLM runtime fix failed after retries. Returning original code.")
    return (original_code, last_response if return_raw_response else None)


def fix_eval_error_with_llm(
    solution_path: str,
    code: str,
    competition_id: str,
    data_root: str,
    grader_error: Optional[str],
    score: Optional[float],
    submission_path: Optional[str],
    full_grading_json: Optional[str],
    model: Optional[str] = None,
    temperature: float = 0.2,
    retries: int = 3,
    validate_syntax: bool = True,
    max_tokens: Optional[int] = None,
    system_prompt: str = "You are a precise code editing assistant. Only output SEARCH/REPLACE diffs.",
    return_raw_response: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Fix a solution file whose submission was rejected during grading (eval errors),
    using the grading JSON information from error_eval_*.json.
    """
    original_code = code
    temp = temperature
    last_response: Optional[str] = None

    for attempt in range(1, max(1, retries) + 1):
        prompt_text = build_eval_error_fix_prompt(
            solution_path=solution_path,
            code=code,
            competition_id=competition_id,
            grader_error=grader_error,
            score=score,
            submission_path=submission_path,
            full_grading_json=full_grading_json,
            data_root=data_root,
        )

        logger.info(
            f"[LLM][EVAL-FIX] Calling chat_complete (attempt {attempt}/{retries}, "
            f"temp={temp}, max_tokens={max_tokens}, solution={solution_path})"
        )

        try:
            response = chat_complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                ],
                model=model,
                temperature=max(0.0, min(1.0, temp)),
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"[LLM][EVAL-FIX] chat_complete failed: {e}")
            continue

        last_response = (response or "")
        logger.info(f"[LLM][EVAL-FIX] Response length: {len(last_response)}")

        diff_blocks = extract_diff_blocks(last_response)
        if diff_blocks:
            logger.info(f"[LLM][EVAL-FIX] Extracted {len(diff_blocks)} diff blocks")
            applied = sum(1 for s, _ in diff_blocks if s in code)
            modified = apply_diff_to_code(code, diff_blocks)
            if not validate_syntax:
                logger.info(
                    f"[LLM][EVAL-FIX] Applied {applied}/{len(diff_blocks)} diff blocks (no syntax validation)."
                )
                return (modified, last_response if return_raw_response else None)
            try:
                ast.parse(modified)
                logger.info(
                    f"[LLM][EVAL-FIX] Applied {applied}/{len(diff_blocks)} diff blocks, syntax OK."
                )
                return (modified, last_response if return_raw_response else None)
            except Exception as e:
                logger.info(f"[LLM][EVAL-FIX] Syntax check failed after diff (attempt {attempt}): {e}")

        logger.info(
            f"[LLM][EVAL-FIX] No valid SEARCH/REPLACE diff blocks. Retrying ({attempt}/{retries})..."
        )
        temp = max(0.0, temp - 0.1)

    logger.error("[LLM][EVAL-FIX] LLM eval fix failed after retries. Returning original code.")
    return (original_code, last_response if return_raw_response else None)
