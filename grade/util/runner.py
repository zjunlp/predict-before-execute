# Core runner logic for LLM-based grading.
from typing import Any, Dict, List

from ...backend import chat_complete  # type: ignore
from . import prompt as prompt_module  # type: ignore
from . import eval as eval_module  # type: ignore


def process_group(
    task_name: str,
    group_entry: Dict[str, Any],
    solutions_dir: str,
    model: str,
    temperature: float,
    api_key: str = None,
    base_url: str = None,
    tasks_root: str = None,
    desc_dir: str = None,
    da_dir: str = None,
    raw_data_sample_dir: str = None,
    max_retries: int = 2,
    allow_cot: bool = False,
    prompt_boost: bool = False,
    concat_random_task_text: bool = False,
    client_slot: int | None = None,  # NEW: logical slot index for client pool
) -> Dict[str, Any]:
    """
    Run a single group prediction with retry on JSON parse failure, then evaluate.

    max_retries:
            - Outer layer (this function): on JSON parse failure, retry up to max_retries times (with an added prompt).
            - Lower layer (backend.chat_complete): for any LLM request error (HTTP 4xx/5xx/network errors, etc.),
                retry up to max_retries times until success or exhaustion.
    client_slot:
      - If provided, is forwarded to backend.chat_complete for client-pool routing.
      - If None, falls back to single-client behavior.
    """
    user_prompt_text = prompt_module.build_user_prompt(
        task_name,
        group_entry,
        solutions_dir,
        tasks_root=tasks_root,
        desc_dir=desc_dir,
        da_dir=da_dir,
        raw_data_sample_dir=raw_data_sample_dir,
        allow_cot=allow_cot,
        prompt_boost=prompt_boost,
        concat_random_task_text=concat_random_task_text,
    )
    messages = [
        {
            "role": "system",
            "content": (
                prompt_module.BENCH_SYSTEM_PROMPT_COT
                if allow_cot
                else prompt_module.BENCH_SYSTEM_PROMPT
            ),
        },
        {"role": "user", "content": user_prompt_text},
    ]

    n = len(group_entry.get("paths", []))
    raw_responses: List[str] = []
    parsed: Dict[str, Any] = {}
    last_err: str = ""
    interaction_log: List[Dict[str, Any]] = []

    # Outer loop: retry on JSON parse failure (add a prompt asking for correct format)
    for _ in range(max_retries + 1):
        snapshot = [{"role": m["role"], "content": m["content"]} for m in messages]
        try:
            resp_text = chat_complete(
                messages,
                model=model,
                temperature=temperature,
                api_key=api_key,
                base_url=base_url,
                client_slot=client_slot,          # NEW: forward to backend
                max_retries=max_retries,          # NEW: lower-level LLM retry count (all errors)
            )
        except Exception as e:
            # If all lower-level retries fail, mark as error and stop JSON-retry loop
            resp_text = f"[llm_request_failed] {e}"
            interaction_log.append({"messages": snapshot, "response": resp_text})
            raw_responses.append(resp_text)
            last_err = str(e)
            # Mark as LLM call failure (distinct from JSON parse failure)
            parsed = {
                "error": f"llm_request_failed: {last_err}",
                "raw": resp_text,
            }
            break

        interaction_log.append({"messages": snapshot, "response": resp_text})
        raw_responses.append(resp_text)
        try:
            parsed = prompt_module.parse_response(resp_text, n)
            last_err = ""
            break
        except Exception as e:
            last_err = str(e)
            if allow_cot:
                retry_msg = (
                    "Your previous output could not be parsed. You may include reasoning first, "
                    "but you must end your answer with a single RAW JSON object that strictly matches the RESPONSE FORMAT. "
                    "Do not write anything after the JSON."
                )
            else:
                retry_msg = (
                    "Your previous output was not valid JSON or missed required keys. "
                    "Reply again with RAW JSON ONLY, strictly following the specified RESPONSE FORMAT."
                )
            messages.append({"role": "user", "content": retry_msg})

    result: Dict[str, Any] = {"parsed": parsed, "n": n}
    result["raw_responses"] = raw_responses
    result["raw_response"] = raw_responses[-1] if raw_responses else ""
    result["interaction_log"] = interaction_log
    result["final_messages"] = messages
    if last_err and not parsed:
        # Preserve the parse_failed marker for JSON parse failures
        result["parsed"] = {"error": f"parse_failed: {last_err}", "raw": result["raw_response"]}

    gt_best = group_entry.get("best_index")
    gt_full = group_entry.get("full_ranking", [])
    if n == 2:
        if "predicted_best_index" in result["parsed"]:
            result["accuracy"] = eval_module.accuracy_n2(result["parsed"]["predicted_best_index"], gt_best)
    else:
        pr = result["parsed"].get("predicted_ranking", [])
        if isinstance(pr, list):
            pr = [int(x) for x in pr]
            missing = [i for i in range(n) if i not in pr]
            pr_full = pr + missing
            result["spearman_rho"] = eval_module.spearman_rho_from_rankings(pr_full, gt_full)
            if isinstance(gt_full, list) and len(gt_full) > 0:
                precision: Dict[int, float] = {}
                max_k = min(len(gt_full), len(pr_full))
                for k in range(1, max_k + 1):
                    precision[k] = eval_module.precision_at_k(pr_full, gt_full, k)
                result["precision"] = precision
            result["predicted_ranking_full"] = pr_full
    return result
