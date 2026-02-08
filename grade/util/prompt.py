# Prompt construction utilities for skip_bench grading.
import json
import os
from typing import Dict, List, Any, Optional
from textwrap import dedent  # new: for cleaner multiline blocks
import random  # new: for selecting random other task text

# System prompt (static)
BENCH_SYSTEM_PROMPT = (
    "You are an ML code and data analysis expert tasked with predicting the relative "
    "performance of provided ML solutions without executing any code. Base your judgment "
    "on the task description and the shown code snippets only. Never assume external ground-truth, "
    "never execute code, and do not include any text beyond the required raw JSON output."
)
# COT-enabled system prompt
BENCH_SYSTEM_PROMPT_COT = (
    "You are an ML code and data analysis expert tasked with predicting the relative "
    "performance of provided ML solutions without executing any code. Base your judgment "
    "on the task description and the shown code snippets only. Never assume external ground-truth. "
    "You should include brief reasoning before the final answer. End your answer with a single JSON object "
    "that strictly matches the specified response format."
)

# User prompt templates (static)
BENCH_USER_PROMPT_TEMPLATE = """{header_sections}Important instructions:
{instructions_block}
Provided solutions:
{solutions_block}
"""

SOLUTION_SNIPPET_TEMPLATE = """### Solution {index}: path={display_path}
```python
{snippet}
```"""


def _read_code_snippet(path: str, max_lines: int = None) -> str:
    """
    Read the full file content. If max_lines is a positive integer, truncate to that many lines.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            if not max_lines or max_lines <= 0:
                return f.read()
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line.rstrip("\n"))
        return "\n".join(lines)
    except Exception:
        return f"# Could not read file: {path}"

def _task_key(task_name: str) -> str:
    """
    Convert a task name to the file key used by description/DA assets.
    Example: 'aptos2019-blindness-detection' -> 'aptos2019_blindness_detection'
    """
    key = task_name.strip().lower()
    key = key.replace("-", "_").replace(" ", "_")
    return key

# New: candidate key generator, compatible with hyphens/underscores/original names
def _candidate_keys(task_name: str) -> List[str]:
    base = (task_name or "").strip()
    k_us = base.lower().replace("-", "_").replace(" ", "_")
    k_dash = base.lower().replace("_", "-").replace(" ", "-")
    keys: List[str] = []
    for k in (k_us, k_dash, base, base.lower()):
        if k and k not in keys:
            keys.append(k)
    return keys

def _load_text_file(path: str, fallback: str = "") -> str:
    if not path or not os.path.exists(path):
        return fallback or ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
        return content if content else (fallback or "")
    except Exception:
        return fallback or ""

def load_task_resources(
    task_name: str,
    tasks_root: Optional[str] = None,
    desc_dir: Optional[str] = None,
    da_dir: Optional[str] = None,
    raw_data_sample_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Load task description and data analysis/raw data sample text using either:
      - Mode 1: tasks_root containing 'description' and 'data_analysis/result'
      - Mode 2: separate desc_dir and da_dir
      - Mode 3: separate desc_dir and raw_data_sample_dir
    Tries multiple name variants: underscores, hyphens, original, lowercase original.
    Returns dict with keys: 'task_desc', 'data_analysis' (or 'raw_data_sample'), 'mode'
    """
    desc_path = None
    da_path = None
    raw_data_path = None
    mode = None
    keys = _candidate_keys(task_name)

    # Mode 3 has highest priority if raw_data_sample_dir is provided
    if raw_data_sample_dir:
        mode = "raw_data_sample"
        for key in keys:
            if not desc_path and desc_dir:
                p = os.path.join(desc_dir, f"description_{key}.md")
                if os.path.exists(p):
                    desc_path = p
            if not raw_data_path:
                # Try both raw_data_sample_*.txt and da_result_*.txt naming patterns
                for prefix in ["raw_data_sample_", "da_result_"]:
                    p = os.path.join(raw_data_sample_dir, f"{prefix}{key}.txt")
                    if os.path.exists(p):
                        raw_data_path = p
                        break
            if desc_path and raw_data_path:
                break
    # Mode 2 has priority if explicitly provided
    elif desc_dir or da_dir:
        mode = "data_analysis"
        for key in keys:
            if desc_dir and not desc_path:
                p = os.path.join(desc_dir, f"description_{key}.md")
                if os.path.exists(p):
                    desc_path = p
            if da_dir and not da_path:
                p = os.path.join(da_dir, f"da_result_{key}.txt")
                if os.path.exists(p):
                    da_path = p
            if desc_path and da_path:
                break
    elif tasks_root:
        mode = "data_analysis"
        for key in keys:
            if not desc_path:
                p = os.path.join(tasks_root, "description", f"description_{key}.md")
                if os.path.exists(p):
                    desc_path = p
            if not da_path:
                p = os.path.join(tasks_root, "data_analysis", "result", f"da_result_{key}.txt")
                if os.path.exists(p):
                    da_path = p
            if desc_path and da_path:
                break

    # return empty string when file missing/empty rather than "N/A"
    task_desc = _load_text_file(desc_path or "", fallback="")
    data_analysis = _load_text_file(da_path or "", fallback="")
    raw_data_sample = _load_text_file(raw_data_path or "", fallback="")
    
    return {
        "task_desc": task_desc,
        "data_analysis": data_analysis,
        "raw_data_sample": raw_data_sample,
        "mode": mode or "data_analysis",
        "desc_path": desc_path or "",
        "da_path": da_path or "",
        "raw_data_path": raw_data_path or "",
    }

def build_user_prompt(
    task_name: str,
    group_entry: Dict[str, Any],
    solutions_dir: str,
    tasks_root: Optional[str] = None,
    desc_dir: Optional[str] = None,
    da_dir: Optional[str] = None,
    raw_data_sample_dir: Optional[str] = None,
    allow_cot: bool = False,
    prompt_boost: Optional[bool] = None,
    concat_random_task_text: bool = False,
) -> str:
    """
    Build the user prompt from static templates.
    For n == 2: expect {"predicted_best_index": <0 or 1>}
    For n > 2: expect {"predicted_ranking": [<index0>, <index1>, ...]}
    """
    paths: List[str] = group_entry.get("paths", [])
    n = len(paths)

    # Load task description and data analysis/raw data sample via either mode
    resources = load_task_resources(
        task_name, 
        tasks_root=tasks_root, 
        desc_dir=desc_dir, 
        da_dir=da_dir,
        raw_data_sample_dir=raw_data_sample_dir,
    )
    task_desc = (resources["task_desc"] or "").strip()
    data_content = ""
    data_label = ""
    desc_path_cur = resources.get("desc_path") or ""
    data_path_cur = ""
    
    if resources["mode"] == "raw_data_sample":
        data_content = (resources["raw_data_sample"] or "").strip()
        data_label = "Raw data sample"
        data_path_cur = resources.get("raw_data_path") or ""
    else:
        data_content = (resources["data_analysis"] or "").strip()
        data_label = "Data analysis"
        data_path_cur = resources.get("da_path") or ""

    # Change: when concat_random_task_text is enabled, replace with a random other task's content (not append)
    if concat_random_task_text:
        # Description directory priority remains the same
        desc_search_dir = None
        if desc_dir:
            desc_search_dir = desc_dir
        elif tasks_root:
            desc_candidate = os.path.join(tasks_root, "description")
            if os.path.isdir(desc_candidate):
                desc_search_dir = desc_candidate
        if desc_search_dir and os.path.isdir(desc_search_dir):
            desc_files = [os.path.join(desc_search_dir, f) for f in os.listdir(desc_search_dir)
                          if f.startswith("description_") and f.endswith(".md")]
            # Exclude the current task's description file
            desc_files = [p for p in desc_files if os.path.abspath(p) != os.path.abspath(desc_path_cur)]
            if desc_files:
                noise_desc_path = random.choice(desc_files)
                noise_desc = _load_text_file(noise_desc_path, fallback="")
                # If original description existed, replace it; otherwise keep original behavior (do not invent)
                if noise_desc and task_desc:
                    task_desc = noise_desc

        # Data content directory (depends on mode)
        data_search_dir = None
        if resources["mode"] == "raw_data_sample" and raw_data_sample_dir:
            data_search_dir = raw_data_sample_dir
            # Try both naming patterns
            data_prefixes = ["raw_data_sample_", "da_result_"]
        elif da_dir:
            data_search_dir = da_dir
            data_prefixes = ["da_result_"]
        elif tasks_root:
            if resources["mode"] == "raw_data_sample":
                data_candidate = os.path.join(tasks_root, "raw_data_sample", "result")
                data_prefixes = ["raw_data_sample_", "da_result_"]
            else:
                data_candidate = os.path.join(tasks_root, "data_analysis", "result")
                data_prefixes = ["da_result_"]
            if os.path.isdir(data_candidate):
                data_search_dir = data_candidate
        
        if data_search_dir and os.path.isdir(data_search_dir):
            data_files = []
            for prefix in data_prefixes:
                data_files.extend([
                    os.path.join(data_search_dir, f) for f in os.listdir(data_search_dir)
                    if f.startswith(prefix) and (f.endswith(".txt") or f.endswith(".md"))
                ])
            # Exclude the current task's data file
            data_files = [p for p in data_files if os.path.abspath(p) != os.path.abspath(data_path_cur)]
            if data_files:
                noise_data_path = random.choice(data_files)
                noise_data = _load_text_file(noise_data_path, fallback="")
                # If original data_content existed, replace it; otherwise keep original behavior
                if noise_data and data_content:
                    data_content = noise_data

    # Response format
    if n == 2:
        response_format = '{"predicted_best_index": <0 or 1>, "confidence": <optional float>}'
    else:
        response_format = '{"predicted_ranking": [<index0>, <index1>, ..., <index{n-1}>], "confidence": <optional float>}'
        # Explicitly require a complete ranking
        response_format += f'\nNote: predicted_ranking must contain ALL indices from 0 to {n-1} in your predicted order (best to worst).'

    # Output rule line (cot vs non-cot)
    if allow_cot:
        output_rule = "- You should include brief reasoning before the final JSON. End with a single JSON object matching the response format. Do not write anything after the JSON."
    else:
        output_rule = "- Output raw JSON only (no extra text, no Markdown fences)."

    # Build header_sections according to rules:
    # - both present: include "Task: {task_name}" then Task description and Data analysis/Raw data sample
    # - only task_desc: include only Task description
    # - only data_content: include only Data analysis/Raw data sample
    # - neither: header_sections is empty
    if task_desc and data_content:
        header_sections = (
            f"Task: {task_name}\n\n"
            f"Task description:\n{task_desc}\n\n"
            f"{data_label}:\n{data_content}\n\n"
        )
    elif task_desc:
        header_sections = f"Task description:\n{task_desc}\n\n"
    elif data_content:
        header_sections = f"{data_label}:\n{data_content}\n\n"
    else:
        header_sections = ""

    # Build dynamic instructions block
    sources = []
    if task_desc:
        sources.append("task description")
    if data_content:
        if resources["mode"] == "raw_data_sample":
            sources.append("raw data sample")
        else:
            sources.append("data analysis")
    sources.append("code snippets")

    if len(sources) == 1:
        sources_str = sources[0]
    elif len(sources) == 2:
        sources_str = " and ".join(sources)
    else:
        sources_str = ", ".join(sources[:-1]) + ", and " + sources[-1]

    instructions_lines = [
        "- Predict which solution will perform best (or provide a full ranking) WITHOUT running code.",
        f"- Use only the {sources_str} below.",
        f"- Response format: {response_format}",
        "- Indices correspond to the order of the listed solutions (0..n-1).",
    ]

    # Equal-importance and trade-off guidance (only when prompt_boost is enabled)
    if prompt_boost:
        if task_desc and data_content:
            if resources["mode"] == "raw_data_sample":
                instructions_lines.insert(
                    2,
                    "- Treat the task description and raw data sample as equally important to the code; analyze them separately, surface their underlying implications, and provide a balanced, trade-off judgment. ",
                )
                instructions_lines.insert(
                    3,
                    "- Connect raw data characteristics to the following code analysis: If the data sample indicates properties (e.g., distribution, format, scale), explain how the architecture addresses them, forming a data→why→method choice causal chain.",
                )
            else:
                instructions_lines.insert(
                    2,
                    "- Treat the task description and data analysis as equally important to the code; analyze them separately, surface their underlying implications, and provide a balanced, trade-off judgment. ",
                )
                instructions_lines.insert(
                    3,
                    "- Connect data analysis to the following code analysis: If data analysis indicates properties , explain how the architecture addresses them , forming a data→why→method choice causal chain.",
                )
            instructions_lines.insert(
                4,
                "- Forbid the 'complexity-wins' shortcut: Do not claim \"deeper/more complex/with attention is better\" as the sole reason. If used, justify why it holds under the current data distribution and training details, and provide a counterexample scenario.",
            )
        elif task_desc and not data_content:
            instructions_lines.insert(
                2,
                "- Treat the task description as equally important to the code; analyze it carefully, surface its underlying implications, and provide a balanced, trade-off judgment.",
            )
            instructions_lines.insert(
                3,
                "- Connect data analysis to the following code analysis: If data analysis indicates properties , explain how the architecture addresses them , forming a data→why→method choice causal chain.",
            )
            instructions_lines.insert(
                4,
                "- Forbid the \"complexity-wins\" shortcut: Do not claim \"deeper/more complex/with attention is better\" as the sole reason. If used, justify why it holds under the current data distribution and training details, and provide a counterexample scenario.",
            )
        elif data_content and not task_desc:
            data_type = "raw data sample" if resources["mode"] == "raw_data_sample" else "data analysis"
            instructions_lines.insert(
                2,
                f"- Treat the {data_type} as equally important to the code; analyze it carefully, surface its underlying implications, and provide a balanced, trade-off judgment.",
            )
            instructions_lines.insert(
                3,
                "- Forbid the \"complexity-wins\" shortcut: Do not claim \"deeper/more complex/with attention is better\" as the sole reason.",
            )
        # If neither present, skip this bullet.

    # Append output rule
    instructions_lines.append(output_rule)
    instructions_block = "\n".join(instructions_lines) + "\n"

    # Solutions block (loop over all solutions)
    def _short_display_name(p: str) -> str:
        """
        Return the last 4 chars of the filename (without extension) plus '.py'.
        If the basename (without ext) is shorter than 4 chars, use it whole.
        """
        name = os.path.basename(p)
        name_no_ext, _ = os.path.splitext(name)
        last4 = name_no_ext[-4:] if len(name_no_ext) >= 4 else name_no_ext
        return f"{last4}.py"

    solutions_block = "\n\n".join(
        SOLUTION_SNIPPET_TEMPLATE.format(
            index=i,
            display_path=_short_display_name(p),
            snippet=_read_code_snippet(p),
        )
        for i, p in enumerate(paths)
    )

    # Final user prompt: inject header + dynamic instructions
    user_prompt = BENCH_USER_PROMPT_TEMPLATE.format(
        header_sections=header_sections,
        instructions_block=instructions_block,
        solutions_block=solutions_block,
    )
    return user_prompt

# Backwards-compatible alias if external code still calls build_prompt
def build_prompt(
    task_name: str,
    group_entry: Dict[str, Any],
    solutions_dir: str,
    tasks_root: Optional[str] = None,
    desc_dir: Optional[str] = None,
    da_dir: Optional[str] = None,
    raw_data_sample_dir: Optional[str] = None,
    allow_cot: bool = False,
    prompt_boost: Optional[bool] = None,
    concat_random_task_text: bool = False,
) -> str:
    return build_user_prompt(
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

def parse_response(raw: str, n: int) -> Dict[str, Any]:
    """
    Parse the LLM response as JSON.
    - For n == 2: expect key "predicted_best_index" or "predicted_ranking" (first item -> best).
    - For n > 2: expect key "predicted_ranking"; if only "predicted_best_index" exists, convert to list.
    Tolerant to extra text: prefer the last JSON object in the text; if that fails, try the first.
    """
    raw = raw.strip()
    # Try direct json.loads
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
        # Prefer the last JSON object
        try:
            end = raw.rfind("}")
            if end != -1:
                start = raw.rfind("{", 0, end + 1)
                if start != -1 and end > start:
                    parsed = json.loads(raw[start : end + 1])
        except Exception:
            parsed = None
        if parsed is None:
            # Fallback: first JSON object
            start_first = raw.find("{")
            end_last = raw.rfind("}")
            if start_first != -1 and end_last != -1 and end_last > start_first:
                try:
                    parsed = json.loads(raw[start_first : end_last + 1])
                except Exception:
                    raise ValueError("Could not parse JSON from LLM response.")
            else:
                raise ValueError("Could not parse JSON from LLM response.")

    # Basic validation and normalization
    if n == 2:
        if "predicted_best_index" not in parsed and "predicted_ranking" not in parsed:
            raise ValueError("Response missing predicted_best_index or predicted_ranking.")
        if "predicted_ranking" in parsed and "predicted_best_index" not in parsed:
            pr = parsed["predicted_ranking"]
            if isinstance(pr, list) and len(pr) >= 1:
                parsed["predicted_best_index"] = int(pr[0])
    else:
        if "predicted_ranking" not in parsed:
            if "predicted_best_index" in parsed:
                parsed["predicted_ranking"] = [int(parsed["predicted_best_index"])]
            else:
                raise ValueError("Response missing predicted_ranking.")

    if "predicted_ranking" in parsed:
        parsed["predicted_ranking"] = [int(x) for x in parsed["predicted_ranking"]]
    if "predicted_best_index" in parsed:
        parsed["predicted_best_index"] = int(parsed["predicted_best_index"])
    if "confidence" in parsed:
        try:
            parsed["confidence"] = float(parsed["confidence"])
        except Exception:
            parsed.pop("confidence", None)

    # New: after successful parsing, keep only needed fields and drop reasoning and other keys
    if n == 2:
        cleaned: Dict[str, Any] = {}
        if "predicted_best_index" in parsed:
            cleaned["predicted_best_index"] = parsed["predicted_best_index"]
        if "confidence" in parsed:
            cleaned["confidence"] = parsed["confidence"]
    else:
        cleaned = {}
        if "predicted_ranking" in parsed:
            cleaned["predicted_ranking"] = parsed["predicted_ranking"]
        if "confidence" in parsed:
            cleaned["confidence"] = parsed["confidence"]

    return cleaned