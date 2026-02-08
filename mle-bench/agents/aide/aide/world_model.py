"""
World Model for predicting relative performance of ML solutions without execution.
Uses tournament-style pairwise comparisons with confidence thresholds.
"""

import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .backend import query
from .journal import Node

logger = logging.getLogger("aide")


# ============== Prompt utilities (adapted from prompt.py) ==============

BENCH_SYSTEM_PROMPT_COT = (
    "You are an ML code and data analysis expert tasked with predicting the relative "
    "performance of provided ML solutions without executing any code. Base your judgment "
    "on the task description and the shown code snippets only. Never assume external ground-truth. "
    "You should include brief reasoning before the final answer. End your answer with a single JSON object "
    "that strictly matches the specified response format."
)

BENCH_USER_PROMPT_TEMPLATE = """{header_sections}Important instructions:
{instructions_block}
Provided solutions:
{solutions_block}
"""

SOLUTION_SNIPPET_TEMPLATE = """### Solution {index}: path={display_path}
```python
{snippet}
```"""


def _load_text_file(path: str, fallback: str = "") -> str:
    if not path or not os.path.exists(path):
        return fallback or ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
        return content if content else (fallback or "")
    except Exception:
        return fallback or ""


def _candidate_keys(task_name: str) -> List[str]:
    base = (task_name or "").strip()
    k_us = base.lower().replace("-", "_").replace(" ", "_")
    k_dash = base.lower().replace("_", "-").replace(" ", "-")
    keys: List[str] = []
    for k in (k_us, k_dash, base, base.lower()):
        if k and k not in keys:
            keys.append(k)
    return keys


def load_task_resources(
    task_name: str,
    tasks_root: Optional[str] = None,
    desc_dir: Optional[str] = None,
    da_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Load task description and data analysis text files.
    
    Priority order:
    1. /home/data/da_report.txt (MLE-bench mounted data dir)
    2. Explicit da_dir or desc_dir
    3. tasks_root structure
    """
    desc_path = None
    da_path = None
    keys = _candidate_keys(task_name)

    # Priority 1: Check /home/data/ for da_report.txt (MLE-bench mounted location)
    mlebench_da_path = "/home/data/da_report.txt"
    if os.path.exists(mlebench_da_path):
        da_path = mlebench_da_path
        logger.info(f"Found data analysis report at MLE-bench location: {mlebench_da_path}")
    
    # Priority 2: Explicit directories
    if desc_dir or da_dir:
        for key in keys:
            if not desc_path and desc_dir:
                p = os.path.join(desc_dir, f"description_{key}.md")
                if os.path.exists(p):
                    desc_path = p
            if not da_path and da_dir:
                p = os.path.join(da_dir, f"da_result_{key}.txt")
                if os.path.exists(p):
                    da_path = p
            if desc_path and da_path:
                break
    
    # Priority 3: tasks_root structure
    elif tasks_root and not da_path:  # Only search if da_path not found yet
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

    task_desc = _load_text_file(desc_path or "", fallback="")
    data_analysis = _load_text_file(da_path or "", fallback="")
    
    return {
        "task_desc": task_desc,
        "data_analysis": data_analysis,
        "desc_path": desc_path or "",
        "da_path": da_path or "",
    }


def build_pairwise_prompt(
    task_desc: str,
    data_analysis: str,
    code_a: str,
    code_b: str,
    task_name: str = "",
) -> str:
    """Build a pairwise comparison prompt for two solutions."""
    
    # Build header sections
    if task_desc and data_analysis:
        header_sections = (
            f"Task: {task_name}\n\n"
            f"Task description:\n{task_desc}\n\n"
            f"Data analysis:\n{data_analysis}\n\n"
        )
    elif task_desc:
        header_sections = f"Task description:\n{task_desc}\n\n"
    elif data_analysis:
        header_sections = f"Data analysis:\n{data_analysis}\n\n"
    else:
        header_sections = ""

    # Build instructions
    sources = []
    if task_desc:
        sources.append("task description")
    if data_analysis:
        sources.append("data analysis")
    sources.append("code snippets")
    
    if len(sources) == 1:
        sources_str = sources[0]
    elif len(sources) == 2:
        sources_str = " and ".join(sources)
    else:
        sources_str = ", ".join(sources[:-1]) + ", and " + sources[-1]

    response_format = '{"predicted_best_index": <0 or 1>, "confidence": <float between 0 and 1>}'
    
    instructions_lines = [
        "- Predict which solution will perform best WITHOUT running code.",
        f"- Use only the {sources_str} below.",
        f"- Response format: {response_format}",
    ]
    
    if task_desc and data_analysis:
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

    instructions_lines.append(
        "- Include brief reasoning before the final JSON. End with a single JSON object."
    )
    
    instructions_block = "\n".join(instructions_lines) + "\n"

    # Build solutions block
    solutions_block = (
        f"### Solution 0:\n```python\n\n{code_a}\n```\n\n"
        f"### Solution 1:\n```python\n\n{code_b}\n```"
    )

    user_prompt = BENCH_USER_PROMPT_TEMPLATE.format(
        header_sections=header_sections,
        instructions_block=instructions_block,
        solutions_block=solutions_block,
    )
    return user_prompt


def parse_pairwise_response(raw: str) -> Dict[str, Any]:
    """Parse LLM response for pairwise comparison."""
    import json
    
    raw = raw.strip()
    parsed = None
    
    # Try direct parse
    try:
        parsed = json.loads(raw)
    except Exception:
        pass
    
    if parsed is None:
        # Try to find last JSON object
        try:
            end = raw.rfind("}")
            if end != -1:
                start = raw.rfind("{", 0, end + 1)
                if start != -1 and end > start:
                    parsed = json.loads(raw[start:end + 1])
        except Exception:
            pass
    
    if parsed is None:
        # Fallback: first JSON object
        start_first = raw.find("{")
        end_last = raw.rfind("}")
        if start_first != -1 and end_last != -1 and end_last > start_first:
            try:
                parsed = json.loads(raw[start_first:end_last + 1])
            except Exception:
                raise ValueError("Could not parse JSON from LLM response.")
        else:
            raise ValueError("Could not parse JSON from LLM response.")
    
    result = {}
    if "predicted_best_index" in parsed:
        result["predicted_best_index"] = int(parsed["predicted_best_index"])
    if "confidence" in parsed:
        try:
            result["confidence"] = float(parsed["confidence"])
        except Exception:
            result["confidence"] = 0.5
    else:
        result["confidence"] = 0.5
    
    return result


@dataclass
class ComparisonResult:
    """Result of a pairwise comparison."""
    winner_idx: Optional[int]  # None if confidence too low
    loser_idx: Optional[int]
    confidence: float
    round_num: int
    is_valid: bool  # True if confidence >= threshold
    raw_response: str = ""
    reasoning: str = ""


@dataclass
class WorldModelConfig:
    """Configuration for World Model predictions."""
    model: str = "DeepSeek-V3.2-Thinking"
    temperature: float = 0.2
    confidence_threshold: float = 0.6
    exec_probability: float = 0.3  # Probability of actually executing after prediction
    max_concurrent_requests: int = 128
    tasks_root: Optional[str] = None
    desc_dir: Optional[str] = None
    da_dir: Optional[str] = None


class WorldModel:
    """
    World Model for predicting solution performance using tournament-style comparisons.
    """
    # Global cache for pairwise comparisons
    _comparison_cache: Dict[Tuple[int, int], ComparisonResult] = {}

    def __init__(
        self,
        task_desc: str,
        data_preview: str,
        task_name: str = "",
        config: Optional[WorldModelConfig] = None,
    ):
        self.task_desc = task_desc
        self.data_preview = data_preview  # This can be the data analysis report
        self.task_name = task_name
        self.config = config or WorldModelConfig()
        self.comparison_history: List[ComparisonResult] = []

    # --- Single-comparison internal helper (preserve original logic) ---
    def _predict_pairwise_single(
        self,
        node_a: Node,
        node_b: Node,
        round_num: int = 0,
    ) -> ComparisonResult:
        """
        Predict which of two nodes will perform better.
        Returns ComparisonResult with winner/loser if confidence >= threshold.
        """
        # Compute hash keys for the pair
        code_a_hash = hash(node_a.code)
        code_b_hash = hash(node_b.code)
        pair_key = (min(code_a_hash, code_b_hash), max(code_a_hash, code_b_hash))

        # Check cache
        if pair_key in self._comparison_cache:
            logger.info(f"Cache hit for pair: {pair_key}")
            cached_result = self._comparison_cache[pair_key]
            return ComparisonResult(
                winner_idx=cached_result.winner_idx,
                loser_idx=cached_result.loser_idx,
                confidence=cached_result.confidence,
                round_num=round_num,
                is_valid=cached_result.is_valid,
                raw_response=cached_result.raw_response,
                reasoning=cached_result.reasoning,
            )

        prompt = build_pairwise_prompt(
            task_desc=self.task_desc,
            data_analysis=self.data_preview,
            code_a=node_a.code,
            code_b=node_b.code,
            task_name=self.task_name,
        )
        
        try:
            response = query(
                system_message=BENCH_SYSTEM_PROMPT_COT,
                user_message=prompt,
            # NOTE: model/temperature kept same as before
                model=self.config.model,
                temperature=self.config.temperature,
            )
            
            # Handle tuple return from query
            if isinstance(response, tuple):
                raw_response = response[0] if response else ""
            else:
                raw_response = response
            
            parsed = parse_pairwise_response(raw_response)
            predicted_idx = parsed.get("predicted_best_index", 0)
            confidence = parsed.get("confidence", 0.5)
            
            is_valid = confidence >= self.config.confidence_threshold
            
            if is_valid:
                winner_idx = predicted_idx
                loser_idx = 1 - predicted_idx
            else:
                winner_idx = None
                loser_idx = None
            
            result = ComparisonResult(
                winner_idx=winner_idx,
                loser_idx=loser_idx,
                confidence=confidence,
                round_num=round_num,
                is_valid=is_valid,
                raw_response=raw_response,
            )
        except Exception as e:
            logger.error(f"World Model prediction failed: {e}")
            result = ComparisonResult(
                winner_idx=None,
                loser_idx=None,
                confidence=0.0,
                round_num=round_num,
                is_valid=False,
                raw_response=str(e),
            )

        # Cache the result
        self._comparison_cache[pair_key] = result
        return result

    def predict_pairwise(
        self,
        node_a: Node,
        node_b: Node,
        round_num: int = 0,
    ) -> ComparisonResult:
        """
        Backwards-compatible wrapper calling the single-comparison helper.
        """
        result = self._predict_pairwise_single(node_a, node_b, round_num)
        self.comparison_history.append(result)
        return result

    # --- NEW: Batch parallel comparison interface ---
    def _batch_predict_pairwise(
        self,
        pairs: List[Tuple[Node, Node, int]],
        max_workers: int | None = None,
    ) -> List[ComparisonResult]:
        """
        Run multiple pairwise predictions in parallel.

        Args:
            pairs: list of (node_a, node_b, round_num)
        """
        if not pairs:
            return []

        # Simple thread pool implementation; use default when max_workers is None
        results: List[ComparisonResult] = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_idx = {
                ex.submit(self._predict_pairwise_single, a, b, r): idx
                for idx, (a, b, r) in enumerate(pairs)
            }
            # Preallocate results list to preserve order matching pairs
            tmp: List[Optional[ComparisonResult]] = [None] * len(pairs)
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    logger.error(f"World Model parallel prediction failed: {e}")
                        # Fallback: return an invalid comparison
                    a, b, r = pairs[idx]
                    res = ComparisonResult(
                        winner_idx=None,
                        loser_idx=None,
                        confidence=0.0,
                        round_num=r,
                        is_valid=False,
                        raw_response=str(e),
                    )
                tmp[idx] = res

        # Populate comparison_history and return
        for r in tmp:
            assert r is not None
            self.comparison_history.append(r)
            results.append(r)

        return results

    def tournament_rank(
        self,
        candidates: List[Node],
    ) -> Tuple[List[Node], List[ComparisonResult], Dict[str, Dict[str, int]]]:
        """
        Rank candidates using tournament-style elimination with confidence threshold.
        
        Returns:
            - Sorted list of nodes (best first)
            - List of all comparison results
            - Rank metadata per node:
                {
                    node_id: {
                        "rank": int,   # display rank (same for inconclusive group)
                        "round": int,  # elimination / last-round number (for display)
                    }
                }
        """
        total_start_time = time.time()
        
        if len(candidates) <= 1:
            # rank_meta: single node has rank 0, round 0
            rank_meta = {}
            if candidates:
                rank_meta[candidates[0].id] = {"rank": 0, "round": 0}
            return candidates, [], rank_meta

        n = len(candidates)
        comparisons: List[ComparisonResult] = []

        # Track: node -> (is_eliminated, elimination_round, wins)
        node_status: Dict[str, Dict] = {
            node.id: {"node": node, "eliminated": False, "round": 0, "wins": 0}
            for node in candidates
        }

        active_nodes = list(candidates)
        round_num = 0

        while len(active_nodes) > 1:
            round_start_time = time.time()

            round_num += 1
            next_round_nodes: List[Node] = []
            pending_nodes: List[Node] = []  # Nodes with inconclusive comparisons this round

            # --- Step 1: Build all pairwise comparisons for this round (parallel LLM calls) ---
            pair_specs: List[Tuple[Node, Node, int]] = []
            pair_index_to_nodes: List[Tuple[Node, Node]] = []

            i = 0
            while i < len(active_nodes):
                if i + 1 < len(active_nodes):
                    node_a = active_nodes[i]
                    node_b = active_nodes[i + 1]
                    pair_specs.append((node_a, node_b, round_num))
                    pair_index_to_nodes.append((node_a, node_b))
                    i += 2
                else:
                    # Odd node gets a bye
                    next_round_nodes.append(active_nodes[i])
                    i += 1

            # --- Step 2: Run all pairwise comparisons for this round in parallel ---
            round_results: List[ComparisonResult] = self._batch_predict_pairwise(
                pair_specs,
                max_workers=self.config.max_concurrent_requests,
            )
            comparisons.extend(round_results)

            # --- Step 3: Update tournament state based on results ---
            for (node_a, node_b), result in zip(pair_index_to_nodes, round_results):
                if result.is_valid:
                    # We have a confident winner
                    if result.winner_idx == 0:
                        winner, loser = node_a, node_b
                    else:
                        winner, loser = node_b, node_a

                    next_round_nodes.append(winner)
                    node_status[winner.id]["wins"] += 1
                    node_status[loser.id]["eliminated"] = True
                    node_status[loser.id]["round"] = round_num

                    logger.info(
                        f"Round {round_num}: Node {winner.id[:8]} beats {loser.id[:8]} "
                        f"(confidence: {result.confidence:.2f})"
                    )
                else:
                    # Low confidence - both nodes continue but marked as pending
                    pending_nodes.extend([node_a, node_b])
                    logger.info(
                        f"Round {round_num}: Node {node_a.id[:8]} vs {node_b.id[:8]} "
                        f"inconclusive (confidence: {result.confidence:.2f})"
                    )

            # --- Step 4: Parallel processing of pending nodes (optimization) ---
            pending_pair_specs: List[Tuple[Node, Node, int]] = []
            pending_map: List[Node] = [] 

            # 1. Prepare pairing data
            for pending_node in pending_nodes:
                if next_round_nodes:
                    # Randomly select an advanced node to compare with
                    compare_with = random.choice(next_round_nodes)
                    pending_pair_specs.append((pending_node, compare_with, round_num))
                    pending_map.append(pending_node)
                else:
                    # If there are no advanced nodes to compare with (edge case), advance directly
                    next_round_nodes.append(pending_node)

            # 2. Execute 'revival matches' in parallel
            if pending_pair_specs:
                pending_results = self._batch_predict_pairwise(
                    pending_pair_specs,
                    max_workers=self.config.max_concurrent_requests
                )
                comparisons.extend(pending_results)

                # 3. Handle results
                for pending_node, result in zip(pending_map, pending_results):
                    if result.is_valid:
                        if result.winner_idx == 0:
                            # Pending node won -> advance
                            next_round_nodes.append(pending_node)
                            node_status[pending_node.id]["wins"] += 1
                            logger.info(
                                f"Round {round_num} (Re-match): Node {pending_node.id[:8]} beats winner "
                                f"(confidence: {result.confidence:.2f})"
                            )
                        else:
                            # Pending node lost -> eliminate
                            node_status[pending_node.id]["eliminated"] = True
                            node_status[pending_node.id]["round"] = round_num
                            logger.info(
                                f"Round {round_num} (Re-match): Node {pending_node.id[:8]} lost to winner "
                                f"(confidence: {result.confidence:.2f})"
                            )
                    else:
                        # Still not confident -> force advance (carry to next round)
                        node_status[pending_node.id]["round"] = round_num
                        next_round_nodes.append(pending_node)
                        logger.info(
                            f"Round {round_num} (Re-match): Node {pending_node.id[:8]} still inconclusive, advancing."
                        )

            active_nodes = next_round_nodes

            round_cost = time.time() - round_start_time
            logger.info(f"Round {round_num} finished. Duration: {round_cost:.2f}s")

            # Safety: prevent infinite loops
            if round_num > n * 2:
                logger.warning("Tournament exceeded maximum rounds, breaking")
                break

        # For nodes not marked eliminated, record round as final round (round_num)
        for nid, info in node_status.items():
            if not info["eliminated"]:
                # For nodes that survived to the end, set round to the final round
                info["round"] = max(info["round"], round_num)

        all_node_info = list(node_status.values())

        # Sort: first by not eliminated, then by wins (desc), then by round (desc)
        def sort_key(info):
            if not info["eliminated"]:
                return (2, info["wins"], info["round"])
            else:
                return (1, info["wins"], info["round"])

        all_node_info.sort(key=sort_key, reverse=True)
        sorted_nodes = [info["node"] for info in all_node_info]

        # === Generate display ranks (ties + round) ===
        # Use (wins, round) as grouping key; assign same rank to same group
        # Rank starts at 0: best group rank=0, next group rank=1, etc.
        rank_meta: Dict[str, Dict[str, int]] = {}
        current_rank = 0
        last_key = None

        for info in all_node_info:
            k = (info["wins"], info["round"])
            if last_key is None:
                # Rank 0
                last_key = k
            elif k != last_key:
                # Rank increases by 1 for a new group
                current_rank += 1
                last_key = k
            nid = info["node"].id
            rank_meta[nid] = {"rank": current_rank, "round": info["round"]}
        
        total_cost = time.time() - total_start_time
        logger.info(f"World Model prediction completed in {total_cost:.2f}s")

        return sorted_nodes, comparisons, rank_meta

    def should_execute(self) -> bool:
        """
        Decide whether to execute based on probability.
        This is used for candidates beyond Top-K.
        """
        return random.random() < self.config.exec_probability
