import shutil
import logging
import random
import time
import os
from typing import Any, Callable, cast, List, Optional
from pathlib import Path
import multiprocessing

import humanize
from .backend import FuncSpec, compile_prompt_to_md, query
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .utils import data_preview
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import (
    extract_code,
    extract_text_up_to_code,
    wrap_code,
    ensure_entrypoint_call,
    truncate_log,
)
from .world_model import WorldModel, WorldModelConfig
import torch  # Ensure PyTorch is installed for GPU detection

logger = logging.getLogger("aide")


def format_time(time_in_sec: int):
    return f"{time_in_sec // 3600}hrs {(time_in_sec % 3600) // 60}mins {time_in_sec % 60}secs"


ExecCallbackType = Callable[[str, bool], ExecutionResult]

review_func_spec = FuncSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "has_csv_submission": {
                "type": "boolean",
                "description": "true if the code saves the predictions on the test data"
                " in a `submission.csv` file in the `./submission/` directory, otherwise false."
                " Note that the file MUST be saved in the ./submission/ directory for this to be evaluated as true."
                " Otherwise, it should be evaluated as false."
                " You can assume the ./submission/ directory exists and is writable.",
            },
            "summary": {
                "type": "string",
                "description": "write a short summary (2-3 sentences) describing "
                " the empirical findings. Alternatively mention if there is a bug or"
                " the submission.csv was not properly produced."
                " DO NOT suggest fixes or improvements.",
            },
            "metric": {
                "type": "number",
                "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null.",
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
            },
        },
        "required": [
            "is_bug",
            "has_csv_submission",
            "summary",
            "metric",
            "lower_is_better",
        ],
    },
    description="Submit a review evaluating the output of the training script.",
)


class Agent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
    ):
        super().__init__()
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = cfg.agent
        self.journal = journal
        self.data_preview: str | None = None
        self.data_analysis_report: str | None = None
        self.start_time = time.time()
        self.current_step = 0
        
        # World Model configuration (NEW)
        self.use_world_model = getattr(self.acfg, 'use_world_model', False)
        self.wm_config = WorldModelConfig(
            model=getattr(self.acfg, 'world_model_model', 'DeepSeek-V3.2-Thinking'),
            temperature=getattr(self.acfg, 'world_model_temp', 0.2),
            confidence_threshold=getattr(self.acfg, 'world_model_confidence_threshold', 0.6),
            exec_probability=getattr(self.acfg, 'world_model_exec_probability', 0.3),
            da_dir=getattr(self.acfg, 'data_analysis_dir', None),
        )
        self.improve_num_candidates = getattr(self.acfg, 'improve_num_candidates', 3)
        self.improve_top_k = getattr(self.acfg, 'improve_top_k', 1)  # NEW: separate parameter
        
        # All nodes directory for saving artifacts
        # NOTE:
        #   - start.sh does: ln -s /home/logs/all_nodes ${AGENT_DIR}/workspaces/exp/all_nodes
        #   - cfg.workspace_dir == ${AGENT_DIR}/workspaces/exp
        #   => self.all_nodes_dir points to a symlink whose target is /home/logs/all_nodes
        #   => /home/logs/all_nodes is inside LOGS_DIR and will be extracted by run.py
        self.all_nodes_dir = self.cfg.workspace_dir / "all_nodes"
        self.all_nodes_dir.mkdir(exist_ok=True, parents=True)

    def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        search_cfg = self.acfg.search

        # initial drafting
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.info("[search policy] drafting new node (not enough drafts)")
            return None

        # debugging
        if random.random() < search_cfg.debug_prob:
            # nodes that are buggy + leaf nodes + debug depth < max debug depth
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                node_to_debug = random.choice(debuggable_nodes)
                logger.info(f"[search policy] debugging node {node_to_debug.id}")
                return node_to_debug

        # back to drafting if no nodes to improve
        good_nodes = self.journal.good_nodes
        if not good_nodes:
            logger.info("[search policy] drafting new node (no good nodes)")
            return None

        # greedy
        greedy_node = self.journal.get_best_node()
        logger.info(f"[search policy] greedy node selected: node {greedy_node.id}")
        return greedy_node

    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy",
            "pandas",
            "scikit-learn",
            "statsmodels",
            "xgboost",
            "lightGBM",
            "torch",
            "torchvision",
            "torch-geometric",
            "bayesian-optimization",
            "timm",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        # Check for GPU availability
        gpu_available = torch.cuda.is_available()
        gpu_message = (
            "A GPU is available and should be used for computations where possible."
            if gpu_available
            else "No GPU is available; computations will be performed on the CPU."
        )

        # Limit CPU core usage to 1/6 of total cores
        total_cores = multiprocessing.cpu_count()
        max_cores = max(1, total_cores // 6)
        cpu_message = (
            f"Use multiple CPU cores for computations, but limit the number of cores to {max_cores}."
        )

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow.",
            "Hardware Constraints": f"{gpu_message} {cpu_message}",
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        tot_time_elapsed = time.time() - self.start_time
        tot_time_remaining = self.acfg.time_limit - tot_time_elapsed
        exec_timeout = int(min(self.cfg.exec.timeout, tot_time_remaining))

        impl_guideline = [
            f"<TOTAL_TIME_REMAINING: {format_time(tot_time_remaining)}>",
            f"<TOTAL_STEPS_REMAINING: {self.acfg.steps - self.current_step}>",
            "The code should **implement the proposed solution**, **print the value of the evaluation metric computed on a hold-out validation set**,",
            "**AND MOST IMPORTANTLY SAVE PREDICTIONS ON THE PROVIDED UNLABELED TEST DATA IN A `submission.csv` FILE IN THE ./submission/ DIRECTORY.**",
            "The code should be a single-file python program that is self-contained and can be executed as-is.",
            "No parts of the code should be skipped, don't terminate the before finishing the script.",
            "Your response should only contain a single code block.",
            f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(exec_timeout)}.",
            'All the provided input data is stored in "./input" directory.',
            '**You MUST submit predictions on the provided unlabeled test data in a `submission.csv` file** file in the "./working" directory as described in the task description** This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!',
            'You can also use the "./working" directory to store any temporary files that your code needs to create.',
            "REMEMBER THE ./submission/submission.csv FILE!!!!! The correct directory is important too.",
        ]
        if self.acfg.expose_prediction:
            impl_guideline.append(
                "The implementation should include a predict() function, "
                "allowing users to seamlessly reuse the code to make predictions on new data. "
                "The prediction function should be well-documented, especially the function signature."
            )

        if self.acfg.k_fold_validation > 1:
            impl_guideline.append(
                f"The evaluation should be based on {self.acfg.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
            )

        return {"Implementation guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
            )
        }

    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        for _ in range(retries):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.acfg.code.model,
                temperature=self.acfg.code.temp,
            )

            # Defensive check: LLM may return None or non-string
            if not isinstance(completion_text, str):
                logger.warning(
                    "LLM completion_text is not a string (got %r), retrying...",
                    type(completion_text),
                )
                continue

            # Log the raw LLM response
            logger.info(f"Raw LLM response:\n{completion_text}\n")

            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # First try to fix entrypoint: if main defined but no if __name__ == "__main__"
                fixed_code = ensure_entrypoint_call(code)
                if fixed_code != code:
                    logger.info(
                        "Auto-added __main__ entrypoint guard to generated code "
                        "(def main(...) detected without if __name__ == '__main__')."
                    )
                    code = fixed_code

                logger.info(
                    f"Plan generation succeeded, proposed plan:\n\n{nl_text}\n\n"
                )
                logger.info(
                    f"Code generation succeeded, proposed code:\n\n{code}\n\n"
                )
                # merge all code blocks into a single string
                return nl_text, code

            logger.info("Plan + code extraction failed, retrying...")

        logger.info("Final plan + code extraction attempt failed, giving up...")
        # Make sure to return strings, even if LLM never provided valid text
        return "", completion_text or ""

    def _draft(self) -> Node:
        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "In order to win this competition, you need to come up with an excellent and creative plan "
            "for a solution and then implement this solution in Python. We will now provide a description of the task."
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. "
                "In order to complete this task, you need to come up with an excellent and creative plan "
                "for a solution and then implement this solution in Python. We will now provide a description of the task."
            )
        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution sketch guideline": [
                "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.",
                "Take the Memory section into consideration when proposing the design,"
                " don't propose the same modelling solution but keep the evaluation the same.",
                "The solution sketch should be 3-5 sentences.",
                "Propose an evaluation metric that is reasonable for this task.",
                "Don't suggest to do EDA.",
                "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan, code = self.plan_and_code_query(prompt)
        new_node = Node(plan=plan, code=code)
        logger.info(f"Drafted new node {new_node.id}")
        return new_node

    def _debug(self, parent_node: Node) -> Node:
        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "Your previous solution had a bug and/or did not produce a submission.csv, "
            "so based on the information below, you should revise it in order to fix this. "
            "Your response should be an implementation outline in natural language,"
            " followed by a single markdown code block which implements the bugfix/solution."
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. "
                "Your previous solution had a bug and/or did not produce a submission.csv, "
                "so based on the information below, you should revise it in order to fix this. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            )
        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            # Use full_term_out to ensure debug agent sees complete error messages
            # [MODIFIED] Truncate output to avoid context overflow
            "Execution output": wrap_code(truncate_log(log_content=parent_node.full_term_out), lang=""),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan, code = self.plan_and_code_query(prompt)
        new_node = Node(plan=plan, code=code, parent=parent_node)
        logger.info(f"Debugged node {parent_node.id} to create new node {new_node.id}")
        return new_node

    def _improve_batch(self, parent_node: Node, num_candidates: int = 3) -> List[Node]:
        """
        Generate multiple improvement candidates for World Model ranking.
        """
        candidates = []
        existing_plans = []
        
        for i in range(num_candidates):
            introduction = (
                "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
                "solution below and should improve it in order to further increase the (test time) performance. "
                "For this you should first outline a brief plan in natural language for how the solution can be improved and "
                "then implement this improvement in Python based on the provided previous solution. "
            )
            if self.acfg.obfuscate:
                introduction = (
                    "You are an expert machine learning engineer attempting a task. You are provided with a previously developed "
                    "solution below and should improve it in order to further increase the (test time) performance. "
                    "For this you should first outline a brief plan in natural language for how the solution can be improved and "
                    "then implement this improvement in Python based on the provided previous solution. "
                )
            
            prompt: Any = {
                "Introduction": introduction,
                "Task description": self.task_desc,
                "Memory": self.journal.generate_summary(),
                "Instructions": {},
            }
            prompt["Previous solution"] = {
                "Code": wrap_code(parent_node.code),
            }

            prompt["Instructions"] |= self._prompt_resp_fmt
            
            # Add constraint to avoid duplicate improvement directions
            avoid_directions = ""
            if existing_plans:
                avoid_directions = (
                    "\n- IMPORTANT: The following improvement directions have already been proposed. "
                    "You MUST propose a DIFFERENT improvement:\n" +
                    "\n".join([f"  * {plan[:200]}..." for plan in existing_plans])
                )
            
            prompt["Instructions"] |= {
                "Solution improvement sketch guideline": [
                    "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
                    "You should be very specific and should only propose a single actionable improvement.",
                    "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
                    "Take the Memory section into consideration when proposing the improvement.",
                    "The solution sketch should be 3-5 sentences.",
                    "Don't suggest to do EDA.",
                ] + ([avoid_directions] if avoid_directions else []),
            }
            prompt["Instructions"] |= self._prompt_impl_guideline

            plan, code = self.plan_and_code_query(prompt)
            
            if plan and code:
                new_node = Node(plan=plan, code=code, parent=parent_node)
                candidates.append(new_node)
                existing_plans.append(plan)
                logger.info(f"Generated improvement candidate {i+1}/{num_candidates}: {new_node.id[:8]}")
            else:
                logger.warning(f"Failed to generate improvement candidate {i+1}/{num_candidates}")
        
        return candidates

    def _improve(self, parent_node: Node) -> Node:
        introduction = (
            "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
            "solution below and should improve it in order to further increase the (test time) performance. "
            "For this you should first outline a brief plan in natural language for how the solution can be improved and "
            "then implement this improvement in Python based on the provided previous solution. "
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. You are provided with a previously developed "
                "solution below and should improve it in order to further increase the (test time) performance. "
                "For this you should first outline a brief plan in natural language for how the solution can be improved and "
                "then implement this improvement in Python based on the provided previous solution. "
            )
        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution improvement sketch guideline": [
                "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
                "You should be very specific and should only propose a single actionable improvement.",
                "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
                "Take the Memory section into consideration when proposing the improvement.",
                "The solution sketch should be 3-5 sentences.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        plan, code = self.plan_and_code_query(prompt)
        new_node = Node(plan=plan, code=code, parent=parent_node)
        logger.info(f"Improved node {parent_node.id} to create new node {new_node.id}")
        return new_node

    def load_data_analysis_report(self, report_path: Optional[str] = None):
        """
        Load data analysis report from file for World Model.
        
        Priority order:
        1. /home/data/da_report.txt (MLE-bench mounted location)
        2. Explicit report_path
        3. data_analysis_dir/{task_name} pattern
        4. Fallback to data_preview
        """
        # Priority 1: Check MLE-bench mounted location
        mlebench_da_path = "/home/data/da_report.txt"
        if os.path.exists(mlebench_da_path):
            try:
                with open(mlebench_da_path, 'r', encoding='utf-8') as f:
                    self.data_analysis_report = f.read().strip()
                logger.info(f"Loaded data analysis report from MLE-bench location: {mlebench_da_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load from MLE-bench location: {e}")
        
        # Priority 2: Explicit path
        if report_path and os.path.exists(report_path):
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    self.data_analysis_report = f.read().strip()
                logger.info(f"Loaded data analysis report from {report_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load data analysis report: {e}")
        
        # Priority 3: Try data_analysis_dir with task name pattern
        da_dir = getattr(self.acfg, 'data_analysis_dir', None)
        if da_dir and os.path.isdir(da_dir):
            # Try to infer task name from config
            task_name = getattr(self.cfg, 'task_name', '')
            if task_name:
                # Try multiple name variants
                from .world_model import _candidate_keys
                for key in _candidate_keys(task_name):
                    da_path = os.path.join(da_dir, f"da_result_{key}.txt")
                    if os.path.exists(da_path):
                        try:
                            with open(da_path, 'r', encoding='utf-8') as f:
                                self.data_analysis_report = f.read().strip()
                            logger.info(f"Loaded data analysis report from {da_path}")
                            return
                        except Exception as e:
                            logger.warning(f"Failed to load from {da_path}: {e}")
        
        # Fallback: use data_preview
        logger.warning("No data analysis report found, falling back to data_preview")
        self.data_analysis_report = self.data_preview

    def step(self, exec_callback: ExecCallbackType):
        # clear the submission dir from previous steps
        shutil.rmtree(self.cfg.workspace_dir / "submission", ignore_errors=True)
        (self.cfg.workspace_dir / "submission").mkdir(exist_ok=True)

        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()
            # Also try to load data analysis report for World Model
            if self.use_world_model and self.data_analysis_report is None:
                da_path = getattr(self.acfg, 'data_analysis_path', None)
                self.load_data_analysis_report(da_path)

        parent_node = self.search_policy()
        logger.info(f"Agent is generating code, parent node type: {type(parent_node)}")

        if parent_node is None:
            # Draft stage - no World Model
            result_node = self._draft()
            result_node = self.parse_exec_result(
                node=result_node,
                exec_result=exec_callback(result_node.code, True),
            )
            self._post_process_node(result_node)
            self.journal.append(result_node)
            # Save node artifacts
            node_dir = self.all_nodes_dir / f"node_{result_node.id[:8]}"
            self._save_node_artifacts(result_node, node_dir)

        elif parent_node.is_buggy:
            # Debug stage - no World Model
            result_node = self._debug(parent_node)
            result_node = self.parse_exec_result(
                node=result_node,
                exec_result=exec_callback(result_node.code, True),
            )
            self._post_process_node(result_node)
            self.journal.append(result_node)
            # Save node artifacts
            node_dir = self.all_nodes_dir / f"node_{result_node.id[:8]}"
            self._save_node_artifacts(result_node, node_dir)

        else:
            # Improve stage - use World Model if enabled
            if self.use_world_model:
                self._improve_with_world_model(parent_node, exec_callback)
            else:
                result_node = self._improve(parent_node)
                result_node = self.parse_exec_result(
                    node=result_node,
                    exec_result=exec_callback(result_node.code, True),
                )
                self._post_process_node(result_node)
                self.journal.append(result_node)
                # Save node artifacts
                node_dir = self.all_nodes_dir / f"node_{result_node.id[:8]}"
                self._save_node_artifacts(result_node, node_dir)

        self._update_best_solution()
        self.current_step += 1

    def _save_node_artifacts(self, node: Node, node_dir: Path):
        """Save all artifacts for a node (code, submission.csv, terminal output, etc.)."""
        node_dir.mkdir(exist_ok=True, parents=True)
        
        # Save code
        with open(node_dir / "code.py", "w") as f:
            f.write(node.code or "")
        
        # Save plan
        with open(node_dir / "plan.txt", "w") as f:
            f.write(node.plan or "")
        
        # Save terminal output (if executed)
        if node._term_out and not node.is_skipped:
            with open(node_dir / "terminal_output.txt", "w") as f:
                f.write("".join(node._term_out))
        
        # Save submission.csv (if exists)
        submission_path = self.cfg.workspace_dir / "submission" / "submission.csv"
        if submission_path.exists() and not node.is_skipped:
            shutil.copy(submission_path, node_dir / "submission.csv")
        
        # Save node metadata
        metadata = {
            "node_id": node.id,
            "step": node.step,
            "stage": node.stage_name,
            "is_skipped": node.is_skipped,
            "is_buggy": node.is_buggy,
            "metric": str(node.metric) if node.metric else None,
            "analysis": node.analysis,
            "parent_id": node.parent.id if node.parent else None,
            "exec_time": node.exec_time,
            "exc_type": node.exc_type,
            "wm_predicted_rank": node.wm_predicted_rank,
            "wm_comparison_results": node.wm_comparison_results,
        }
        
        import json
        with open(node_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _improve_with_world_model(self, parent_node: Node, exec_callback: ExecCallbackType):
        """
        Improve using World Model predictions with tournament ranking.
        
        Logic:
        - Generate N candidates (improve_num_candidates)
        - Rank them using World Model
        - For Top-K (improve_top_k): execute with probability (exec_probability)
        - For beyond Top-K: skip directly (no execution)
        """
        # 1. Generate multiple candidates
        candidates = self._improve_batch(parent_node, self.improve_num_candidates)
        
        if not candidates:
            logger.warning("No improvement candidates generated, falling back to standard improve")
            result_node = self._improve(parent_node)
            result_node = self.parse_exec_result(
                node=result_node,
                exec_result=exec_callback(result_node.code, True),
            )
            self._post_process_node(result_node)
            self.journal.append(result_node)
            node_dir = self.all_nodes_dir / f"node_{result_node.id[:8]}"
            self._save_node_artifacts(result_node, node_dir)
            return
        
        if len(candidates) == 1:
            # Only one candidate, just execute it
            result_node = candidates[0]
            result_node = self.parse_exec_result(
                node=result_node,
                exec_result=exec_callback(result_node.code, True),
            )
            self._post_process_node(result_node)
            self.journal.append(result_node)
            node_dir = self.all_nodes_dir / f"node_{result_node.id[:8]}"
            self._save_node_artifacts(result_node, node_dir)
            return
        
        # 2. Use World Model to rank candidates
        world_model = WorldModel(
            task_desc=self.task_desc,
            data_preview=self.data_analysis_report or self.data_preview or "",
            task_name=getattr(self.cfg, 'task_name', ''),
            config=self.wm_config,
        )
        
        ranked_candidates, comparisons, rank_meta = world_model.tournament_rank(candidates)
        
        # Store comparison results and rank info in nodes
        for node in ranked_candidates:
            meta = rank_meta.get(node.id, {"rank": None, "round": None})
            node.wm_predicted_rank = meta["rank"]
            node.wm_round = meta["round"]
            node.wm_comparison_results = [
                {
                    "winner": c.winner_idx,
                    "confidence": c.confidence,
                    "round": c.round_num,
                    "valid": c.is_valid,
                }
                for c in comparisons
            ]
        
        # 3. Execution policy:
        #    - Rank 0 ~ (top_k-1): execute with probability
        #    - Rank >= top_k: skip directly
        executed_count = 0
        for i, node in enumerate(ranked_candidates):
            # Use node.wm_predicted_rank to determine Top-K status
            node_rank = node.wm_predicted_rank if node.wm_predicted_rank is not None else i
            if node_rank < self.improve_top_k:
                # Within Top-K: execute with probability
                should_exec = world_model.should_execute()
                if should_exec:
                    logger.info(
                        f"Executing candidate (rank={node_rank}, Top-{self.improve_top_k}, prob triggered): {node.id[:8]}"
                    )
                else:
                    logger.info(
                        f"Skipping candidate (rank={node_rank}, Top-{self.improve_top_k}, prob not triggered): {node.id[:8]}"
                    )
            else:
                # Beyond Top-K: always skip
                should_exec = False
                logger.info(
                    f"Skipping candidate (rank={node_rank}, beyond Top-{self.improve_top_k}): {node.id[:8]}"
                )
            
            if should_exec:
                # Actually execute this candidate
                node = self.parse_exec_result(
                    node=node,
                    exec_result=exec_callback(node.code, True),
                )
                self._post_process_node(node)
                executed_count += 1
            else:
                # Skip execution - mark as skipped
                node.is_skipped = True
                node.is_buggy = False
                node.metric = None
                node.analysis = f"Skipped by World Model (predicted rank: {node_rank}, round: {node.wm_round})"
            
            self.journal.append(node)
            
            # Save node artifacts (both executed and skipped)
            node_dir = self.all_nodes_dir / f"node_{node.id[:8]}"
            self._save_node_artifacts(node, node_dir)
        
        # Safety check: ensure at least one was executed
        # If all Top-K failed probability check, force execute Rank 0
        if executed_count == 0 and ranked_candidates:
            logger.warning("No nodes executed due to probability. Forcing execution of Rank 0 candidate.")
            # Find the node with rank=0 in rank_meta; if not found, use the first one
            top_node = None
            for node in ranked_candidates:
                if node.wm_predicted_rank == 0:
                    top_node = node
                    break
            if top_node is None:
                top_node = ranked_candidates[0]
            # Remove from journal and re-execute
            if top_node in self.journal.nodes:
                self.journal.nodes.remove(top_node)
            top_node.is_skipped = False
            top_node = self.parse_exec_result(
                node=top_node,
                exec_result=exec_callback(top_node.code, True),
            )
            self._post_process_node(top_node)
            self.journal.append(top_node)
            # Re-save artifacts
            node_dir = self.all_nodes_dir / f"node_{top_node.id[:8]}"
            self._save_node_artifacts(top_node, node_dir)

    def _post_process_node(self, result_node: Node):
        """Post-process an executed node (check for submission.csv, etc.)."""
        if not result_node.is_buggy and not result_node.is_skipped:
            submission_path = self.cfg.workspace_dir / "submission" / "submission.csv"
            if not submission_path.exists():
                logger.warning(
                    "Post-process: submission.csv is missing on disk, "
                    f"marking node {result_node.id} as buggy. "
                    f"(workspace_dir={self.cfg.workspace_dir})"
                )
                result_node.is_buggy = True
                result_node.metric = WorstMetricValue()
                logger.info(
                    f"Actually, node {result_node.id} did not produce a submission.csv"
                )

    def _update_best_solution(self):
        """Update best solution cache if current best changed."""
        best_node = self.journal.get_best_node()
        if best_node is not None:
            # Check if this is a newly added best node
            latest_executed = [n for n in self.journal.nodes if not n.is_skipped]
            if latest_executed and best_node.id == latest_executed[-1].id:
                logger.info(f"Node {best_node.id} is the best node so far")
                best_solution_dir = self.cfg.workspace_dir / "best_solution"
                best_solution_dir.mkdir(exist_ok=True, parents=True)
                best_submission_dir = self.cfg.workspace_dir / "best_submission"
                best_submission_dir.mkdir(exist_ok=True, parents=True)
                
                submission_path = self.cfg.workspace_dir / "submission" / "submission.csv"
                if submission_path.exists():
                    shutil.copy(submission_path, best_submission_dir)
                
                with open(best_solution_dir / "solution.py", "w") as f:
                    f.write(best_node.code)
                with open(best_solution_dir / "node_info.txt", "w") as f:
                    f.write(
                        f"node_id: {str(best_node.id)}\n\nmetric: {str(best_node.metric)}\n\nsolution:\n{best_node.plan}"
                    )

    def update_data_preview(
        self,
    ):
        self.data_preview = data_preview.generate(self.cfg.workspace_dir)

    def parse_exec_result(self, node: Node, exec_result: ExecutionResult) -> Node:
        logger.info(f"Agent is parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)

        # Output original ExecutionResult.term_out info for comparison
        raw_out = "".join(exec_result.term_out or [])
        logger.info(
            "Raw execution output for node %s: %d chars, first 1000 chars:\n%s",
            node.id,
            len(raw_out),
            raw_out[:1000],
        )

        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "You have written code to solve this task and now need to evaluate the output of the code execution. "
            "You should determine if there were any bugs as well as report the empirical findings."
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. "
                "You have written code to solve this task and now need to evaluate the output of the code execution. "
                "You should determine if there were any bugs as well as report the empirical findings."
            )
        prompt = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Implementation": wrap_code(node.code),
            "Execution output": wrap_code(node.term_out, lang=""),
        }

        response = cast(
            dict,
            query(
                system_message=prompt,
                user_message=None,
                func_spec=review_func_spec,
                model=self.acfg.feedback.model,
                temperature=self.acfg.feedback.temp,
            ),
        )
        
        logger.info(f"Verification results:\n\n{response}\n\n")

        # if the metric isn't a float then fill the metric with the worst metric
        if not isinstance(response.get("metric", None), float):
            response["metric"] = None

        # do an extra check, to catch cases where judge fails
        has_csv_submission = (
            self.cfg.workspace_dir / "submission" / "submission.csv"
        ).exists()

        node.analysis = response["summary"]

        # Detailed logging for bug decision conditions
        cond_is_bug = bool(response.get("is_bug"))
        cond_exc = node.exc_type is not None
        cond_metric_none = response["metric"] is None
        cond_judge_no_csv = response.get("has_csv_submission") is False
        cond_disk_no_csv = has_csv_submission is False

        logger.info(
            "Bug decision details for node %s: "
            "is_bug=%s, exc_type=%s, metric_is_none=%s, "
            "judge_has_csv=%s, file_has_csv=%s",
            node.id,
            cond_is_bug,
            node.exc_type,
            cond_metric_none,
            response.get("has_csv_submission"),
            has_csv_submission,
        )

        node.is_buggy = (
            cond_is_bug
            or cond_exc
            or cond_metric_none
            or cond_judge_no_csv
            or cond_disk_no_csv
        )

        if node.is_buggy:
            logger.info(
                "Parsed results: Node %s is buggy and/or did not produce a submission.csv "
                "(reasons: is_bug=%s, exc_type_not_none=%s, metric_is_none=%s, "
                "judge_has_csv_false=%s, disk_has_csv_false=%s)",
                node.id,
                cond_is_bug,
                cond_exc,
                cond_metric_none,
                cond_judge_no_csv,
                cond_disk_no_csv,
            )
            node.metric = WorstMetricValue()
        else:
            logger.info(f"Parsed results: Node {node.id} is not buggy")
            node.metric = MetricValue(
                response["metric"], maximize=not response["lower_is_better"]
            )

        return node
