# Adapted from mle-bench (https://github.com/openai/mle-bench)

# Helper classes and utilities for grading submissions and computing competition reports.

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from .common import get_logger, import_fn

logger = get_logger(__name__)

class InvalidSubmissionError(Exception):
    pass

class Grader:
    def __init__(self, name: str, grade_fn: str) -> None:
        self.name = name
        self.grade_fn = import_fn(grade_fn)
        # Cache the latest error message from grading (if any) for higher-level reporting.
        self.last_error_message: Optional[str] = None
        assert isinstance(self.name, str), "Grader name must be a string."
        assert len(self.name) > 0, "Grader name cannot be empty."

    def is_lower_better(self, leaderboard: pd.DataFrame) -> bool:
        # Convert all types to number
        scores = pd.to_numeric(leaderboard["score"], errors="coerce").dropna()
        if len(scores) < 2:
            return False
        top_score = scores.iloc[0]
        bottom_score = scores.iloc[-1]
        return bool(top_score < bottom_score)

    @staticmethod
    def from_dict(data: dict) -> "Grader":
        return Grader(**data)

    def __call__(self, submission: pd.DataFrame, answers: Any) -> Optional[float]:
        # Reset cached error before each grading call.
        self.last_error_message = None
        try:
            score = self.grade_fn(submission, answers)
        except InvalidSubmissionError as e:
            logger.warning(f"Invalid submission: {e}")
            self.last_error_message = f"Invalid submission: {e}"
            return None
        except Exception as e:
            msg = f"Unexpected error during grading: {e}"
            logger.error(msg)
            # Cache the same error message that is logged, so upper layers can write it to error_eval.txt.
            self.last_error_message = msg
            return None
        rounded_score = round(score, 5)
        return rounded_score

    def rank_score(self, score: Optional[float], leaderboard: pd.DataFrame) -> dict:
        assert "score" in leaderboard.columns, "Leaderboard must have a `score` column."
        
        # Convert scores to numeric and clean them to ensure accurate computation.
        scores = pd.to_numeric(leaderboard["score"], errors="coerce").dropna()
        num_teams = len(leaderboard)
        
        # Re-evaluate whether lower scores are better, consistent with processed scores.
        if num_teams >= 2:
            lower_is_better = bool(scores.iloc[0] < scores.iloc[-1])
        else:
            lower_is_better = False

        def get_score_at_position(position: int) -> float:
            if position - 1 >= len(scores) or position < 1:
                raise IndexError("Position out of bounds in the leaderboard.")
            return scores.iloc[position - 1]

        def get_thresholds(n: int) -> tuple[float, float, float, float]:
            if 1 <= n < 100:
                gold_threshold = get_score_at_position(max(1, int(n * 0.1)))
                silver_threshold = get_score_at_position(max(1, int(n * 0.2)))
                bronze_threshold = get_score_at_position(max(1, int(n * 0.4)))
            elif 100 <= n < 250:
                gold_threshold = get_score_at_position(10)
                silver_threshold = get_score_at_position(max(1, int(n * 0.2)))
                bronze_threshold = get_score_at_position(max(1, int(n * 0.4)))
            elif 250 <= n < 1000:
                gold_threshold = get_score_at_position(10 + int(n * 0.002))
                silver_threshold = get_score_at_position(50)
                bronze_threshold = get_score_at_position(100)
            elif n >= 1000:
                gold_threshold = get_score_at_position(10 + int(n * 0.002))
                silver_threshold = get_score_at_position(max(1, int(n * 0.05)))
                bronze_threshold = get_score_at_position(max(1, int(n * 0.1)))
            else:
                raise ValueError("Number of teams in leaderboard must be greater than 0.")
            median_threshold = scores.median()
            return float(gold_threshold), float(silver_threshold), float(bronze_threshold), float(median_threshold)

        gold_threshold, silver_threshold, bronze_threshold, median_threshold = get_thresholds(num_teams)

        # Default beat_ratio is 0.0 (when there is no valid score).
        beat_ratio: float = 0.0

        if score is None:
            return {
                "gold_medal": False,
                "silver_medal": False,
                "bronze_medal": False,
                "above_median": False,
                "gold_threshold": gold_threshold,
                "silver_threshold": silver_threshold,
                "bronze_threshold": bronze_threshold,
                "median_threshold": median_threshold,
                "beat_ratio": beat_ratio,
            }

        gold_medal = score <= gold_threshold if lower_is_better else score >= gold_threshold
        silver_medal = not gold_medal and (score <= silver_threshold if lower_is_better else score >= silver_threshold)
        bronze_medal = (
            not gold_medal and not silver_medal and (score <= bronze_threshold if lower_is_better else score >= bronze_threshold)
        )
        above_median = score < median_threshold if lower_is_better else score > median_threshold

        # ------------- Compute beat_ratio -------------
        # Define "beating human participants" as the fraction of teams with strictly worse scores.
        if lower_is_better:
            # Lower score is better: beat those whose scores are strictly greater (worse) than the current score.
            num_beaten = scores[scores <= score].count()
        else:
            # Higher score is better: beat those whose scores are strictly smaller (worse) than the current score.
            num_beaten = scores[scores >= score].count()

        beat_ratio = float((num_teams - num_beaten) / num_teams)

        return {
            "gold_medal": bool(gold_medal),
            "silver_medal": bool(silver_medal),
            "bronze_medal": bool(bronze_medal),
            "above_median": bool(above_median),
            "gold_threshold": float(gold_threshold),
            "silver_threshold": float(silver_threshold),
            "bronze_threshold": float(bronze_threshold),
            "median_threshold": float(median_threshold),
            "beat_ratio": float(beat_ratio),
        }

@dataclass(frozen=True)
class CompetitionReport:
    competition_id: str
    score: float | None
    gold_threshold: float
    silver_threshold: float
    bronze_threshold: float
    median_threshold: float
    any_medal: bool
    gold_medal: bool
    silver_medal: bool
    bronze_medal: bool
    above_median: bool
    beat_ratio: float
    submission_exists: bool
    valid_submission: bool
    is_lower_better: bool
    created_at: datetime
    submission_path: str
    # If grading encounters an exception/invalid submission, carry the error message recorded in Grader (may be None).
    grader_error: Optional[str] = None
    # Added: fraction of human participants beaten (0.0 ~ 1.0).

    def to_dict(self) -> dict:
        return {
            "competition_id": self.competition_id,
            "score": float(self.score) if self.score is not None else None,
            "gold_threshold": float(self.gold_threshold),
            "silver_threshold": float(self.silver_threshold),
            "bronze_threshold": float(self.bronze_threshold),
            "median_threshold": float(self.median_threshold),
            "any_medal": bool(self.any_medal),
            "gold_medal": bool(self.gold_medal),
            "silver_medal": bool(self.silver_medal),
            "bronze_medal": bool(self.bronze_medal),
            "above_median": bool(self.above_median),
            "submission_exists": bool(self.submission_exists),
            "valid_submission": bool(self.valid_submission),
            "is_lower_better": bool(self.is_lower_better),
            "created_at": self.created_at.isoformat(),
            "submission_path": self.submission_path,
            "grader_error": self.grader_error,
            "beat_ratio": float(self.beat_ratio),
        }
