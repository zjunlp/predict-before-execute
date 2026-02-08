# Adapted from mle-bench (https://github.com/openai/mle-bench)

# Utility functions for loading leaderboards and checking dataset preparation status.

from pathlib import Path

import pandas as pd

from .common import get_logger, is_empty

logger = get_logger(__name__)

def get_leaderboard(leaderboard_path: Path) -> pd.DataFrame:
    assert leaderboard_path.exists(), f"Leaderboard not found: {leaderboard_path}"
    return pd.read_csv(leaderboard_path)

def is_dataset_prepared(answers: Path, public_dir: Path, private_dir: Path, grading_only: bool = False) -> bool:
    if not grading_only:
        if not public_dir.is_dir():
            logger.warning("Public directory does not exist.")
            return False
        if is_empty(public_dir):
            logger.warning("Public directory is empty.")
            return False

    if not private_dir.is_dir():
        logger.warning("Private directory does not exist.")
        return False
    if is_empty(private_dir):
        logger.warning("Private directory is empty.")
        return False

    if not answers.is_file():
        logger.warning("Answers file does not exist.")
        return False

    return True
