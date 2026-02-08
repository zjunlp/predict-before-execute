# Adapted from mle-bench (https://github.com/openai/mle-bench)

# Registry for loading competition configurations and constructing Competition objects.

from dataclasses import dataclass
from pathlib import Path
import sys

import yaml

from ..util.grade_helpers import Grader

@dataclass(frozen=True)
class Competition:
    id: str
    grader: Grader
    answers: Path
    leaderboard: Path
    private_dir: Path
    public_dir: Path

class SimpleRegistry:
    def __init__(self, data_dir: Path, competitions_dir: Path):
        self._data_dir = data_dir.resolve()
        self._competitions_dir = competitions_dir.resolve()

        # Make grader modules importable: add the parent of competitions_dir (mlebench root) to sys.path.
        root = self._competitions_dir.parent  # .../mlebench
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    def list_competition_ids(self) -> list[str]:
        return sorted([p.name for p in self._competitions_dir.iterdir() if (p / "config.yaml").exists()])

    def get_competition(self, competition_id: str) -> Competition:
        cfg_path = self._competitions_dir / competition_id / "config.yaml"
        assert cfg_path.exists(), f"Config not found: {cfg_path}"
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        grader = Grader.from_dict(cfg["grader"])

        answers = self._data_dir / cfg["dataset"]["answers"]
        private_dir = self._data_dir / competition_id / "prepared" / "private"
        public_dir = self._data_dir / competition_id / "prepared" / "public"
        leaderboard = self._competitions_dir / competition_id / "leaderboard.csv"

        return Competition(
            id=cfg["id"],
            grader=grader,
            answers=answers,
            leaderboard=leaderboard,
            private_dir=private_dir,
            public_dir=public_dir,
        )
