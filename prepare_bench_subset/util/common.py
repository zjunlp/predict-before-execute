# Adapted from mle-bench (https://github.com/mle-bench/mlebench)

# Common utility helpers for logging, reading data files, timestamps, and dynamic imports.

import importlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

def purple(s: str) -> str:
    return f"\033[1;35m{s}\033[0m"

def get_logger(name: str, level: int = logging.INFO, filename: Optional[Path] = None) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        filename=str(filename) if filename else None,
    )
    return logging.getLogger(name)

logger = get_logger(__name__)

def read_jsonl(file_path: str, skip_commented_out_lines: bool = False) -> list[dict]:
    result = []
    with open(file_path, "r") as f:
        if skip_commented_out_lines:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("//"):
                    continue
                result.append(json.loads(line))
        else:
            return [json.loads(line) for line in f]
    return result

def read_csv(*args, **kwargs) -> pd.DataFrame:
    try:
        new_default_kwargs = {"float_precision": "round_trip"}
        new_kwargs = {**new_default_kwargs, **kwargs}
        return pd.read_csv(*args, **new_kwargs)
    except pd.errors.EmptyDataError:
        logger.warning(f"CSV file empty! {args[0]}")
        return pd.DataFrame()

def load_answers(path_to_answers: Path) -> Any:
    if path_to_answers.suffix == ".csv":
        return read_csv(path_to_answers)
    if path_to_answers.suffix == ".jsonl":
        return read_jsonl(str(path_to_answers))
    raise ValueError(f"Unsupported file format for answers: {path_to_answers}")

def get_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%S-%Z", time.gmtime())

def is_empty(dir_path: Path) -> bool:
    return not any(dir_path.iterdir())

def import_fn(fn_import_string: str) -> Callable:
    module_name, fn_name = fn_import_string.split(":")
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    return fn
