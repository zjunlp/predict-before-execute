# Token counting utilities for prompts and files.
import argparse
import sys
from pathlib import Path

def count_with_tiktoken(text: str, model: str = "gpt-4") -> int:
    try:
        import tiktoken
    except Exception:
        raise
    # try to get encoding for model, fallback to gpt2
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("gpt2")
    return len(enc.encode(text))

def count_with_whitespace(text: str) -> int:
    # simple fallback: split on any whitespace
    return len(text.split())

def count_tokens_in_file(path: Path, model: str = "gpt-4") -> tuple[int, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        cnt = count_with_tiktoken(text, model=model)
        method = f"tiktoken (model={model})"
    except Exception:
        cnt = count_with_whitespace(text)
        method = "whitespace-split (fallback)"
    return cnt, method

# new: generic helper for other modules (no filesystem)
def count_tokens_in_text(text: str, model: str = "gpt-4") -> tuple[int, str]:
    """
    Count tokens in a raw text string using tiktoken for the given model,
    falling back to whitespace split if tiktoken is unavailable.
    Returns (count, method_description).
    """
    try:
        cnt = count_with_tiktoken(text, model=model)
        method = f"tiktoken (model={model})"
    except Exception:
        cnt = count_with_whitespace(text)
        method = "whitespace-split (fallback)"
    return cnt, method

def main(argv=None):
    p = argparse.ArgumentParser(description="Count tokens in a text file (tries tiktoken, falls back to whitespace).")
    p.add_argument("file", nargs="?", default="/datadisk/zjs/skip_bench/temp/tokens.txt", help="Path to the tokens.txt file")
    p.add_argument("--model", default="gpt-4", help="Model name for tiktoken encoding (if available)")
    args = p.parse_args(argv)

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(2)

    count, method = count_tokens_in_file(path, model=args.model)
    print(f"File: {path}")
    print(f"Counting method: {method}")
    print(f"Total tokens: {count}")

if __name__ == "__main__":
    main()
