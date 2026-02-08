# Utility helpers for SEARCH/REPLACE diff parsing, application, and fenced code extraction.
import logging
import re
from typing import List, Tuple, Optional

logger = logging.getLogger("debug")


def wrap_code(code: str, lang: str = "python") -> str:
    """Wrap code into a fenced markdown block."""
    return f"```{lang}\n{code}\n```"


def extract_diff_blocks(response: str) -> List[Tuple[str, str]]:
    """Extract SEARCH/REPLACE diff blocks from an LLM response.

    Block format:
    <<<<<<< SEARCH
    # exact code to replace (must match exactly)
    =======
    # new code
    >>>>>>> REPLACE
    """
    pattern = r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE"
    matches = re.findall(pattern, response, re.DOTALL)
    return [(search.strip(), replace.strip()) for search, replace in matches]


def has_diff(response: str) -> bool:
    """Return True if the response contains at least one SEARCH/REPLACE diff block."""
    return bool(extract_diff_blocks(response))


def apply_diff_to_code(code: str, diff_blocks: List[Tuple[str, str]]) -> str:
    """Apply diff blocks to code; remove leftover markers if any."""
    modified_code = code
    for search, replace in diff_blocks:
        if search in modified_code:
            modified_code = modified_code.replace(search, replace)
        else:
            logger.warning(f"Search block not found in code:\n{search}")

    # Clean up any remaining diff markers just in case
    modified_code = re.sub(
        r"<<<<<<< SEARCH.*?=======.*?>>>>>>> REPLACE",
        "",
        modified_code,
        flags=re.DOTALL,
    )
    modified_code = re.sub(r"<<<<<<< SEARCH", "", modified_code)
    modified_code = re.sub(r"=======", "", modified_code)
    modified_code = re.sub(r">>>>>>> REPLACE", "", modified_code)
    return modified_code


def extract_code_block(text: str) -> Optional[str]:
    """
    Extract the first fenced code block content (any language).

    NOTE: for the main LLM editing paths we now REQUIRE SEARCH/REPLACE diffs,
    so this helper should not be used as a fallback to accept full rewrites.
    """
    # prefer fenced code block
    m = re.search(r"```(?:\w+)?\n(.*?)\n```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # fallback: treat the whole text as code if no fences present
    stripped = text.strip()
    if stripped:
        return stripped
    return None