import json
import re

import black


def wrap_code(code: str, lang="python") -> str:
    """Wraps code with three backticks."""
    return f"```{lang}\n{code}\n```"


def is_valid_python_script(script):
    """Check if a script is a valid Python script."""
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def looks_like_code(block: str) -> bool:
    """
    Heuristic: does this text look like Python code rather than pure natural language?
    不要求一定能 compile 通过，只要包含典型代码结构就认为是代码。
    """
    if not block.strip():
        return False

    # 有这些关键词非常像代码
    code_keywords = [
        "import ",
        "from ",
        "def ",
        "class ",
        "for ",
        "while ",
        "if ",
        "elif ",
        "else:",
        "return ",
        "with ",
        "try:",
        "except ",
        "raise ",
    ]
    if any(kw in block for kw in code_keywords):
        return True

    # 有典型符号也比较像代码
    if re.search(r"[=\[\]{}():]", block):
        return True

    # 含有缩进 + 冒号的行也像代码
    for line in block.splitlines():
        if line.lstrip().startswith("#"):
            return True
        if re.match(r"^\s+\w+.*:", line):
            return True

    return False


def extract_jsons(text):
    """Extract all JSON objects from the text. Caveat: This function cannot handle nested JSON objects."""
    json_objects = []
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for match in matches:
        try:
            json_obj = json.loads(match)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            pass

    # Sometimes chatgpt-turbo forget the last curly bracket, so we try to add it back when no json is found
    if len(json_objects) == 0 and not text.endswith("}"):
        json_objects = extract_jsons(text + "}")
        if len(json_objects) > 0:
            return json_objects

    return json_objects


def trim_long_string(string, threshold=5100, k=2500):
    """
    Pretty-print helper: only for truncating long strings when *displaying* them
    in prompts or logs. It should never be used to overwrite the underlying
    stored output (_term_out), otherwise later consumers will only ever see the
    truncated marker instead of the real traceback.
    """
    # Check if the length of the string is longer than the threshold
    if len(string) > threshold:
        # Output the first k and last k characters
        first_k_chars = string[:k]
        last_k_chars = string[-k:]

        truncated_len = len(string) - 2 * k

        return f"{first_k_chars}\n ... [{truncated_len} characters truncated] ... \n{last_k_chars}"
    else:
        return string


def extract_code(text: str) -> str:
    """Extract python code blocks from the text.

    目标：
    - 优先使用 markdown 里的 ```python``` / ``` 块；
    - 避免把明显的自然语言 reasoning 当成代码；
    - 但不过于严格，不强制每块都能 compile 通过，防止 code block 稍微不完整就整段丢掉。
    """
    # 防御式编程：LLM 可能返回 None 或非字符串
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    parsed_blocks: list[str] = []

    # 1) 先找所有 ```python ... ``` 块
    matches = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    for block in matches:
        candidate = block.strip()
        if candidate and looks_like_code(candidate):
            parsed_blocks.append(candidate)

    # 2) 如果没找到 python 标注的 code block，再找通用 ``` ... ``` 块
    if not parsed_blocks:
        generic_matches = re.findall(r"```(?!python)(.*?)```", text, re.DOTALL)
        for block in generic_matches:
            candidate = block.strip()
            if candidate and looks_like_code(candidate):
                parsed_blocks.append(candidate)

    # 3) 如果依然没有任何块，考虑整个文本是不是代码（原始语义）
    if not parsed_blocks:
        stripped = text.strip()
        if looks_like_code(stripped):
            parsed_blocks.append(stripped)

    if not parsed_blocks:
        # 实在没有任何像代码的内容，就返回空字符串，由上层决定怎么处理
        return ""

    # 只取第一块代码作为“主实现”，避免把后面的残缺代码拼进来
    main_code = parsed_blocks[0]

    # 最后尝试用 black 美化一下；即使 SyntaxError 也不丢弃原文
    try:
        return format_code(main_code)
    except Exception:
        return main_code


def extract_text_up_to_code(s):
    """Extract (presumed) natural language text up to the start of the first code block."""
    if "```" not in s:
        return s.strip()
    return s[: s.find("```")].strip()


def format_code(code) -> str:
    """Format Python code using Black."""
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        return code


def ensure_entrypoint_call(code: str) -> str:
    """
    Ensure that a script with a `main`-like function actually calls it.

    Heuristics:
    - If there is a `def main(` definition (top-level) AND
    - There is NO `if __name__ == "__main__"` block in the file,
      THEN append a standard entrypoint block at the end:

          if __name__ == "__main__":
              main()

    仅做兜底，不会修改已有的入口逻辑。
    """
    # Already has a canonical entrypoint guard
    if re.search(r'if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:', code):
        return code

    # Check for a top-level `def main(` (粗略 heuristic)
    if re.search(r'^\s*def\s+main\s*\(', code, flags=re.MULTILINE):
        # Avoid duplicating if already calling main() at top-level without guard
        # (这类情况一般也是可接受的)
        if re.search(r'^\s*main\s*\(\s*\)\s*$', code, flags=re.MULTILINE):
            return code

        # Append a standard guard at the end
        entry_block = "\n\n\nif __name__ == \"__main__\":\n    main()\n"
        return code.rstrip() + entry_block

    return code


def truncate_log(log_content: str, max_length: int = 2000) -> str:
    """
    Truncate log content to a specified maximum length.
    If the log exceeds the max_length, keep the first and last portions.
    """
    if isinstance(log_content, list):
        log_content = "".join(log_content)

    if len(log_content) <= max_length:
        return log_content

    half_length = max_length // 2
    return f"{log_content[:half_length]}...\n[TRUNCATED]\n...{log_content[-half_length:]}"
