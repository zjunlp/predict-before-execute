import os
from typing import List, Dict, Optional, Iterator, Tuple
import httpx  # NEW

try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("Please install the OpenAI Python SDK: pip install openai>=1.0.0") from e

# Required configuration (defaults can be overridden by environment variables)
# Read sensitive configuration from environment variables only.
# Set environment variables: OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or None
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None
DEFAULT_MODEL = os.getenv("OPENAI_MODEL") or "DeepSeek-V3.2"

def _resolve_config(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
):
    """
    Resolve final (api_key, base_url, model) with clear priority:

    1. Explicit function arguments (api_key/base_url/model)  -- from CLI or caller
      2. Environment variables: OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
      3. Module-level defaults: OPENAI_API_KEY / OPENAI_BASE_URL / DEFAULT_MODEL
    """
    # 1) api_key
    if api_key and str(api_key).strip():
        final_api_key = api_key
    else:
        final_api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY

    # 2) base_url
    if base_url and str(base_url).strip():
        final_base_url = base_url
    else:
        final_base_url = os.getenv("OPENAI_BASE_URL") or OPENAI_BASE_URL

    # 3) model
    if model and str(model).strip():
        final_model = model
    else:
        final_model = os.getenv("OPENAI_MODEL") or DEFAULT_MODEL

    return final_api_key, final_base_url, final_model

# NEW: Use only HTTP/HTTPS proxies and ignore system-level SOCKS/ALL_PROXY
def _build_http_client_from_env() -> Optional[httpx.Client]:
    """
    Build a sync httpx.Client using only HTTP/HTTPS proxies from environment variables.
    Ignore system-level SOCKS or ALL_PROXY settings by using trust_env=False.
    """
    def _valid_http_scheme(url: Optional[str]) -> bool:
        if not url:
            return False
        u = url.lower().strip()
        return u.startswith("http://") or u.startswith("https://")

    http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy") or os.getenv("OPENAI_HTTP_PROXY")
    https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or os.getenv("OPENAI_HTTPS_PROXY")

    # Only accept http/https schemes; ignore socks or other proxy schemes
    if not _valid_http_scheme(http_proxy):
        http_proxy = None
    if not _valid_http_scheme(https_proxy):
        https_proxy = None

    proxy_url = https_proxy or http_proxy

    # Use trust_env=False to avoid picking up system-wide ALL_PROXY (socks) etc.
    try:
        return httpx.Client(proxy=proxy_url, trust_env=False)
    except TypeError:
        proxies = {}
        if http_proxy:
            proxies["http://"] = http_proxy
        if proxy_url:
            proxies["https://"] = proxy_url
        return httpx.Client(proxies=proxies or None, trust_env=False)

_client: Optional[OpenAI] = None
_async_http_client: Optional[httpx.AsyncClient] = None  # NEW: shared async client

# NEW: client pool + per-client concurrency limit
_MAX_CONCURRENCY_PER_CLIENT = int(os.getenv("OPENAI_MAX_CONCURRENCY_PER_CLIENT", "64") or "64")
_client_pool: Dict[int, OpenAI] = {}  # pool_index -> OpenAI


def get_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    """
    Backwards-compatible single-client getter.
    Kept for callers that don't care about client pools.
    """
    global _client
    # NOTE: Reuse the singleton only when caller does not provide api_key/base_url;
    # if a new key/url is provided, recreate the client.
    if _client is not None and api_key is None and base_url is None:
        return _client
    api_key_resolved, base_url_resolved, _ = _resolve_config(api_key=api_key, base_url=base_url)
    http_client = _build_http_client_from_env()
    _client = OpenAI(api_key=api_key_resolved, base_url=base_url_resolved, http_client=http_client) \
        if http_client else OpenAI(api_key=api_key_resolved, base_url=base_url_resolved)
    return _client

# NEW: client-pool aware getter
def get_client_for_slot(
    slot_index: int,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OpenAI:
    """
    Map a logical 'slot index' (e.g., group index or worker index) to a client in a small pool.
    The idea:
      - We assume each OpenAI client (or its underlying HTTP pool) can safely handle
        up to _MAX_CONCURRENCY_PER_CLIENT concurrent requests.
      - Given a global concurrency (e.g., --parallel), caller assigns each request
        a slot_index (0..parallel-1).
      - We map this slot_index to a pool_index = slot_index // _MAX_CONCURRENCY_PER_CLIENT,
        and lazily create an OpenAI client for that pool_index.

    This allows a single process to maintain multiple HTTP client instances, each
    handling a bounded amount of concurrent load, which may help avoid server-side
    per-client throttling.
    """
    global _client_pool
    if slot_index < 0:
        pool_index = 0
    else:
        pool_index = slot_index // max(1, _MAX_CONCURRENCY_PER_CLIENT)

    # Reuse existing client only when caller does not provide api_key/base_url
    if pool_index in _client_pool and api_key is None and base_url is None:
        return _client_pool[pool_index]

    api_key_resolved, base_url_resolved, _ = _resolve_config(api_key=api_key, base_url=base_url)
    http_client = _build_http_client_from_env()
    client = OpenAI(api_key=api_key_resolved, base_url=base_url_resolved, http_client=http_client) \
        if http_client else OpenAI(api_key=api_key_resolved, base_url=base_url_resolved)
    _client_pool[pool_index] = client
    return client

# NEW: async OpenAI client helper (reuse only the underlying httpx.AsyncClient, preserving sync logic)
def _get_async_http_client() -> httpx.AsyncClient:
    """
    Lazily create a process-wide AsyncClient using the same proxy/env rules as sync client.
    Avoids per-request TCP connection overhead and can significantly increase concurrency throughput.
    """
    global _async_http_client
    if _async_http_client is not None:
        return _async_http_client

    # Reuse the logic from _build_http_client_from_env but construct an AsyncClient
    def _valid_http_scheme(url: Optional[str]) -> bool:
        if not url:
            return False
        u = url.lower().strip()
        return u.startswith("http://") or u.startswith("https://")

    http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy") or os.getenv("OPENAI_HTTP_PROXY")
    https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or os.getenv("OPENAI_HTTPS_PROXY")
    if not _valid_http_scheme(http_proxy):
        http_proxy = None
    if not _valid_http_scheme(https_proxy):
        https_proxy = None
    proxy_url = https_proxy or http_proxy

    try:
        _async_http_client = httpx.AsyncClient(proxy=proxy_url, trust_env=False, timeout=60.0)
    except TypeError:
        proxies = {}
        if http_proxy:
            proxies["http://"] = http_proxy
        if proxy_url:
            proxies["https://"] = proxy_url
        _async_http_client = httpx.AsyncClient(proxies=proxies or None, trust_env=False, timeout=60.0)
    return _async_http_client

async def async_chat_complete(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """
    Async version of chat_complete using httpx.AsyncClient + OpenAI-compatible API.
    Note: This only supports the standard chat.completions endpoint and does not
    handle DeepSeek "thinking" special logic. Intended for benchmark/concurrency scripts.
    """
    api_key_resolved, base_url_resolved, model_resolved = _resolve_config(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    client = _get_async_http_client()
    url = (base_url_resolved or "").rstrip("/") + "/chat/completions"
    payload: Dict[str, object] = {
        "model": model_resolved,
        "messages": messages,
        "temperature": float(temperature),
        "stream": False,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    headers = {
        "Authorization": f"Bearer {api_key_resolved}",
        "Content-Type": "application/json",
    }
    resp = await client.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    # OpenAI-compatible shape: { choices: [ { message: { content: "..." } } ] }
    try:
        content = data["choices"][0]["message"]["content"] or ""
    except Exception:
        # Be tolerant of unexpected response shapes to avoid aborting a load test
        content = str(data)
    return content.strip()

async def async_chat_complete_many(
    batched_messages: List[Tuple[List[Dict[str, str]], Dict[str, object]]],
    default_model: Optional[str] = None,
    default_temperature: float = 0.2,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> List[str]:
    """
    High-concurrency batch caller:
      batched_messages: list of (messages, options), options like {"model": "...", "temperature": 0.2}
    Returns a list of content strings corresponding to each request (order preserved).
    Example:
      payloads = []
      for group in groups:
          payloads.append((messages_for_group, {"model": "gpt-4o", "temperature": 0.2}))
      contents = asyncio.run(async_chat_complete_many(payloads))
    """
    import asyncio

    async def _one(idx: int, msgs: List[Dict[str, str]], opts: Dict[str, object]) -> Tuple[int, str]:
        model = str(opts.get("model") or default_model or "")
        temp = float(opts.get("temperature") or default_temperature)
        # Reuse async_chat_complete here to centralize error handling
        try:
            text = await async_chat_complete(
                messages=msgs,
                model=model or None,
                temperature=temp,
                max_tokens=opts.get("max_tokens"),  # type: ignore[arg-type]
                api_key=api_key,
                base_url=base_url,
            )
        except Exception as e:
            text = f"[async_chat_complete error] {e}"
        return idx, text

    tasks = [
        _one(i, msgs, opts)
        for i, (msgs, opts) in enumerate(batched_messages)
    ]
    results: List[str] = ["" for _ in range(len(tasks))]
    for i, text in await asyncio.gather(*tasks):
        results[i] = text
    return results

def chat_complete(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    client_slot: Optional[int] = None,  # NEW: logical slot index for client-pool routing
    max_retries: Optional[int] = 0,
) -> str:
    """
    messages example:
      [{"role": "system", "content": "You are a helpful assistant."},
       {"role": "user", "content": "Hello"}]

    client_slot:
      - If provided, map to a specific OpenAI client instance in a small pool.
      - If None, fall back to the legacy single-client behavior.

    max_retries:
      - Number of extra attempts if the underlying API call raises an exception
        (HTTP 4xx/5xx, transport error, SDK error, etc.).
      - The request payload (messages / model / temperature / max_tokens) is NOT changed
        between retries.
    """
    # Resolve final model name (consider CLI/env overrides).
    # Use _resolve_config for consistent precedence; api_key/base_url are optional here.
    _, _, model_resolved = _resolve_config(api_key=api_key, base_url=base_url, model=model)
    model_lower = (model_resolved or "").lower()

    # If using DeepSeek "thinking" models, route to dedicated thinking channel:
    # - deepseek-v3.2-thinking
    # - deepseek-reasoner
    if model_lower in ("deepseek-v3.2-thinking", "deepseek-reasoner", "qwen3-235b-a22b-thinking-2507", "qwen3-30b-a3b-thinking-2507"):
        # Use dedicated thinking API instead of relying on prompt-based COT hacks
        result = chat_complete_with_thinking_ds(
            messages=messages,
            model=model_resolved,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )
        reasoning = (result.get("reasoning_content") or "").strip()
        content = (result.get("content") or "").strip()

        # To remain compatible with existing parse_response logic:
        # - Do not assume content is only JSON
        # - Return reasoning and final answer together so logs contain the full chain-of-thought
        if reasoning and content:
            combined = f"{reasoning}\n\n{content}"
        elif reasoning:
            combined = reasoning
        else:
            combined = content
        return combined

    # For regular models: choose single client or client pool based on client_slot
    if client_slot is not None:
        client = get_client_for_slot(client_slot, api_key=api_key, base_url=base_url)
    else:
        client = get_client(api_key=api_key, base_url=base_url)

    # Defensive handling of max_retries: allow None or non-integer
    try:
        mr_int = int(max_retries) if max_retries is not None else 0
    except Exception:
        mr_int = 0
    attempts = max(0, mr_int) + 1

    last_err: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            # Only print debug info on retries
            if attempt > 0:
                print(f"[DEBUG] Attempt {attempt + 1}/{attempts} for LLM request.")
            resp = client.chat.completions.create(
                model=model_resolved,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            # Retry on any exception (400/429/500/transport errors, etc.)
            last_err = e
            # Print more detailed error information to aid debugging
            detail = getattr(e, "message", "") or getattr(e, "args", [""])[0]
            print(f"[DEBUG] Attempt {attempt + 1} failed with error: {detail!r}")
            continue

    if last_err is not None:
        print(f"[DEBUG] All {attempts} attempts failed. Raising last error.")
        raise last_err
    return ""

def chat_complete_stream(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Iterator[str]:
    client = get_client(api_key=api_key, base_url=base_url)
    _, _, model_resolved = _resolve_config(model=model)
    stream = client.chat.completions.create(
        model=model_resolved,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = getattr(chunk.choices[0].delta, "content", None)
        if delta:
            yield delta

def chat_complete_with_thinking_ds(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    Call DeepSeek thinking mode.
    Returns:
      {
        "reasoning_content": "<chain-of-thought, may be None>",
        "content": "<final answer, may be None>"
      }
    """
    # Resolve configuration from CLI/env/defaults and create a client
    client = get_client(api_key=api_key, base_url=base_url)
    _, _, model_resolved = _resolve_config(api_key=api_key, base_url=base_url, model=model)

    resp = client.chat.completions.create(
        model=model_resolved,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"thinking": {"type": "enabled"}},
    )

    msg = resp.choices[0].message
    reasoning_content = getattr(msg, "reasoning_content", None)
    content = getattr(msg, "content", None)
    return {
        "reasoning_content": (reasoning_content or "").strip() if reasoning_content else None,
        "content": (content or "").strip() if content else None,
    }

def ask(prompt: str, system: Optional[str] = None, **kwargs) -> str:
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat_complete(messages, **kwargs)

# Optional quick test:
if __name__ == "__main__":
    try:
        print(ask("Say hello in one short sentence."))
    except Exception as e:
        print(f"API call failed: {e}")
