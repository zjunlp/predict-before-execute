"""Backend for OpenAI API."""

import json
import logging
import time
from funcy import notnone, once, select_values
import openai

from .utils import OutputType, opt_messages_to_list, backoff_create, FuncSpec

logger = logging.getLogger("aide")

_client: openai.OpenAI = None  # type: ignore

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@once
def _setup_openai_client():
    global _client
    _client = openai.OpenAI(max_retries=0)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FuncSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    Query the OpenAI API, optionally with function calling.
    If the model doesn't support function calling, gracefully degrade to text generation.
    
    Special handling for DeepSeek Thinking models (deepseek-reasoner, deepseek-v3.2-thinking):
    - Uses thinking-specific API endpoint with extra_body parameter
    - Returns reasoning_content + content concatenated (for text-only queries)
    - For function calling: extracts JSON from content and parses it
    """
    _setup_openai_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)
    
    if "max_tokens" not in filtered_kwargs:
        filtered_kwargs["max_tokens"] = 8192

    # Convert system/user messages to the format required by the client
    messages = opt_messages_to_list(system_message, user_message)

    # Check if this is a DeepSeek Thinking model
    model_name = filtered_kwargs.get("model", "").lower()
    is_deepseek_thinking = model_name in ("deepseek-reasoner", "deepseek-v3.2-thinking")

    # DeepSeek Thinking models don't support OpenAI-style function calling
    # We need to handle it differently: request JSON output and parse manually
    if is_deepseek_thinking:
        t0 = time.time()
        logger.info(f"Detected DeepSeek Thinking model: {model_name}, using manual JSON parsing mode")
        
        # If function calling is requested, append the schema to the prompt
        if func_spec is not None:
            # Add function spec to the user message as JSON schema
            schema_prompt = (
                f"\n\nYou must respond with a JSON object matching this schema:\n"
                f"```json\n{json.dumps(func_spec.json_schema, indent=2)}\n```\n"
                f"Function description: {func_spec.description}\n"
                f"End your response with a single valid JSON object that matches the schema exactly."
            )
            # Append to the last user message
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += schema_prompt
            else:
                messages.append({"role": "user", "content": schema_prompt})
        
        # Add thinking parameter
        filtered_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        
        try:
            completion = backoff_create(
                _client.chat.completions.create,
                OPENAI_TIMEOUT_EXCEPTIONS,
                messages=messages,
                **filtered_kwargs,
            )
        except Exception as e:
            logger.error(f"DeepSeek Thinking API call failed: {e}")
            raise
        
        req_time = time.time() - t0  # Will be calculated below
        choice = completion.choices[0]
        
        # Extract reasoning and content
        reasoning_content = getattr(choice.message, "reasoning_content", None)
        content = getattr(choice.message, "content", None)
        
        reasoning = (reasoning_content or "").strip()
        answer = (content or "").strip()
        
        # If function calling is requested, try to parse JSON from content
        if func_spec is not None:
            # Try to extract JSON from the answer
            try:
                # Look for JSON object in the content
                json_start = answer.rfind("{")
                json_end = answer.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    json_str = answer[json_start:json_end]
                    output = json.loads(json_str)
                    logger.info(f"Successfully extracted function call JSON from DeepSeek Thinking response")
                else:
                    # Fallback: try to parse the entire answer as JSON
                    output = json.loads(answer)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse JSON from DeepSeek Thinking response: {e}")
                logger.error(f"Response content: {answer[:500]}...")
                # Fallback: return the raw text (will cause downstream error, but more informative)
                output = answer
        else:
            # Text-only query: combine reasoning and answer
            if reasoning and answer:
                output = f"{reasoning}\n\n{answer}"
            elif reasoning:
                output = reasoning
            else:
                output = answer
        
        logger.info(f"DeepSeek Thinking response parsed (reasoning: {len(reasoning)} chars, answer: {len(answer)} chars)")
        
    else:
        t0 = time.time()
        # Standard OpenAI models: use native function calling
        if func_spec is not None:
            filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
            filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

        try:
            completion = backoff_create(
                _client.chat.completions.create,
                OPENAI_TIMEOUT_EXCEPTIONS,
                messages=messages,
                **filtered_kwargs,
            )
        except openai.BadRequestError as e:
            # Check whether the error indicates that function calling is not supported
            if "function calling" in str(e).lower() or "tools" in str(e).lower():
                logger.warning(
                    "Function calling was attempted but is not supported by this model. "
                    "Falling back to plain text generation."
                )
                # Remove function-calling parameters and retry
                filtered_kwargs.pop("tools", None)
                filtered_kwargs.pop("tool_choice", None)

                # Retry without function calling
                completion = backoff_create(
                    _client.chat.completions.create,
                    OPENAI_TIMEOUT_EXCEPTIONS,
                    messages=messages,
                    **filtered_kwargs,
                )
            else:
                # If it's some other error, re-raise
                raise
        
        req_time = time.time() - t0  # Will be calculated below
        choice = completion.choices[0]
        
        # Parse standard OpenAI response
        if func_spec is None or "tools" not in filtered_kwargs:
            # No function calling was ultimately used
            output = choice.message.content
        else:
            # Attempt to extract tool calls
            tool_calls = getattr(choice.message, "tool_calls", None)
            if not tool_calls:
                logger.warning(
                    "No function call was used despite function spec. Fallback to text.\n"
                    f"Message content: {choice.message.content}"
                )
                output = choice.message.content
            else:
                first_call = tool_calls[0]
                # Optional: verify that the function name matches
                if first_call.function.name != func_spec.name:
                    logger.warning(
                        f"Function name mismatch: expected {func_spec.name}, "
                        f"got {first_call.function.name}. Fallback to text."
                    )
                    output = choice.message.content
                else:
                    try:
                        output = json.loads(first_call.function.arguments)
                    except json.JSONDecodeError as ex:
                        logger.error(
                            "Error decoding function arguments:\n"
                            f"{first_call.function.arguments}"
                        )
                        raise ex

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info