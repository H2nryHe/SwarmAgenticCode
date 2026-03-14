import os
import time

from langchain_openai import ChatOpenAI


DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_MAX_RETRIES = 6
DEFAULT_INVOKE_ATTEMPTS = 3
DEFAULT_RETRY_DELAY_SECONDS = 2.0


def _get_float_env(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_int_env(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def build_chat_openai(model, temperature):
    timeout_seconds = _get_float_env("TP_OPENAI_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
    max_retries = _get_int_env("TP_OPENAI_MAX_RETRIES", DEFAULT_MAX_RETRIES)
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        request_timeout=timeout_seconds,
        max_retries=max_retries,
    )


def invoke_with_retries(chain, payload, description="LLM call"):
    attempts = max(1, _get_int_env("TP_LLM_INVOKE_ATTEMPTS", DEFAULT_INVOKE_ATTEMPTS))
    base_delay_seconds = max(
        0.0,
        _get_float_env("TP_LLM_RETRY_DELAY_SECONDS", DEFAULT_RETRY_DELAY_SECONDS),
    )
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            return chain.invoke(payload)
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay_seconds = base_delay_seconds * attempt
            print(
                f"{description} failed on attempt {attempt}/{attempts} "
                f"with {type(exc).__name__}: {exc}. Retrying in {delay_seconds:.1f}s."
            )
            time.sleep(delay_seconds)

    raise last_error
