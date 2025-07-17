import asyncio
import json
import logging
import os
import time
from typing import Any, Dict

from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from tenacity import (
    before_sleep_log,
    retry,
    RetryError,
    stop_after_attempt,
    wait_exponential,
)

from src.config_manager import load_config


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_llm(provider: str, model: str, api_key: str = None, **kwargs) -> BaseLLM | None:
    """returns a language model instance based on the provider."""
    if provider == "openai":
        return ChatOpenAI(model_name=model, api_key=api_key, **kwargs)
    if provider == "gemini":
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, **kwargs)
    if provider == "anthropic":
        return ChatAnthropic(model=model, api_key=api_key, **kwargs)
    if provider == "ollama":
        return ChatOllama(model=model, **kwargs)
    raise ValueError(f"unsupported llm provider: {provider}")


def get_api_key(provider: str) -> str | None:
    """Gets the API key for a given provider from environment variables."""
    key_map = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    return os.environ.get(key_map.get(provider))


async def summarize(
    db_writer_queue: asyncio.Queue,
    semaphore: asyncio.Semaphore,
    sample_id: int,
    log_content: str,
    provider: str,
    model_config: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """
    Summarizes a log file using a specified LLM, with retries and performance tracking.
    """
    model_name = model_config["name"]
    api_key = get_api_key(provider)
    if not api_key and provider in ["openai", "gemini", "anthropic"]:
        logging.warning(f"api key for {provider} not found. skipping.")
        return
    prompt_template_str = config["prompt_template"]
    llm_parameters = model_config.get("parameters", {})
    retry_settings = config["app_settings"]["retry_settings"]

    prompt = PromptTemplate(template=prompt_template_str, input_variables=["log_content"])
    llm = get_llm(provider, model_name, api_key, **llm_parameters)
    if not llm:
        return

    llm_chain = prompt | llm

    summary, success, error_message = "", False, None
    input_tokens, output_tokens, latency_ms = 0, 0, 0.0
    retry_attempts = 0

    @retry(
        stop=stop_after_attempt(retry_settings["max_attempts"]),
        wait=wait_exponential(
            multiplier=retry_settings["initial_backoff_seconds"],
            max=retry_settings["max_backoff_seconds"],
        ),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.INFO),
        reraise=True,
    )
    async def _generate_summary():
        return await llm_chain.ainvoke({"log_content": log_content})

    start_time = time.monotonic()
    try:
        async with semaphore:
            result = await _generate_summary()
        latency_ms = (time.monotonic() - start_time) * 1000
        summary = result.content.strip()

        token_usage = result.usage_metadata

        input_tokens = token_usage.get("input_tokens", 0)
        output_tokens = token_usage.get("output_tokens", 0)
        success = True
        retry_attempts = _generate_summary.retry.statistics.get("attempt_number", 1) - 1

    except RetryError as e:
        latency_ms = (time.monotonic() - start_time) * 1000
        error_message = f"LLM call failed after multiple retries: {str(e.last_attempt.exception())}"
        retry_attempts = e.last_attempt.retry.statistics.get("attempt_number", 1) - 1
        logging.error("Error for sample %s with model %s: %s", sample_id, model_name, error_message)
    except Exception as e:
        latency_ms = (time.monotonic() - start_time) * 1000
        error_message = f"An unexpected error occurred: {str(e)}"
        retry_attempts = 0
        logging.error("Unexpected error for sample %s with model %s: %s", sample_id, model_name, error_message, exc_info=True)

    await db_writer_queue.put({
        "action": "save_model_response",
        "payload": {
            "log_sample_id": sample_id,
            "model": f"{provider}:{model_name}",
            "summary": summary,
            "prompt_used": prompt.format(log_content=log_content),
            "parameters_used": json.dumps(llm_parameters),
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "success": success,
            "error_message": error_message,
            "retry_attempts": retry_attempts,
        },
    })
