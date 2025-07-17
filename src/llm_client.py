import asyncio
import logging
import os
from typing import Any, Dict

from aiolimiter import AsyncLimiter
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from src.config_manager import load_config


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_llm(provider: str, model: str, api_key: str = None) -> BaseLLM:
    """returns a language model instance based on the provider."""
    if provider == "openai":
        return ChatOpenAI(model_name=model, api_key=api_key, temperature=0)
    if provider == "gemini":
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0)
    if provider == "anthropic":
        return ChatAnthropic(model=model, api_key=api_key, temperature=0)
    if provider == "ollama":
        return ChatOllama(model=model)
    raise ValueError(f"unsupported llm provider: {provider}")


def get_api_key_for_provider(provider: str) -> str | None:
    """Gets the API key for a given provider from environment variables."""
    key_map = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    return os.environ.get(key_map.get(provider))


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
async def process(
    payload: str,
    llm_chain: LLMChain,
    rate_limiter: AsyncLimiter,
) -> str:
    """processes log file (or chunk) with the llm."""
    async with rate_limiter:
        try:
            logging.info("processing chunk with llm.")
            response = await llm_chain.arun(payload)
            logging.info("successfully processed chunk.")
            return response.strip()
        except Exception as e:
            logging.error("llm processing failed: %s", e, exc_info=True)
            raise


async def summarize(
    db_writer_queue: asyncio.Queue,
    semaphore: asyncio.Semaphore,
    sample_id: int,
    log_content: str,
    provider: str,
    model_config: Dict[str, Any],
) -> None:
    """summarizes a log file using the specified llm."""
    model_name = model_config["name"]
    api_key = get_api_key_for_provider(provider)

    async with semaphore:
        try:
            llm = get_llm(
                provider, model_name, api_key
            )
            template = """
            summarize the following ansible log content.
            focus on the key actions, errors, and outcomes.
            be concise and clear.
            log content:
            "{log_content}"
            summary:
            """
            prompt = PromptTemplate(template=template, input_variables=["log_content"])
            llm_chain = LLMChain(prompt=prompt, llm=llm)

            summary = await process(
                log_content, llm_chain, AsyncLimiter(10, 60) # Placeholder limiter
            )

            await db_writer_queue.put({
                "action": "save_model_response",
                "payload": {
                    "log_sample_id": sample_id,
                    "summary": summary,
                    "model": f"{provider}:{model_name}",
                },
            })
            logging.info(
                "summary for sample_id %d with model %s completed.", sample_id, model_name
            )
        except (ValueError, Exception) as e:
            logging.error(
                "error during summarization for sample_id %d with model %s: %s",
                sample_id,
                model_name,
                e,
                exc_info=True,
            )
            await db_writer_queue.put({
                "action": "log_error",
                "payload": {
                    "sample_id": sample_id,
                    "error_message": f"error using {model_name}: {e}",
                },
            })
