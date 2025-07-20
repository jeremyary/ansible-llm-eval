"""workflow orchestration module for the llm evaluation pipeline."""
import asyncio
import logging
from typing import Any, Dict, TypedDict

from langgraph.graph import END, StateGraph

from src.config_manager import load_config
from src.db_manager import get_samples
from src.ingestion import process_files
from src.llm_client import summarize
from src.rate_limiter import TokenRateLimiter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class WorkflowState(TypedDict):
    """represents the state of the workflow."""
    config: Dict[str, Any]
    db_path: str
    db_writer_queue: asyncio.Queue
    stop_event: asyncio.Event


async def ingest_data_node(state: WorkflowState) -> WorkflowState:
    """ingests log data from the specified directory."""
    config = state["config"]
    
    # In UX development mode, check if we already have data and skip ingestion if we do
    if config.get("app_settings", {}).get("ux_development_mode", False):
        # Check if we already have log samples in the database
        try:
            import aiosqlite
            async with aiosqlite.connect(state["db_path"]) as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("SELECT COUNT(*) FROM log_samples")
                    result = await cursor.fetchone()
                    sample_count = result[0] if result else 0
                    
            if sample_count > 0:
                logging.info(f"UX development mode: found {sample_count} existing log samples, skipping data ingestion.")
                return state
        except Exception as e:
            logging.warning(f"Could not check existing data: {e}. Proceeding with ingestion.")
    
    logging.info("starting data ingestion.")
    await process_files(
        state["config"]["data_ingestion"],
        state["db_writer_queue"],
    )
    logging.info("data ingestion complete, waiting for db writer.")
    await state["db_writer_queue"].join()
    logging.info("db writer finished, proceeding to summarization.")
    return state


async def summarize_logs_node(state: WorkflowState) -> WorkflowState:
    """summarizes the ingested logs using the llm."""
    logging.info("starting log summarization.")
    config = state["config"]
    
    # Check if UX development mode is enabled
    if config.get("app_settings", {}).get("ux_development_mode", False):
        logging.info("UX development mode enabled - skipping LLM calls.")
        return state
    
    num_models = len(config["llm"]["models"])
    pending_summaries = await get_samples(state["db_path"], num_models)
    logging.info(f"found {len(pending_summaries)} log samples to summarize.")

    if not pending_summaries:
        return state

    semaphore = asyncio.Semaphore(config["llm"]["concurrent_requests"])
    models_to_run = config["llm"]["models"]

    limiters = {
        provider: TokenRateLimiter(model_config.get("rate_limit_tpm", 10000))
        for provider, model_config in models_to_run.items()
    }

    logging.info(f"summarizing with models: {list(models_to_run.keys())}")

    tasks = []
    for row in pending_summaries:
        for provider, model_config in models_to_run.items():
            tasks.append(
                summarize(
                    state["db_writer_queue"],
                    semaphore,
                    limiters[provider],
                    row["id"],
                    row["content"],
                    provider,
                    model_config,
                    state["config"],
                )
            )

    logging.info(f"created {len(tasks)} summarization tasks. waiting for completion...")
    await asyncio.gather(*tasks)
    logging.info("log summarization batch complete.")
    logging.info("waiting for db writer to save summaries...")
    await state["db_writer_queue"].join()
    logging.info("db writer finished saving summaries.")

    return state


async def should_continue_node(state: WorkflowState) -> str:
    """determines whether the workflow should continue."""
    # In UX development mode, always end after ingestion
    if state["config"].get("app_settings", {}).get("ux_development_mode", False):
        logging.info("UX development mode enabled - ending workflow after data ingestion.")
        return "end"
        
    num_models = len(state["config"]["llm"]["models"])
    pending_summaries = await get_samples(state["db_path"], num_models)
    if not pending_summaries:
        logging.info("no more summaries to process. ending workflow.")
        return "end"
    logging.info("more summaries to process. continuing workflow.")
    return "continue"


def create_workflow() -> StateGraph:
    """creates and inits the langgraph workflow."""
    workflow = StateGraph(WorkflowState)

    workflow.add_node("ingest", ingest_data_node)
    workflow.add_node("summarize", summarize_logs_node)

    workflow.add_edge("ingest", "summarize")

    workflow.add_conditional_edges(
        "summarize",
        should_continue_node,
        {
            "continue": "summarize",
            "end": END,
        },
    )

    workflow.set_entry_point("ingest")
    return workflow


async def run_workflow(
    db_writer_queue: asyncio.Queue,
    stop_event: asyncio.Event,
) -> None:
    """initializes and runs the workflow."""
    config = load_config()
    db_path = config["database"]["path"]

    initial_state: WorkflowState = {
        "config": config,
        "db_path": db_path,
        "db_writer_queue": db_writer_queue,
        "stop_event": stop_event,
    }

    workflow_graph = create_workflow()
    runnable = workflow_graph.compile()
    await runnable.ainvoke(initial_state)

    # ensure the db writer queue is empty before exiting
    await db_writer_queue.join()


if __name__ == "__main__":
    # this is a test runner, requires a running db writer
    from src.db_writer import writer_job

    async def test_run() -> None:
        """provides a test runner for the workflow."""
        queue = asyncio.Queue()
        event = asyncio.Event()
        writer = asyncio.create_task(writer_job(queue, "test.db", event))
        await run_workflow(queue, event)
        event.set()
        await writer

    asyncio.run(test_run())
