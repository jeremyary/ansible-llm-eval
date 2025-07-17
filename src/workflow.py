"""workflow orchestration module for the llm evaluation pipeline."""
import asyncio
import logging
from typing import Any, Dict, TypedDict

from langgraph.graph import END, StateGraph

from src.config_manager import load_config
from src.db_manager import get_samples
from src.ingestion import process_files
from src.llm_client import summarize

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
    logging.info("starting data ingestion.")
    await process_files(
        state["config"]["data_ingestion"],
        state["db_writer_queue"],
    )
    logging.info("data ingestion complete.")
    return state


async def summarize_logs_node(state: WorkflowState) -> WorkflowState:
    """summarizes the ingested logs using the llm."""
    logging.info("starting log summarization.")
    config = state["config"]
    pending_summaries = await get_samples(state["db_path"])
    semaphore = asyncio.Semaphore(config["llm"]["concurrent_requests"])
    rate_limit = config["llm"]["rate_limit_per_minute"]
    models_to_run = config["llm"]["models"]

    tasks = []
    for row in pending_summaries:
        for provider, model_config in models_to_run.items():
            tasks.append(
                summarize(
                    state["db_writer_queue"],
                    semaphore,
                    row["id"],
                    row["content"],
                    provider,
                    model_config,
                    state["config"],
                )
            )

    if tasks:
        await asyncio.gather(*tasks)
        logging.info("log summarization batch complete.")
    else:
        logging.info("no pending summaries to process.")

    return state


async def should_continue_node(state: WorkflowState) -> str:
    """determines whether the workflow should continue."""
    pending_summaries = await get_samples(state["db_path"])
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


async def run_workflow() -> None:
    """initializes and runs the workflow."""
    config = load_config()
    db_path = config["database"]["path"]

    initial_state: WorkflowState = {
        "config": config,
        "db_path": db_path,
        "db_writer_queue": asyncio.Queue(),
        "stop_event": asyncio.Event(),
    }

    workflow_graph = create_workflow()
    runnable = workflow_graph.compile()
    await runnable.ainvoke(initial_state)

    # ensure the db writer queue is empty before exiting
    await initial_state["db_writer_queue"].join()
    initial_state["stop_event"].set()


if __name__ == "__main__":
    asyncio.run(run_workflow())
