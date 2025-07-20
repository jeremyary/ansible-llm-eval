"""main entry point for the llm evaluation application."""
import asyncio
import logging
import os
import subprocess
from dotenv import load_dotenv

from src.config_manager import load_config
from src.db_manager import seed
from src.db_writer import writer_job
from src.workflow import run_workflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


async def main() -> None:
    """main function to run the application."""
    load_dotenv()

    # If the LangSmith API key is set, configure tracing
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "ansible-llm-eval"
        logging.info("LangSmith tracing enabled for project 'ansible-llm-eval'.")

    config = load_config()
    db_path = config["database"]["path"]

    # Initialize the database based on config settings
    await seed(db_path, config)

    db_writer_queue = asyncio.Queue()
    stop_event = asyncio.Event()

    # start the database writer as a background task
    writer_task = asyncio.create_task(
        writer_job(db_writer_queue, db_path, stop_event)
    )
    logging.info("database writer started.")

    # run the main workflow
    await run_workflow(db_writer_queue, stop_event)

    # stop the database writer
    stop_event.set()
    await writer_task
    logging.info("database writer stopped.")

    # launch the reporting app
    logging.info("launching the reporting app.")

    reporting_config = config.get("reporting", {})
    host = reporting_config.get("host", "0.0.0.0")
    port = reporting_config.get("port", 1111)

    # Use localhost for the viewing URL if host is 0.0.0.0
    view_host = "localhost" if host == "0.0.0.0" else host

    command = [
        "streamlit",
        "run",
        "src/reporting_app.py",
        "--server.address", host,
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]

    with subprocess.Popen(command) as proc:
        logging.info("reporting app process started with pid: %s", proc.pid)
        logging.info(
            f"You can view the report by navigating to http://{view_host}:{port} in your browser."
        )
        proc.wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("application shutting down.") 