import asyncio
import logging

from src.db_manager import (
    seed,
    save_error,
    save_sample,
    save_response,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


async def writer_job(
    queue: asyncio.Queue,
    db_path: str,
    stop_event: asyncio.Event
) -> None:
    """asynchronous job to write data to the database from a queue."""
    while not stop_event.is_set() or not queue.empty():
        try:
            item = await asyncio.wait_for(queue.get(), timeout=1.0)
            if item is None:
                continue

            action = item.get("action")
            payload = item.get("payload")

            if action == "initialize_db":
                await seed(db_path, payload)
            elif action == "save_sample":
                await save_sample(db_path, **payload)
            elif action == "save_model_response":
                await save_response(db_path, **payload)
            elif action == "log_error":
                await save_error(db_path, **payload)
            else:
                logging.warning("unknown action in db writer queue: %s", action)

            queue.task_done()
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            logging.error("error processing db queue item: %s", e, exc_info=True)


async def main_test_db_writer() -> None:
    """provides a main function for testing the db writer."""
    db_path = "test_db_writer.db"
    queue = asyncio.Queue()
    stop_event = asyncio.Event()

    # start the writer job as a background task
    writer_task = asyncio.create_task(writer_job(queue, db_path, stop_event))

    # give the writer a moment to start up
    await asyncio.sleep(0.1)

    # stop the writer
    stop_event.set()
    await writer_task


if __name__ == "__main__":
    asyncio.run(main_test_db_writer())
