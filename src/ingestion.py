import asyncio
import logging
import os
import uuid
from typing import List, Dict, Any

import aiofiles
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config_manager import load_config
from src.db_manager import seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


async def read_file(file_path: str) -> str:
    """asynchronously reads the content of a log file."""
    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            return await f.read()
    except IOError as e:
        logging.error("error reading file %s: %s", file_path, e)
        return ""


def chunk(
    content: str,
    chunk_size: int,
    chunk_overlap: int
) -> List[str]:
    """splits log content into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(content)


async def process_files(
    config: Dict[str, Any], db_writer_queue: asyncio.Queue
) -> None:
    """processes all log files in a directory."""
    log_directory = config["log_directory"]
    enable_chunking = config.get("enable_chunking", True)
    chunk_size = config.get("chunk_size", 10000)
    chunk_overlap = config.get("chunk_overlap", 500)

    try:
        for filename in os.listdir(log_directory):
            if not filename.endswith(".log"):
                continue
            file_path = os.path.join(log_directory, filename)
            log_content = await read_file(file_path)
            if not log_content:
                continue

            if enable_chunking:
                chunks = chunk(
                    log_content, chunk_size, chunk_overlap
                )
            else:
                chunks = [log_content]

            parent_log_id = str(uuid.uuid4())
            for i, chunk in enumerate(chunks):
                sample_id = str(uuid.uuid4())
                await db_writer_queue.put({
                    "action": "save_sample",
                    "payload": {
                        "sample_id": sample_id,
                        "filename": file_path,
                        "parent_log_id": parent_log_id,
                        "chunk_index": i,
                        "content": chunk,
                    },
                })
    except OSError as e:
        logging.error("failed to process log files: %s", e, exc_info=True)
