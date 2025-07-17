import logging
import os
import uuid
from typing import Any, Dict, List

import aiosqlite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


async def seed(db_path: str, config: Dict[str, Any]) -> None:
    """initializes the database and creates tables."""
    if config["database"]["clear_on_startup"]:
        logging.info("clearing database on startup.")
        try:
            os.remove(db_path)
        except FileNotFoundError:
            logging.info("database file not found, creating a new one.")

    async with aiosqlite.connect(db_path) as conn:
        await conn.execute("PRAGMA journal_mode=WAL;")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS log_samples (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                parent_log_id TEXT,
                chunk_index INTEGER,
                content TEXT NOT NULL
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_responses (
                id TEXT PRIMARY KEY,
                log_sample_id TEXT NOT NULL,
                model TEXT NOT NULL,
                summary TEXT,
                FOREIGN KEY (log_sample_id) REFERENCES log_samples(id)
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                sample_id INTEGER,
                error_message TEXT,
                FOREIGN KEY(sample_id) REFERENCES log_samples(id)
            )
            """
        )
        await conn.commit()
    logging.info("database initialized successfully.")


async def get_samples(db_path: str) -> List[Dict[str, Any]]:
    """fetches all log samples from the database."""
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.cursor() as cursor:
            await cursor.execute("SELECT id, content FROM log_samples")
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def save_sample(
    db_path: str,
    sample_id: str,
    filename: str,
    parent_log_id: str,
    chunk_index: int,
    content: str,
) -> None:
    """saves a new log sample to the database."""
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            "INSERT INTO log_samples (id, filename, parent_log_id, chunk_index, content) VALUES (?, ?, ?, ?, ?)",
            (sample_id, filename, parent_log_id, chunk_index, content),
        )
        await conn.commit()


async def save_response(
    db_path: str, log_sample_id: str, model: str, summary: str
) -> None:
    """saves a model's summary for a given log sample."""
    response_id = str(uuid.uuid4())
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            "INSERT INTO model_responses (id, log_sample_id, model, summary) VALUES (?, ?, ?, ?)",
            (response_id, log_sample_id, model, summary),
        )
        await conn.commit()


async def save_error(db_path: str, sample_id: str, error_message: str) -> None:
    """saves an error message for a given sample."""
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            "INSERT INTO errors (sample_id, error_message) VALUES (?, ?)",
            (sample_id, error_message),
        )
        await conn.commit() 
