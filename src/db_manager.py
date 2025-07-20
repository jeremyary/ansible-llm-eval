import logging
import os
import time
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
                run_id TEXT,
                summary TEXT,
                prompt_used TEXT,
                parameters_used TEXT,
                latency_ms REAL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                success INTEGER NOT NULL,
                error_message TEXT,
                retry_attempts INTEGER NOT NULL,
                timestamp REAL NOT NULL,
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


async def get_samples(db_path: str, num_models: int) -> List[Dict[str, Any]]:
    """fetches log samples from the database that have not been fully processed."""
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                SELECT ls.id, ls.content
                FROM log_samples ls
                LEFT JOIN model_responses mr ON ls.id = mr.log_sample_id
                GROUP BY ls.id
                HAVING COUNT(mr.id) < ?
                """,
                (num_models,),
            )
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
    db_path: str,
    log_sample_id: str,
    model: str,
    run_id: str,
    summary: str,
    prompt_used: str,
    parameters_used: str,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    success: bool,
    error_message: str,
    retry_attempts: int,
) -> None:
    """saves a model's summary and performance metrics for a given log sample."""
    response_id = str(uuid.uuid4())
    timestamp = time.time()
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            """INSERT INTO model_responses (
                id, log_sample_id, model, run_id, summary, prompt_used, parameters_used,
                latency_ms, input_tokens, output_tokens, success, error_message,
                retry_attempts, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                response_id,
                log_sample_id,
                model,
                run_id,
                summary,
                prompt_used,
                parameters_used,
                latency_ms,
                input_tokens,
                output_tokens,
                int(success),
                error_message,
                retry_attempts,
                timestamp,
            ),
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
