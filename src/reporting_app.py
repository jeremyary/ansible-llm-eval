"""streamlit reporting app for visualizing llm evaluation results."""
import json
import sqlite3
from typing import Any, Dict, Tuple

import pandas as pd
import streamlit as st

from src.config_manager import load_config


def get_db_connection() -> sqlite3.Connection:
    """establishes a connection to the sqlite database."""
    config = load_config()
    db_path = config["database"]["path"]
    return sqlite3.connect(db_path)


def load_data(conn: sqlite3.Connection) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """loads and joins data from the database."""
    query = """
    SELECT
        ls.id as sample_id,
        ls.filename,
        ls.parent_log_id,
        ls.chunk_index,
        ls.content as original_content,
        mr.model,
        mr.summary,
        mr.latency_ms,
        mr.input_tokens,
        mr.output_tokens,
        mr.success,
        mr.retry_attempts,
        mr.error_message
    FROM
        log_samples ls
    LEFT JOIN
        model_responses mr ON ls.id = mr.log_sample_id
    """
    results_df = pd.read_sql_query(query, conn)
    errors_df = pd.read_sql_query("SELECT * FROM errors", conn)
    return results_df, errors_df


def calculate_statistics(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Calculates detailed summary statistics for each model."""
    if df.empty or 'model' not in df.columns or df['model'].dropna().empty:
        return pd.DataFrame()

    stats = []
    for model_name, group in df.dropna(subset=['model']).groupby('model'):
        total_attempts = len(group)
        successful_summaries = group['success'].sum()
        provider, _ = model_name.split(":", 1)
        pricing_info = config.get("llm", {}).get("models", {}).get(provider, {}).get("pricing", {})
        
        input_price = pricing_info.get("input_per_k_tokens", 0)
        output_price = pricing_info.get("output_per_k_tokens", 0)

        total_input_tokens = group['input_tokens'].sum()
        total_output_tokens = group['output_tokens'].sum()
        
        cost = ((total_input_tokens / 1000) * input_price) + \
               ((total_output_tokens / 1000) * output_price)

        stats.append({
            "Model": model_name,
            "Success Rate (%)": (successful_summaries / total_attempts) * 100 if total_attempts > 0 else 0,
            "Avg Latency (ms)": group['latency_ms'].mean(),
            "Total Cost (USD)": cost,
            "Avg Retries": group['retry_attempts'].mean(),
            "Total Input Tokens": total_input_tokens,
            "Total Output Tokens": total_output_tokens,
        })
    
    return pd.DataFrame(stats).set_index("Model")


def run_app() -> None:
    """runs the streamlit reporting application."""
    st.set_page_config(page_title="llm evaluation report", layout="wide")
    st.title("ansible log summarization evaluation report")

    config = load_config()
    conn = get_db_connection()
    results_df, errors_df = load_data(conn)
    conn.close()

    if results_df.empty:
        st.warning("no evaluation results found in the database.")
        return

    st.subheader("Model Performance Overview")
    stats_df = calculate_statistics(results_df, config)
    st.dataframe(
        stats_df.style.format({
            "Success Rate (%)": "{:.2f}",
            "Avg Latency (ms)": "{:.0f}",
            "Total Cost (USD)": "${:.4f}",
            "Avg Retries": "{:.2f}",
        })
    )

    st.subheader("Detailed View")
    unique_files = results_df[["parent_log_id", "filename"]].drop_duplicates()
    selected_parent_id = st.selectbox(
        "select a log file to view details:",
        options=unique_files["parent_log_id"],
        format_func=lambda x: f"{unique_files[unique_files['parent_log_id'] == x]['filename'].iloc[0]}",
    )

    if selected_parent_id:
        sample_chunks = results_df[
            results_df["parent_log_id"] == selected_parent_id
        ].sort_values(by="chunk_index")

        for _, chunk_row in sample_chunks.drop_duplicates(subset=['sample_id']).iterrows():
            st.markdown(f"---")
            st.markdown(f"#### Chunk {chunk_row['chunk_index']} (ID: `{chunk_row['sample_id'][:8]}`)")
            st.text_area(
                "original log content",
                chunk_row["original_content"],
                height=200,
                key=f"content_{chunk_row['sample_id']}",
            )

            model_responses = sample_chunks[sample_chunks['sample_id'] == chunk_row['sample_id']]
            
            for _, response_row in model_responses.iterrows():
                if pd.notna(response_row['model']):
                    with st.expander(f"**Model:** `{response_row['model']}` - {'Success' if response_row['success'] else 'Failed'}"):
                        st.text_area(
                            "Generated Summary",
                            response_row["summary"] or "No summary generated.",
                            height=100,
                            key=f"summary_{response_row['sample_id']}_{response_row['model']}",
                        )
                        if not response_row['success']:
                            st.error(f"**Error:** {response_row['error_message']}")

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Latency (ms)", f"{response_row['latency_ms']:.0f}")
                        col2.metric("Input Tokens", response_row['input_tokens'])
                        col3.metric("Output Tokens", response_row['output_tokens'])
                        col4.metric("Retries", response_row['retry_attempts'])


if __name__ == "__main__":
    run_app()