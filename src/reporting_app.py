"""streamlit reporting app for visualizing llm evaluation results."""
import sqlite3
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from src.config_manager import load_config


def get_db_connection() -> sqlite3.Connection:
    """establishes a connection to the sqlite database."""
    config = load_config()
    db_path = config["database"]["path"]
    return sqlite3.connect(db_path)


def load_data(conn: sqlite3.Connection) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """loads and joins data from log_samples and model_responses."""
    query = """
    SELECT
        ls.id as sample_id,
        ls.filename,
        ls.parent_log_id,
        ls.chunk_index,
        ls.content as original_content,
        mr.model,
        mr.summary
    FROM
        log_samples ls
    LEFT JOIN
        model_responses mr ON ls.id = mr.log_sample_id
    """
    results_df = pd.read_sql_query(query, conn)
    errors_df = pd.read_sql_query("SELECT * FROM errors", conn)
    return results_df, errors_df


def calculate_statistics(results_df: pd.DataFrame, errors_df: pd.DataFrame) -> Dict[str, Any]:
    """calculates summary statistics from the dataframes."""
    total_samples = results_df["sample_id"].nunique()
    successful_summaries = results_df["summary"].notna().sum()
    failed_summaries = len(errors_df)
    total_attempts = len(results_df)
    success_rate = (successful_summaries / total_attempts) * 100 if total_attempts > 0 else 0
    return {
        "total_samples": total_samples,
        "successful_summaries": successful_summaries,
        "failed_summaries": failed_summaries,
        "success_rate": success_rate,
    }


def display_sidebar(
    results_df: pd.DataFrame
) -> Tuple[List[str], str]:
    """configures and displays the streamlit sidebar for filtering."""
    st.sidebar.title("filters")
    models = results_df["model"].dropna().unique().tolist()
    selected_models = st.sidebar.multiselect(
        "filter by model:", options=models, default=models
    )

    statuses = ["all", "successful", "failed"]
    selected_status = st.sidebar.selectbox(
        "filter by status:", options=statuses, default="all"
    )

    return selected_models, selected_status


def filter_data(
    df: pd.DataFrame, selected_models: List[str], selected_status: str
) -> pd.DataFrame:
    """filters the dataframe based on user selections."""
    filtered_df = df[df["model"].isin(selected_models)]
    if selected_status == "successful":
        return filtered_df[filtered_df["summary"].notna()]
    if selected_status == "failed":
        return filtered_df[filtered_df["summary"].isna()]
    return filtered_df


def run_app() -> None:
    """runs the streamlit reporting application."""
    st.set_page_config(page_title="llm evaluation report", layout="wide")
    st.title("ansible log summarization evaluation report")

    conn = get_db_connection()
    results_df, errors_df = load_data(conn)
    conn.close()

    if results_df.empty:
        st.warning("no evaluation results found in the database.")
        return

    stats = calculate_statistics(results_df, errors_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("total samples", stats["total_samples"])
    col2.metric("successful summaries", stats["successful_summaries"])
    col3.metric("failed summaries", stats["failed_summaries"])
    col4.metric("success rate", f"{stats['success_rate']:.2f}%")

    selected_models, selected_status = display_sidebar(results_df)
    filtered_df = filter_data(results_df, selected_models, selected_status)

    st.subheader("evaluation results")
    # Display a summary table with one row per original log file
    summary_df = (
        filtered_df.groupby("parent_log_id")
        .agg(
            filename=("filename", "first"),
            total_chunks=("chunk_index", "nunique"),
            successful_summaries=("summary", "count"),
            models=("model", lambda x: ", ".join(x.dropna().unique())),
        )
    )
    st.dataframe(summary_df)

    if not errors_df.empty:
        st.subheader("error logs")
        st.dataframe(errors_df)

    if not filtered_df.empty:
        st.subheader("detailed view")
        # Create a readable identifier for the select box
        unique_files = (
            filtered_df[["parent_log_id", "filename"]]
            .drop_duplicates()
        )
        selected_parent_id = st.selectbox(
            "select a log file to view details:",
            options=unique_files["parent_log_id"],
            format_func=lambda x: f"{unique_files[unique_files['parent_log_id'] == x]['filename'].iloc[0]}",
        )

        if selected_parent_id:
            # Get all chunks for the selected file
            sample_chunks = filtered_df[
                filtered_df["parent_log_id"] == selected_parent_id
            ].sort_values(by="chunk_index")

            for _, chunk_row in sample_chunks.drop_duplicates(subset=['sample_id']).iterrows():
                st.markdown(f"---")
                st.markdown(f"#### Chunk {chunk_row['chunk_index']}")
                st.text_area(
                    "original log content",
                    chunk_row["original_content"],
                    height=200,
                    key=f"content_{chunk_row['sample_id']}",
                )

                # Get all model responses for this specific chunk
                model_responses = sample_chunks[sample_chunks['sample_id'] == chunk_row['sample_id']]
                
                for _, response_row in model_responses.iterrows():
                    with st.container():
                        st.markdown(f"**model:** `{response_row['model']}`")
                        st.text_area(
                            "generated summary",
                            response_row["summary"] or "no summary generated.",
                            height=100,
                            key=f"summary_{response_row['sample_id']}_{response_row['model']}",
                        )


if __name__ == "__main__":
    run_app()