"""streamlit reporting app for visualizing llm evaluation results."""
import json
import os
import sqlite3
from typing import Any, Dict, Tuple

import pandas as pd
import streamlit as st
from langsmith import Client

from config_manager import load_config


def submit_langsmith_feedback_async(run_id: str, feedback_value: bool) -> None:
    """Submits feedback to LangSmith for the given run_id."""
    if not run_id or not os.getenv("LANGSMITH_API_KEY"):
        return
    
    try:
        import threading
        
        def send_feedback():
            try:
                client = Client()
                client.create_feedback(
                    run_id=run_id,
                    key="user_rating",
                    score=1.0 if feedback_value else 0.0,
                    value="positive" if feedback_value else "negative"
                )
            except Exception as e:
                # Log error but don't show in UI since this is async
                print(f"Failed to submit LangSmith feedback: {str(e)}")
        
        # Send feedback in background thread to avoid blocking UI
        thread = threading.Thread(target=send_feedback)
        thread.daemon = True
        thread.start()
        
    except Exception as e:
        print(f"Failed to start feedback thread: {str(e)}")


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
        mr.run_id,
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


def should_collapse_chunk(results_df: pd.DataFrame, sample_id: str) -> bool:
    """Determines if a chunk should be collapsed based on all providers reporting no errors."""
    model_responses = results_df[
        (results_df['sample_id'] == sample_id) &
        (pd.notna(results_df['model'])) &
        (results_df['success'] == True)
    ]
    
    if model_responses.empty:
        return False
    
    # Get the three expected providers
    providers = set()
    summaries = {}
    for _, response_row in model_responses.iterrows():
        provider, _ = response_row['model'].split(":", 1)
        providers.add(provider)
        summaries[provider] = response_row['summary'].strip() if response_row['summary'] else ""
    
    # Check if we have all three providers
    expected_providers = {"openai", "anthropic", "gemini"}
    if not expected_providers.issubset(providers):
        return False
    
    # Check if all providers reported "No errors found."
    for provider in expected_providers:
        if summaries.get(provider, "") != "No errors found.":
            return False
    
    return True


def render_chunk_content(chunk_row, model_responses):
    """Renders the content and model responses for a chunk."""
    st.text_area(
        "original log content",
        chunk_row["original_content"],
        height=400,
        key=f"content_{chunk_row['sample_id']}",
        label_visibility="collapsed"
    )    

    if not model_responses.empty:
        for _, response_row in model_responses.iterrows():
            provider, _ = response_row['model'].split(":", 1)
            run_id = response_row.get('run_id')

            # Create columns for provider name, toggle, and stats
            col1, col2, col3 = st.columns([0.075, 0.075, 0.85])
            
            with col1:
                if provider == "openai":
                    provider_html = f'<span class="openai-provider-name">{provider.upper()}</span>'
                elif provider == "anthropic":
                    provider_html = f'<span class="anthropic-provider-name">{provider.upper()}</span>'
                elif provider == "gemini":
                    provider_html = f'<span class="google-provider-name">{provider.upper()}</span>'
                else:
                    provider_html = f'<span class="provider-name">{provider.upper()}</span>'
                st.markdown(provider_html, unsafe_allow_html=True)
            
            with col2:
                # LangSmith feedback toggle (defaults to False/off)
                if run_id and os.getenv("LANGSMITH_API_KEY"):
                    feedback_key = f"feedback_{chunk_row['sample_id']}_{provider}"
                    previous_key = f"{feedback_key}_previous"
                    
                    feedback_value = st.toggle(
                        "Accurate?", 
                        value=False, 
                        key=feedback_key
                    )
                    
                    # Only submit feedback when toggle actually changes (not on initial render)
                    if previous_key in st.session_state and st.session_state[previous_key] != feedback_value:
                        submit_langsmith_feedback_async(run_id, feedback_value)
                    
                    # Store current value for next comparison
                    st.session_state[previous_key] = feedback_value
                else:
                    st.write("")  # Empty space when LangSmith not available
            
            with col3:
                if response_row['success']:
                    stats = (
                        f"latency: {response_row['latency_ms']:.0f}ms &nbsp;|&nbsp; "
                        f"tokens: {response_row['input_tokens']}/{response_row['output_tokens']} &nbsp;|&nbsp; "
                        f"retries: {response_row['retry_attempts']}"
                    )
                    stats_html = f'<span class="stats-text">[ {stats} ]</span>'
                    st.markdown(stats_html, unsafe_allow_html=True)
                else:
                    st.write("")  # Empty space for failed responses

            summary_text = f"{response_row['summary']}" or "No summary generated."
            suggestion_marker = "[SUGGESTION]"

            if suggestion_marker in summary_text:
                parts = summary_text.split(suggestion_marker, 1)
                main_summary = parts[0].strip()
                suggestion = parts[1].strip()
                summary_html = f"""
                <div class="summary-text">
                    {main_summary}</br>
                    <div class="suggestion-text">{suggestion}</div>
                </div>
                """
            else:
                summary_html = f"""
                <div class="summary-text">
                    <p>{summary_text}</p>
                </div>
                """
            st.markdown(summary_html, unsafe_allow_html=True)

            if not response_row['success']:
                st.error(f"**Error:** {response_row['error_message']}")


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
    st.set_page_config(page_title="Ansible log evaluation report", layout="wide")
    
    st.markdown("""
        <style>
        .stVerticalBlock {
            padding-right: 6em;
            padding-left: 4em;
        }
        .summary-box {
            background-color: #262730;
            border: 1px solid #444;
            border-radius: 7px;
            padding: 10px 10px 1px 10px;
            margin-top:100px;
            margin-bottom: 7px;
        }
        .summary-box p {
            margin-bottom: 7px;
        }
        .provider-name {
            font-weight: bold;
            color: #55a8f2;
        }
        .openai-provider-name {
            font-weight: bold;
            color: #10a37f;
        }
        .anthropic-provider-name {
            font-weight: bold;
            color: #d97757;
        }
        .google-provider-name {
            font-weight: bold;
            color: #356cf1;
        }
        .stats-text {
            font-size: 0.9em;
            color: #656565;
            font-style: normal;
        }
        .stButton button {
            padding: 0.1em 0.5em;
            font-size: 0.9em;
            line-height: 1.2;
            height: auto;
        }
        .stTextArea textarea {
            color: dimgray;
            padding: 2em;
            background-color: #1a1c24;
        }
        .stTextArea {
            margin-bottom: 1em;
            margin-left: 0.25em;
        }
        .suggestion-text {
            font-weight: bold;
            color: seagreen;
        }
        .stMainBlockContainer {
            padding-top: 3rem;
        }
        .summary-text {
            margin-left: 1em;
            margin-right: 1em;
            margin-bottom: 1em;
            margin-top: 0;
            border-radius: 10px;
            border: 2px solid #333;
            padding: 20px;
        }
        .collapsed-summary {
            font-size: 0.9em;
            color: #888;
            margin-left: 1em;
            font-style: italic;
        }
        .stExpander {
            border: 1px solid #444;
            border-radius: 5px;
            background-color: #1a1c24;
        }
        .stExpander > div > div > div > div {
            background-color: #1a1c24;
        }
        .stCheckbox {
            margin-top: 0 !important;
        }
        .stCheckbox > label:has(input[type="checkbox"]:not(:checked)) > div:first-child {
            background-color: #dc3545 !important; /* Red when OFF */
        }
        .stCheckbox > label:has(input[type="checkbox"]:checked) > div:first-child {
            background-color: #28a745 !important; /* Green when ON */
        }
        .stColumn {
            padding: 0 0 0 0;
        }
        .stVerticalBlock {
            padding-right: 0;
            padding-left: 0;
        }
        .stMarkdown > div > p {
            margin-bottom: 0;
        }
        .stHorizontalBlock {
            padding-bottom: 0;
            margin-bottom: 0;
        }
        </style>
    """, unsafe_allow_html=True)

    config = load_config()
    conn = get_db_connection()
    results_df, errors_df = load_data(conn)
    conn.close()

    if results_df.empty:
        st.warning("no evaluation results found in the database.")
        return

    total_latency_ms = results_df['latency_ms'].sum()
    total_time_seconds = total_latency_ms / 1000

    # format the time
    hours, rem = divmod(total_time_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    
    time_str = ""
    if hours:
        time_str += f"{int(hours)}h "
    if minutes:
        time_str += f"{int(minutes)}m "
    
    time_str += f"{seconds:.0f}s"
    
    st.markdown(f"### Performance Summary &nbsp; <span class='stats-text'>(total time: {time_str})</span>", unsafe_allow_html=True)

    stats_df = calculate_statistics(results_df, config)
    st.dataframe(
        stats_df.style.format({
            "Success Rate (%)": "{:.0f}",
            "Avg Latency (ms)": "{:.0f}",
            "Total Cost (USD)": "${:.4f}",
            "Avg Retries": "{:.0f}",
        })
    )
    
    # Sort all chunks by filename and chunk index for a flat, scrollable list
    all_chunks = results_df.sort_values(
        by=["filename", "chunk_index"]
    ).drop_duplicates(subset=['sample_id'])

    # Count collapsed vs expanded chunks
    collapsed_count = 0
    total_count = len(all_chunks)

    for _, chunk_row in all_chunks.iterrows():
        sample_id = chunk_row['sample_id']
        should_collapse = should_collapse_chunk(results_df, sample_id)
        
        if should_collapse:
            collapsed_count += 1

        st.markdown(f"</br></br>", unsafe_allow_html=True)
        stats = f"chunk {chunk_row['chunk_index'] + 1}"
        
        model_responses = results_df[
            (results_df['sample_id'] == sample_id) &
            (pd.notna(results_df['model']))
        ].sort_values(by="model")

        if should_collapse:
            # Collapsed view - use expander
            header = f"✅ {os.path.basename(chunk_row['filename'])} ({stats}) - No errors found"
            with st.expander(header, expanded=False):
                st.markdown('<div class="collapsed-summary">All providers reported no errors in this chunk. Expand to view details.</div>', unsafe_allow_html=True)
                render_chunk_content(chunk_row, model_responses)
        else:
            # Normal expanded view
            st.markdown(f"### {os.path.basename(chunk_row['filename'])} &nbsp; ({stats})", unsafe_allow_html=True)
            render_chunk_content(chunk_row, model_responses)

    # Show summary of collapsed chunks
    if collapsed_count > 0:
        st.markdown("</br>", unsafe_allow_html=True)
        st.info(f"📋 **Summary**: {collapsed_count} out of {total_count} chunks had no errors reported by all providers and are shown collapsed. Click to expand any collapsed entry to view details.")
          
if __name__ == "__main__":
    run_app()