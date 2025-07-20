"""streamlit reporting app for visualizing llm evaluation results."""
import json
import os
import sqlite3
from typing import Any, Dict, Tuple

import pandas as pd
import streamlit as st

from config_manager import load_config


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
            margin-top:0;
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
            border-radius: 10px;
            border: 2px solid #333;
            padding: 20px;
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

    for _, chunk_row in all_chunks.iterrows():

        st.markdown(f"</br></br>", unsafe_allow_html=True)
        stats = f"chunk {chunk_row['chunk_index'] + 1}"
        st.markdown(f"### {os.path.basename(chunk_row['filename'])} &nbsp; ({stats})", unsafe_allow_html=True)

        st.text_area(
            "original log content",
            chunk_row["original_content"],
            height=400,
            key=f"content_{chunk_row['sample_id']}",
            label_visibility="collapsed"
        )    

        model_responses = results_df[
            (results_df['sample_id'] == chunk_row['sample_id']) &
            (pd.notna(results_df['model']))
        ].sort_values(by="model")

        if not model_responses.empty:
            for _, response_row in model_responses.iterrows():
                provider, _ = response_row['model'].split(":", 1)

                # stats_html = ""
                if response_row['success']:
                    stats = (
                        f"latency: {response_row['latency_ms']:.0f}ms &nbsp;|&nbsp; "
                        f"tokens: {response_row['input_tokens']}/{response_row['output_tokens']} &nbsp;|&nbsp; "
                        f"retries: {response_row['retry_attempts']}"
                    )
                    stats_html = f'&nbsp; <span class="stats-text">[ {stats} ]</span>'

                if provider == "openai":
                    provider_html = f'<span class="openai-provider-name">{provider.upper()}</span>'
                elif provider == "anthropic":
                    provider_html = f'<span class="anthropic-provider-name">{provider.upper()}</span>'
                elif provider == "google":
                    provider_html = f'<span class="google-provider-name">{provider.upper()}</span>'
                else:
                    provider_html = f'<span class="provider-name">{provider.upper()}</span>'
                
                line_html = f"<span class='provider-title-row'>{provider_html} {stats_html}</span>"
                st.markdown(line_html, unsafe_allow_html=True)

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
        
    js_script = """
    <script>
        function styleAllSliders() {
            const sliders = document.querySelectorAll('.stSlider');

            sliders.forEach(slider => {
                const thumb = slider.querySelector('[role="slider"]');
                if (!thumb) return;

                const min = parseInt(thumb.getAttribute('aria-valuemin'), 10);
                const max = parseInt(thumb.getAttribute('aria-valuemax'), 10);
                const value = parseInt(thumb.getAttribute('aria-valuemax'), 10);

                const percentage = ((value - min) / (max - min));

                let r, g, b;
                if (percentage < 0.5) {
                    r = 255;
                    g = Math.round(510 * percentage);
                } else {
                    r = Math.round(510 - 510 * percentage);
                    g = 255;
                }
                b = 0;
                const color = `rgb(${r}, ${g}, ${b})`;

                const trackFill = slider.querySelector('div[data-baseweb="slider"] > div:nth-child(2)');
                const thumbElement = slider.querySelector('div[data-baseweb="slider"] > div:nth-child(3)');

                if (trackFill) {
                    trackFill.style.background = color;
                }
                if (thumbElement) {
                    thumbElement.style.backgroundColor = color;
                }
            });
        }

        const observer = new MutationObserver((mutations) => {
            styleAllSliders();
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        // Initial run
        styleAllSliders();
    </script>
    """
    st.markdown(js_script, unsafe_allow_html=True)

if __name__ == "__main__":
    run_app()