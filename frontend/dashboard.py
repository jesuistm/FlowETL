import streamlit as st
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
from datetime import datetime
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def initialize_session_state():
    """Initialize session state variables for caching. This bypasses the problem where 
    streamlit resets the session state when new interactions are detected."""

    if 'processed_dataframe' not in st.session_state:
        st.session_state.processed_dataframe = None

    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

    if 'dataset_name' not in st.session_state:
        st.session_state.dataset_name = None

    if 'dqr_before' not in st.session_state:
        st.session_state.dqr_before = None

    if 'dqr_after' not in st.session_state:
        st.session_state.dqr_after = None


def clear_cache():
    """Clear session data. This action is triggered only on browser reloads or new requests to /transform"""
    st.session_state.processed_dataframe = None
    st.session_state.query_history = []
    st.session_state.dataset_name = None
    st.session_state.dqr_before = None
    st.session_state.dqr_after = None

def fig_to_base64(fig):
    """convert plotly figure (from data analysis response) into base64 image so that it can be embedded in chat response"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%; border-radius:10px; margin-top:10px;">'


def render_query_response(query, response, error=False):
    """Render a query-response pair in chat-like format"""

    # user query bubble
    with st.container():
        col1, col2 = st.columns([1, 5])
        with col2:
            st.markdown(
                f"""
                <div style="background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px 0;">
                    <strong>Query</strong><br>{query}
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # data analysis response bubble
    with st.container():
        col1, col2 = st.columns([5, 1])
        with col1:
            if error:
                st.markdown(
                    f"""
                    <div style="background-color: #ffebee; padding: 10px; border-radius: 10px; margin: 5px 0;">
                        <strong>❌ Error</strong><br>Could not fulfill your request.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:

                content = ""

                # add summary if available
                if response.get('summary'):
                    content += f"<p>{response['summary']}</p>"

                # add plot if available
                if response.get('plot'):
                    content += fig_to_base64(response['plot'])

                # Render everything inside HERE
                st.markdown(
                    f"""
                    <div style="background-color: #f5f5f5; padding: 10px; border-radius: 10px; margin: 5px 0;">
                        <strong>Response</strong><br>{content}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

def main():
    """Run the frontend application"""

    # configure page settings
    st.set_page_config(page_title="FlowETL", initial_sidebar_state="collapsed")
    
    # initialize session state
    initialize_session_state()
    
    st.title("FlowETL")
    st.text("Transform and query your data using natural language")

    # user inputs: dataset uploader and task description text area
    input_dataset = st.file_uploader(label="Upload your dataset", type=["csv"], accept_multiple_files=False, key="input_dataset")
    task_description = st.text_area(label="Describe what you want FlowETL to do", key="task_description", placeholder="e.g., Fill all missing values, then handle all outliers in the 'salary' column...")

    # transform button
    col1, col2, col3 = st.columns([2, 1, 4]) 
    with col1:
        trigger_btn = st.button(
            label="Transform", 
            key="trigger_btn", 
            disabled=not (input_dataset and task_description), 
            help="Ensure both the dataset and task description are provided", 
            type="primary"
        )

    if trigger_btn:
        # Clear previous cache when new transformation is triggered
        clear_cache()
        
        # Abstract the input dataset to a pandas dataframe
        abstraction = pd.read_csv(input_dataset)
        
        # Extract the uploaded dataset name
        dataset_name = input_dataset.name
        st.session_state.dataset_name = dataset_name

        with st.spinner("Preparing your dataset..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/transform", 
                    json={
                        "dataset_name": dataset_name,
                        "abstraction": abstraction.to_json(), 
                        "task": task_description
                    }
                )

                response_body = response.json()
                if response.status_code == 200:
                    json_response_df = response_body.get('processed_abstraction', None)
                    
                    if json_response_df:
                        processed_dataframe = pd.DataFrame(json.loads(json_response_df))
                        # Cache the processed dataframe
                        st.session_state.processed_dataframe = processed_dataframe
                        
                        # Cache DQR reports
                        st.session_state.dqr_before = response_body.get('data-quality-before', None)
                        st.session_state.dqr_after = response_body.get('data-quality-after', None)
                        
                        st.success("Dataset transformed successfully!")
                else:
                    st.error("Error occurred during transformation. Please check the runtime logs.")

            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")

    # Display cached results if available
    if st.session_state.processed_dataframe is not None:
        st.divider()
        
        st.markdown("**🔍 Transformed Dataset**")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.text(f"Preview of the transformed dataset: {st.session_state.dataset_name}")
        with col2:
            # Download button for processed dataframe
            CSV = st.session_state.processed_dataframe.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download", 
                data=CSV, 
                file_name=f"output_{st.session_state.dataset_name}", 
                mime="text/csv"
            )
        
        # Render the processed dataframe
        st.dataframe(st.session_state.processed_dataframe.head())
        
        # Data Quality Reports Section
        st.divider()
        st.markdown("**📊 Data Quality Reports**")
        st.text("Data quality reports computed before and after processing your dataset, computed with respect to your task requirements.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.dqr_before:
                st.markdown("**Pre-transformation Report**")
                st.download_button(
                    label="📥 Download",
                    data=json.dumps(st.session_state.dqr_before, indent=2),
                    file_name=f"dqr_before_{st.session_state.dataset_name}.json",
                    mime="application/json",
                    key="download_dqr_before"
                )
                with st.expander("View Report", expanded=False):
                    st.json(st.session_state.dqr_before)
        
        with col2:
            if st.session_state.dqr_after:
                st.markdown("**Post-transformation Report**")
                st.download_button(
                    label="📥 Download",
                    data=json.dumps(st.session_state.dqr_after, indent=2),
                    file_name=f"dqr_after_{st.session_state.dataset_name}.json",
                    mime="application/json",
                    key="download_dqr_after"
                )
                with st.expander("View Report", expanded=False):
                    st.json(st.session_state.dqr_after)
        
        # Data Analysis Section
        st.divider()
        st.markdown("**🔍 Data Analysis**")
        st.text("Query your transformed dataset using natural language")
        
        # Query input
        query = st.text_area(
            label="Enter your analysis query", 
            key="data_analysis_query",
            placeholder="e.g., Show me the distribution of sales by region, or Calculate the average price per category"
        )
        
        col1, col2, col3 = st.columns([2, 1, 4])
        with col1:
            analyze_btn = st.button(
                label="Query", 
                key="trigger_analysis_btn", 
                disabled=not query, 
                type="primary"
            )
        
        if analyze_btn and query:
            with st.spinner("Processing your query..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/analyze", 
                        json={
                            "abstraction": st.session_state.processed_dataframe.to_json(), 
                            "task": query
                        }
                    )

                    response_body = response.json()
                    if response.status_code == 200:
                        # Prepare the response object
                        query_result = {
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'query': query,
                            'summary': response_body.get('summary', 'No summary available'),
                            'plot': None
                        }
                        
                        # Check if there is a plotting function to be executed
                        plotting_code_string = response_body.get('plot_code', None)
                        if plotting_code_string is not None:
                            try:
                                # Extract the plotting function and run it on the dataset
                                namespace = {
                                    'plt': plt, 
                                    'pd': pd, 
                                    'pandas' : pd,
                                    'numpy' : np,
                                    'np': np, 
                                    '__builtins__': __builtins__
                                }
                                exec(plotting_code_string, namespace)
                                plotting_function = namespace['plot_data']
                                
                                # Generate the plot
                                fig = plotting_function(st.session_state.processed_dataframe)
                                query_result['plot'] = fig
                            except Exception as e:
                                st.warning(f"Could not generate plot: {e}")
                        
                        # Add to query history
                        st.session_state.query_history.append(query_result)
                        
                    else:
                        # Add error to history
                        st.session_state.query_history.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'query': query,
                            'error': response_body.get('detail', 'Unknown error occurred'),
                            'is_error': True
                        })

                except requests.exceptions.RequestException as e:
                    # Add error to history
                    st.session_state.query_history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'query': query,
                        'error': str(e),
                        'is_error': True
                    })
        
        # Display query history in chat-like format
        if st.session_state.query_history:
            st.divider()
            st.markdown("### 💬 Query History")
            
            # Display queries in reverse chronological order
            for item in reversed(st.session_state.query_history):
                if item.get('is_error', False):
                    render_query_response(item['query'], item['error'], error=True)
                else:
                    render_query_response(item['query'], item)
                st.markdown("---")

if __name__ == "__main__":
    main()