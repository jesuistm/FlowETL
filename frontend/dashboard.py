import streamlit as st
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    """Run the frontend application"""

    # configure page settings
    st.set_page_config(page_title="FlowETL", page_icon="🔁", initial_sidebar_state=None)
        
    st.title("FlowETL")
    st.text("Transform and query your data using natural language")


    data_engineer, data_analyst = st.tabs(["Prepare your dataset", "Analyse your dataset"])

    # dataset preparation tab
    with data_engineer:
        # user inputs : dataset uploader and task description text area
        input_dataset = st.file_uploader(label="Upload your dataset", type=["csv"], accept_multiple_files=False, key="input_dataset")
        task_description = st.text_area(label="Describe what you want FlowETL to do", key="task_description")

        # trigger transformation only if both user inputs are provided
        trigger_btn = st.button(
            label="Transform", 
            key="trigger_btn", 
            disabled=not (input_dataset and task_description), 
            help="Ensure both the dataset and task description are provided", 
            type="primary"
        )

        if trigger_btn:

            # abstract the input dataset to a pandas dataframe
            abstraction = pd.read_csv(input_dataset)

            # extract the uploaded dataset name
            dataset_name = input_dataset.name

            try:
                response = requests.post(
                    "http://localhost:8000/transform", 
                    json={
                        "dataset_name" : dataset_name,
                        "abstraction": abstraction.to_json(), 
                        "task": task_description
                    }
                )

                response_body = response.json()
                if response.status_code == 200:
                    
                    processed_dataframe = pd.DataFrame(json.loads(response_body.get('processed_abstraction', None)))

                    # download button for processed dataframe
                    CSV = processed_dataframe.to_csv(index=False).encode("utf-8")
                    st.download_button( label="📥 Download", data=CSV, file_name=f"output_{dataset_name}", mime="text/csv" )

                    st.text("Preview of the transformed dataset")
                    # render the processed dataframe to screen
                    st.dataframe(processed_dataframe.head())

                else:
                    # FastAPI uses HTTPException by default, hence we assume an error returns the "detail" key
                    st.error(response_body.get("detail"))

            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
    
    # data analysis tab
    with data_analyst:
        input_dataset = st.file_uploader(label="Upload your dataset", type=["csv"], accept_multiple_files=False, key="dataset_to_analyse")
        query = st.text_area(label="Describe your data analysis query with natural language", key="data_analysis_task")

        trigger_analysis_btn = st.button(
            label="Query", 
            key="trigger_analysis_btn", 
            disabled=not (input_dataset and query), 
            help="Ensure both the dataset to be analysed and your query are provided", 
            type="primary"
        )

        if trigger_analysis_btn:
            
            # abstract the input dataset to a pandas dataframe
            abstraction = pd.read_csv(input_dataset)

            try:
                response = requests.post("http://localhost:8000/analyze", json={ "abstraction": abstraction.to_json(), "task": query })

                response_body = response.json()
                if response.status_code == 200:

                    # check if there is a plotting function to be executed on the code
                    plotting_code_string = response_body.get('plot_code', None)
                    if plotting_code_string is not None:
                        # extract the plotting function and run it on the datatse
                        namespace = {'plt' : plt, 'pd' : pd, 'np' : np, '__builtins__' : __builtins__}
                        exec(plotting_code_string, namespace)
                        plotting_function = namespace['plot_data']

                        # show the plot
                        st.pyplot(plotting_function(abstraction))

                    # render the query response in natural language
                    st.markdown(response_body['summary'])

                else:
                    # FastAPI uses HTTPException by default, hence we assume an error returns the "detail" key
                    st.error(response_body.get("detail"))

            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")

if __name__ == "__main__":
    main()