import streamlit as st
import pandas as pd
import requests
import json

# imports for type hints 
from typing import Any, Optional, List
from streamlit.runtime.uploaded_file_manager import UploadedFile


def extract_list(value : Any) -> Optional[List[Any]]:
    """Recursively traverses a python dictionary and returns the value of type list"""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for subvalue in value.values():
            result = extract_list(subvalue)
            if result: 
                return result

def abstract_dataset(dataset: UploadedFile) -> Any:
    """Abstract the uploaded dataset into a Pandas dataframe"""
    
    if dataset.name.endswith(".csv"):
        return pd.read_csv(dataset)

    if dataset.name.endswith(".json"):

        # extract the first list of objects, as it is assumed to contain the objects of interest
        dataset_contents = json.load(dataset)
        objects = extract_list(dataset_contents)

        # compute the headers of the abstraction as the union of keys across all objects
        abstraction_headers = list(set([ attribute for object in objects for attribute in object.keys() ]))

        # abstract each object by expanding its key-value pairs to match the headers
        abstraction_rows = []
        for object in objects:
            # use a placeholder to extend each object
            extended_object_values = [ object[attribute] if attribute in object else "--" for attribute in abstraction_headers  ]
            abstraction_rows.append(extended_object_values)

        # intitalise a pandas dataframe from the abstracted dataset contents 
        return pd.DataFrame(abstraction_rows, columns=abstraction_headers)

def main():
    """Run the frontend application"""

    # configure page settings
    st.set_page_config(page_title="FlowETL", page_icon="🔁", initial_sidebar_state=None)
        
    st.title("FlowETL")
    st.text("Transform your data using natural language")

    # user inputs : dataset uploader and task description text area
    input_dataset = st.file_uploader(label="Upload your dataset", type=["csv", "json"], accept_multiple_files=False, key="input_dataset")
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
        abstraction = abstract_dataset(input_dataset)

        # take a 25% sample of the abstraction - this makes processing quicker
        # we assume that the plan generated will be successfully applied to the entire dataset
        sampled_abstraction = abstraction.sample(frac=0.25).to_json()

        try:
            response = requests.post(
                "http://localhost:8000/", 
                json={
                    "abstraction": sampled_abstraction, 
                    "task_description": task_description
                }
            )

            response_body = response.json()
            if response.status_code == 200:
                st.success("Plan generated but not validated")
                st.json(response_body["plan"])
            else:
                # FastAPI uses HTTPException by default, hence we assume an error returns the "detail" key
                st.error(response_body.get("detail"))

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

if __name__ == "__main__":
    main()