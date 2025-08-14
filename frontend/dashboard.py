import streamlit as st


if __name__ == "__main__":

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
        
        # TODO - send the artifacts to the backend for processing, then render the response
        st.text(f"received the following : input dataset : {input_dataset.name}, task desc : {task_description}")