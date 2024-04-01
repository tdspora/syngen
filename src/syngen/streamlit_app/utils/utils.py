import os

import pandas as pd
import streamlit as st


UPLOAD_DIRECTORY = "uploaded_files"
MODEL_ARTIFACTS = "model_artifacts"


def show_data(uploaded_file):
    """
    Show sample data from the uploaded file
    """
    file_name = uploaded_file.name
    file_path = os.path.join(UPLOAD_DIRECTORY, file_name)
    if not os.path.exists(UPLOAD_DIRECTORY):
        os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
    with open(file_path, "wb") as file_object:
        file_object.write(uploaded_file.getvalue())
    df = pd.read_csv(file_path)
    st.write(f"Preview of {file_name}:", df.head())
    st.write(f"Rows: {df.shape[0]}, columns: {df.shape[1]}")


def get_running_status():
    """
    Get the status of the process of a model training and generation data
    """
    if "gen_button" in st.session_state and st.session_state.gen_button is True:
        return True
    else:
        return False


def set_session_state():
    """
    Set a session state for the Streamlit app
    """
    if "disabled" not in st.session_state:
        st.session_state.disabled = False
