import os
import shutil
import threading

import streamlit as st

from syngen.ml.data_loaders import DataLoader


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
    df, schema = DataLoader(file_path).load_data()
    st.write(f"Preview of {file_name}:", df.head())
    st.write(f"Rows: {df.shape[0]}, columns: {df.shape[1]}")


def get_running_status():
    """
    Get the status of the process of a model training and generation data
    """
    if (
            ("gen_button" in st.session_state and st.session_state.gen_button is True) or
            [thread for thread in threading.enumerate() if thread.name == "train_and_infer"]
    ):
        return True
    else:
        return False


def set_session_state():
    """
    Set a session state for the Streamlit app
    """
    if "disabled" not in st.session_state:
        st.session_state.disabled = False


def cleanup_artifacts():
    """
    Clean up the artifacts
    """
    shutil.rmtree(UPLOAD_DIRECTORY, ignore_errors=True)
    shutil.rmtree(MODEL_ARTIFACTS, ignore_errors=True)
