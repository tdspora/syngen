# streamlit_app.py
import streamlit as st
import os
import pandas as pd
from syngen.ml.worker import Worker

import queue
from loguru import logger
import threading

# Setup the log queue
log_queue = queue.Queue()

def log_sink(message):
    log_queue.put(message.record["message"])
logger.add(log_sink)

# Create the path for uploaded CSV files
UPLOAD_DIRECTORY = "uploaded_files"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

st.title("Syngen")
st.write("Upload CSV files and train your model.")

# Define file uploader
uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=False)

dataframes = {}
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save each uploaded file to the local directory
        file_path = os.path.join(UPLOAD_DIRECTORY, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        df = pd.read_csv(file_path)
        dataframes[uploaded_file.name] = df
        st.write(f"Preview of {uploaded_file.name}:", df.head())

# Main page for training parameters
epochs = st.number_input('Epochs', min_value=1, value=10)
row_limit = st.number_input('Row Limit', min_value=1, max_value=None, value=1000)

@logger.catch
def train_model():
    with st.spinner('Training is in progress...'):
        logger.info("Starting model training...")
        for uploaded_file in uploaded_files:
            file_path = os.path.join(UPLOAD_DIRECTORY, uploaded_file.name)
            settings = {
                "source": file_path,
                "epochs": epochs,
                "drop_null": False,  # Assuming this needs to remain static
                "row_limit": row_limit,  # Using the row_limit from the main page
                "batch_size": 32,  # Assuming this needs to remain static
                "print_report": False  # Assuming this needs to remain static
            }
            worker = Worker(
                table_name=uploaded_file.name,
                settings=settings,
                metadata_path=None,
                log_level='DEBUG',
                type_of_process="train"
            )
            worker.launch_train()
        logger.info("Model training completed.")

# Button to start the training process
if st.button('Start Model Training'):
    if uploaded_files:
        training_thread = threading.Thread(target=train_model)
        training_thread.start()

        # Display logs from the log queue
        with st.spinner('Waiting for the training to complete...'):
            with st.expander("Logs"):
                while True:
                    if not log_queue.empty():
                        with st.code("logs"):
                            st.text(log_queue.get())
                    elif not training_thread.is_alive():
                        st.success("Training completed.")
                        break
    else:
        st.warning("Please upload CSV files to proceed.")

# for the error related to Thread 'Thread-82': missing ScriptRunContext
# check: https://stackoverflow.com/questions/64568709/python-use-concurrent-futures-in-streamlit-got-missing-reportcontext