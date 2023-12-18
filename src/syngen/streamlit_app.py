# streamlit_app.py
import streamlit as st
import os
import pandas as pd
from syngen.ml.worker import Worker
import time
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
uploaded_file = st.file_uploader(
    "Upload CSV file",
    type="csv",
    accept_multiple_files=False
)
dataframe = {}  # Define the "dataframe" variable as an empty dictionary

if uploaded_file:
    # Save the uploaded file to the local directory
    file_path = os.path.join(UPLOAD_DIRECTORY, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getvalue())

    # Read the CSV into a DataFrame
    df = pd.read_csv(file_path)
    dataframe[uploaded_file.name] = df

    # Show a preview of the uploaded file
    st.write(f"Preview of {uploaded_file.name}:", df.head())

# Main page for training parameters
epochs = st.number_input('Epochs', min_value=1, value=1)
size_limit = st.number_input('Size Limit',
                             min_value=1,
                             max_value=None,
                             value=1000)


@logger.catch
def train_model():
    logger.info("Starting model training...")
    file_path = os.path.join(UPLOAD_DIRECTORY, uploaded_file.name)
    settings = {
        "source": file_path,
        "epochs": epochs,
        "row_limit": 1000,
        "drop_null": False,
        "batch_size": 32,
        "print_report": False
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


def infer_model():
    try:
        logger.info("Starting data generation...")
        settings = {
            "size": size_limit,
            "batch_size": 32,
            "run_parallel": False,
            "random_seed": None,
            "print_report": False
        }
        worker = Worker(
            table_name=uploaded_file.name,
            settings=settings,
            metadata_path=None,
            log_level='DEBUG',
            type_of_process="infer"
        )
        worker.launch_infer()
        logger.info("Data generation completed.")
    except Exception as e:
        logger.error(f"Error during infer: {e}")
        log_queue.put(f"Error during infer: {e}")


# Function to run both training and inference in sequence

def train_and_infer():
    train_model()
    infer_model()


# Button to start the training and inference process
if st.button('Generate data'):
    if uploaded_file:
        # Start the training and inference thread
        thread = threading.Thread(target=train_and_infer)
        thread.start()

        # Display logs from the log queue in different format on streamlit
#        with st.spinner('Waiting for the training to complete...'):
#            with st.expander("Logs"):
#                while True:
#                    if not log_queue.empty():
#                        with st.code("logs"):
#                            st.text(log_queue.get())
#                    elif not training_thread.is_alive():
#                        st.success("Training completed.")
#                        break
#                    time.sleep(0.001)

        # Display logs from the log queue
        with st.spinner('Waiting for the processes to complete...'):
            with st.expander("Logs"):
                st.code("Logs will be displayed here...")
                while thread.is_alive():
                    if not log_queue.empty():
                        log = log_queue.get()
                        st.code(log, language='log')
                    time.sleep(0.001)
                st.success("Data generation completed.")
    else:
        st.warning("Please upload a CSV file to proceed.")
