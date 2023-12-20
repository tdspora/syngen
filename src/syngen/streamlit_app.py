import os
import threading
import time
import queue
from slugify import slugify

import pandas as pd
from loguru import logger
import streamlit as st
from syngen.ml.worker import Worker
from syngen.ml.utils import setup_logger, fetch_log_message

# Setup the log queue
log_queue = queue.Queue()


def log_sink(message):
    log_message = fetch_log_message(message)
    log_queue.put(log_message)


# Create the path for uploaded CSV files
UPLOAD_DIRECTORY = "uploaded_files"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

st.title("Syngen")
uploaded_file = st.file_uploader(
    "Upload CSV file",
    type="csv",
    accept_multiple_files=False
)
dataframe = {}
file_name = str()
table_name = str()

if uploaded_file:
    file_name = uploaded_file.name
    table_name = os.path.splitext(file_name)[0]
    # Save the uploaded file to the local directory
    file_path = os.path.join(UPLOAD_DIRECTORY, file_name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Read the CSV into a DataFrame
    df = pd.read_csv(file_path)
    dataframe[table_name] = df

    # Show a preview of the uploaded file
    st.write(f"Preview of {file_name}:", df.head())
    st.write(f"Rows: {df.shape[0]}, columns: {df.shape[1]}")

# Main page for training parameters
epochs = st.number_input("Epochs", min_value=1, value=1)
size_limit = st.number_input("Size Limit",
                             min_value=1,
                             max_value=None,
                             value=1000)
log_level = "INFO"


@logger.catch
def train_model():
    logger.info("Starting model training...")
    file_path = os.path.join(UPLOAD_DIRECTORY, file_name)
    settings = {
        "source": file_path,
        "epochs": epochs,
        "row_limit": 10000,
        "drop_null": False,
        "batch_size": 32,
        "print_report": False
    }
    worker = Worker(
        table_name=table_name,
        settings=settings,
        metadata_path=None,
        log_level="INFO",
        type_of_process="train"
    )
    worker.launch_train()
    logger.info("Model training completed.")


@logger.catch
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
            table_name=table_name,
            settings=settings,
            metadata_path=None,
            log_level="INFO",
            type_of_process="infer"
        )
        worker.launch_infer()
        logger.info("Data generation completed.")
    except Exception as e:
        logger.error(f"Error during infer: {e}")
        log_queue.put(f"Error during infer: {e}")


# Function to run both training and inference in sequence

def train_and_infer():
    os.environ["LOGURU_LEVEL"] = log_level
    os.environ["SUCCESS_LOG_FILE"] = f"{UPLOAD_DIRECTORY}/{table_name}.log"
    setup_logger()
    logger.add(log_sink, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    train_model()
    infer_model()


# Button to start the training and inference process
if st.button("Generate data"):
    thread = None
    if uploaded_file:
        # Start the training and inference thread
        thread = threading.Thread(target=train_and_infer)
        thread.start()

        # Display logs from the log queue
        with st.spinner("Waiting for the processes to complete..."):
            with st.expander("Logs"):
                st.code("Logs will be displayed here...")
                while thread.is_alive():
                    if not log_queue.empty():
                        log = log_queue.get()
                        st.code(log, language="log")
                    time.sleep(0.001)
                st.success("Data generation completed.")
    else:
        st.warning("Please upload a CSV file to proceed.")
sl_table_name = slugify(table_name)
path_to_generated_data = f"model_artifacts/tmp_store/{sl_table_name}/merged_infer_{sl_table_name}.csv"
path_to_logs = f"{UPLOAD_DIRECTORY}/{table_name}.log"
if os.path.exists(path_to_generated_data):
    with open(path_to_generated_data, "rb") as f:
        st.download_button("Download the generated data", f, file_name=f"generated_{file_name}")
if os.path.exists(path_to_logs):
    with open(path_to_logs, "rb") as f:
        st.download_button("Download the logs", f, file_name=f"logs.log")
