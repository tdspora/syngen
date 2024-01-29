import os
import traceback
import shutil
import threading
import time
from datetime import datetime
from queue import Queue
from slugify import slugify

import pandas as pd
from loguru import logger
import streamlit as st

from syngen.ml.worker import Worker
from syngen.ml.utils import fetch_log_message, ProgressBarHandler
from streamlit_option_menu import option_menu


UPLOAD_DIRECTORY = "uploaded_files"


class StreamlitHandler:
    """
    A class for handling the Streamlit app
    """
    def __init__(self, uploaded_file):
        self.log_queue = Queue()
        self.progress_handler = ProgressBarHandler()
        self.log_error_queue = Queue()
        self.epochs = int()
        self.size_limit = int()
        self.print_report = bool()
        self.file_name = uploaded_file.name
        self.table_name = os.path.splitext(self.file_name)[0]
        self.file_path = os.path.join(UPLOAD_DIRECTORY, self.file_name)
        self.sl_table_name = slugify(self.table_name)
        self.path_to_generated_data = (f"model_artifacts/tmp_store/{self.sl_table_name}/"
                                       f"merged_infer_{self.sl_table_name}.csv")
        self.path_to_logs = (f"model_artifacts/tmp_store/{self.sl_table_name}_"
                             f"{slugify(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}.log")
        self.path_to_report = (f"model_artifacts/tmp_store/{self.sl_table_name}/"
                               f"draws/accuracy_report.html")

    def set_parameters(self, epochs, size_limit, print_report):
        """
        Set parameters for a model training and data generation
        """
        self.epochs = epochs
        self.size_limit = size_limit
        self.print_report = print_report

    def set_env_variables(self):
        """
        Set environment variables
        """
        os.environ["PATH_TO_GENERATED_DATA"] = self.path_to_generated_data
        os.environ["PATH_TO_REPORT"] = self.path_to_report
        os.environ["SUCCESS_LOG_FILE"] = self.path_to_logs
        os.environ["PRINT_REPORT"] = "True" if self.print_report else str()
        os.environ["TABLE_NAME"] = self.sl_table_name

    def set_logger(self):
        """
        Set a logger to see logs, and collect log messages
        with the log level - 'INFO' in a log file and stdout
        """
        logger.add(self.file_sink, level="INFO")
        logger.add(self.log_sink, level="INFO")

    def log_sink(self, message):
        """
        Put log messages to a log queue
        """
        log_message = fetch_log_message(message)
        self.log_queue.put(log_message)

    def file_sink(self, message):
        """
        Write log messages to a log file
        """
        os.makedirs(os.path.dirname(self.path_to_logs), exist_ok=True)
        with open(self.path_to_logs, "a") as log_file:
            log_message = fetch_log_message(message)
            log_file.write(log_message + "\n")

    def train_model(self):
        """
        Launch a model training
        """
        try:
            self.set_logger()
            logger.info("Starting model training...")
            settings = {
                "source": self.file_path,
                "epochs": self.epochs,
                "row_limit": 10000,
                "drop_null": False,
                "batch_size": 32,
                "print_report": False
            }
            worker = Worker(
                table_name=self.table_name,
                settings=settings,
                metadata_path=None,
                log_level="INFO",
                type_of_process="train"
            )
            ProgressBarHandler().set_progress(0.01)
            worker.launch_train()
            logger.info("Model training completed")
        except Exception:
            logger.error(f"Error during train: {traceback.format_exc()}")
            self.log_error_queue.put(f"Error during train: {traceback.format_exc()}")

    def infer_model(self):
        """
        Launch a data generation
        """
        try:
            logger.info("Starting data generation...")
            settings = {
                "size": self.size_limit,
                "batch_size": 32,
                "run_parallel": False,
                "random_seed": None,
                "print_report": self.print_report,
                "get_infer_metrics": False
            }
            worker = Worker(
                table_name=self.table_name,
                settings=settings,
                metadata_path=None,
                log_level="INFO",
                type_of_process="infer"
            )
            worker.launch_infer()
            logger.info("Data generation completed")
        except Exception:
            logger.error(f"Error during infer: {traceback.format_exc()}")
            self.log_error_queue.put(f"Error during infer: {traceback.format_exc()}")

    def train_and_infer(self):
        """
        Launch a model training and data generation
        """
        self.train_model()
        self.infer_model()


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
    if st.checkbox("Show sample data"):
        df = pd.read_csv(file_path)
        st.write(f"Preview of {file_name}:", df.head())
        st.write(f"Rows: {df.shape[0]}, columns: {df.shape[1]}")


def generate_button(label, path_to_file, download_name):
    """
    Generate a download button
    """
    if os.path.exists(path_to_file):
        with open(path_to_file, "rb") as f:
            st.download_button(
                label,
                f,
                file_name=download_name,
            )


def get_running_status():
    """
    Get the status of the proces of a model training and generation data
    """
    if "gen_button" in st.session_state and st.session_state.gen_button is True:
        st.session_state.running = True
        return True
    else:
        st.session_state.running = False
        return False


def main():
    path_to_logo = f"{os.path.join(os.path.dirname(__file__))}/img/logo.svg"
    st.set_page_config(
        page_title="SynGen UI",
        page_icon=path_to_logo
    )
    st.sidebar.image(path_to_logo, use_column_width=True)
    st.markdown(f"""
        <style>
        {"".join(open(f"{os.path.join(os.path.dirname(__file__))}/css/font_style.css").readlines())}
        </style>
        """, unsafe_allow_html=True)
    label = (
        """
    <style>
        div[data-testid="stFileUploader"]>section[data-testid="stFileUploadDropzone"]>button[data-testid="baseButton-secondary"] {
           color:white;
        }
        div[data-testid="stFileUploader"]>section[data-testid="stFileUploadDropzone"]>button[data-testid="baseButton-secondary"]::after {
            content: "Browse a file";
            color:black;
            display: block;
            position: absolute;
        }
        div[data-testid="stFileDropzoneInstructions"]>div>span {
           visibility:hidden;
        }
        div[data-testid="stFileDropzoneInstructions"]>div>span::after {
           content:"Drop a file here";
           visibility:visible;
           display:block;
        }
         div[data-testid="stFileDropzoneInstructions"]>div>small {
           visibility:hidden;
        }
        div[data-testid="stFileDropzoneInstructions"]>div>small::before {
           content:"Limit 200MB per file";
           visibility:visible;
           display:block;
        }
    </style>
    """
    )
    st.markdown(label, unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {
            visibility: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    with st.sidebar:
        selected = option_menu("", ["Basic", "Advanced", "DOCS", "Authorization"],
                               icons=["'play'", "'gear'", "'journals'", "'person-check'"],
                               default_index=0,
                               menu_icon=None,
                               styles={
                                   "container": {"font-family": "Open Sans"}
                               }
                               )

    if selected == "Basic":
        st.title("SynGen UI")
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type="csv",
            accept_multiple_files=False
        )
        if not uploaded_file:
            shutil.rmtree(UPLOAD_DIRECTORY, ignore_errors=True)
            shutil.rmtree("model_artifacts", ignore_errors=True)
            st.warning("Please upload a CSV file to proceed")
        if uploaded_file:
            show_data(uploaded_file)
            epochs = st.number_input("Epochs", min_value=1, value=1)
            size_limit = st.number_input(
                "Rows to generate", min_value=1, max_value=None, value=1000
            )
            print_report = st.checkbox("Create an accuracy report", value=False)
            if st.button(
                    "Generate data", type="primary", key="gen_button", disabled=get_running_status()
            ):
                app = StreamlitHandler(uploaded_file)
                app.set_parameters(epochs, size_limit, print_report)
                app.set_env_variables()
                runner = threading.Thread(target=app.train_and_infer, name="train_and_infer")
                st.session_state.running = True
                lock = threading.Lock()
                with lock:
                    runner.start()
                current_progress = 0
                prg = st.progress(current_progress)

                while runner.is_alive():
                    with st.expander("Logs"):
                        while True:
                            if not app.log_queue.empty():
                                with st.code("logs", language="log"):
                                    log = app.log_queue.get()
                                    st.text(log)
                                    current_progress, message = app.progress_handler.info
                                    prg.progress(value=current_progress, text=message)
                            elif not runner.is_alive():
                                break
                            time.sleep(0.001)
                if not app.log_error_queue.empty():
                    st.exception(app.log_error_queue.get())
                elif app.log_error_queue.empty() and not runner.is_alive():
                    prg.progress(100)
                    app.progress_handler.reset_instance()
                    st.success("Data generation completed")
            with st.container():
                col1, col2, col3 = st.columns([0.6, 0.4, 0.6], )
                with col1:
                    generate_button(
                        "Download generated data",
                        os.getenv("PATH_TO_GENERATED_DATA", ""),
                        f"generated_{os.getenv('TABLE_NAME', '')}.csv"
                    )
                with col2:
                    generate_button(
                        "Download logs",
                        os.getenv("SUCCESS_LOG_FILE", ""),
                        f"logs_{os.getenv('TABLE_NAME', '')}.log"
                    )
                if os.getenv("PRINT_REPORT", ""):
                    with col3:
                        generate_button(
                            "Download report",
                            os.getenv("PATH_TO_REPORT", ""),
                            f"accuracy_report_{os.getenv('TABLE_NAME', '')}.html"
                        )


if __name__ == "__main__":
    main()
