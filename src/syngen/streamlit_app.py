import os
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
from syngen.ml.utils import fetch_log_message
from streamlit_option_menu import option_menu


class StreamlitHandler:
    def __init__(self, uploaded_file):
        self.upload_directory = "uploaded_files"
        self.log_queue = Queue()
        self.log_error_queue = Queue()
        self.epochs = int()
        self.size_limit = int()
        self.print_report = bool()
        self.file_name = uploaded_file.name
        self.table_name = os.path.splitext(self.file_name)[0]
        self.file_path = os.path.join(self.upload_directory, self.file_name)
        self.sl_table_name = slugify(self.table_name)
        self.path_to_generated_data = (f"model_artifacts/tmp_store/{self.sl_table_name}/"
                                       f"merged_infer_{self.sl_table_name}.csv")
        self.path_to_logs = (f"model_artifacts/tmp_store/{self.sl_table_name}_"
                             f"{slugify(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}.log")
        self.path_to_report = f"model_artifacts/tmp_store/{self.sl_table_name}/draws/accuracy_report.html"

    def set_parameters(self, epochs, size_limit, print_report):
        self.epochs = epochs
        self.size_limit = size_limit
        self.print_report = print_report

    def set_env_variables(self):
        os.environ["PATH_TO_GENERATED_DATA"] = self.path_to_generated_data
        os.environ["PATH_TO_REPORT"] = self.path_to_report
        os.environ["SUCCESS_LOG_FILE"] = self.path_to_logs
        os.environ["PRINT_REPORT"] = "True" if self.print_report else str()
        os.environ["TABLE_NAME"] = self.sl_table_name

    def set_logger(self):
        logger.add(self.file_sink, level="INFO")
        logger.add(self.log_sink)

    def log_sink(self, message):
        log_message = fetch_log_message(message)
        self.log_queue.put(log_message)

    def file_sink(self, message):
        os.makedirs(os.path.dirname(self.path_to_logs), exist_ok=True)
        with open(self.path_to_logs, "a") as log_file:
            log_message = fetch_log_message(message)
            log_file.write(log_message + "\n")

    def get_progress_status(self, log_message):
        mapping = {
            f"Training process of the table - {self.table_name} has started": 5,
            f"Training of the table - {self.table_name} was completed": 40,
            f"Infer process of the table - {self.table_name} has started": 50
        }
        if log_message in mapping.keys():
            current_progress = mapping[log_message]
            return current_progress

    def handle_training_process(self, progress_bar):
        for i in range(5, 40):
            time.sleep(self.epochs / 35 if self.epochs < 35 else 1)
            progress_bar.progress(i + 1, text="Training model...")

    def handle_infer_process(self, log, progress_bar, current_progress):
        log_message = f"Infer process of the table - {self.table_name} has started"
        if log_message in log:
            current_progress = 50
            progress_bar.progress(current_progress, text=log_message)
        sleep_time = int((self.size_limit / 32) / 40)
        for i in range(50, 90):
            time.sleep(sleep_time)
            progress_bar.progress(i + 1, text="Generating data...")
        return current_progress

    def train_model(self):
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
            worker.launch_train()
            logger.info("Model training completed")
        except Exception as e:
            logger.error(f"Error during train: {e}")
            self.log_error_queue.put(f"Error during train: {e}")

    def infer_model(self):
        try:
            logger.info("Starting data generation...")
            settings = {
                "size": self.size_limit,
                "batch_size": 32,
                "run_parallel": False,
                "random_seed": None,
                "print_report": self.print_report
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
        except Exception as e:
            logger.error(f"Error during infer: {e}")
            self.log_error_queue.put(f"Error during infer: {e}")

    @staticmethod
    def load_data(label, path_to_file, download_name):
        if os.path.exists(path_to_file):
            with open(path_to_file, "rb") as f:
                st.download_button(label, f, file_name=download_name)

    def train_and_infer(self):
        try:
            self.train_model()
            self.infer_model()
        except Exception as e:
            self.log_error_queue.put(e)


def show_data(uploaded_file):
    file_name = uploaded_file.name
    file_path = os.path.join("uploaded_files", file_name)
    if not os.path.exists("uploaded_files"):
        os.makedirs("uploaded_files", exist_ok=True)
    with open(file_path, "wb") as file_object:
        file_object.write(uploaded_file.getvalue())
    if st.checkbox("Show sample data"):
        df = pd.read_csv(file_path)
        st.write(f"Preview of {file_name}:", df.head())
        st.write(f"Rows: {df.shape[0]}, columns: {df.shape[1]}")


def generate_button(label, path_to_file, download_name):
    if os.path.exists(path_to_file):
        with open(path_to_file, "rb") as f:
            st.download_button(
                label,
                f,
                file_name=download_name,
            )


def get_running_status():
    if "gen_button" in st.session_state and st.session_state.gen_button is True:
        st.session_state.running = True
        return True
    else:
        st.session_state.running = False
        return False


def main():
    st.set_page_config(
        page_title="SynGen UI",
        page_icon="./.streamlit/img/logo.svg"
    )
    st.sidebar.image("./.streamlit/img/logo.svg", use_column_width=True)
    st.markdown(f"""
        <style>
        {"".join(open("./.streamlit/css/font_style.css").readlines())}
        </style>
        """, unsafe_allow_html=True)
    label = (
        """
    <style>
        div[data-testid="stFileUploader"]>section[data-testid="stFileUploadDropzone"]>button[data-testid="baseButton-secondary"] {
           color:white;
        }
        div[data-testid="stFileUploader"]>section[data-testid="stFileUploadDropzone"]>button[data-testid="baseButton-secondary"]::after {
            content: "Browse";
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
            "Upload CSV file",
            type="csv",
            accept_multiple_files=False
        )
        if not uploaded_file:
            shutil.rmtree("uploaded_files", ignore_errors=True)
            shutil.rmtree("model_artifacts", ignore_errors=True)
            st.warning("Please upload a CSV file to proceed.")
        if uploaded_file:
            show_data(uploaded_file)
            epochs = st.number_input("Epochs", min_value=1, value=1)
            size_limit = st.number_input("Rows to generate", min_value=1, max_value=None, value=1000)
            print_report = st.checkbox("Create an accuracy report", value=False)
            if st.button("Generate data", type="primary", key="gen_button", disabled=get_running_status()):
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

                # Progress bar
                while runner.is_alive():
                    with st.spinner("Waiting for the process to complete..."):
                        with st.expander("Logs"):
                            while True:
                                if not app.log_queue.empty():
                                    with st.code("logs", language="log"):
                                        log = app.log_queue.get()
                                        st.text(log)
                                        if f"Training process of the table - {app.table_name} has started" in log:
                                            current_progress = 5
                                            prg.progress(
                                                current_progress,
                                                text=f"Training process of the table - {app.table_name} has started"
                                            )
                                        elif current_progress >= 5 and current_progress < 40:
                                            for i in range(5, 40):
                                                time.sleep(epochs / 35 if epochs < 35 else 1)
                                                prg.progress(i + 1, text="Training model...")
                                            current_progress = 45
                                        elif f"Training of the table - {app.table_name} was completed" in log:
                                            current_progress = 45
                                            prg.progress(current_progress)
                                        elif f"Infer process of the table - {app.table_name} has started" in log:
                                            current_progress = 50
                                            prg.progress(
                                                current_progress,
                                                text=f"Infer process of the table - {app.table_name} has started"
                                            )
                                        elif current_progress >= 50 and current_progress < 100:
                                            sleep_time = int((size_limit / 32) / 40)
                                            for i in range(50, 100):
                                                time.sleep(sleep_time)
                                                prg.progress(i + 1, text="Generating data...")
                                elif not runner.is_alive():
                                    break
                                time.sleep(0.001)
                if not app.log_error_queue.empty():
                    st.exception(app.log_error_queue.get())
                elif app.log_error_queue.empty() and not runner.is_alive():
                    prg.progress(100, text="Data generation completed")
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
