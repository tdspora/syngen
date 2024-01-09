import os
import threading
import time
from queue import Queue
from slugify import slugify

import pandas as pd
from loguru import logger
import streamlit as st
from syngen.ml.worker import Worker
from syngen.ml.utils import fetch_log_message, create_log_file
from streamlit_option_menu import option_menu


class StreamlitHandler:
    def __init__(self, uploaded_file):
        self.upload_directory = "uploaded_files"
        self.log_queue = Queue()
        self.log_error_queue = Queue()
        self.epochs = int()
        self.size_limit = int()
        self.print_report = bool()
        self.log_format = ("<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
                           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                           "<level>{message}</level>")
        self.uploaded_file = uploaded_file
        self.file_name = uploaded_file.name
        self.table_name = os.path.splitext(self.file_name)[0]
        self.file_path = os.path.join(self.upload_directory, self.file_name)
        sl_table_name = slugify(self.table_name)
        self.path_to_generated_data = f"model_artifacts/tmp_store/{sl_table_name}/merged_infer_{sl_table_name}.csv"
        self.path_to_logs = f"uploaded_files/logs_{sl_table_name}.log"
        self.path_to_report = f"model_artifacts/tmp_store/{sl_table_name}/draws/accuracy_report.html"

    def set_parameters(self, epochs, size_limit, print_report):
        self.epochs = epochs
        self.size_limit = size_limit
        self.print_report = print_report

    def set_logger(self):
        logger.add(self.file_sink, level="INFO")
        logger.add(self.log_sink, format=self.log_format)

    def show_data(self):
        if not os.path.exists("uploaded_files"):
            os.makedirs("uploaded_files", exist_ok=True)
        with open(self.file_path, "wb") as file_object:
            file_object.write(self.uploaded_file.getvalue())
        if st.checkbox("Show dataframe"):
            df = pd.read_csv(self.file_path)
            st.write(f"Preview of {self.file_name}:", df.head())
            st.write(f"Rows: {df.shape[0]}, columns: {df.shape[1]}")

    def log_sink(self, message):
        log_message = fetch_log_message(message)
        self.log_queue.put(log_message)

    def file_sink(self, message):
        os.makedirs(os.path.dirname(self.path_to_logs), exist_ok=True)
        os.environ["SUCCESS_LOG_FILE"] = self.path_to_logs
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

    @staticmethod
    def generate_button(label, path_to_file, download_name):
        if os.path.exists(path_to_file):
            with open(path_to_file, "rb") as f:
                st.download_button(
                    label,
                    f,
                    file_name=download_name,
                )

    def train_and_infer(self):
        try:
            self.train_model()
            self.infer_model()
        except Exception as e:
            self.log_error_queue.put(e)


def main():
    st.sidebar.image("./.streamlit/img/logo.svg", use_column_width=True)
    st.markdown(f"""
        <style>
        {"".join(open("./.streamlit/css/font_style.css").readlines())}
        </style>
        """, unsafe_allow_html=True)
    with st.sidebar:
        selected = option_menu("", ["Demo", "Advanced", "DOCS", "Authorization"],
                               icons=["'play'", "'gear'", "'journals'", "'person-check'"],
                               default_index=0,
                               menu_icon=None,
                               styles={
                                   "container": {"font-family": "Open Sans"}
                               }
                               )

    if selected == "Demo":
        st.title("Syngen")
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type="csv",
            accept_multiple_files=False
        )
        if not uploaded_file:
            st.warning("Please upload a CSV file to proceed.")
        if uploaded_file:
            app = StreamlitHandler(uploaded_file)
            app.show_data()
            epochs = st.number_input("Epochs", min_value=1, value=1)
            size_limit = st.number_input("Size Limit", min_value=1, max_value=None, value=1000)
            print_report = st.checkbox("Create the accuracy report", value=False)
            app.set_parameters(epochs, size_limit, print_report)
            if st.button("Generate data", type="primary"):
                thread = threading.Thread(target=app.train_and_infer)
                thread.start()
                current_progress = 0
                prg = st.progress(current_progress)

                # Progress bar
                while thread.is_alive():
                    if not app.log_queue.empty():
                        log = app.log_queue.get()
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
                    time.sleep(0.001)
                if not app.log_error_queue.empty():
                    st.exception(app.log_error_queue.get())
                elif app.log_error_queue.empty():
                    st.success("Data generation completed")
            with st.container():
                app.generate_button(
                    "Download the generated data",
                    app.path_to_generated_data,
                    f"generated_{app.table_name}.csv"
                )
                app.generate_button(
                    "Download the report",
                    app.path_to_report,
                    f"accuracy_report_{app.table_name}.html"
                )
                app.generate_button(
                    "Download the logs",
                    app.path_to_logs,
                    f"logs_{app.table_name}.log"
                )


if __name__ == "__main__":
    main()
