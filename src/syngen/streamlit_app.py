import os
import shutil
import threading
import time
from queue import Queue
from slugify import slugify

import pandas as pd
from loguru import logger
import streamlit as st
from syngen.ml.worker import Worker
from syngen.ml.utils import file_sink, fetch_log_message


class StreamlitHandler:
    def __init__(self, uploaded_file, epochs, size_limit):
        self.upload_directory = "uploaded_files"
        self.log_queue = Queue()
        self.epochs = epochs
        self.size_limit = size_limit
        self.log_format = ("<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
                           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                           "<level>{message}</level>")
        self.uploaded_file = uploaded_file
        self.file_name = uploaded_file.name
        self.table_name = os.path.splitext(self.file_name)[0]
        self.file_path = os.path.join(self.upload_directory, self.file_name)
        sl_table_name = slugify(self.table_name)
        self.path_to_generated_data = f"model_artifacts/tmp_store/{sl_table_name}/merged_infer_{sl_table_name}.csv"
        self.path_to_logs = f"{self.upload_directory}/{self.table_name}.log"
        self.path_to_report = f"model_artifacts/tmp_store/{sl_table_name}/draws/accuracy_report.html"

    def prepare_data(self):
        self.restore_artifacts()

    def set_logger(self):
        os.environ["LOGURU_LEVEL"] = "INFO"
        os.environ["SUCCESS_LOG_FILE"] = f"{self.upload_directory}/{self.table_name}.log"
        logger.add(file_sink, level="INFO")
        logger.add(self.log_sink, format=self.log_format)

    def show_data(self):
        df = pd.read_csv(self.file_path)

        # Show a preview of the uploaded file
        st.write(f"Preview of {self.file_name}:", df.head())
        st.write(f"Rows: {df.shape[0]}, columns: {df.shape[1]}")

    def log_sink(self, message):
        log_message = fetch_log_message(message)
        self.log_queue.put(log_message)

    def get_progress_status(self, log_message):
        mapping = {
            f"Training process of the table - {self.table_name} has started": 5,
            f"Training of the table - {self.table_name} was completed": 40,
            f"Infer process of the table - {self.table_name} has started": 50
        }
        if log_message in mapping.keys():
            current_progress = mapping[log_message]
            return current_progress

    # def handle_training_start(self, log, progress_bar, current_progress):
    #     log_message = f"Training process of the table - {self.table_name} has started"
    #     if log_message in log:
    #         current_progress = 5
    #         progress_bar.progress(current_progress, text=log_message)
    #     return current_progress

    def handle_training_process(self, progress_bar):
        for i in range(5, 40):
            time.sleep(self.epochs / 35 if self.epochs < 35 else 1)
            progress_bar.progress(i + 1, text="Training model...")

    # def handle_training_end(self, log, progress_bar, current_progress):
    #     log_message = f"Training of the table - {self.table_name} was completed"
    #     if log_message in log:
    #         current_progress = 45
    #         progress_bar.progress(current_progress, text=log_message)
    #     return current_progress

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

    def show_progress_bar(self, thread: threading.Thread):
        current_progress = 0
        progress_bar = st.progress(current_progress)

        while thread.is_alive():
            # Display logs from the log queue
            with st.spinner("Waiting for the processes to complete..."):
                with st.expander("Logs"):
                    while thread.is_alive():
                        if not self.log_queue.empty():
                            log = self.log_queue.get()
                            st.code(log, language='log')
                        time.sleep(0.001)
            if not self.log_queue.empty():
                log_message = self.log_queue.get()

                if current_progress is not None and 40 > current_progress > 5:
                    for i in range(5, 40):
                        time.sleep(self.epochs / 35 if self.epochs < 35 else 1)
                        progress_bar.progress(i + 1, text="Training model...")
                elif current_progress is not None and current_progress == 50:
                    sleep_time = int((self.size_limit / 32) / 40)
                    for i in range(50, 90):
                        time.sleep(sleep_time)
                        progress_bar.progress(i + 1, text="Generating data...")
                else:
                    current_progress = self.get_progress_status(log_message)
                    if current_progress is not None:
                        progress_bar.progress(current_progress, text=log_message)

            time.sleep(0.001)

        progress_bar.progress(100)
        st.success("Data generation completed.")

    # def show_progress_bar(self, thread: threading.Thread):
    #     x = 0
    #     prg = st.progress(x)
    #
    #     # Progress bar
    #     while thread.is_alive():
    #         if not self.log_queue.empty():
    #             log = self.log_queue.get()
    #             if log_message := f"Training process of the table - {self.table_name} has started" in log:
    #                 x = 5
    #                 print(log_message)
    #                 prg.progress(x, text=log_message)
    #             elif x >= 5 and x < 40:
    #                 for i in range(5, 40):
    #                     time.sleep(self.epochs / 35 if self.epochs < 35 else 1)
    #                     prg.progress(i + 1, text="Training model...")
    #                 x = 45
    #             elif log_message := f"Training of the table - {self.table_name} was completed" in log:
    #                 x = 45
    #                 prg.progress(x, text=log_message)
    #             elif log_message := f"Infer process of the table - {self.table_name} has started" in log:
    #                 x = 50
    #                 prg.progress(x, text=log_message)
    #             elif x >= 50 and x < 90:
    #                 sleep_time = int((self.size_limit / 32) / 40)
    #                 for i in range(50, 90):
    #                     time.sleep(sleep_time)
    #                     prg.progress(i + 1, text="Generating data...")
    #         time.sleep(0.001)
    #     st.success("Data generation completed.")
    #     prg.progress(100)

    @logger.catch
    def train_model(self):
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
        logger.info("Model training completed.")

    @logger.catch
    def infer_model(self):
        try:
            logger.info("Starting data generation...")
            settings = {
                "size": self.size_limit,
                "batch_size": 32,
                "run_parallel": False,
                "random_seed": None,
                "print_report": True
            }
            worker = Worker(
                table_name=self.table_name,
                settings=settings,
                metadata_path=None,
                log_level="INFO",
                type_of_process="infer"
            )
            worker.launch_infer()
            logger.info("Data generation completed.")
        except Exception as e:
            logger.error(f"Error during infer: {e}")
            self.log_queue.put(f"Error during infer: {e}")

    @staticmethod
    def generate_download_button(label, path_to_file, download_name, key):
        if os.path.exists(path_to_file) and st.status[key]:
            with open(path_to_file, "rb") as f:
                st.download_button(label, f, file_name=download_name)

    def restore_artifacts(self):
        if os.path.exists(self.path_to_generated_data):
            os.remove(self.path_to_generated_data)
        if os.path.exists(self.path_to_report):
            os.remove(self.path_to_report)
        if os.path.exists(self.path_to_logs):
            os.remove(self.path_to_logs)

    def download_artifacts(self):
        if st.button("Download the generated data", key="data"):
            self.generate_download_button("Download the generated data",
                                          self.path_to_generated_data,
                                          f"generated_{self.file_name}",
                                          "data")
        if st.button("Download the report", key="report"):
            self.generate_download_button("Download the generated data",
                                          self.path_to_generated_data,
                                          f"generated_{self.file_name}",
                                          "report")
        if st.button("Download the logs", key="logs"):
            self.generate_download_button("Download the logs",
                                          self.path_to_logs,
                                          "logs.log",
                                          "logs")

    def train_and_infer(self):
        self.set_logger()
        self.train_model()
        self.infer_model()


def show_data(uploaded_file):
    if os.path.exists("uploaded_files"):
        shutil.rmtree("uploaded_files")
    file_path = os.path.join("uploaded_files", uploaded_file.name)
    if not os.path.exists("uploaded_files"):
        os.makedirs("uploaded_files", exist_ok=True)
    with open(file_path, "wb") as file_object:
        file_object.write(uploaded_file.getvalue())
    df = pd.read_csv(file_path)
    st.write(f"Preview of {uploaded_file.name}:", df.head())
    st.write(f"Rows: {df.shape[0]}, columns: {df.shape[1]}")


def main():
    st.title("Syngen")
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type="csv",
        accept_multiple_files=False
    )
    if uploaded_file:
        show_data(uploaded_file)
        epochs = st.number_input("Epochs", min_value=1, value=1)
        size_limit = st.number_input("Size Limit", min_value=1, max_value=None, value=1000)
        if st.button("Generate data"):
            app = StreamlitHandler(uploaded_file, epochs, size_limit)
            app.prepare_data()

            thread = threading.Thread(target=app.train_and_infer)
            thread.start()
            app.show_progress_bar(thread)
            app.download_artifacts()
        else:
            st.warning("Please upload a CSV file to proceed.")


if __name__ == "__main__":
    main()
