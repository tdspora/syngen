import os
from datetime import datetime
import traceback
from queue import Queue

from loguru import logger
from slugify import slugify
import streamlit as st

from syngen.ml.worker import Worker
from syngen.ml.utils import fetch_log_message, ProgressBarHandler
import streamlit.components.v1 as components
from syngen.streamlit_app.utils import (
    show_data,
    get_running_status,
    set_session_state,
    cleanup_artifacts,
)

UPLOAD_DIRECTORY = "uploaded_files"
TIMESTAMP = slugify(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class StreamlitHandler:
    """
    A class for handling the Streamlit app
    """

    def __init__(self, uploaded_file, epochs: int, size_limit: int, print_report: bool):
        self.log_queue = Queue()
        self.progress_handler = ProgressBarHandler()
        self.log_error_queue = Queue()
        self.epochs = epochs
        self.size_limit = size_limit
        self.print_report = print_report
        self.file_name = uploaded_file.name
        self.table_name = os.path.splitext(self.file_name)[0]
        self.file_path = os.path.join(UPLOAD_DIRECTORY, self.file_name)
        self.sl_table_name = slugify(self.table_name)
        self.path_to_generated_data = (f"model_artifacts/tmp_store/{self.sl_table_name}/"
                                       f"merged_infer_{self.sl_table_name}.csv")
        self.path_to_report = (f"model_artifacts/tmp_store/{self.sl_table_name}/"
                               f"draws/accuracy_report.html")

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
        path_to_logs = f"model_artifacts/tmp_store/{self.sl_table_name}_{TIMESTAMP}.log"
        os.environ["SUCCESS_LOG_FILE"] = path_to_logs
        os.makedirs(os.path.dirname(path_to_logs), exist_ok=True)
        with open(path_to_logs, "a") as log_file:
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

    @staticmethod
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

    def open_report(self):
        """
        Open the accuracy report in the iframe
        """
        if os.path.exists(self.path_to_report) and self.print_report:
            with open(self.path_to_report, "r") as report:
                report_content = report.read()
            with st.expander("View the accuracy report"):
                components.html(report_content, 680, 1000, True)

    def generate_buttons(self):
        """
        Generate download buttons for downloading artifacts
        """
        self.generate_button(
            "Download generated data",
            self.path_to_generated_data,
            f"generated_data_{self.sl_table_name}.csv"
        )
        self.generate_button(
            "Download logs",
            os.getenv("SUCCESS_LOG_FILE", ""),
            f"logs_{self.sl_table_name}.log"
        )
        if self.print_report:
            self.generate_button(
                "Download the accuracy report",
                self.path_to_report,
                f"accuracy_report_{self.sl_table_name}.html"
            )
            self.open_report()

