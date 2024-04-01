import os
from datetime import datetime
import traceback
from queue import Queue
import sys

from loguru import logger
from slugify import slugify
import streamlit as st
from streamlit.elements.widgets.file_uploader import UploadedFile

from syngen.ml.worker import Worker
from syngen.ml.utils import fetch_log_message, ProgressBarHandler

UPLOAD_DIRECTORY = "uploaded_files"


class StreamlitHandler:
    """
    A class for handling the Streamlit app
    """
    def __init__(
            self,
            epochs: int,
            size_limit: int,
            print_report: bool,
            uploaded_file: UploadedFile
    ):
        self.epochs = epochs
        self.size_limit = size_limit
        self.print_report = print_report
        self.uploaded_file = uploaded_file
        self.file_name = self.uploaded_file.name
        self.table_name = os.path.splitext(self.file_name)[0]
        self.file_path = os.path.join(UPLOAD_DIRECTORY, self.file_name)
        self.sl_table_name = slugify(self.table_name)
        self.progress_handler = ProgressBarHandler()
        self.log_queue = Queue()
        self.log_error_queue = Queue()
        self.path_to_logs = (f"model_artifacts/tmp_store/{self.sl_table_name}_"
                             f"{slugify(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}.log")
        self.path_to_generated_data = (f"model_artifacts/tmp_store/{self.sl_table_name}/"
                                       f"merged_infer_{self.sl_table_name}.csv")
        self.path_to_report = (f"model_artifacts/tmp_store/{self.sl_table_name}/"
                               f"draws/accuracy_report.html")
        os.environ["SUCCESS_LOG_FILE"] = self.path_to_logs

    def reset_log_queues(self):
        """
        Clean up log queues
        """
        self.log_queue = Queue()
        self.log_error_queue = Queue()

    def set_logger(self):
        """
        Set a logger to see logs, and collect log messages
        with the log level - 'INFO' in a log file and stdout
        """
        logger.remove()
        logger.add(self.console_sink, colorize=True, level="INFO")
        logger.add(self.file_sink, level="INFO")

    def console_sink(self, message):
        """
        Put log messages to a log queue
        """
        log_message = fetch_log_message(message)
        sys.stdout.write(log_message + "\n")
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
                type_of_process="test"
            )
            ProgressBarHandler().set_progress(0.01)
            worker.launch_train()
            logger.info("Model training completed")
        except Exception as e:
            error_message = f"Error during train: {traceback.format_exc()}"
            logger.error(error_message)
            self.log_error_queue.put(e)

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
        except Exception as e:
            error_message = f"Error during infer: {traceback.format_exc()}"
            logger.error(error_message)
            self.log_error_queue.put(e)

    def train_and_infer(self):
        """
        Launch a model training and data generation
        """
        self.set_logger()
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
                "Download report",
                self.path_to_report,
                f"accuracy_report_{self.sl_table_name}.html"
            )
