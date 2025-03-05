from typing import Literal
import os
from datetime import datetime
from queue import Queue
import sys
import traceback

from loguru import logger
from slugify import slugify
import streamlit as st
from streamlit.elements.widgets.file_uploader import UploadedFile
import streamlit.components.v1 as components

from syngen.ml.worker import Worker
from syngen.ml.utils import fetch_log_message, ProgressBarHandler


UPLOAD_DIRECTORY = "uploaded_files"
TIMESTAMP = slugify(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class StreamlitHandler:
    """
    A class for handling the Streamlit app
    """
    def __init__(
            self,
            epochs: int,
            size_limit: int,
            reports: bool,
            uploaded_file: UploadedFile
    ):
        self.epochs = epochs
        self.size_limit = size_limit
        self.reports = ["accuracy"] if reports else []
        self.uploaded_file = uploaded_file
        self.file_name = self.uploaded_file.name
        self.table_name = os.path.splitext(self.file_name)[0]
        self.file_path = os.path.join(UPLOAD_DIRECTORY, self.file_name)
        self.sl_table_name = slugify(self.table_name)
        self.progress_handler = ProgressBarHandler()
        self.log_queue = Queue()
        self.log_error_queue = Queue()
        self.path_to_generated_data = (f"model_artifacts/tmp_store/{self.sl_table_name}/"
                                       f"merged_infer_{self.sl_table_name}.csv")
        self.path_to_report = (f"model_artifacts/tmp_store/{self.sl_table_name}/"
                               f"reports/accuracy_report.html")
        self.train_settings = {
            "source": self.file_path,
            "epochs": self.epochs,
            "row_limit": 10000,
            "drop_null": False,
            "batch_size": 32,
            "reports": []
        }
        self.infer_settings = {
            "size": self.size_limit,
            "batch_size": 32,
            "run_parallel": False,
            "random_seed": None,
            "reports": self.reports,
        }

    def set_logger(self):
        """
        Set a logger to see logs, and collect log messages
        with the log level - 'INFO' in a log file and stdout
        """
        logger.remove()
        logger.add(self.console_sink, colorize=True, level="INFO")
        logger.add(self.file_sink, level="INFO")

    def console_sink(self, message: str):
        """
        Put log messages to a log queue
        """
        log_message = fetch_log_message(message)
        sys.stdout.write(log_message + "\n")
        self.log_queue.put(log_message)

    def file_sink(self, message: str):
        """
        Write log messages to a log file
        """
        path_to_logs = f"model_artifacts/tmp_store/{self.sl_table_name}_{TIMESTAMP}.log"
        os.environ["SUCCESS_LOG_FILE"] = path_to_logs
        os.makedirs(os.path.dirname(path_to_logs), exist_ok=True)
        with open(path_to_logs, "a") as log_file:
            log_message = fetch_log_message(message)
            log_file.write(log_message + "\n")

    def _get_worker(self, process_type: Literal["train", "infer"]):
        """
        Get a Worker object

        Parameters
        ----------
        process_type : str
            The type of process ("train" or "infer").

        Returns
        -------
        Worker
            The Worker object.
        """
        worker = Worker(
            table_name=self.table_name,
            settings=self.train_settings if process_type == "train" else self.infer_settings,
            metadata_path=None,
            log_level="INFO",
            type_of_process=process_type
        )
        return worker

    def train_model(self):
        """
        Launch a model training
        """
        try:
            logger.info("Starting model training...")
            worker = self._get_worker("train")
            ProgressBarHandler().set_progress(0.01)
            worker.launch_train()
            logger.info("Model training completed")
        except Exception as e:
            error_message = (f"The error raised during the training process - "
                             f"{traceback.format_exc()}")
            logger.error(error_message)
            self.log_error_queue.put(e)
            raise e

    def infer_model(self):
        """
        Launch a data generation
        """
        try:
            logger.info("Starting data generation...")
            worker = self._get_worker("infer")
            worker.launch_infer()
            logger.info("Data generation completed")
        except Exception as e:
            error_message = (f"The error raised during the inference process - "
                             f"{traceback.format_exc()}")
            logger.error(error_message)
            self.log_error_queue.put(e)
            raise e

    def set_monitoring(self):
        """
        Reset the progress bar and set up the logger
        """
        self.progress_handler.reset_instance()
        self.set_logger()

    def train_and_infer(self):
        """
        Launch a model training and data generation
        """
        self.set_monitoring()
        self.train_model()
        self.infer_model()

    @staticmethod
    def generate_button(label: str, path_to_file: str, download_name: str):
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
        if os.path.exists(self.path_to_report) and self.reports:
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
        if self.reports:
            self.generate_button(
                "Download the accuracy report",
                self.path_to_report,
                f"accuracy_report_{self.sl_table_name}.html"
            )
            self.open_report()
