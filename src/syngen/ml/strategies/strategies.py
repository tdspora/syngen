from abc import ABC, abstractmethod
import os
import traceback
from loguru import logger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from syngen.ml.handlers import RootHandler
from syngen.ml.reporters import Report, AccuracyReporter, SampleAccuracyReporter
from syngen.ml.config import TrainConfig, InferConfig
from syngen.ml.handlers import LongTextsHandler, VaeTrainHandler, VaeInferHandler
from syngen.ml.vae import VanillaVAEWrapper
from syngen.ml.data_loaders import BinaryLoader
from syngen.ml.mlflow_tracker.mlflow_tracker import MlflowTracker
from syngen.ml.utils import get_initial_table_name


class Strategy(ABC):
    """
    Abstract class for the strategies of training or infer process
    """

    def __init__(self):
        self.handler = None
        self.config = None
        self.metadata = None
        self.table_name = None

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_config(self):
        pass

    @abstractmethod
    def add_handler(self, *args, **kwargs):
        pass

    @abstractmethod
    def add_reporters(self):
        """
        Set up reporter which used in order to create the sampling report during training process
        """
        pass

    def set_metadata(self, metadata):
        if metadata:
            self.metadata = metadata
            return self
        if self.config.table_name:
            metadata = {"table_name": self.config.table_name}
            self.metadata = metadata
            return self
        else:
            raise AttributeError("Either table name or path to metadata MUST be provided")


class TrainStrategy(Strategy, ABC):
    """
    Class of the strategies of training process
    """

    def _save_training_config(self):
        BinaryLoader(
            path=self.config.paths["train_config_pickle_path"]
        ).save_data(data=self.config)

    def set_config(self, **kwargs):
        """
        Set up configuration for training process
        """
        configuration = TrainConfig(**kwargs)
        self.config = configuration
        self.config.preprocess_data()
        self._save_training_config()
        return self

    def add_handler(self):
        """
        Set up the handler which used in training process
        """
        root_handler = RootHandler(
            metadata=self.metadata,
            table_name=self.config.table_name,
            paths=self.config.paths,
            loader=self.config.loader
        )

        vae_handler = VaeTrainHandler(
            metadata=self.metadata,
            table_name=self.config.table_name,
            schema=self.config.schema,
            paths=self.config.paths,
            wrapper_name=VanillaVAEWrapper.__name__,
            epochs=self.config.epochs,
            row_subset=self.config.row_subset,
            drop_null=self.config.drop_null,
            batch_size=self.config.batch_size,
            print_report=self.config.print_report,
            type_of_process="train",
        )

        long_text_handler = LongTextsHandler(
            metadata=self.metadata,
            table_name=self.config.table_name,
            schema=self.config.schema,
            paths=self.config.paths,
        )

        root_handler.set_next(vae_handler).set_next(long_text_handler)

        self.handler = root_handler
        return self

    def add_reporters(self, **kwargs):
        table_name = self.config.table_name
        source = self.config.paths["source_path"]
        loader = self.config.loader
        if (
                not table_name.endswith("_fk")
                and source is not None
                and loader is None
                and os.path.exists(source)
                and self.config.print_report
        ):
            sample_reporter = SampleAccuracyReporter(
                table_name=get_initial_table_name(table_name),
                paths=self.config.paths,
                config=self.config.to_dict(),
            )
            Report().register_reporter(table=table_name, reporter=sample_reporter)

        return self

    def run(self, **kwargs):
        """
        Launch the training process
        """
        try:
            table = kwargs["table_name"]
            # Start the separate run for the preprocess stage
            # included preprocessing of the original data, identification of data types of columns,
            # fit and transform of the assigned features
            MlflowTracker().start_run(
                run_name=f"{table}-PREPROCESS",
                tags={"table_name": table, "process": "preprocess"},
            )
            self.set_config(
                source=kwargs["source"],
                epochs=kwargs["epochs"],
                drop_null=kwargs["drop_null"],
                row_limit=kwargs["row_limit"],
                table_name=table,
                metadata_path=kwargs["metadata_path"],
                print_report=kwargs["print_report"],
                batch_size=kwargs["batch_size"],
                loader=kwargs["loader"]
            )

            self.add_reporters().set_metadata(kwargs["metadata"]).add_handler()
            self.handler.handle()
            # End the separate run for the training stage
            MlflowTracker().end_run()
        except Exception:
            logger.error(
                f"Training of the table - \"{kwargs['table_name']}\" failed on running stage.\n"
                f"The traceback of the error - {traceback.format_exc()}"
            )
            raise
        else:
            logger.info(f"Training of the table - {kwargs['table_name']} was completed")


class InferStrategy(Strategy):
    """
    Class of the strategies of infer process
    """

    def set_config(self, **kwargs):
        """
        Set up the configuration for infer process
        """
        configuration = InferConfig(**kwargs)
        self.config = configuration
        return self

    def add_handler(self, type_of_process: str):
        """
        Set up the handler which used in infer process
        """

        self.handler = VaeInferHandler(
            metadata=self.metadata,
            metadata_path=self.config.metadata_path,
            table_name=self.config.table_name,
            paths=self.config.paths,
            wrapper_name=VanillaVAEWrapper.__name__,
            size=self.config.size,
            random_seed=self.config.random_seed,
            batch_size=self.config.batch_size,
            run_parallel=self.config.run_parallel,
            print_report=self.config.print_report,
            get_infer_metrics=self.config.get_infer_metrics,
            log_level=self.config.log_level,
            type_of_process=type_of_process,
            loader=self.config.loader
        )
        return self

    def add_reporters(self):
        table_name = self.config.table_name
        if (
                not table_name.endswith("_fk") and
                (self.config.print_report or self.config.get_infer_metrics)
        ):
            accuracy_reporter = AccuracyReporter(
                table_name=get_initial_table_name(table_name),
                paths=self.config.paths,
                config=self.config.to_dict(),
                loader=self.config.loader
            )
            Report().register_reporter(table=table_name, reporter=accuracy_reporter)

        return self

    def run(self, **kwargs):
        """
        Launch the infer process
        """
        try:

            self.set_config(
                destination=kwargs["destination"],
                size=kwargs["size"],
                table_name=kwargs["table_name"],
                metadata_path=kwargs["metadata_path"],
                run_parallel=kwargs["run_parallel"],
                batch_size=kwargs["batch_size"],
                random_seed=kwargs["random_seed"],
                print_report=kwargs["print_report"],
                get_infer_metrics=kwargs["get_infer_metrics"],
                log_level=kwargs["log_level"],
                both_keys=kwargs["both_keys"],
                loader=kwargs["loader"]
            )

            MlflowTracker().log_params(self.config.to_dict())

            self.add_reporters().set_metadata(kwargs["metadata"]).add_handler(
                type_of_process=kwargs["type_of_process"]
            )
            self.handler.handle()
        except Exception:
            logger.error(
                f"Generation of the table - \"{kwargs['table_name']}\" failed on running stage.\n"
                f"The traceback of the error - {traceback.format_exc()}"
            )
            raise
        else:
            logger.info(
                f"Synthesis of the table - {kwargs['table_name']} was completed. "
                f"Synthetic data saved in {self.handler.paths['path_to_merged_infer']}"
            )
