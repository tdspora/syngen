from abc import ABC, abstractmethod
import os
import traceback
from loguru import logger
from copy import deepcopy

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from syngen.ml.handlers import RootHandler
from syngen.ml.reporters import Report, AccuracyReporter, SampleAccuracyReporter
from syngen.ml.config import TrainConfig, InferConfig
from syngen.ml.handlers import LongTextsHandler, VaeTrainHandler, VaeInferHandler
from syngen.ml.vae import VanillaVAEWrapper
from syngen.ml.data_loaders import BinaryLoader
from syngen.ml.mlflow_tracker.mlflow_tracker import MlflowTracker
from syngen.ml.utils import get_initial_table_name, clean_up_metadata


class Strategy(ABC):
    """
    Abstract class for the strategies of training or infer process
    """

    def __init__(self):
        self.handler = None
        self.config = None
        self.metadata = None

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


class TrainStrategy(Strategy, ABC):
    """
    Class of a strategy defined in a training process
    """

    def _save_training_config(self):
        metadata = deepcopy(self.config.metadata)
        self.config.metadata = clean_up_metadata(metadata=metadata)

        BinaryLoader(
            path=self.config.paths["train_config_pickle_path"]
        ).save_data(data=self.config)

    def set_config(self, **kwargs):
        """
        Set up configuration for training process
        """
        configuration = TrainConfig(**kwargs)
        self.config = configuration
        self.metadata = deepcopy(self.config.metadata)
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
            reports=self.config.reports,
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
        # TODO: now the reporter isn't added if the flatten metadata exists
        # This should be refactored in the future
        table_name = self.config.table_name
        flatten_metadata_exists = os.path.exists(self.config.paths["path_to_flatten_metadata"])
        if (
                not table_name.endswith("_fk")
                and "sample" in self.config.reports
                and not flatten_metadata_exists
        ):
            sample_reporter = SampleAccuracyReporter(
                table_name=get_initial_table_name(table_name),
                paths=self.config.paths,
                config=self.config.to_dict(),
                metadata=self.metadata,
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
            self.set_config(**kwargs)
            self.add_reporters().add_handler()
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
    Class of a strategy defined in the infer process
    """

    def set_config(self, **kwargs):
        """
        Set up the configuration for infer process
        """
        configuration = InferConfig(**kwargs)
        self.config = configuration
        self.metadata = deepcopy(self.config.metadata)
        return self

    def add_handler(self, type_of_process: str):
        """
        Set up the handler which used in infer process
        """
        self.handler = VaeInferHandler(
            metadata_path=self.config.metadata_path,
            metadata=self.metadata,
            table_name=self.config.table_name,
            paths=self.config.paths,
            wrapper_name=VanillaVAEWrapper.__name__,
            size=self.config.size,
            random_seed=self.config.random_seed,
            batch_size=self.config.batch_size,
            run_parallel=self.config.run_parallel,
            reports=self.config.reports,
            log_level=self.config.log_level,
            type_of_process=type_of_process,
            loader=self.config.loader
        )
        return self

    def add_reporters(self):
        table_name = self.config.table_name
        if (
                not table_name.endswith("_fk") and
                any([item in ["accuracy", "metrics_only"] for item in self.config.reports])
        ):
            accuracy_reporter = AccuracyReporter(
                table_name=get_initial_table_name(table_name),
                paths=self.config.paths,
                config=self.config.to_dict(),
                metadata=self.metadata,
                loader=self.config.loader,
            )
            Report().register_reporter(table=table_name, reporter=accuracy_reporter)

        return self

    def run(self, **kwargs):
        """
        Launch the infer process
        """
        table_name = kwargs["table_name"]
        try:
            self.set_config(**kwargs)
            MlflowTracker().log_params(self.config.to_dict())
            self.add_reporters()
            self.add_handler(type_of_process=kwargs["type_of_process"])
            self.handler.handle()
        except Exception:
            logger.error(
                f"Generation of the table - \"{table_name}\" failed on running stage.\n"
                f"The traceback of the error - {traceback.format_exc()}"
            )
            raise
        else:
            logger.info(
                f"Synthesis of the table - \"{table_name}\" was completed. "
                f"Synthetic data saved in {self.handler.paths['path_to_merged_infer']}"
            )
