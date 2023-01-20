from abc import ABC, abstractmethod
from typing import Optional, Dict
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from loguru import logger
from syngen.ml.data_loaders import DataLoader
from syngen.ml.train_chain import RootHandler
from syngen.ml.reporters import Report, AccuracyReporter, SampleAccuracyReporter
from syngen.ml.config import TrainConfig, InferConfig
from syngen.ml.train_chain import VaeTrainHandler, VaeInferHandler
from syngen.ml.strategies import TrainStrategy, InferStrategy
from syngen.ml.vae import VanillaVAEWrapper


class Interface(ABC):
    """
    Abstract class for the interface of training or infer process
    """
    def __init__(self):
        self.handler = None
        self.config = None
        self.strategy = None
        self.metadata = None
        self.table_name = None

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_config(self):
        pass

    @abstractmethod
    def set_handler(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_strategy(self):
        pass

    def set_reporters(self):
        """
        Set up reporter which used in order to create the sampling report during training process
        """
        if isinstance(self.config, TrainConfig):
            if self.config.print_report and self.config.row_limit:
                sample_reporter = SampleAccuracyReporter(
                    metadata={"table_name": self.config.table_name},
                    paths=self.config.set_paths()
                )
                Report().register_reporter(sample_reporter)

        if self.config.print_report:
            accuracy_reporter = AccuracyReporter(
                metadata={"table_name": self.config.table_name},
                paths=self.config.set_paths()
            )
            Report().register_reporter(accuracy_reporter)

        return self

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


class TrainInterface(Interface, ABC):
    """
    Class of the interface of training process
    """

    def set_config(self, **kwargs):
        """
        Set up configuration for training process
        """
        self.config = TrainConfig(**kwargs)
        return self

    def set_handler(self, schema, *args):
        """
        Set up the handler which used in training process
        """
        paths = self.config.set_paths()

        root_handler = RootHandler(
            metadata=self.metadata,
            table_name=self.config.table_name,
            paths=paths
        )

        vae_handler = VaeTrainHandler(
            metadata=self.metadata,
            table_name=self.config.table_name,
            schema=schema,
            paths=paths,
            wrapper_name=VanillaVAEWrapper.__name__
        )

        root_handler.set_next(vae_handler)
        self.handler = root_handler
        return self

    def set_strategy(self, **kwargs):
        """
        Set up the strategy for training process
        """
        self.strategy = TrainStrategy(**kwargs)
        return self

    def run(
            self,
            metadata,
            source: str,
            epochs: int,
            drop_null: bool,
            row_limit: int,
            table_name: str,
            metadata_path: str,
            print_report: bool,
            batch_size: int
    ):
        """
        Launch the training process
        """
        self.set_config(
            source=source,
            epochs=epochs,
            drop_null=drop_null,
            row_limit=row_limit,
            table_name=table_name,
            metadata_path=metadata_path,
            print_report=print_report,
            batch_size=batch_size
        )

        data, schema = DataLoader(source).load_data()
        # remove completely empty columns
        data = data.dropna(how="all", axis=1)
        if schema is not None:
            schema["fields"] = {
                column: data_type for column, data_type in schema.get("fields", {}).items() if column in data.columns
            }

        self.set_reporters().\
            set_metadata(metadata).\
            set_handler(schema).\
            set_strategy(
            paths=self.config.set_paths(),
            handler=self.handler
        )

        logger.info("Generator: 'vae', mode: 'train'")

        self.strategy.run(
            data,
            epochs=self.config.epochs,
            row_subset=self.config.row_limit,
            batch_size=self.config.batch_size,
            drop_null=self.config.drop_null
        )


class InferInterface(Interface):
    """
    Class of the interface of infer process
    """
    def set_config(self, **kwargs):
        """
        Set up the configuration for infer process
        """
        self.config = InferConfig(**kwargs)
        return self

    def set_handler(self):
        """
        Set up the handler which used in infer process
        """

        self.handler = VaeInferHandler(
            metadata=self.metadata,
            table_name=self.config.table_name,
            paths=self.config.set_paths(),
            wrapper_name=VanillaVAEWrapper.__name__,
            random_seed=self.config.random_seed
        )
        return self

    def set_strategy(self, **kwargs):
        """
        Set up the strategy for infer process
        """
        self.strategy = InferStrategy(**kwargs)

    def run(
            self,
            metadata: Optional[Dict],
            size: int,
            table_name: str,
            metadata_path: str,
            run_parallel: bool,
            batch_size: Optional[int],
            random_seed: Optional[int],
            print_report: bool,
            both_keys: bool
    ):
        """
        Launch the infer process
        """
        self.set_config(
            size=size,
            table_name=table_name,
            metadata_path=metadata_path,
            run_parallel=run_parallel,
            batch_size=batch_size,
            random_seed=random_seed,
            print_report=print_report,
            both_keys=both_keys,
        ).\
            set_reporters(). \
            set_metadata(metadata).\
            set_handler().\
            set_strategy(
                size=self.config.size,
                run_parallel=self.config.run_parallel,
                batch_size=self.config.batch_size,
                metadata_path=self.config.metadata_path,
                random_seed=self.config.random_seed,
                handler=self.handler
            )

        self.strategy.run()
