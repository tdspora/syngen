import traceback
from typing import Optional, Dict
import pandas as pd
from loguru import logger

from syngen.ml.train_chain import RootHandler
from syngen.ml.train_chain import VaeInferHandler


class TrainStrategy:
    def __init__(
            self,
            paths: Dict[str, str],
            handler: RootHandler,
    ):
        self.tmp_store_path = paths['tmp_store_path']
        self.handler = handler

    def run(
            self,
            data: pd.DataFrame,
            epochs: int,
            row_subset: Optional[int],
            batch_size: int,
            drop_null: bool,
    ):
        logger.debug(f"Train model with parameters: epochs={epochs}, drop_null={drop_null}")
        try:
            self.handler.handle(
                data,
                epochs=epochs,
                batch_size=batch_size,
                row_subset=row_subset,
                drop_null=drop_null
            )

        except Exception as e:
            logger.info(f"Training of the table - {self.handler.table_name} failed on running stage.")
            logger.error(e)
            logger.error(traceback.format_exc())
            raise

        logger.info(f"Training of the table - {self.handler.table_name} was completed")


class InferStrategy:
    def __init__(
            self,
            size: int,
            run_parallel: bool,
            metadata_path: Optional[str],
            print_report: bool,
            batch_size: Optional[int],
            handler: VaeInferHandler,
    ):
        self.handler = handler
        self.size = size
        self.run_parallel = run_parallel
        self.print_report = print_report
        self.metadata_path = metadata_path
        self.batch_size = batch_size

    def run(self):
        self.handler.handle(
            size=self.size,
            run_parallel=self.run_parallel,
            batch_size=self.batch_size,
            print_report=self.print_report,
            metadata_path=self.metadata_path,
        )
