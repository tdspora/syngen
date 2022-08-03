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
            keys_mode: bool = False
    ):
        self.tmp_store_path = paths['tmp_store_path']
        self.keys_mode = keys_mode
        self.handler = handler

    def run(
            self,
            data: pd.DataFrame,
            epochs: int,
            row_subset: Optional[int],
            batch_size: int,
            dropna: bool,
            keys_mode: Optional[bool]
    ):
        logger.debug(f"Train model with parameters: epochs={epochs}, dropna={dropna}")
        try:
            self.handler.handle(
                data,
                epochs=epochs,
                batch_size=batch_size,
                row_subset=row_subset,
                keys_mode=keys_mode,
                dropna=dropna
            )

        except Exception as e:
            logger.info("Training failed on running stage.")
            logger.error(e)
            logger.error(traceback.format_exc())
            raise

        logger.info("Training was completed")


class InferStrategy:
    def __init__(
            self,
            size: int,
            run_parallel: bool,
            metadata_path: Optional[str],
            print_report: bool,
            batch_size: Optional[int],
            handler: VaeInferHandler,
            keys_mode: Optional[bool] = None
    ):
        self.handler = handler
        self.size = size
        self.run_parallel = run_parallel
        self.print_report = print_report
        self.keys_mode = keys_mode
        self.metadata_path = metadata_path
        self.batch_size = batch_size

    def run(self):
        self.handler.handle(
            size=self.size,
            run_parallel=self.run_parallel,
            batch_size=self.batch_size,
            print_report=self.print_report,
            keys_mode=self.keys_mode,
            metadata_path=self.metadata_path,
        )
