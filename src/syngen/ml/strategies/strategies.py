import traceback
from typing import Dict
import pandas as pd
from loguru import logger

from syngen.ml.train_chain import RootHandler
from syngen.ml.train_chain import VaeInferHandler


class TrainStrategy:
    def __init__(self, paths: Dict[str, str], handler: RootHandler):
        self.tmp_store_path = paths['tmp_store_path']
        self.handler = handler

    def run(self, data: pd.DataFrame):
        try:
            self.handler.handle(data)

        except Exception as e:
            logger.info(f"Training of the table - {self.handler.table_name} failed on running stage.")
            logger.error(e)
            logger.error(traceback.format_exc())
            raise

        logger.info(f"Training of the table - {self.handler.table_name} was completed")


class InferStrategy:
    def __init__(self, handler: VaeInferHandler):
        self.handler = handler

    def run(self):
        try:
            self.handler.handle()
        except Exception as e:
            logger.info(f"Generation of the table - {self.handler.table_name} failed on running stage.")
            logger.error(e)
            logger.error(traceback.format_exc())
            raise

        logger.info(
            f"Synthesis of the table - {self.handler.table_name} was completed. "
            f"Synthetic data saved in {self.handler.path_to_merged_infer}"
        )
