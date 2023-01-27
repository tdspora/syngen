from typing import Tuple, Optional, Dict
from abc import ABC, abstractmethod
import os
import math

import pandas as pd
import numpy as np
from loguru import logger
from numpy.random import seed, choice
from pathos.multiprocessing import ProcessingPool
import dill

from syngen.ml.vae import *
from syngen.ml.data_loaders import DataLoader


class AbstractHandler(ABC):
    @abstractmethod
    def set_next(self, handler):
        pass

    @abstractmethod
    def handle(self, data: pd.DataFrame, **kwargs):
        pass


class BaseHandler(AbstractHandler):
    def __init__(self, metadata: dict, paths: dict, table_name: str):
        self.metadata = metadata
        self.paths = paths
        self.table_name = table_name
        if self.table_name is None:
            raise KeyError("No table name was provided.")

    _next_handler: AbstractHandler = None

    def set_next(self, handler: AbstractHandler) -> AbstractHandler:
        self._next_handler = handler
        return handler

    @abstractmethod
    def handle(self, data: pd.DataFrame, **kwargs):

        if self._next_handler:
            return self._next_handler.handle(data, **kwargs)

        return None

    @staticmethod
    def create_wrapper(cls_name, data: pd.DataFrame, schema: Optional[Dict], **kwargs):
        return globals()[cls_name](
            data, schema, kwargs["metadata"], kwargs["table_name"], kwargs["paths"], kwargs["process"]
        )


class RootHandler(BaseHandler):
    def __init__(self, metadata: dict, paths: dict, table_name: str):
        super().__init__(metadata, paths, table_name)

    def handle(self, **kwargs):
        data, schema = DataLoader(self.paths["input_data_path"]).load_data()
        return super().handle(data, **kwargs)


class VaeTrainHandler(BaseHandler):
    def __init__(
            self,
            metadata: dict,
            paths: dict,
            table_name: str,
            schema: Optional[Dict],
            wrapper_name: str,
            epochs: int,
            row_subset: int,
            drop_null: bool,
            batch_size: int
    ):
        super().__init__(metadata, paths, table_name)
        self.wrapper_name = wrapper_name
        self.schema = schema
        self.epochs = epochs
        self.row_subset = row_subset
        self.drop_null = drop_null
        self.batch_size = batch_size
        self.state_path = self.paths["state_path"]

    def __fit_model(
            self,
            data: pd.DataFrame
    ):
        os.makedirs(self.state_path, exist_ok=True)
        logger.info("Start VAE training")
        if data is None:
            logger.error("For mode = 'train' path must be provided")
            raise ValueError("Can't read data from path or path is None")

        self.model = self.create_wrapper(
            self.wrapper_name,
            data,
            self.schema,
            metadata=self.metadata,
            table_name=self.table_name,
            paths=self.paths,
            process="train"
        )

        self.model.batch_size = min(self.batch_size, len(data))

        logger.debug(
            f"Train model with parameters: epochs={self.epochs}, row_subset={self.row_subset}, "
            f"drop_null={self.drop_null}, batch_size={self.batch_size}")
        self.model.fit_on_df(
            data,
            epochs=self.epochs,
        )

        self.model.save_state(self.state_path)
        logger.info("Finished VAE training")

    def handle(self, data: pd.DataFrame, **kwargs):
        self.__fit_model(data)
        return super().handle(data, **kwargs)


class VaeInferHandler(BaseHandler):
    def __init__(
            self,
            metadata: dict,
            metadata_path: Optional[str],
            paths: dict,
            table_name: str,
            wrapper_name: str,
            size: Optional[int],
            random_seed: Optional[int],
            batch_size: int,
            run_parallel: bool,
            print_report: bool
    ):
        super().__init__(metadata, paths, table_name)
        self.metadata_path = metadata_path
        self.random_seed = random_seed
        self.random_seeds_list = []
        if random_seed:
            seed(random_seed)
        self.vae = None
        self.size = size
        self.batch_size = batch_size
        self.run_parallel = run_parallel
        self.print_report = print_report
        self.wrapper_name = wrapper_name
        self.vae_state_path = self.paths["state_path"]
        self.path_to_merged_infer = self.paths["path_to_merged_infer"]
        self.fk_kde_path = self.paths["fk_kde_path"]

    def _prepare_dir(self):
        tmp_store_path = self.paths["tmp_store_path"]
        os.makedirs(tmp_store_path, exist_ok=True)

    def _is_pk(self):
        is_pk = self.table_name.endswith("_pk")
        return is_pk

    def _concat_slices_with_unique_pk(self, df_slices: list):
        if self.metadata and self.table_name in self.metadata:
            config_of_keys = self.metadata.get(self.table_name).get("keys", {})
            for key in config_of_keys.keys():
                column = config_of_keys.get(key).get("columns")[0]
                if config_of_keys.get(key).get("type") == "PK" and not isinstance(df_slices[0][column][0], str):
                    cumm_len = 0
                    for i, frame in enumerate(df_slices):
                        frame[column] = frame[column].map(lambda pk_val: pk_val + cumm_len)
                        cumm_len += len(frame)
        return pd.concat(df_slices, ignore_index=True)

    def run_separate(self, params: Tuple):
        i, size = params

        if self.random_seed:
            seed(self.random_seeds_list[i])

        input_data_existed = DataLoader(self.paths["input_data_path"]).has_existed_path

        if input_data_existed:
            data, schema = DataLoader(self.paths["input_data_path"]).load_data()
        else:
            data = pd.DataFrame()
            schema = None

        self.vae = self.create_wrapper(
            self.wrapper_name,
            data,
            schema,
            metadata={"table_name": self.table_name},
            table_name=self.table_name,
            paths=self.paths,
            process="infer"
        )
        self.vae.load_state(self.vae_state_path)
        synthetic_infer = self.vae.predict_sampled_df(size)
        return synthetic_infer

    @staticmethod
    def split_by_batches(size, nodes):
        quote = int(size / nodes)
        data = [quote] * nodes
        data.append((size - nodes * quote) + data.pop())
        return data

    def run(self, size: int, run_parallel: bool):
        logger.info("Start data synthesis")
        if run_parallel:
            pool = ProcessingPool()
            if self.random_seed:
                self.random_seeds_list = choice(
                    range(0, max(100, pool.nodes)), pool.nodes, replace=False
                )

            frames = pool.map(
                self.run_separate, enumerate(self.split_by_batches(size, pool.nodes))
            )
            generated = self._concat_slices_with_unique_pk(frames)
        else:
            if self.random_seed:
                self.random_seeds_list = [self.random_seed]
            generated = self.run_separate((0, size))
        return generated

    def kde_gen(self, pk_table, pk_column_label, size, fk_label):
        pk = pk_table[pk_column_label]

        try:
            with open(f"{self.fk_kde_path}{fk_label}.pkl", "rb") as file:
                kde = dill.load(file)
            pk = pk.dropna()
            numeric_pk = np.arange(len(pk)) if pk.dtype == "object" else pk
            fk_pdf = np.maximum(kde.evaluate(numeric_pk), 1e-12)
            synth_fk = np.random.choice(pk, size=size, p=fk_pdf / sum(fk_pdf), replace=True)
            synth_fk = pd.DataFrame({fk_label: synth_fk}).reset_index(drop=True)

        except FileNotFoundError:
            logger.warning(f"The mapper for the {fk_label} text key is not found. Making simple sampling")
            synth_fk = pk.sample(size, replace=True).reset_index(drop=True)
            synth_fk.rename(fk_label, inplace=True)

        return synth_fk

    def generate_keys(self, generated, size, metadata, table_name):
        metadata_of_table = metadata.get(table_name)
        if "keys" not in metadata_of_table:
            return None
        config_of_keys = metadata_of_table.get("keys")
        for key in config_of_keys.keys():
            if config_of_keys.get(key).get("type") == "FK":
                pk_table = config_of_keys.get(key).get("references").get("table")
                pk_path = f"model_artifacts/tmp_store/{pk_table}/merged_infer_{pk_table}.csv"
                if not os.path.exists(pk_path):
                    raise FileNotFoundError(
                        "The table with a primary key specified in the metadata file does not "
                        "exist or is not trained. Ensure that the metadata contains the "
                        "name of referenced table with a primary key in the foreign key declaration section."
                    )
                pk_table_data, pk_table_schema = DataLoader(pk_path).load_data()
                pk_column_label = config_of_keys.get(key).get("references").get("columns")[0]
                logger.info(f"The {pk_column_label} assigned as a foreign_key feature")

                synth_fk = self.kde_gen(pk_table_data, pk_column_label, size, config_of_keys.get(key).get("columns")[0])
                generated = generated.reset_index(drop=True)

                null_column_name = f"{key}_null"
                if null_column_name in generated.columns:
                    not_null_column_mask = generated[null_column_name].astype("float64") <= 0.5
                    synth_fk = synth_fk.where(not_null_column_mask, np.nan)
                    generated = generated.drop(null_column_name, axis=1)

                generated = pd.concat([generated, synth_fk], axis=1)
        return generated

    def handle(
            self,
            **kwargs
    ):
        self._prepare_dir()

        batch_num = math.ceil(self.size / self.batch_size)
        logger.debug(
            f"Infer model with parameters: size={self.size}, run_parallel={self.run_parallel}, "
            f"batch_size={self.batch_size}, random_seed={self.random_seed}, print_report={self.print_report}"
        )
        logger.info(f"Total of {batch_num} batch(es)")
        batches = self.split_by_batches(self.size, batch_num)
        prepared_batches = [self.run(batch, self.run_parallel) for batch in batches]
        prepared_data = self._concat_slices_with_unique_pk(prepared_batches) if len(prepared_batches) > 0 else pd.DataFrame()

        is_pk = self._is_pk()
        if self.metadata_path is not None:
            if not is_pk:
                generated_data = self.generate_keys(prepared_data, self.size, self.metadata, self.table_name)
                if generated_data is None:
                    prepared_data.to_csv(self.path_to_merged_infer, index=False)
                else:
                    generated_data.to_csv(self.path_to_merged_infer, index=False)
            else:
                prepared_data.to_csv(self.path_to_merged_infer, index=False)
        if self.metadata_path is None:
            prepared_data.to_csv(self.path_to_merged_infer, index=False)
