from typing import Tuple, Optional, Dict, List
from abc import ABC, abstractmethod
import os
import math
from ulid import ULID
from uuid import UUID
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from numpy.random import seed, choice
from pathos.multiprocessing import ProcessingPool
import dill
from scipy.stats import gaussian_kde
from collections import OrderedDict
from tensorflow.keras.preprocessing.text import Tokenizer
from slugify import slugify
from loguru import logger

from syngen.ml.vae import *  # noqa: F403
from syngen.ml.data_loaders import DataLoader
from syngen.ml.utils import fetch_dataset, check_if_features_assigned, generate_uuid
from syngen.ml.context import get_context
from syngen.ml.utils import setup_logger


class AbstractHandler(ABC):
    @abstractmethod
    def set_next(self, handler):
        pass

    @abstractmethod
    def handle(self, data: pd.DataFrame, **kwargs):
        pass


@dataclass
class BaseHandler(AbstractHandler):
    metadata: Dict
    paths: Dict
    table_name: str
    _next_handler: Optional[AbstractHandler] = field(init=False)

    def __post_init__(self):
        self._next_handler = None

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
            data,
            schema,
            metadata=kwargs["metadata"],
            table_name=kwargs["table_name"],
            paths=kwargs["paths"],
            batch_size=kwargs["batch_size"],
            process=kwargs["process"],
        )


@dataclass
class RootHandler(BaseHandler):
    def handle(self, **kwargs):
        data, schema = DataLoader(self.paths["input_data_path"]).load_data()
        return super().handle(data, **kwargs)


@dataclass
class LongTextsHandler(BaseHandler):
    schema: Optional[Dict]

    @staticmethod
    def series_count_words(x):
        return len(str(x).split())

    def _prepare_dir(self):
        os.makedirs(self.paths["no_ml_state_path"], exist_ok=True)

    def _save_no_ml_checkpoints(self, features: Dict):
        with open(f'{self.paths["no_ml_state_path"]}kde_params.pkl', "wb") as file:
            dill.dump(features, file)

    def handle(self, data: pd.DataFrame, **kwargs):
        self._prepare_dir()

        dataset = fetch_dataset(self.paths["dataset_pickle_path"])
        long_text_columns = dataset.long_text_columns

        if len(long_text_columns) > 0:
            features = {}
            for col in long_text_columns:
                tokenizer = Tokenizer(lower=False, char_level=True)
                if type(data[col].dropna().values[0]) is bytes:
                    text_col = data[col].str.decode("utf-8", errors="ignore")
                else:
                    text_col = data[col]
                text_col = text_col.fillna("")
                tokenizer.fit_on_texts(text_col)

                indexes = OrderedDict((k, v) for k, v in tokenizer.word_index.items() if k != " ")
                counts = OrderedDict((k, v) for k, v in tokenizer.word_counts.items() if k != " ")
                ordered_indexes = OrderedDict((k, indexes[k]) for k in counts.keys())
                text_structure = np.array(
                    [text_col.str.len(), text_col.apply(self.series_count_words)]
                )
                noise_to_prevent_singularity = np.random.uniform(
                    low=-1e-4,
                    high=1e-4,
                    size=(text_structure.shape[0], text_structure.shape[1]),
                )
                bw_width = text_structure.shape[1] / text_structure.shape[1] ** 1.3
                kde = gaussian_kde(
                    (text_structure + noise_to_prevent_singularity).astype("float64"),
                    bw_method=bw_width,
                )
                features[col] = {
                    "counts": counts,
                    "indexes": ordered_indexes,
                    "kde": kde,
                }

            self._save_no_ml_checkpoints(features)

        else:
            logger.info("No columns to train kde over found")
        return super().handle(data, **kwargs)


@dataclass
class VaeTrainHandler(BaseHandler):
    wrapper_name: str
    schema: Dict
    epochs: int
    row_subset: int
    drop_null: bool
    batch_size: int

    def __fit_model(self, data: pd.DataFrame):
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
            batch_size=self.batch_size,
            process="train",
        )
        self.model.batch_size = min(self.batch_size, len(data))

        logger.debug(
            f"Train model with parameters: epochs={self.epochs}, row_subset={self.row_subset}, "
            f"drop_null={self.drop_null}, batch_size={self.batch_size}"
        )

        self.model.fit_on_df(
            data,
            epochs=self.epochs,
        )

        if not check_if_features_assigned(self.paths["dataset_pickle_path"]):
            return

        self.model.save_state(self.paths["state_path"])
        logger.info("Finished VAE training")

    def __prepare_dir(self):
        os.makedirs(self.paths["fk_kde_path"], exist_ok=True)

    def handle(self, data: pd.DataFrame, **kwargs):
        self.__prepare_dir()
        self.__fit_model(data)
        return super().handle(data, **kwargs)


@dataclass
class VaeInferHandler(BaseHandler):
    metadata_path: str
    random_seed: Optional[int]
    size: int
    batch_size: int
    run_parallel: bool
    print_report: bool
    wrapper_name: str
    log_level: str
    type_of_process: str
    random_seed_list: List = field(init=False)
    vae: Optional[VAEWrapper] = field(init=False)  # noqa: F405
    has_vae: bool = field(init=False)
    has_no_ml: bool = field(init=False)

    def __post_init__(self):
        if self.random_seed:
            seed(self.random_seed)
        self.random_seeds_list = list()
        self.vae = None
        self.dataset = fetch_dataset(self.paths["dataset_pickle_path"])
        self.has_vae = len(self.dataset.features) > 0
        self.has_no_ml = os.path.exists(f'{self.paths["path_to_no_ml"]}')

    @staticmethod
    def synth_word(size, indexes, counts):
        return "".join(
            np.random.choice(
                np.array(list(indexes)),
                size=size,
                p=np.array(list(counts.values())) / sum(np.array(list(counts.values()))),
            )
        )

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
                if config_of_keys.get(key).get("type") == "PK" and not isinstance(
                    df_slices[0][column][0], (str, UUID, ULID)
                ):
                    cumm_len = 0
                    for i, frame in enumerate(df_slices):
                        frame[column] = frame[column].map(lambda pk_val: pk_val + cumm_len)
                        cumm_len += len(frame)
        return pd.concat(df_slices, ignore_index=True)

    def generate_vae(self, size, data, schema):
        self.vae = self.create_wrapper(
            self.wrapper_name,
            data,
            schema,
            metadata={"table_name": self.table_name},
            table_name=self.table_name,
            paths=self.paths,
            batch_size=self.batch_size,
            process="infer",
        )
        self.vae.load_state(self.paths["state_path"])
        synthetic_infer = self.vae.predict_sampled_df(size)
        return synthetic_infer

    def generate_long_texts(self, size, synthetic_infer):
        with open(f'{self.paths["path_to_no_ml"]}', "rb") as file:
            features = dill.load(file)
        for col in features.keys():
            kde = features[col]["kde"]
            text_structures = np.maximum(kde.resample(size).astype("int32"), 0)
            indexes = features[col]["indexes"]
            counts = features[col]["counts"]
            generated_column = [
                " ".join(
                    [
                        self.synth_word(s, indexes, counts)
                        for s in np.maximum(np.random.normal(i / j, 1, j).astype("int32"), 2)
                    ]
                )
                for i, j in zip(*text_structures)
            ]
            synthetic_infer[col] = generated_column
        return synthetic_infer

    def run_separate(self, params: Tuple):
        os.environ["LOGURU_LEVEL"] = self.log_level
        setup_logger()
        i, size = params

        if self.random_seed:
            seed(self.random_seeds_list[i])

        input_data_existed = DataLoader(self.paths["input_data_path"]).has_existed_path

        if input_data_existed:
            data, schema = DataLoader(self.paths["input_data_path"]).load_data()
        else:
            data = pd.DataFrame()
            schema = None

        synthetic_infer = pd.DataFrame()
        if self.has_vae:
            synthetic_infer = self.generate_vae(size, data, schema)
        if self.has_no_ml:
            synthetic_infer = self.generate_long_texts(size, synthetic_infer)

        uuid_columns = self.dataset.uuid_columns
        if uuid_columns:
            synthetic_infer = generate_uuid(size, self.dataset, uuid_columns, synthetic_infer)

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
            with open(f'{self.paths["fk_kde_path"]}{fk_label}.pkl', "rb") as file:
                kde = dill.load(file)
            pk = pk.dropna()
            numeric_pk = np.arange(len(pk)) if pk.dtype == "object" else pk
            fk_pdf = np.maximum(kde.evaluate(numeric_pk), 1e-12)
            synth_fk = np.random.choice(pk, size=size, p=fk_pdf / sum(fk_pdf), replace=True)
            synth_fk = pd.DataFrame({fk_label: synth_fk}).reset_index(drop=True)

        except FileNotFoundError:
            logger.warning(
                f"The mapper for the {fk_label} text key is not found. Making simple sampling"
            )
            synth_fk = pk.sample(size, replace=True).reset_index(drop=True)
            synth_fk = synth_fk.rename(fk_label)

        return synth_fk

    def _get_pk_path(self, pk_table, table_name) -> str:
        """
        Set the path to synthetic data of corresponding pk table
        """
        destination_to_pk_table = None
        if self.type_of_process == "infer":
            infer_settings = self.metadata[pk_table].get("infer_settings", {})
            destination_to_pk_table = infer_settings.get("destination")

            if destination_to_pk_table is None:
                destination_to_pk_table = (
                    f"model_artifacts/tmp_store/{slugify(pk_table)}/"
                    f"merged_infer_{slugify(pk_table)}.csv"
                )

        if self.type_of_process == "train":
            destination_to_pk_table = self.paths["path_to_merged_infer"].replace(
                slugify(table_name), slugify(pk_table)
            )
        if not os.path.exists(destination_to_pk_table):
            raise FileNotFoundError(
                "The table with a primary key specified in the metadata file does not "
                "exist or is not trained. Ensure that the metadata contains the "
                "name of referenced table with a primary key in the foreign key "
                "declaration section."
            )

        return destination_to_pk_table

    def generate_keys(self, generated, size, metadata, table_name):
        metadata_of_table = metadata.get(table_name)
        if "keys" not in metadata_of_table:
            return None
        config_of_keys = metadata_of_table.get("keys")
        for key in config_of_keys.keys():
            if config_of_keys.get(key).get("type") == "FK":
                fk_column_name = config_of_keys.get(key).get("columns")[0]
                pk_table = config_of_keys.get(key).get("references").get("table")
                pk_path = self._get_pk_path(pk_table=pk_table, table_name=table_name)
                pk_table_data, pk_table_schema = DataLoader(pk_path).load_data()
                pk_column_label = config_of_keys.get(key).get("references").get("columns")[0]
                logger.info(f"The {pk_column_label} assigned as a foreign_key feature")

                synth_fk = self.kde_gen(pk_table_data, pk_column_label, size, fk_column_name)
                generated = generated.reset_index(drop=True)

                null_column_name = f"{key}_null"
                if null_column_name in generated.columns:
                    not_null_column_mask = generated[null_column_name].astype("float64") <= 0.5
                    synth_fk = synth_fk.where(not_null_column_mask, np.nan)
                    generated = generated.drop(null_column_name, axis=1)

                generated[fk_column_name] = synth_fk
        return generated

    def _restore_empty_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Restore empty columns in the generated table
        """
        empty_columns = self.dataset.dropped_columns
        empty_df = pd.DataFrame(index=df.index, columns=empty_columns)
        df = pd.concat([df, empty_df], axis=1)
        return df

    def handle(self, **kwargs):
        self._prepare_dir()

        batch_num = math.ceil(self.size / self.batch_size)
        logger.debug(
            f"Infer model with parameters: size={self.size}, run_parallel={self.run_parallel}, "
            f"batch_size={self.batch_size}, random_seed={self.random_seed}, "
            f"print_report={self.print_report}"
        )
        logger.info(f"Total of {batch_num} batch(es)")
        batches = self.split_by_batches(self.size, batch_num)
        prepared_batches = [self.run(batch, self.run_parallel) for batch in batches]
        prepared_data = (
            self._concat_slices_with_unique_pk(prepared_batches)
            if len(prepared_batches) > 0
            else pd.DataFrame()
        )
        prepared_data = self._restore_empty_columns(prepared_data)

        is_pk = self._is_pk()

        if self.metadata_path is not None:
            if not is_pk:
                generated_data = self.generate_keys(
                    prepared_data, self.size, self.metadata, self.table_name
                )
                generated_data = generated_data[self.dataset.order_of_columns]
                if generated_data is None:
                    DataLoader(self.paths["path_to_merged_infer"]).save_data(
                        self.paths["path_to_merged_infer"],
                        prepared_data,
                        format=get_context().get_config(),
                    )
                else:
                    DataLoader(self.paths["path_to_merged_infer"]).save_data(
                        self.paths["path_to_merged_infer"],
                        generated_data,
                        format=get_context().get_config(),
                    )
            else:
                DataLoader(self.paths["path_to_merged_infer"]).save_data(
                    self.paths["path_to_merged_infer"],
                    prepared_data,
                    format=get_context().get_config(),
                )
        if self.metadata_path is None:
            prepared_data = prepared_data[self.dataset.order_of_columns]
            DataLoader(self.paths["path_to_merged_infer"]).save_data(
                self.paths["path_to_merged_infer"],
                prepared_data,
                format=get_context().get_config(),
            )
