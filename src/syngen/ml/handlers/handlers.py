from typing import Tuple, Optional, Dict, List, Callable
from abc import ABC, abstractmethod
import os
import math
from ulid import ULID
from uuid import UUID

import multiprocessing as mp
import functools
import psutil
import gc

import pandas as pd
import numpy as np
from numpy.random import seed, choice
import dill
from scipy.stats import gaussian_kde
from collections import OrderedDict
from tensorflow.keras.preprocessing.text import Tokenizer
from slugify import slugify
from loguru import logger
from attrs import define, field

from syngen.ml.vae import *  # noqa: F403
from syngen.ml.data_loaders import DataLoader, DataFrameFetcher
from syngen.ml.reporters import Report
from syngen.ml.vae.models.dataset import Dataset
from syngen.ml.utils import (
    fetch_config,
    check_if_features_assigned,
    get_initial_table_name,
    ProgressBarHandler,
    timing,
)
from syngen.ml.context import get_context


class AbstractHandler(ABC):
    @abstractmethod
    def set_next(self, handler):
        pass

    @abstractmethod
    def handle(self, data: pd.DataFrame, **kwargs):
        pass


@define
class BaseHandler(AbstractHandler):
    metadata: Dict = field(kw_only=True)
    paths: Dict = field(kw_only=True)
    table_name: str = field(kw_only=True)
    loader: Optional[Callable[[str], pd.DataFrame]] = None
    _next_handler: Optional[AbstractHandler] = None

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
            main_process=kwargs["main_process"],
            process=kwargs["process"],
        )

    def fetch_data(self) -> Tuple[pd.DataFrame, Optional[Dict]]:
        """
        Fetch the data
        """
        data_loader = DataLoader(self.paths["input_data_path"])
        data = pd.DataFrame()
        schema = None
        if data_loader.has_existed_path:
            data, schema = data_loader.load_data()
        elif self.loader:
            data, schema = DataFrameFetcher(
                loader=self.loader,
                table_name=self.table_name
            ).fetch_data()
        return data, schema


@define
class RootHandler(BaseHandler):

    def handle(self, **kwargs):
        data, schema = super().fetch_data()
        return super().handle(data, **kwargs)


@define
class LongTextsHandler(BaseHandler):
    schema: Optional[Dict] = field(kw_only=True)

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

        dataset = fetch_config(self.paths["dataset_pickle_path"])
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


@define
class VaeTrainHandler(BaseHandler):
    wrapper_name: str = field(kw_only=True)
    schema: Dict = field(kw_only=True)
    epochs: int = field(kw_only=True)
    row_subset: int = field(kw_only=True)
    drop_null: bool = field(kw_only=True)
    batch_size: int = field(kw_only=True)
    type_of_process: str = field(kw_only=True)
    reports: List[str] = field(kw_only=True)

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
            main_process=self.type_of_process,
            process="train",
        )
        self.model.batch_size = min(self.batch_size, len(data))
        list_of_reports = [f'"{report}"' for report in self.reports]
        list_of_reports = ', '.join(list_of_reports) if list_of_reports else '"none"'
        logger.debug(
            f"Train model with parameters: epochs={self.epochs}, "
            f"row_subset={self.row_subset}, drop_null={self.drop_null}, "
            f"batch_size={self.batch_size}, reports - {list_of_reports}"
        )

        self.model.fit_on_df(epochs=self.epochs)

        if not check_if_features_assigned(self.paths["dataset_pickle_path"]):
            return

        self.model.save_state(self.paths["state_path"])
        log_message = "Finished VAE training"
        logger.info(log_message)
        ProgressBarHandler().set_progress(message=log_message)

    def __prepare_dir(self):
        os.makedirs(self.paths["fk_kde_path"], exist_ok=True)

    def handle(self, data: pd.DataFrame, **kwargs):
        self.__prepare_dir()
        self.__fit_model(data)
        return super().handle(data, **kwargs)


@define
class VaeInferHandler(BaseHandler):
    metadata_path: str = field(kw_only=True)
    random_seed: Optional[int] = field(kw_only=True)
    size: int = field(kw_only=True)
    batch_size: int = field(kw_only=True)
    run_parallel: bool = field(kw_only=True)
    reports: List[str] = field(kw_only=True)
    wrapper_name: str = field(kw_only=True)
    log_level: str = field(kw_only=True)
    type_of_process: str = field(kw_only=True)
    random_seeds_list: List = field(init=False)
    vae: Optional[VAEWrapper] = field(init=False)  # noqa: F405
    dataset: Dataset = field(init=False)
    original_schema: Dict = field(init=False)
    has_vae: bool = field(init=False)
    has_no_ml: bool = field(init=False)
    batch_num: int = field(init=False)

    def __attrs_post_init__(self):
        if self.random_seed:
            seed(self.random_seed)
        self.batch_num = math.ceil(self.size / self.batch_size)
        self.random_seeds_list = list()
        self.vae = None
        self.dataset = fetch_config(self.paths["dataset_pickle_path"])
        path_to_schema = self.paths["original_schema_path"]
        self.original_schema = (
            fetch_config(path_to_schema) if os.path.exists(path_to_schema) else None
        )

        self.has_vae = len(self.dataset.features) > 0

        data, schema = self.fetch_data()

        self.has_no_ml = os.path.exists(f'{self.paths["path_to_no_ml"]}')

        # set it to None here to avoid serialization issues
        self._pool = None

        # initialize the VAE model if it is not a parallel run
        if self.has_vae and not self.run_parallel:
            self.vae = self._get_wrapper(data, schema)

        if self.has_vae and self.run_parallel:
            self._setup_parallel_processing(data, schema)

    def _cleanup_pool(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    @timing
    def _setup_parallel_processing(self, data, schema):
        # to avoid errors with pkl loading
        ##ANCHOR MANUAL CONTROL HERE
        mp.set_start_method('spawn', force=True)

        logger.info("Running in parallel mode")

        logger.info("Note: Running in parallel mode causes "
                    "some log messages to appear multiple times. "
                    "This is expected behavior as the model is loaded "
                    "on multiple devices to ensure efficient processing."
                    )

        if self.batch_num > 1:
            n_jobs = min(self.batch_num, mp.cpu_count())
        else:
            self.batch_num = min(self.size, mp.cpu_count())

            # equal batches for each process in last batch keep the remainder
            self.batch_size = math.ceil(self.size / self.batch_num)

            # ensure that we have at least 1 record in the last batch
            if self.batch_size * (self.batch_num - 1) >= self.size:
                self.batch_size = math.floor(self.size / self.batch_num)

            logger.info(
                f"Splitting data into {self.batch_num} "
                f"batches with batch_size={self.batch_size} "
                f"to run in parallel mode"
            )

            n_jobs = self.batch_num

        self._pool = mp.Pool(
            processes=n_jobs,
            initializer=self.worker_init,
            initargs=(data, schema, self._get_wrapper)
        )

    @staticmethod
    def worker_init(data, schema, get_wrapper_func):
        global vae_model
        vae_model = get_wrapper_func(data, schema)

    @staticmethod
    def worker_process(params, random_seed,
                       random_seeds_list, run_separate_func):
        global vae_model
        i, size = params
        if random_seed:
            seed(random_seeds_list[i % len(random_seeds_list)])

        result = run_separate_func((i, size), vae_model)
        return result

    @staticmethod
    def synth_word(size, indexes, counts):
        return "".join(
            np.random.choice(
                np.array(list(indexes)),
                size=size,
                p=np.array(list(counts.values())) / sum(np.array(list(counts.values()))),
            )
        )

    def _get_wrapper(self, data: pd.DataFrame, schema: Dict):
        """
        Create and get the wrapper for the VAE model
        """
        return self.create_wrapper(
            self.wrapper_name,
            data,
            schema,
            metadata=self.metadata,
            table_name=self.table_name,
            paths=self.paths,
            batch_size=self.batch_size,
            main_process=self.type_of_process,
            process="infer",
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
            logger.warning(f"config_of_keys: {config_of_keys}")
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

    def generate_vae(self, size, vae_model):
        synthetic_infer = vae_model.predict_sampled_df(size)
        return synthetic_infer

    # @timing
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
            # current_process = os.getpid()
            # logger.debug(f"Long text for column '{col}' is generated for process {current_process}.")
            synthetic_infer[col] = generated_column
        return synthetic_infer

    def run_separate(self, params: Tuple, vae_model):
        i, size = params

        if self.batch_num > 1:
            seed(self.random_seeds_list[i])
            # logger.warning(f"Set random seed {self.random_seeds_list[i]} for batch {i}")

        synthetic_infer = pd.DataFrame()

        if self.has_vae:
            synthetic_infer = self.generate_vae(size, vae_model)

        if self.has_no_ml:
            logger.info(f'Long texts generation for {self.table_name} started.')
            synthetic_infer = self.generate_long_texts(size, synthetic_infer)

        return synthetic_infer

    def split_by_batches(self):
        """
        Split the total size into batches
        based on the specified number of nodes.

        This method divides the total size
        into a specified number of batches (nodes).
        Each batch will have a size equal to batch_size,
        except for the last batch, which will contain the remaining size
        """
        full_batch_size = self.batch_size
        nodes = self.batch_num
        data = [full_batch_size] * (nodes - 1)
        data.append(self.size - full_batch_size * (nodes - 1))
        return data

    def check_memory_usage(self, current_usage, target_usage=80, current_batch_size=None):
        if current_usage > target_usage and current_batch_size is not None:
            reduction_factor = target_usage / current_usage
            new_batch_size = max(1, math.floor(current_batch_size * reduction_factor))
            logger.info(f"Memory usage is {current_usage}%. Reducing batch size from {current_batch_size} to {new_batch_size}")
            return new_batch_size
        return current_batch_size

    def run(self, size: int, run_parallel: bool):
        logger.info("Start data synthesis")
        batches = list(enumerate(self.split_by_batches()))
        logger.warning(f"batches: {batches}")
        delta = ProgressBarHandler().delta / self.batch_num

        if self.has_vae:
            logger.info(f'VAE generation for {self.table_name} started')

        if self.batch_num > 1:
            self._set_random_seeds()

        if run_parallel:
            worker_func = functools.partial(
                self.worker_process,
                random_seed=self.random_seed,
                random_seeds_list=self.random_seeds_list,
                run_separate_func=self.run_separate
            )

            frames = []
            for result in self._pool.imap_unordered(
                    worker_func,
                    ((i, batch_size) for i, batch_size in batches)
                    ):
                frames.append(result)

                memory_usage = psutil.virtual_memory().percent
                logger.info(
                            f"{len(frames)} batches "
                            f"out of {self.batch_num} are processed. "
                            f"Memory usage: {memory_usage}%"
                            )
                if memory_usage > 93:  # and (len(frames) < mp.cpu_count() - 2):
                    logger.warning(
                        f"High memory usage detected: {memory_usage}%. "
                        f"To avoid memory overflow, reduce the batch size and rerun. "
                        f"Current batch_size={self.batch_size}. "
                        f"Recommended batch_size={self.batch_size // 4}. "
                        f"Stopping the process to avoid memory overflow."
                    )
                    raise MemoryError(
                        f"High memory usage detected: {memory_usage}%. "
                        f"To avoid memory overflow, reduce the batch size and rerun. "
                        f"Current batch_size={self.batch_size}. "
                        f"Recommended batch_size={self.batch_size // 4}. "
                        f"Stopping the process to avoid memory overflow."
                    )
            logger.info(f"Finished processing all batches. Memory usage: {memory_usage}%")
            self._cleanup_pool()

            logger.warning(f"In run method all frames are ready")

            prepared_data = self._concat_slices_with_unique_pk(frames)
            logger.warning(f"Frames are concatinated with unique pk")

        else:
            prepared_batches = []
            for i, batch_size in batches:
                log_message = (
                    f"Data synthesis for the table - '{self.table_name}'. "
                    f"Generating the batch {i + 1} of {self.batch_num}"
                )
                ProgressBarHandler().set_progress(
                    progress=ProgressBarHandler().progress + delta,
                    delta=delta,
                    message=log_message,
                )
                logger.info(log_message)

                # if self.random_seed:
                #     logger.warning(f"self.random_seeds_list: {self.random_seeds_list}")
                #     self.random_seeds_list.append(self.random_seed)
                #     logger.warning(f"self.random_seeds_list: {self.random_seeds_list}")
                logger.info(f" Memory usage 1: {psutil.virtual_memory().percent}%")
                prepared_batch = self.run_separate((i, batch_size), self.vae)
                memory_usage = psutil.virtual_memory().percent
                logger.info(
                            f"{i + 1} batches "
                            f"out of {self.batch_num} are processed. "
                            f"Memory usage: {memory_usage}%"
                            )
                prepared_batches.append(prepared_batch)
                logger.info(f" Memory usage before concat_slices: {psutil.virtual_memory().percent}%")

            logger.info(f"Finished processing all batches. Memory usage: {memory_usage}%")
            prepared_data = (
                        self._concat_slices_with_unique_pk(prepared_batches)
                        if len(prepared_batches) > 0
                        else pd.DataFrame()
                )

        return prepared_data

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
        initial_table_name = get_initial_table_name(table_name)
        if self.type_of_process == "train":
            destination_to_pk_table = self.paths["path_to_merged_infer"].replace(
                slugify(initial_table_name), slugify(pk_table)
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

        empty_df = pd.DataFrame(index=df.index, columns=list(empty_columns))
        df = pd.concat([df, empty_df], axis=1)

        return df

    @timing
    def handle(self, **kwargs):
        self._prepare_dir()
        list_of_reports = [f'"{report}"' for report in self.reports]
        list_of_reports = ', '.join(list_of_reports) if list_of_reports else '"none"'
        log_message = (
            f"Infer model with parameters: size={self.size}, "
            f"run_parallel={self.run_parallel}, batch_size={self.batch_size}, "
            f"random_seed={self.random_seed}"
        )
        if self.type_of_process == "infer":
            log_message += f", reports - {list_of_reports}"
        logger.debug(log_message)
        logger.info(f"Total of {self.batch_num} batch(es)")
        prepared_data = self.run(self.size, self.run_parallel)

        prepared_data = self._restore_empty_columns(prepared_data)
        # workaround for the case when all columns are dropped
        # with technical column
        tech_columns = list(self.dataset.tech_columns)
        if tech_columns:
            prepared_data = prepared_data.drop(tech_columns, axis=1)
            logger.debug(
                "Technical columns "
                f"{tech_columns} were removed "
                "from the generated table."
            )
            Report().unregister_reporters(self.table_name)
            logger.info(
                "Since there were no columns suitable for training, "
                "reports will not be generated "
                f"for the table '{self.table_name}'."
            )

        is_pk = self._is_pk()

        if self.metadata_path is not None:
            if not is_pk:
                generated_data = self.generate_keys(
                    prepared_data, self.size, self.metadata, self.table_name
                )
                generated_data = generated_data[self.dataset.order_of_columns]

                if generated_data is None:
                    DataLoader(self.paths["path_to_merged_infer"]).save_data(
                        prepared_data,
                        schema=self.original_schema,
                        format=get_context().get_config(),
                    )
                else:
                    DataLoader(self.paths["path_to_merged_infer"]).save_data(
                        generated_data,
                        schema=self.original_schema,
                        format=get_context().get_config(),
                    )
            else:
                DataLoader(self.paths["path_to_merged_infer"]).save_data(
                    prepared_data,
                    schema=self.original_schema,
                    format=get_context().get_config(),
                )
        if self.metadata_path is None:
            prepared_data = prepared_data[self.dataset.order_of_columns]
            DataLoader(self.paths["path_to_merged_infer"]).save_data(
                prepared_data,
                schema=self.original_schema,
                format=get_context().get_config(),
            )
        self._cleanup_pool()

    # TODO - set random seeds get reed of them all through the code
    def _set_random_seeds(self):
        logger.warning(f"self.batch_num: {self.batch_num}")
        if self.random_seed or self.batch_num > 1:
            seed(self.random_seed)
            num_seeds = self.batch_num
            self.random_seeds_list = choice(
                range(0, max(100, num_seeds)), num_seeds, replace=False
            )
            # self.random_seeds_list = [self.random_seed] * self.batch_num

            # num_seeds = max(self.batch_num, self._pool._processes if self.run_parallel else 1)
            # self.random_seeds_list = choice(range(0, max(100, num_seeds)), num_seeds, replace=False)
            # logger.warning(f"Random seeds: {self.random_seeds_list}")
            # logger.warning(f"Len Random seed list: {len(self.random_seeds_list)}")
        else:
            self.random_seeds_list = []
        return self.random_seeds_list
