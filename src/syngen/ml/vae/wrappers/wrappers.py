import os
from datetime import datetime
from typing import Tuple, List, Optional, Dict
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field

import warnings
import pickle
import tensorflow as tf
from tensorflow.python.data.experimental import AutoShardPolicy
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import time
import tqdm
import pandas as pd
import numpy as np
from loguru import logger
from slugify import slugify

from syngen.ml.vae.models.model import CVAE
from syngen.ml.vae.models import Dataset
from syngen.ml.mlflow_tracker import MlflowTracker
from syngen.ml.utils import (
    generate_uuid,
    fetch_config,
    check_if_features_assigned,
    ProgressBarHandler
)
from syngen.ml.data_loaders import DataLoader

warnings.filterwarnings("ignore")

BATCH_SIZE_DEFAULT = 32


class BaseWrapper(ABC):
    """
    Abstract class for wrappers
    """

    @abstractmethod
    def fit_on_df(self, df: pd.DataFrame, epochs: int, columns_subset: List[str] = None):
        pass

    @abstractmethod
    def predict_sampled_df(self, n: int) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_state(self, path: str):
        pass


@dataclass
class VAEWrapper(BaseWrapper):
    df: pd.DataFrame
    schema: Optional[Dict]
    metadata: Dict
    table_name: str
    paths: Dict
    process: str
    main_process: str
    batch_size: int
    log_level: str
    losses_info: pd.DataFrame = pd.DataFrame()
    dataset: Dataset = field(init=False)
    vae: CVAE = field(init=False, default=None)
    model: Model = field(init=False, default=None)
    num_batches: int = field(init=False)
    feature_types: Dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        if self.process == "train":
            self._prepare_dir()
            self.dataset = Dataset(
                df=self.df,
                schema=self.schema,
                metadata=self.metadata,
                table_name=self.table_name,
                main_process=self.main_process,
                paths=self.paths,
            )
            self.dataset.set_metadata()
            self.df = self.dataset.pipeline()
            self._save_dataset()
        elif self.process == "infer":
            self.dataset = fetch_config(self.paths["dataset_pickle_path"])
            self._update_dataset()
            self._save_dataset()

    def _update_dataset(self):
        """
        Update dataset object related to the current process
        """
        self.dataset.paths = self.paths
        self.dataset.main_process = self.main_process

    def _save_dataset(self):
        """
        Save dataset object on the disk
        """
        with open(self.paths["dataset_pickle_path"], "wb") as f:
            f.write(pickle.dumps(self.dataset))

    @staticmethod
    def _prepare_dir():
        os.makedirs("model_artifacts/tmp_store/losses", exist_ok=True)

    def _restore_zero_values(self, df):
        for column in self.dataset.zero_num_column_names:
            if column.endswith("_zero"):
                # remove _zero to get original column name
                num_column_name = column[:-5]
                num_column = df[num_column_name].copy()
                zero_column_mask = df[column].astype("float") >= 0.5
                num_column = num_column.where(zero_column_mask, 0)
                num_zero_values = (num_column == 0).sum()
                df[num_column_name] = num_column
                df = df.drop(column, axis=1)
                logger.info(
                    f"Column {num_column_name} has {num_zero_values} "
                    f"({round(num_zero_values * 100 / len(num_column), 2)}%) "
                    f"zero values generated"
                )
        return df

    def _restore_nan_values(self, df):
        for column in self.dataset.null_num_column_names:
            if column.endswith("_null"):
                # remove _null to get original column name
                num_column_name = column[:-5]
                num_column = df[num_column_name].copy()
                not_null_column_mask = df[column].astype("float64") <= 0.5
                num_column = num_column.where(not_null_column_mask, np.nan)
                df[num_column_name] = num_column
                df = df.drop(column, axis=1)
                num_nan_values = num_column.isna().sum()
                logger.info(
                    f"Column {num_column_name} has {num_nan_values} "
                    f"({round(num_nan_values * 100 / len(num_column), 2)}%) "
                    f"empty values generated"
                )
        return df

    def _restore_nan_labels(self, df):
        for column_name, nan_label in self.dataset.nan_labels_dict.items():
            if nan_label is None:
                nan_label = np.nan
            if column_name in df.select_dtypes(int).columns:
                df[column_name] = df[column_name].astype("object")
            df[column_name] = df[column_name].fillna(nan_label)
        return df

    def fit_on_df(
        self,
        epochs: int,
        columns_subset: List[str] = None,  # TODO columns_subset does not work
    ):
        if not check_if_features_assigned(self.paths["dataset_pickle_path"]):
            return

        if columns_subset is None:
            columns_subset = self.df.columns
        else:
            # if a column is in columns_subset, its null column should also be added if present
            columns_subset += [
                col
                for col in self.df.columns
                if col.endswith(("_null", "_zero")) and (col[:-5] in columns_subset)
            ]

        df = self.df.loc[:, list(set(columns_subset))]

        train_dataset = self._create_batched_dataset(df)
        self.num_batches = len(train_dataset)
        self.model = self.vae.model
        self.feature_types = self.vae.feature_types

        self.optimizer = self.__create_optimizer()
        self.loss_metric = self._create_loss()

        # Start of the run of training process
        MlflowTracker().start_run(
            run_name=f"{self.table_name}-TRAIN",
            tags={"table_name": self.table_name, "process": "train"},
        )
        config = fetch_config(self.paths["train_config_pickle_path"])
        MlflowTracker().log_params(config.to_dict())

        self._train(train_dataset, epochs)

        MlflowTracker().end_run()

        MlflowTracker().start_run(
            run_name=f"{self.table_name}-POSTPROCESS",
            tags={"table_name": self.table_name, "process": "postprocess"},
        )
        self.fit_sampler(df)

    def _calculate_loss_by_type(
        self,
        feature_losses: Dict,
        feature_type: str
    ) -> float:
        """
        Group features and calculate the loss by the type
        """
        return sum(
            loss for name, loss in feature_losses.items()
            if self.feature_types[name] == feature_type
        )

    def _get_grouped_losses(self, feature_losses, epoch) -> Dict:
        """
        Get the mean numerical, categorical, and text losses for every epoch
        """
        num_loss = self._calculate_loss_by_type(
            feature_losses,
            feature_type="numeric"
        )

        categorical_loss = self._calculate_loss_by_type(
            feature_losses,
            feature_type="categorical"
        )

        text_loss = self._calculate_loss_by_type(
            feature_losses,
            feature_type="text"
        )
        logger.trace(
            f"The 'numeric_loss' - {num_loss}, "
            f"the 'categorical_loss' - {categorical_loss}, "
            f"the 'text_loss' - {text_loss} in the {epoch} epoch"
        )
        MlflowTracker().log_metric(
            "numeric_loss", num_loss, step=epoch
        )
        MlflowTracker().log_metric(
            "categorical_loss", categorical_loss, step=epoch
        )
        MlflowTracker().log_metric(
            "text_loss", text_loss, step=epoch
        )
        return {
            "numeric_loss": num_loss,
            "categorical_loss": categorical_loss,
            "text_loss": text_loss
        }

    def _get_mean_feature_losses(self, total_feature_losses: Dict):
        """
        Get the mean loss of every feature
        """
        return {
            name: np.mean(loss / self.num_batches)
            for name, loss in total_feature_losses.items()
        }

    def _get_ending(self, feature_name):
        """
        Get the appropriate ending for the name of the loss of the certain feature
        """
        endings = {
            "categorical": "cat",
            "numeric": "num",
            "text": "text"
        }
        feature_type = self.feature_types.get(feature_name)
        return endings.get(feature_type)

    def _monitor_feature_losses(self, mean_feature_losses, epoch):
        """
        Monitor the mean value of the loss of every feature for every epoch
        """
        for name, loss in mean_feature_losses.items():
            ending = self._get_ending(feature_name=name)
            MlflowTracker().log_metric(
                f"{slugify(name)}_loss_{ending}", loss, step=epoch
            )

    def _fetch_feature_losses_info(
            self,
            feature_losses: Dict,
            epoch: int
    ) -> pd.DataFrame:
        """
        Fetch the information related to the loss for every feature in a certain epoch
        """
        timestamp = slugify(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        table_name = f"{self.table_name}_{timestamp}"

        rows = []
        for name, loss in feature_losses.items():
            ending = self._get_ending(feature_name=name)
            row = {
                "table_name": table_name,
                "epoch": epoch,
                "column_name": name if ending else "",
                "column_type": self.feature_types.get(name, "general"),
                "loss_name": f"{name}_loss_{ending}" if ending else name,
                "value": loss,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def _accumulate_feature_losses(feature_losses: Dict, total_feature_losses: Dict):
        """
        Accumulate the loss for every feature
        """
        for key, value in feature_losses.items():
            if key in total_feature_losses:
                total_feature_losses[key] += value
            else:
                total_feature_losses[key] = value
        return total_feature_losses

    def _update_losses_info(self, feature_losses, epoch):
        """
        Add the information about losses of all features fetched during the certain epoch
        """
        data = self._fetch_feature_losses_info(feature_losses, epoch)
        self.losses_info = pd.concat([self.losses_info, data])

    def __save_losses(self):
        """
        Save the information about losses of every feature in every epoch
        """
        path = self.paths["losses_path"]
        DataLoader(path).save_data(path, df=self.losses_info)

    def _gather_losses_info(self, total_feature_losses, mean_loss, mean_kl_loss, epoch):
        """
        Gather the information of losses related to every feature,
        numeric loss, categorical loss and text loss
        """
        mean_feature_losses = self._get_mean_feature_losses(total_feature_losses)
        self._update_losses_info(mean_feature_losses, epoch)
        self._monitor_feature_losses(
            mean_feature_losses,
            epoch
        )
        losses = self._get_grouped_losses(mean_feature_losses, epoch)
        losses.update({"kl_loss": mean_kl_loss, "total_loss": mean_loss})
        self._update_losses_info(losses, epoch)

    def _log_losses_info_to_mlflow(self):
        """
        Log 'losses.csv' to mlflow
        """
        path = self.paths["losses_path"]
        try:
            MlflowTracker().log_artifact(path)
        except Exception as error:
            logger.warning(
                f"Logging the report to mlflow has failed due to a permission error. "
                f"Error details: {error}.\n"
                f"The report will be saved locally in '{path}'"
            )
            pass

    def _train(self, dataset, epochs: int):
        step = self._train_step

        loss_grows_num_epochs = 0
        prev_total_loss = float("inf")
        es_min_delta = 0.005
        es_patience = 10
        pth = Path(self.paths["state_path"])
        # loss that corresponds to the best saved weights
        saved_weights_loss = float("inf")

        delta = ProgressBarHandler().delta / (epochs * 2)
        for epoch in range(epochs):
            log_message = (
                f"Training process of the table - '{self.table_name}' "
                f"on the epoch: {epoch}"
            )
            ProgressBarHandler().set_progress(
                progress=ProgressBarHandler().progress + delta,
                message=log_message
            )
            total_loss = 0.0
            total_feature_losses = dict()
            total_kl_loss = 0.0
            t1 = time.time()

            # Iterate over the batches of the dataset.
            for i, x_batch_train in tqdm.tqdm(iterable=enumerate(dataset)):
                loss, kl_loss, feature_losses = step(x_batch_train)
                total_loss += loss
                total_kl_loss += kl_loss
                total_feature_losses = self._accumulate_feature_losses(
                    feature_losses,
                    total_feature_losses
                )

            mean_loss = np.mean(total_loss / self.num_batches)
            mean_kl_loss = np.mean(total_kl_loss / self.num_batches)

            if mean_loss >= prev_total_loss - es_min_delta:
                loss_grows_num_epochs += 1
            else:
                self.model.save_weights(str(pth / "vae_best_weights_tmp.ckpt"))
                loss_grows_num_epochs = 0
                # loss that corresponds to the best saved weights
                saved_weights_loss = mean_loss

            log_message = (
                f"epoch: {epoch}, total loss: {mean_loss}, time: {(time.time() - t1):.4f} sec"
            )
            logger.info(log_message)

            ProgressBarHandler().set_progress(
                progress=ProgressBarHandler().progress + delta,
                message=log_message
            )

            self._gather_losses_info(total_feature_losses, mean_loss, mean_kl_loss, epoch)
            logger.trace(f"The 'kl_loss' - {mean_kl_loss} in {epoch} epoch")

            MlflowTracker().log_metric("loss", mean_loss, step=epoch)
            MlflowTracker().log_metric("saved_weights_loss", saved_weights_loss, step=epoch)
            MlflowTracker().log_metric("kl_loss", mean_kl_loss, step=epoch)

            prev_total_loss = mean_loss

            if loss_grows_num_epochs == es_patience:
                self.model.load_weights(str(pth / "vae_best_weights_tmp.ckpt"))
                logger.info(
                    f"The loss does not become lower for "
                    f"{loss_grows_num_epochs} epochs in a row. "
                    f"Stopping the training."
                )
                break
            epoch += 1
        self.__save_losses()
        self._log_losses_info_to_mlflow()

    @staticmethod
    def _create_optimizer(learning_rate):
        import platform
        if platform.processor() == 'arm':
            logger.info('Mac ARM processor is detected. Legacy Adam optimizer has been created.')
            return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        else:
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def __create_optimizer(self):
        learning_rate = 1e-04 * np.sqrt(self.batch_size / BATCH_SIZE_DEFAULT)
        return self._create_optimizer(learning_rate)

    @staticmethod
    def _create_loss():
        return tf.keras.metrics.Mean()

    def _create_batched_dataset(self, df: pd.DataFrame):
        """
        Define batched dataset for training vae
        """
        transformed_data = self.dataset.transform(df)

        feature_datasets = []
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
        for inp in transformed_data:
            dataset = tf.data.Dataset.from_tensor_slices(inp).with_options(options)
            feature_datasets.append(dataset)

        dataset = tf.data.Dataset.zip(tuple(feature_datasets)).with_options(options)
        return dataset.batch(self.batch_size, drop_remainder=True)

    def _train_step(self, batch: Tuple[tf.Tensor]):
        with tf.GradientTape() as tape:
            self.model(batch)

            # Compute reconstruction loss
            loss = sum(self.model.losses)
            order_of_features = list(self.vae.feature_losses.keys())
            kl_loss = self.model.losses[-1].numpy()
            feature_losses = {
                name: loss.numpy()
                for name, loss in
                zip(order_of_features, self.model.losses[:-1])
            }

        self.optimizer.minimize(
            loss=loss,
            var_list=self.model.trainable_weights,
            tape=tape
        )
        self.loss_metric(loss)
        return loss, kl_loss, feature_losses

    @staticmethod
    def display_losses(feature_losses: Dict):
        for name, l in feature_losses.items():
            plt.plot(l, label=name)

        plt.legend()
        plt.ylim(0, 10)
        return plt.show()

    def fit_sampler(self, df: pd.DataFrame):
        self.vae.fit_sampler(df)

    def predict_sampled_df(self, n: int) -> pd.DataFrame:
        sampled_df = self.vae.sample(n)

        # uuid columns are generated here to restore nan values
        uuid_columns = self.dataset.uuid_columns
        if uuid_columns:
            sampled_df = generate_uuid(n, self.dataset,
                                       uuid_columns, sampled_df)

        sampled_df = self._restore_nan_values(sampled_df)
        sampled_df = self._restore_zero_values(sampled_df)
        sampled_df = self._restore_nan_labels(sampled_df)

        return sampled_df

    def predict_less_likely_samples(self, df: pd.DataFrame, n: int, temp=0.05, variaty=3):
        self.fit_sampler(df)
        return self.vae.less_likely_sample(n, temp, variaty)

    def save_state(self, path: str):
        self.vae.save_state(path)
        logger.info(f"Saved VAE state in {path}")

    def load_state(self, path: str):
        try:
            self.vae.load_state(path)

        except FileNotFoundError:
            raise FileNotFoundError("Missing file with VAE state")

        logger.info(f"Loaded VAE state from {path}")


class VanillaVAEWrapper(VAEWrapper):
    """
    Class that implements end to end train and generation of structured data by CVAE as a model.

    Attributes
    ----------
    model : CVAE
        final model that we will use to generate new data
    """
    def __init__(
            self,
            df: pd.DataFrame,
            schema: Optional[Dict],
            metadata: Dict,
            table_name: str,
            paths: Dict,
            process: str,
            main_process: str,
            batch_size: int,
            latent_dim: int = 10,
            latent_components: int = 30):

        log_level = os.getenv("LOGURU_LEVEL")

        super().__init__(
            df,
            schema,
            metadata,
            table_name,
            paths,
            process,
            main_process,
            batch_size,
            log_level
        )
        self.latent_dim = min(latent_dim, int(len(self.dataset.columns) / 2))
        self.vae = CVAE(
            self.dataset,
            batch_size=self.batch_size,
            latent_dim=latent_dim,
            latent_components=min(latent_components, latent_dim * 2),
            intermediate_dim=128,
        )
        self.vae.build_model()
        if self.process == "infer":
            self.load_state(self.paths["state_path"])
