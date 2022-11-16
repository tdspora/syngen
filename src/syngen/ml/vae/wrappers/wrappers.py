from typing import Tuple, List
from abc import ABC, abstractmethod
import warnings
import pickle
import tensorflow as tf
from tensorflow.python.data.experimental import AutoShardPolicy
import matplotlib.pyplot as plt
import time
import tqdm
from loguru import logger
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np

from syngen.ml.vae.models.model import CVAE
from syngen.ml.vae.models import Dataset

warnings.filterwarnings("ignore")


class BaseWrapper(ABC):
    """
    Abstract class for wrappers
    """

    def __init__(self):
        self.model = None

    @abstractmethod
    def fit_on_df(
        self,
        df: pd.DataFrame,
        row_subset: List[int] = None,
        columns_subset: List[str] = None,
        batch_size: int = 10,
        epochs: int = 30,
        verbose: int = 0,
    ):
        pass

    @abstractmethod
    def predict_sampled_df(self, n: int) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_state(self, path: str):
        pass

    @abstractmethod
    def load_state(self, path: str):
        pass


class VAEWrapper(BaseWrapper):
    """Base class that implements end to end train and generation of structured data.

    Attributes
    ----------
    df
    metadata
    paths
    batch_size
    latent_dim
    latent_components

    Methods
    -------
    _pipeline()
        data preprocessing

    _train(dataset, row_subset, epochs)
        train the VAE and save result in model

    display_losses()
        show train losses curve by each feature

    predict_sampled_df(df, n)
        generate new data based on df that consist of n rows and return the result as pd.DataFrame

    predict_less_likely_samples(df, n, temp=0.05, variaty=3)
        generate new data based on df that consist of n which has less probablity
        computed as log lokelihood and return the result as pd.DataFrame
    """

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
        table_name: str,
        paths: dict,
        batch_size: int = 32,
        latent_dim: int = 30,
        latent_components: int = 30,
    ):
        super(VAEWrapper, self).__init__()
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.latent_components = latent_components
        self.metadata = metadata
        self.table_name = table_name
        self.vae_resources_path = paths["state_path"]
        self.dataset_pickle_path = paths["dataset_pickle_path"]
        self.fk_kde_path = paths["fk_kde_path"]
        self.dataset = Dataset(df, self.metadata, self.table_name, self.fk_kde_path)

    def _pipeline(self):
        self.df = self.dataset.pipeline()

        with open(self.dataset_pickle_path, "wb") as f:
            f.write(pickle.dumps(self.dataset))

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
        return df

    def _restore_nan_labels(self, df):
        for column_name, nan_label in self.dataset.nan_labels_dict.items():
            df[column_name] = df[column_name].fillna(nan_label)
        return df

    @abstractmethod
    def _init_model(self):
        pass

    def fit_on_df(
        self,
        df: pd.DataFrame,
        row_subset: int = None,
        columns_subset: List[str] = None,  # TODO columns_subset does not work
        batch_size: int = 32,
        epochs: int = 30,
        verbose: int = 0,
    ):
        row_subset = row_subset or len(df)

        self._pipeline()
        self._init_model()

        # feature_names = ['mmd'] + [name.name for name in self.dataset.features.values()]

        if columns_subset is None:
            columns_subset = self.df.columns
        else:
            # if a column is in columns_subset, its null column should also be added if present
            columns_subset += [
                col
                for col in self.df.columns
                if col.endswith("_null") and (col[:-5] in columns_subset)
            ]

        df = self.df.loc[:, list(set(columns_subset))]

        train_dataset = self._create_batched_dataset(df)
        self.vae = self.model.model

        self.optimizer = self._create_optimizer()
        self.loss_metric = self._create_loss()
        self._train(train_dataset, row_subset, epochs)

        self.model.model = self.vae
        self.fit_sampler(df.dropna())

    def _train(self, dataset, row_subset, epochs: int):
        step = self._train_step

        self.feature_losses = defaultdict(list)
        loss_grows_num_epochs = 0
        prev_total_loss = float("inf")
        es_min_delta = 0.005
        es_patience = 10
        pth = Path(self.vae_resources_path)

        for epoch in range(epochs):
            num_batches = 0.0
            total_loss = 0.0
            t1 = time.time()

            # Iterate over the batches of the dataset.
            for i, x_batch_train in tqdm.tqdm(iterable=enumerate(dataset)):
                total_loss += step(x_batch_train)
                num_batches += 1

            mean_loss = np.mean(total_loss / num_batches)
            if mean_loss >= prev_total_loss - es_min_delta:
                loss_grows_num_epochs += 1
            else:
                self.vae.save_weights(str(pth / "vae_best_weights_tmp.ckpt"))
                loss_grows_num_epochs = 0

            logger.info(
                f"epoch: {epoch}, loss: {mean_loss}, time: {time.time()-t1}, sec"
            )

            prev_total_loss = mean_loss
            if loss_grows_num_epochs == es_patience:
                self.vae.load_weights(str(pth / "vae_best_weights_tmp.ckpt"))
                logger.info(
                    f"The loss does not become lower for {loss_grows_num_epochs} epochs in a row. Stopping the training."
                )
                break
            epoch += 1

    @staticmethod
    def _create_optimizer():
        learning_rate = 1e-03
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @staticmethod
    def _create_loss():
        return tf.keras.metrics.Mean()

    def _create_batched_dataset(self, df: pd.DataFrame):
        """Define batched dataset for training vae"""
        transformed_data = self.dataset.transform(df)

        feature_datasets = []
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
        for inp in transformed_data:
            dataset = tf.data.Dataset.from_tensor_slices(inp).with_options(options)
            feature_datasets.append(dataset)

        dataset = tf.data.Dataset.zip(tuple(feature_datasets)).with_options(options)
        return dataset.batch(self.batch_size, drop_remainder=True)

    def _train_step(self, batch: Tuple[tf.Tensor]) -> tf.Tensor:
        with tf.GradientTape() as tape:
            self.vae(batch)

            # Compute reconstruction loss
            loss = sum(self.vae.losses)

        self.optimizer.minimize(
            loss=loss, var_list=[self.vae.trainable_weights], tape=tape
        )
        self.loss_metric(loss)
        return loss

    def display_losses(self):
        for name, l in self.feature_losses.items():
            plt.plot(l, label=name)

        plt.legend()
        plt.ylim(0, 10)
        return plt.show()

    def fit_sampler(self, df: pd.DataFrame):
        self.model.fit_sampler(df)

    def predict_sampled_df(self, n: int) -> pd.DataFrame:
        sampled_df = self.model.sample(n)
        sampled_df = self._restore_nan_values(sampled_df)
        return self._restore_nan_labels(sampled_df)

    def predict_less_likely_samples(
        self, df: pd.DataFrame, n: int, temp=0.05, variaty=3
    ):
        self.fit_sampler(df)
        return self.model.less_likely_sample(n, temp, variaty)

    def save_state(self, path: str):
        self.model.save_state(path)
        logger.info(f"Saved VAE state in {path}")

    def load_state(self, path: str):
        try:
            with open(path + "/model_dataset.pkl", "rb") as f:
                self.dataset = pickle.loads(f.read())

            self._init_model()

            state = self.model.load_state(path)

        except (FileNotFoundError, ValueError):
            raise FileNotFoundError("Missing file with VAE state")

        logger.info(f"Loaded VAE state from {path}")
        return state


class VanillaVAEWrapper(VAEWrapper):
    """
    Class that implements end to end train and generation of structured data by CVAE as a model.

    Attributes
    ----------
    model : CVAE
        final model that we will use to generate new data
    """

    def _init_model(self):
        self.model = CVAE(
            self.dataset,
            batch_size=self.batch_size,
            latent_dim=self.latent_dim,
            latent_components=self.latent_components,
            intermediate_dim=128,
        )

        self.model.build_model()
