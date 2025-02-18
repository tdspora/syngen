from typing import Union, List, Optional, Dict, Literal
from abc import ABC
from itertools import combinations
from collections import Counter
import re
import random

import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, accuracy_score
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from slugify import slugify
from loguru import logger

from syngen.ml.utils import (
    timestamp_to_datetime,
    fetch_config
)
matplotlib.use("Agg")


class BaseMetric(ABC):
    def __init__(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        plot: bool,
        reports_path: str,
    ):
        self.original = original
        self.synthetic = synthetic
        self.reports_path = reports_path
        self.plot = plot
        self.value = None

    def get_value(self) -> Union[float, List[float]]:
        return self.value


class JensenShannonDistance(BaseMetric):
    def __init__(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        plot: bool,
        reports_path: str,
    ):
        super().__init__(original, synthetic, plot, reports_path)
        self.cmap = LinearSegmentedColormap.from_list(
            "rg", ["#96195C", "#C13666", "#B24E89", "#9075C1", "#3F93E1", "#E8F4FF"]
        )

    def calculate_all(self, categorical_columns: List[str]):
        self.original[categorical_columns] = self.original[categorical_columns].fillna("")
        self.synthetic[categorical_columns] = self.synthetic[categorical_columns].fillna("")
        self.heatmap, self.labels = self.__compute_vs_columns(categorical_columns)

        if self.plot:
            plt.clf()
            sns.set(
                rc={
                    "xtick.labelsize": self.heatmap.shape[0],
                    "ytick.labelsize": self.heatmap.shape[0],
                    "ytick.major.size": 1.5,
                    "ytick.major.width": 0.2,
                },
                font_scale=2,
            )
            heatmap = sns.heatmap(
                self.heatmap,
                xticklabels=self.labels,
                yticklabels=self.labels,
                cmap=self.cmap,
                vmin=0.0,
                vmax=1.0,
                center=0.5,
                annot=False,
            )
            heatmap.figure.set_size_inches(self.heatmap.shape)
            heatmap.axes.tick_params(axis="x", rotation=90)
            heatmap.axes.tick_params(axis="y", rotation=0)
            heatmap.figure.tight_layout()
            plt.savefig(
                f"{self.reports_path}/accuracy_heatmap.svg",
                bbox_inches="tight",
                format="svg",
            )
            plt.clf()

    @staticmethod
    def calculate_heatmap_median(heatmap):
        heatmap_no_diag = heatmap[~np.eye(heatmap.shape[0], dtype=bool)].reshape(
            heatmap.shape[0], -1
        )
        heatmap_median = np.nanmedian(heatmap_no_diag)  # ignores nan when calculating median
        return heatmap_median

    def _calculate_pair_continuous_vs_continuous(self, first_column, second_column):
        original_score = self.__jensen_shannon_distance(
            self.original[first_column].fillna(self.original[first_column].mean()),
            self.original[second_column].fillna(self.original[second_column].mean()),
        )

        synthetic_score = self.__jensen_shannon_distance(
            self.synthetic[first_column].fillna(self.synthetic[first_column].mean()),
            self.synthetic[second_column].fillna(self.synthetic[second_column].mean()),
        )
        return 1 - abs(original_score - synthetic_score)

    def _calculate_pair_categ_vs_continuous(self, first_column, second_column):
        map_dict = {
            k: i + 1
            for i, k in enumerate(
                set(self.original[first_column]) | set(self.synthetic[first_column])
            )
        }
        original_score = self.__jensen_shannon_distance(
            self.original[first_column].map(map_dict),
            self.original[second_column].fillna(self.original[second_column].mean()),
        )

        synthetic_score = self.__jensen_shannon_distance(
            self.synthetic[first_column].map(map_dict),
            self.synthetic[second_column].fillna(self.synthetic[second_column].mean()),
        )
        return 1 - abs(original_score - synthetic_score)

    def _calculate_pair_categ_vs_categ(self, first_column, second_column):
        map_dict_first = {
            k: i + 1
            for i, k in enumerate(
                set(self.original[first_column]) | set(self.synthetic[first_column])
            )
        }
        map_dict_second = {
            k: i + 1
            for i, k in enumerate(
                set(self.original[second_column]) | set(self.synthetic[second_column])
            )
        }

        original_score = self.__jensen_shannon_distance(
            self.original[first_column].map(map_dict_first),
            self.original[second_column].map(map_dict_second),
        )

        synthetic_score = self.__jensen_shannon_distance(
            self.synthetic[first_column].map(map_dict_first),
            self.synthetic[second_column].map(map_dict_second),
        )
        return 1 - abs(original_score - synthetic_score)

    def _calculate_pair_continuous_vs_categ(self, first_column, second_column):
        map_dict = {
            k: i + 1
            for i, k in enumerate(
                set(self.original[second_column]) | set(self.synthetic[second_column])
            )
        }

        original_score = self.__jensen_shannon_distance(
            self.original[first_column].fillna(self.original[first_column].mean()),
            self.original[second_column].map(map_dict),
        )

        synthetic_score = self.__jensen_shannon_distance(
            self.synthetic[first_column].fillna(self.synthetic[first_column].mean()),
            self.synthetic[second_column].map(map_dict),
        )
        return 1 - abs(original_score - synthetic_score)

    def __compute_vs_columns(self, categorical_columns: List[str]):
        valid_cols = [col for col in self.original.columns if "word_count" not in col]
        mask = list(map(lambda col: col in categorical_columns, valid_cols))
        heatmap_matrix = []
        for i, c in enumerate(valid_cols):
            row = []
            for j, c_ in enumerate(valid_cols):
                if mask[i]:
                    if mask[j]:
                        acc = self._calculate_pair_categ_vs_categ(c, c_)
                    else:
                        acc = self._calculate_pair_categ_vs_continuous(c, c_)
                else:
                    if mask[j]:
                        acc = self._calculate_pair_continuous_vs_categ(c, c_)
                    else:
                        acc = self._calculate_pair_continuous_vs_continuous(c, c_)

                row.append(acc)
            heatmap_matrix.append(row)

        return np.array(heatmap_matrix), valid_cols

    @staticmethod
    def __normalize(dist):
        min_ = dist.min()
        max_ = dist.max()
        if max_ != min_:
            std = (dist - min_) / (max_ - min_)
        else:
            std = dist / max_
        scaled = std * (1 - 0) + 0
        return scaled

    def __jensen_shannon_distance(self, p, q):
        """
        method to compute the Jenson-Shannon Distance
        between two probability distributions. 0 < distance < 1.
        """

        # convert the vectors into numpy arrays in case that they aren't
        p = self.__normalize(np.array(p))
        q = self.__normalize(np.array(q))

        m = (p + q) / 2

        # compute Jensen Shannon Divergence
        divergence = (st.entropy(p, m) + st.entropy(q, m)) / 2

        # compute the Jensen Shannon Distance
        distance = np.sqrt(divergence)

        return distance + 1e-6


class Correlations(BaseMetric):
    def __init__(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        plot: bool,
        reports_path: str,
    ):
        super().__init__(original, synthetic, plot, reports_path)
        self.cmap = LinearSegmentedColormap.from_list(
            "rg",
            list(reversed(["#96195C", "#C13666", "#B24E89", "#9075C1", "#3F93E1", "#E8F4FF"])),
        )

    def calculate_all(self, categorical_columns: List[str], cont_columns: List[str]):
        categorical_columns = [col for col in categorical_columns if "word_count" not in col]
        cont_columns = [col for col in cont_columns if "word_count" not in col]
        for col in categorical_columns:
            map_dict = {
                k: i + 1 for i, k in enumerate(set(self.original[col]) | set(self.synthetic[col]))
            }
            self.original[col] = self.original[col].map(map_dict)
            self.synthetic[col] = self.synthetic[col].map(map_dict)

        self.original_heatmap = self.__calculate_correlations(
            self.original[categorical_columns + cont_columns].apply(
                pd.to_numeric, axis=0, errors="ignore"
            )
        )
        self.synthetic_heatmap = self.__calculate_correlations(
            self.synthetic[categorical_columns + cont_columns].apply(
                pd.to_numeric, axis=0, errors="ignore"
            )
        )
        self.corr_score = self.original_heatmap - self.synthetic_heatmap
        self.corr_score = (
            self.corr_score
            .dropna(how="all")
            .dropna(how="all", axis=1)
        )

        # check if there are any nans left in corr_score
        if self.corr_score.isna().values.any():
            # mask for NaNs in both original_heatmap and synthetic_heatmap
            nan_mask = (
                np.isnan(self.original_heatmap) &
                np.isnan(self.synthetic_heatmap)
            )

            # Set the NaN values in corr_score to 0 where both
            # original_heatmap and synthetic_heatmap have NaNs
            self.corr_score[nan_mask] = 0

        if self.plot:
            self.__create_correlation_heatmap()

        return np.median(self.corr_score)

    def __create_correlation_heatmap(self):
        plt.clf()
        # workaround for the issue not showing annotations for NaN values
        # change NaN values to 2 that is not in the range 0-1
        corr_score = self.corr_score.fillna(2)
        # set mask for NaN values
        nan_mask = corr_score == 2
        # Color for values over vmax
        self.cmap.set_over('gray')

        # annotation matrix where 'NaN' is placed where the data is NaN
        annotations = np.where(nan_mask, 'NaN', '')

        sns.set(rc={"figure.figsize": self.corr_score.shape}, font_scale=2)
        heatmap = sns.heatmap(
            corr_score,
            annot=annotations,
            cmap=self.cmap,
            vmin=0.0,
            vmax=1.0,
            center=0.5,
            square=True,
            fmt='s',
            annot_kws={'color': 'white'}
        )

        heatmap.figure.tight_layout()
        plt.savefig(
            f"{self.reports_path}/correlations_heatmap.svg",
            bbox_inches="tight",
            format="svg",
        )

    @staticmethod
    def __calculate_correlations(data):
        return abs(data.corr(method="spearman"))


class BivariateMetric(BaseMetric):
    def __init__(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        plot: bool,
        reports_path: str,
        dataset_pickle_path: str
    ):
        super().__init__(original, synthetic, plot, reports_path)
        self.cmap = LinearSegmentedColormap.from_list(
            "rg", ["#0D5598", "#3E92E0", "#E8F4FF"]
        )
        self.dataset = fetch_config(dataset_pickle_path)
        nan_labels_dict = self.dataset.nan_labels_dict
        na_values = [f'"{value}"' for value in self.dataset.format.get("na_values", [])]
        if na_values:
            logger.info(
                f"During data loading the 'format' section was used, therefore the values - "
                f"{', '.join(na_values)} have been replaced with 'nan', impacting bivariate plots"
            )
        self.missing_values: Dict[str, str] = (
            {col: nan_labels_dict.get(col, "nan") for col in self.dataset.order_of_columns}
            if nan_labels_dict
            else {}
        )

    @staticmethod
    def _format_date_labels(heatmap_orig_data, heatmap_synthetic_data, axis):
        heatmap_orig, x_tick_labels_orig, y_tick_labels_orig = heatmap_orig_data
        heatmap_synth, x_tick_labels_synth, y_tick_labels_synth = heatmap_synthetic_data
        if axis == "y":
            y_tick_labels_orig = [timestamp_to_datetime(i) for i in y_tick_labels_orig]
            y_tick_labels_synth = [timestamp_to_datetime(i) for i in y_tick_labels_synth]
        else:
            x_tick_labels_orig = [timestamp_to_datetime(i) for i in x_tick_labels_orig]
            x_tick_labels_synth = [timestamp_to_datetime(i) for i in x_tick_labels_synth]
        return (heatmap_orig, x_tick_labels_orig, y_tick_labels_orig), (
            heatmap_synth,
            x_tick_labels_synth,
            y_tick_labels_synth,
        )

    def calculate_all(
        self,
        cont_columns: List[str],
        categorical_columns: List[str],
        date_columns: Optional[List[str]],
        num_not_na_cont_ticks: int = 10,
        max_num_combinations: int = 100,
    ):
        self.date_columns = date_columns if date_columns else []
        self.num_not_na_ticks = num_not_na_cont_ticks
        fetched_columns = set(cont_columns) | set(categorical_columns)
        excluded_cols = {
            col for col in fetched_columns if col.endswith(("_word_count", "_char_len"))
        }
        all_columns = fetched_columns - excluded_cols
        all_column_pairs = list(combinations(all_columns, 2))
        column_pairs = random.sample(
            all_column_pairs, min(max_num_combinations, len(all_column_pairs))
        )
        bi_imgs = {}
        for i, (first_col, second_col) in tqdm.tqdm(
            iterable=enumerate(column_pairs),
            desc="Generating bivariate distributions...",
            total=len(column_pairs),
        ):
            plt.clf()
            fig, self._axes = plt.subplots(
                1, 2, figsize=(30, 15), gridspec_kw={"width_ratios": [7, 8.7]}
            )
            if first_col in cont_columns:
                if second_col in cont_columns:
                    (
                        heatmap_orig_data,
                        heatmap_synthetic_data,
                    ) = self._calculate_pair_continuous_vs_continuous(
                        y_col=second_col, x_col=first_col  # x is first, y is second
                    )
                elif second_col in categorical_columns:
                    (
                        heatmap_orig_data,
                        heatmap_synthetic_data,
                    ) = self._calculate_pair_continuous_vs_categ(
                        cont_col=first_col,
                        categorical_col=second_col,  # x is second, y is first
                    )

                    heatmap_orig_data = (
                        np.transpose(heatmap_orig_data[0]),
                        heatmap_orig_data[2],
                        heatmap_orig_data[1],
                    )
                    heatmap_synthetic_data = (
                        np.transpose(heatmap_synthetic_data[0]),
                        heatmap_synthetic_data[2],
                        heatmap_synthetic_data[1],
                    )  # x is first, y is second
            elif first_col in categorical_columns:
                if second_col in cont_columns:
                    (
                        heatmap_orig_data,
                        heatmap_synthetic_data,
                    ) = self._calculate_pair_continuous_vs_categ(
                        cont_col=second_col,
                        categorical_col=first_col,  # x is first, y is second
                    )
                elif second_col in categorical_columns:
                    (
                        heatmap_orig_data,
                        heatmap_synthetic_data,
                    ) = self._calculate_pair_categ_vs_categ(y_col=second_col, x_col=first_col)

            heatmap_min, heatmap_max = self.get_common_min_max(
                heatmap_orig_data[0], heatmap_synthetic_data[0]
            )

            if first_col in self.date_columns:
                heatmap_orig_data, heatmap_synthetic_data = self._format_date_labels(
                    heatmap_orig_data, heatmap_synthetic_data, "x"
                )
            if second_col in self.date_columns:
                heatmap_orig_data, heatmap_synthetic_data = self._format_date_labels(
                    heatmap_orig_data, heatmap_synthetic_data, "y"
                )

            self._plot_heatmap(heatmap_orig_data, plt_index=0,
                               vrange=(heatmap_min, heatmap_max),
                               features=(first_col, second_col),
                               cbar=False)

            self._plot_heatmap(heatmap_synthetic_data, plt_index=1,
                               vrange=(heatmap_min, heatmap_max),
                               features=(first_col, second_col),
                               cbar=True)
            # first_col is x axis, second_col is y axis
            title = f"{first_col} vs. {second_col}"
            path_to_image = (
                f"{self.reports_path}/bivariate_{slugify(first_col)}_{slugify(second_col)}.svg"
            )
            bi_imgs[title] = path_to_image
            plt.savefig(path_to_image, format="svg")
        return bi_imgs

    @staticmethod
    def get_common_min_max(original, synthetic):
        original_flat = [c for row in original for c in row]
        synthetic_flat = [c for row in synthetic for c in row]
        vmax = max(max(original_flat), max(synthetic_flat))
        vmin = min(min(original_flat), min(synthetic_flat))
        return vmin, vmax

    @staticmethod
    def __format_float_tick_labels(labels: List, nan_label: str = "nan") -> List:
        labels = [nan_label if pd.isna(label) else label for label in labels]
        if all([isinstance(i, float) for i in labels]) and (
            max(labels) > 1e5 or min(labels) < 1e-03
        ):
            return [f"{label:.4e}" for label in labels]
        if all([isinstance(i, float) for i in labels]):
            return [f"{round(i, 4)}" for i in labels]
        return labels

    def _plot_heatmap(
        self,
        heatmap_data: List,
        plt_index: int,
        vrange: tuple[float],
        features: tuple[str],
        cbar=True,
    ):
        vmin, vmax = vrange
        xfeature, yfeature = features
        ax = self._axes.flat[plt_index]
        ax.tick_params(labelsize=14)
        heatmap, x_tick_labels, y_tick_labels = heatmap_data
        x_tick_labels = self.__format_float_tick_labels(
            x_tick_labels,
            self.missing_values.get(xfeature, "nan")
        )
        y_tick_labels = self.__format_float_tick_labels(
            y_tick_labels,
            self.missing_values.get(yfeature, "nan")
        )
        ax = sns.heatmap(
            heatmap,
            xticklabels=x_tick_labels,
            yticklabels=y_tick_labels,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap=self.cmap,
            cbar=cbar,
        )
        if cbar:
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(axis="y", labelsize=20)

    def _get_continuous_ticks(self, col_name: str):
        original_col_values = self.original[col_name].dropna().values
        synthetic_col_values = self.original[col_name].dropna().values
        col_values = np.concatenate((original_col_values, synthetic_col_values))
        return np.percentile(col_values, list(range(0, 101, self.num_not_na_ticks)))

    def _get_categorical_ticks(self, col_name: str, ticks_count: int = 50):
        """
        Select the ticks for a categorical column
        """
        min_ticks = int(ticks_count * 0.2)
        max_ticks = int(ticks_count * 0.2)
        other_ticks = ticks_count - min_ticks - max_ticks

        categ_ticks = sorted(
            set(
                set(self.original[col_name].fillna("?"))
                | set(self.synthetic[col_name].fillna("?"))
            )
        )
        # Select the first min_ticks elements
        selected_min_ticks = categ_ticks[:min_ticks]

        # Select the last max_ticks elements
        selected_max_ticks = categ_ticks[-max_ticks:]

        # Remove the already selected elements from categ_ticks
        remaining_ticks = categ_ticks[min_ticks:-max_ticks]

        # Randomly select other_ticks elements from the remaining list
        selected_other_ticks = random.sample(
            remaining_ticks, min(other_ticks, len(remaining_ticks))
        )

        return sorted(
            set(
                selected_min_ticks + selected_other_ticks + selected_max_ticks
                )
            )

    @staticmethod
    def _format_categorical_labels(labels):
        return [
            str(label) if len(str(label)) < 15 else str(label[:12]) + "..." for label in labels
        ]

    @staticmethod
    def _smooth(dist):
        for i in range(1, len(dist)):
            if dist[i - 1] == dist[i]:
                dist[i:] += 2
        dist.sort()
        return dist

    def _calculate_pair_categ_vs_categ(self, y_col: str, x_col: str):
        def calc_heatmap_data(
            df: pd.DataFrame,
            y_col: str,
            x_col: str,
            xticks: List[str],
            yticks: List[float],
        ):
            heatmap = []
            for y_val in yticks:
                heatmap_row = []
                for x_val in xticks:
                    count = df[df[y_col].eq(y_val) & df[x_col].eq(x_val)].shape[0] / len(df)
                    heatmap_row.append(count)
                heatmap.append(heatmap_row)
            return (
                heatmap,
                self._format_categorical_labels(xtick),
                self._format_categorical_labels(ytick),
            )

        xtick = self._get_categorical_ticks(x_col)
        ytick = self._get_categorical_ticks(y_col)

        original_heatmap = calc_heatmap_data(self.original, y_col, x_col, xtick, ytick)
        synthetic_heatmap = calc_heatmap_data(self.synthetic, y_col, x_col, xtick, ytick)
        return original_heatmap, synthetic_heatmap

    def _calculate_pair_continuous_vs_categ(self, cont_col: str, categorical_col: str):
        def calc_heatmap_data(
            df: pd.DataFrame,
            cont_col: str,
            categorical_col: str,
            xticks: List[float],
            yticks: List[str],
        ):
            heatmap = []
            cont_has_nan = df[cont_col].isnull().values.any()
            cont_ticks = self.num_not_na_ticks + 1 if cont_has_nan else self.num_not_na_ticks
            for i in range(cont_ticks):
                heatmap_row = []
                if i == 0:
                    x_index = (df[cont_col] >= xticks[i]) & (df[cont_col] <= xticks[i + 1])
                elif i < self.num_not_na_ticks:
                    x_index = (df[cont_col] > xticks[i]) & (df[cont_col] <= xticks[i + 1])
                else:
                    x_index = df[cont_col].isna()

                for val in yticks:
                    count = df[x_index & df[categorical_col].eq(val)].shape[0] / len(df)
                    heatmap_row.append(count)
                heatmap.append(heatmap_row)

            if cont_has_nan:
                xticks = np.append(xticks, np.nan)

            return heatmap, self._format_categorical_labels(yticks), xticks[1:]

        xtick = self._smooth(self._get_continuous_ticks(cont_col))
        ytick = self._get_categorical_ticks(categorical_col)

        original_heatmap = calc_heatmap_data(
            self.original,
            cont_col,
            categorical_col,
            xtick,
            ytick
        )
        synthetic_heatmap = calc_heatmap_data(
            self.synthetic,
            cont_col,
            categorical_col,
            xtick,
            ytick
        )

        return original_heatmap, synthetic_heatmap

    def _calculate_pair_continuous_vs_continuous(self, y_col: str, x_col: str):
        def calc_heatmap_data(
            df: pd.DataFrame,
            y_col: str,
            x_col: str,
            xticks: List[float],
            yticks: List[float],
        ):
            heatmap = []
            y_has_nan = df[y_col].isnull().values.any()
            x_has_nan = df[x_col].isnull().values.any()
            y_ticks_num = self.num_not_na_ticks + 1 if y_has_nan else self.num_not_na_ticks
            x_ticks_num = self.num_not_na_ticks + 1 if x_has_nan else self.num_not_na_ticks

            for i in range(y_ticks_num):
                heatmap_row = []
                if i == 0:
                    y_index = (df[y_col] >= yticks[i]) & (df[y_col] <= yticks[i + 1])
                elif i < self.num_not_na_ticks:
                    y_index = (df[y_col] > yticks[i]) & (df[y_col] <= yticks[i + 1])
                else:
                    y_index = df[y_col].isna()

                for j in range(x_ticks_num):
                    if j == 0:
                        x_index = (df[x_col] >= xticks[j]) & (df[x_col] <= xticks[j + 1])
                    elif j < self.num_not_na_ticks:
                        x_index = (df[x_col] > xticks[j]) & (df[x_col] <= xticks[j + 1])
                    else:
                        x_index = df[x_col].isna()

                    count = df[y_index & x_index].shape[0] / len(df)
                    heatmap_row.append(count)

                heatmap.append(heatmap_row)

            if x_has_nan:
                xticks = np.append(xticks, np.nan)
            if y_has_nan:
                yticks = np.append(yticks, np.nan)
            return heatmap, xticks[1:], yticks[1:]

        ytick = self._smooth(self._get_continuous_ticks(y_col))
        xtick = self._smooth(self._get_continuous_ticks(x_col))

        original_heatmap = calc_heatmap_data(self.original, y_col, x_col, xtick, ytick)
        synthetic_heatmap = calc_heatmap_data(self.synthetic, y_col, x_col, xtick, ytick)

        return original_heatmap, synthetic_heatmap


class UnivariateMetric(BaseMetric):
    def __init__(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        plot: bool,
        reports_path: str,
    ):
        super().__init__(original, synthetic, plot, reports_path)

    def _calculate(self, column):
        pass

    def calculate_all(
        self,
        cont_columns: List[str],
        categorical_columns: List[str],
        date_columns: Optional[List[str]],
        print_nan: bool = False,
    ):
        cont_columns = list(cont_columns)
        excluded_cols = {
            col for col in cont_columns if "word_count" if col.endswith("_word_count")
        }
        cont_columns = set(cont_columns) - excluded_cols
        excluded_cols = {
            col for col in categorical_columns if "word_count" if col.endswith("_word_count")
        }
        categorical_columns = set(categorical_columns) - excluded_cols
        images = {}
        uni_categ_images = {}
        for col in cont_columns:
            images.update(
                self.__calc_continuous(column=col, print_nan=print_nan, isdate=col in date_columns)
            )
        for col in categorical_columns:
            uni_categ_images.update(self.__calc_categ(col))
        images.update(uni_categ_images)
        return images

    @staticmethod
    def _get_ratio_counts(ratio_counts, count: int = 30) -> Dict:
        """
        Select the most common, least common and some random items between them
        from the ratio_counts
        """
        ratio_counts = Counter(ratio_counts)

        most_least_count = int(count * 0.2)
        other_items = count - 2 * most_least_count

        most_common_items = ratio_counts.most_common(most_least_count)
        least_common_items = ratio_counts.most_common()[: -most_least_count - 1: -1]
        between_items = ratio_counts.most_common()[most_least_count:-most_least_count]
        selected_between_items = random.sample(between_items, min(other_items, len(between_items)))

        updated_ratio_counts = dict(
            most_common_items + selected_between_items + least_common_items
        )

        return updated_ratio_counts

    def __calc_categ(self, column):
        def plot_dist(column_data, sort=True, full_set=None):
            counts = Counter(column_data)
            if full_set is not None:
                absent_keys = full_set - set(counts.keys())
                if len(absent_keys) > 0:
                    for k in absent_keys:
                        counts[k] = 0

            if sort:
                sorted_keys = sorted(counts, key=lambda x: counts[x])
                counts = {i: counts[i] for i in sorted_keys}

            size = len(column_data)
            counts = {key: (x / size * 100) for key, x in counts.items()}
            return counts

        def sanitize_labels(label):
            return re.sub(r"\$|\^", "", label)

        original_column = self.original[column].fillna("?")
        synthetic_column = self.synthetic[column].fillna("?")
        full_values_set = set(original_column.values) | set(synthetic_column.values)

        original_ratio_counts = plot_dist(original_column, True, full_values_set)
        original_ratio_counts = self._get_ratio_counts(original_ratio_counts)
        original_labels = list(original_ratio_counts.keys())
        original_ratio = list(original_ratio_counts.values())

        synthetic_ratio_counts = plot_dist(synthetic_column, False, full_values_set)
        synthetic_ratio = [
            synthetic_ratio_counts[label]
            for label in original_labels
            if synthetic_ratio_counts.get(label) is not None
        ]

        uni_images = {}

        if self.plot:
            fig = plt.figure(figsize=(9.5, 6.5))

            width = 0.35
            x = np.arange(len(original_labels))
            plt.bar(
                x - width / 2,
                original_ratio,
                width=width,
                label="Original",
                color="#3F93E1",
            )
            plt.bar(
                x + width / 2,
                synthetic_ratio,
                width=width,
                label="Synthetic",
                color="#FF9C54",
            )
            ax = plt.gca()
            ax.spines[["top", "right", "left", "bottom"]].set_color("#E5E9EB")
            ax.yaxis.grid(color="#E5E9EB", linestyle="-", linewidth=0.5)
            ax.set_axisbelow(True)
            ax.set_facecolor("#FFFFFF")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [
                    str(sanitize_labels(label[:30])) + "..."
                    if len(str(label)) > 33
                    else sanitize_labels(str(label))
                    for label in original_labels
                ]
            )
            ax.tick_params(axis="both", which="major", labelsize=9)
            fig.autofmt_xdate()
            ax.set_xlabel("category", fontsize=9)
            ax.set_ylabel("percents", fontsize=9)
            plt.legend(
                ["original", "synthetic"],
                loc="upper left",
                fontsize=9,
                bbox_to_anchor=(0, 1.07),
                ncol=2,
                frameon=False,
            )
            if self.reports_path:
                path_to_image = f"{self.reports_path}/univariate_{slugify(column)}.svg"
                plt.savefig(path_to_image, bbox_inches="tight", format="svg")
                uni_images[column] = path_to_image
        return uni_images

    def __calc_continuous(self, column: str, isdate: bool, print_nan: bool = False):
        original_nan_count = self.original[column].isna().sum()
        synthetic_nan_count = self.synthetic[column].isna().sum()
        original_unique_count = self.original[column].nunique()
        synthetic_unique_count = self.synthetic[column].nunique()

        uni_images = {}

        if self.plot and original_unique_count > 1 and synthetic_unique_count > 1:
            fig, ax = plt.subplots(figsize=(8, 6.5))

            # Kernel Density Estimation plot using Seaborn
            sns.kdeplot(data=self.original, x=column, color="#3F93E1", linewidth=2, ax=ax)
            sns.kdeplot(data=self.synthetic, x=column, color="#FF9C54", linewidth=2, ax=ax)

            ax.set_xlabel("value", fontsize=9)
            ax.set_ylabel("density", fontsize=9)
            ax.tick_params(axis="both", which="major", labelsize=9)
            ax.ticklabel_format(axis="x", style="plain")
            fmt = ax.yaxis.get_major_formatter()
            fmt.set_useMathText(True)
            ax.yaxis.offsetText.set_visible(False)
            ax.yaxis.set_label_text(f"density {fmt.get_offset()}")
            ax.yaxis.get_offset_text().set_fontsize(9)
            plt.xticks(rotation=45)
            ax.spines[["top", "right", "left", "bottom"]].set_color("#E5E9EB")
            ax.grid(color="#E5E9EB", linestyle="-", linewidth=0.5)
            ax.set_axisbelow(True)
            ax.set_facecolor("#FFFFFF")
            plt.legend(
                ["original", "synthetic"],
                loc="upper left",
                fontsize=9,
                bbox_to_anchor=(0, 1.07),
                ncol=2,
                frameon=False,
            )
            if isdate:
                lower_x, upper_x = ax.dataLim._points[:, 0]
                len_x_labels = len(ax.get_xticklabels())

                x_ticks = np.linspace(lower_x, upper_x, len_x_labels)
                x_ticks = [timestamp_to_datetime(i) for i in x_ticks]
                ax.set_xticklabels(x_ticks, rotation=45, ha="right")
            if self.reports_path:
                path_to_image = f"{self.reports_path}/univariate_{slugify(column)}.svg"
                plt.savefig(path_to_image, bbox_inches="tight", format="svg")
                uni_images[column] = path_to_image
        if print_nan:
            logger.info(f"Number of original NaN values in {column}: {original_nan_count}")
            logger.info(f"Number of synthetic NaN values in {column}: {synthetic_nan_count}")
        return uni_images


class Clustering(BaseMetric):
    def __init__(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        plot: bool,
        reports_path: str,
    ):
        super().__init__(original, synthetic, plot, reports_path)

    def calculate_all(self, categorical_columns: List[str], cont_columns: List[str]):
        for col in categorical_columns:
            map_dict = {
                k: i + 1 for i, k in enumerate(set(self.original[col]) | set(self.synthetic[col]))
            }
            self.original[col] = self.original[col].map(map_dict)
            self.synthetic[col] = self.synthetic[col].map(map_dict)

        # Perform clustering of original dataset
        # to get optimal number of clusters
        original_for_clustering = self.original[
            cont_columns + categorical_columns
            ].dropna()
        original_transformed = self.__preprocess_data(original_for_clustering)

        if len(original_transformed) == 0:
            logger.warning(
                "No clustering metric will be formed due to empty DataFrame"
            )
            return None

        optimal_clust_num = self.__get_optimal_number_of_clusters(
            original_transformed
            )

        logger.trace(f"Optimal number of clusters for "
                     f"original dataset: {optimal_clust_num}"
                     )

        row_limit = min(len(self.original), len(self.synthetic))

        # TODO check whether random_state affects the results
        self.merged = (
            pd.concat(
                [
                    self.original[cont_columns + categorical_columns].sample(
                        row_limit,
                        random_state=10
                    ),
                    self.synthetic[cont_columns + categorical_columns].sample(
                        row_limit,
                        random_state=10
                    ),
                ],
                keys=["original", "synthetic"],
            )
            .dropna()
            .reset_index()
        )

        if len(self.merged) == 0:
            logger.warning(
                "No clustering metric will be formed due to empty DataFrame"
            )
            return None
        self.merged_transformed = self.__preprocess_data(self.merged)

        statistics = self.__calculate_clusters(optimal_clust_num)
        statistics.columns = ["cluster", "dataset", "count"]

        diversity_scores = statistics.groupby('cluster').apply(
            self.calculate_diversity
            )
        mean_score = diversity_scores.mean()

        if self.plot:
            plt.clf()
            sns.set(font_scale=3)
            barplot = sns.barplot(
                data=statistics,
                x="cluster",
                y="count",
                hue="dataset",
                palette={"synthetic": "#ff9c54", "original": "#3f93e1"},
                saturation=1,
            )
            barplot.set_facecolor("#FFFFFF")
            ax = plt.gca()
            ax.spines[["top", "right", "left", "bottom"]].set_color("#E5E9EB")
            ax.yaxis.grid(color="#E5E9EB", linestyle="-", linewidth=1)
            original_label = Patch(color="#3f93e1", label="original")
            synthetic_label = Patch(color="#ff9c54", label="synthetic")
            plt.legend(
                handles=[original_label, synthetic_label],
                loc="upper left",
                bbox_to_anchor=(0, 1.09),
                ncol=2,
                frameon=False,
            )
            plt.savefig(
                f"{self.reports_path}/clusters_barplot.svg",
                bbox_inches="tight",
                format="svg",
            )
        return mean_score

    @staticmethod
    def calculate_diversity(statistics):
        """
        Calculate the diversity score for each cluster
        from collected statistics.
        If in cluster only one dataset is present, return 0.
        """
        if statistics['dataset'].nunique() == 2:
            return min(statistics['count']) / max(statistics['count'])
        else:
            return 0

    def __get_optimal_number_of_clusters(self, dataset):
        """
        Calculate the optimal number of clusters using Davies-Bouldin score
        """
        davies_bouldin_scores = []
        max_clusters = min(10, len(dataset))

        for i in range(2, max_clusters):
            clusters = KMeans(n_clusters=i, random_state=10).fit(
                dataset
                )
            labels = clusters.labels_
            score = davies_bouldin_score(dataset, labels)
            davies_bouldin_scores.append(score)

        # Get number of clusters with the lowest Davies-Bouldin score
        # +2 because the range starts from 2
        optimal_clusters = np.argmin(davies_bouldin_scores) + 2

        return optimal_clusters

    def __preprocess_data(self, dataset):
        transformed_dataset = dataset.apply(
            pd.to_numeric, axis=0, errors="ignore"
        ).select_dtypes(include="number")
        scaler = MinMaxScaler()
        transformed_dataset = scaler.fit_transform(transformed_dataset)

        return transformed_dataset

    def __calculate_clusters(self, n):
        clusters = KMeans(n_clusters=n, random_state=10).fit(
            self.merged_transformed
        )
        labels = clusters.labels_
        rows_labels = pd.DataFrame(
            {"origin": self.merged["level_0"], "cluster": labels}
        )

        return rows_labels.groupby(["cluster", "origin"]).size().reset_index()


class Utility(BaseMetric):
    def __init__(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        plot: bool,
        reports_path: str,
        sample_size: int = 100000,
    ):
        super().__init__(original, synthetic, plot, reports_path)

        self.sample_size = sample_size

    @staticmethod
    def check_empty_df(df: pd.DataFrame, df_type: Literal["original", "synthetic"]) -> bool:
        """
        Check if the dataframe is empty after dropping rows with missing values
        """
        if df.empty:
            logger.warning(
                f"Utility metric calculation is skipped: the {df_type} dataframe is empty "
                "after dropping rows with missing values (dropna() function is applied)"
            )
            return True
        return False

    def calculate_all(self, categorical_columns: List[str], cont_columns: List[str]):
        logger.info("Calculating utility metric")

        for col in categorical_columns:
            map_dict = {
                k: i + 1 for i, k in enumerate(
                    set(self.original[col]) | set(self.synthetic[col])
                )
            }
            self.original[col] = self.original[col].map(map_dict)
            self.synthetic[col] = self.synthetic[col].map(map_dict)

        self.original = self.original[cont_columns + categorical_columns].apply(
            pd.to_numeric, axis=0, errors="ignore"
        )
        self.synthetic = self.synthetic[cont_columns + categorical_columns].apply(
            pd.to_numeric, axis=0, errors="ignore"
        )
        self.original = self.original.select_dtypes(include="number").dropna()
        self.synthetic = self.synthetic.select_dtypes(include="number").dropna()

        if self.check_empty_df(self.original, "original"):
            return
        if self.check_empty_df(self.synthetic, "synthetic"):
            return

        self.synthetic = self.synthetic[self.original.columns]

        excluded_cols = [
            col
            for col in categorical_columns + cont_columns
            if self.original[col].nunique() < 2 or col.endswith(("_word_count", "_char_len"))
        ]
        binary_cols = [
            col
            for col in categorical_columns
            if self.original[col].nunique() == 2 and col not in excluded_cols
        ]
        cont_cols = [
            col
            for col in cont_columns
            if col not in binary_cols and col not in excluded_cols
        ]
        categorical_cols = [
            col
            for col in categorical_columns
            if col not in binary_cols and col not in excluded_cols
        ]

        # Check if the datasets after dropna() have more than
        # self.sample_size rows and stratified sampling is needed
        self.is_big_original_data_after_drop_na = (
            len(self.original) > self.sample_size
        )
        self.is_big_synthetic_data_after_drop_na = (
            len(self.synthetic) > self.sample_size
        )

        log_debug_msg = (
            "There are {} records left "
            "in {} data after dropping rows with null values "
            "in numeric columns. "
            "Creating stratified samples of size {} "
            "to calculate utility metric"
        )

        log_info_msg = (
            "There are more than {} records "
            "in {} data. "
            "Creating stratified samples of size {} "
            "to calculate utility metric"
        )

        if self.is_big_original_data_after_drop_na:
            logger.debug(log_debug_msg.format(
                len(self.original),
                'original',
                self.sample_size
                )
            )
            logger.info(
                log_info_msg.format(
                    self.sample_size,
                    'original',
                    self.sample_size
                )
            )

        if self.is_big_synthetic_data_after_drop_na:
            logger.debug(log_debug_msg.format(
                len(self.synthetic),
                'synthetic',
                self.sample_size
                )
            )
            logger.info(
                log_info_msg.format(
                    self.sample_size,
                    'synthetic',
                    self.sample_size
                )
            )

        (
            best_categ,
            score_categ,
            synth_score_categ
        ) = self.__create_multi_class_models(categorical_cols)
        (
            best_binary,
            score_binary,
            synth_score_binary,
        ) = self.__create_binary_class_models(binary_cols)
        (
            best_regres,
            score_regres,
            synth_regres_score
        ) = self.__create_regression_models(cont_cols)

        result = pd.DataFrame(
            {
                "original": [
                    round(score_binary, 3) if best_binary is not None else np.nan,
                    round(score_categ, 3) if best_categ is not None else np.nan,
                    # Do this to avoid zero division
                    round(max(score_regres, 1e-10), 3) if best_regres is not None else np.nan,
                ],
                "synthetic": [
                    round(synth_score_binary, 3) if best_binary is not None else np.nan,
                    round(synth_score_categ, 3) if best_categ is not None else np.nan,
                    abs(round(max(0, synth_regres_score), 3))
                    if best_regres is not None else np.nan,
                ],
                "synth_to_orig_ratio": [
                    round(synth_score_binary / score_binary, 3)
                    if best_binary is not None and score_binary != 0 else (
                        0 if score_binary == 0 else np.nan
                        ),
                    round(synth_score_categ / score_categ, 3)
                    if best_categ is not None and score_categ != 0 else (
                        0 if score_categ == 0 else np.nan
                        ),
                    abs(round(max(0, synth_regres_score) / score_regres, 3))
                    if best_regres is not None and score_regres != 0 else (
                        0 if score_regres == 0 else np.nan
                        ),
                ],
                "type": [
                    "Binary (" + f"{best_binary})" if best_binary is not None else "" + ")",
                    "Multiclass (" + f"{best_categ})" if best_categ is not None else "" + ")",
                    "Regression (" + f"{best_regres})" if best_regres is not None else "" + ")",
                ],
            }
        )
        result = pd.melt(result.dropna(), id_vars=["type", "synth_to_orig_ratio"])
        result.columns = ["Type", "Synth to orig ratio", "Origin", "Model score"]
        result = result.sort_values("Type").reset_index(drop=True)

        if self.plot:
            if result.empty:
                logger.info("No data to provide utility barplot")
            else:
                plt.clf()
                sns.set(font_scale=3)
                barplot = sns.barplot(
                    data=result,
                    x="Type",
                    y="Model score",
                    hue="Origin",
                    palette={"synthetic": "#FF9C54", "original": "#3F93E1"},
                    saturation=1,
                )
                barplot.set_facecolor("#FFFFFF")
                ax = plt.gca()
                ax.spines[["top", "right", "left", "bottom"]].set_color("#E5E9EB")
                ax.yaxis.grid(color="#E5E9EB", linestyle="-", linewidth=1)
                original_label = Patch(color="#3f93e1", label="original")
                synthetic_label = Patch(color="#ff9c54", label="synthetic")
                plt.axhline(y=0, color="#000000", linestyle="-")
                plt.legend(
                    handles=[original_label, synthetic_label],
                    loc="upper left",
                    bbox_to_anchor=(0, 1.09),
                    ncol=2,
                    frameon=False,
                )
                plt.savefig(
                    f"{self.reports_path}/utility_barplot.svg",
                    bbox_inches="tight",
                    format="svg",
                )
        log_msg = (
            "The ratio of synthetic {} accuracy to original is "
            "{}. The model considers the '{}' column "
            "as a target and other columns as predictors"
        )
        if best_binary is not None:
            score = (0 if score_binary == 0 else
                     round(synth_score_binary/score_binary, 4)
                     )
            logger.info(log_msg.format(
                'binary',
                score,
                best_binary
                )
            )
        if best_categ is not None:
            score = (0 if score_categ == 0 else
                     round(synth_score_categ/score_categ, 4)
                     )
            logger.info(log_msg.format(
                'multiclass',
                score,
                best_categ
                )
            )
        if best_regres is not None:
            score = (0 if score_regres == 0 else
                     abs(round(max(0, synth_regres_score) / score_regres, 4))
                     )
            logger.info(log_msg.format(
                'regression',
                score,
                best_regres
                )
            )
        logger.info("Utility metric is calculated")
        return result

    @staticmethod
    def __get_accuracy_score(y_true, y_pred, task_type):
        if task_type != "regression":
            score = accuracy_score(y_true=y_true, y_pred=y_pred)
        else:
            score = r2_score(y_true=y_true, y_pred=y_pred)
        return score

    def __model_process(self, model_object, targets, task_type):
        best_score = -1
        best_target = None
        best_model = None
        synthetic_score = -1

        for i, col in tqdm.tqdm(
            iterable=enumerate(targets),
            desc=f"Calculating utility metric for {task_type}",
            total=len(targets),
        ):
            if self.original.shape[1] < 2:
                logger.info(
                    "There is only one column in the dataset. "
                    "It will not be used in utility metric."
                )
                continue
            original = pd.get_dummies(self.original.drop(col, axis=1))
            original = StandardScaler().fit_transform(original)
            model_y = self.original[col].values

            # If the original dataset is large create a stratified sample
            if self.is_big_original_data_after_drop_na:
                original, _, model_y, _ = self.__perform_train_test_split(
                    original, model_y, self.sample_size
                )

            # validate number of unique values and instances in each class
            if not self._valid_target(col, model_y, task_type):
                continue

            (
                original_train,
                original_test,
                model_y_train,
                model_y_test
            ) = self.__perform_train_test_split(
                original,
                model_y,
                train_size=0.8
            )

            model = model_object.fit(
                X=original_train, y=model_y_train
            )
            score = self.__get_accuracy_score(
                model_y_test,
                model.predict(original_test),
                task_type,
            )
            if score > best_score:
                best_score = score
                best_target = col
                best_model = model

        # Calculate synthetic score
        if best_score > -1:
            if best_score < 0.6:
                logger.info(
                    f"The best score for all possible {task_type} models "
                    f"for the original data is "
                    f"{round(best_score, 4)}, which is below 0.6. "
                    f"The utility metric is unreliable"
                )
            synthetic = pd.get_dummies(self.synthetic.drop(best_target, axis=1))
            synthetic = StandardScaler().fit_transform(synthetic)
            synthetic_y = self.synthetic[best_target].values

            # If the synthetic dataset is large
            # create a stratified sample
            if self.is_big_synthetic_data_after_drop_na:
                synthetic, _, synthetic_y, _ = self.__perform_train_test_split(
                    synthetic, synthetic_y, self.sample_size
                )

            synthetic_score = self.__get_accuracy_score(
                synthetic_y,
                best_model.predict(synthetic),
                task_type,
            )

        return best_target, best_score, synthetic_score

    @staticmethod
    def __perform_train_test_split(data, model_y, train_size):
        """
        Splits the data into train and test sets.
        Tries stratified sampling first; if it fails,
        perform random sampling.
        """
        try:
            data_train, data_test, model_y_train, model_y_test = train_test_split(
                data,
                model_y,
                train_size=train_size,
                stratify=model_y,
                random_state=10
            )
        except ValueError:
            data_train, data_test, model_y_train, model_y_test = train_test_split(
                data,
                model_y,
                train_size=train_size,
                stratify=None,
                random_state=10
            )

        return data_train, data_test, model_y_train, model_y_test

    def __create_binary_class_models(self, binary_targets):
        best_target, score, synthetic_score = self.__model_process(
            LogisticRegression(n_jobs=-1, random_state=10),
            binary_targets, "binary classification"
        )
        return best_target, score, synthetic_score

    def __create_multi_class_models(self, multiclass_targets):
        best_target, score, synthetic_score = self.__model_process(
            RandomForestClassifier(n_jobs=-1, random_state=10),
            multiclass_targets,
            "multiclass classification",
        )
        return best_target, score, synthetic_score

    def __create_regression_models(self, cont_targets):
        best_target, score, synthetic_score = self.__model_process(
            RandomForestRegressor(n_jobs=-1, random_state=10),
            cont_targets, "regression"
        )
        return best_target, score, synthetic_score

    @staticmethod
    def _valid_target(col, model_y, task_type):
        """
        Validate the target column in utility metric calculation.

        Args:
        col (str): The name of the column being checked.
        model_y (array-like): The target values.
        task_type (str): type task

        Returns:
        bool: True if the column is valid
        """
        unique_values, unique_counts = np.unique(model_y, return_counts=True)

        # check if there is only one unique value in the column
        if len(unique_values) < 2:
            logger.info(
                f"Column '{col}' has only one unique value. "
                "It will not be used as target column "
                f"in utility metric calculation for {task_type}."
            )
            return False

        # check if there is more than 1 sample in each class
        if task_type in ["binary classification", "multiclass classification"]:
            if np.min(unique_counts) < 2:
                logger.info(
                    f"Column '{col}' has a class with less than 2 samples. "
                    "It will not be used as target column "
                    f"in utility metric calculation for {task_type}."
                )
                return False

        return True
