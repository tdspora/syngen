from typing import Union, List
from abc import ABC
from itertools import combinations
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
import re

from syngen.ml.pipeline import get_nan_labels, nan_labels_to_float
from syngen.ml.metrics.utils import text_to_continuous


class BaseMetric(ABC):
    def __init__(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        draws_path: str = None,
        plot: bool = True,
    ):
        columns_nan_labels = get_nan_labels(original)
        self.original = nan_labels_to_float(original, columns_nan_labels)
        self.synthetic = nan_labels_to_float(synthetic, columns_nan_labels)
        self.draws_path = draws_path
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
        draws_path: str,
    ):
        super().__init__(original, synthetic)
        self.plot = plot
        self.draws_path = draws_path

    def calculate_all(self, categ_columns: List[str]):
        self.original[categ_columns] = self.original[categ_columns].fillna("")
        self.synthetic[categ_columns] = self.synthetic[categ_columns].fillna("")
        self.heatmap, self.labels = self.__compute_vs_columns(categ_columns)

        if self.plot:
            sns.set(rc={'figure.figsize': (16, 12)})
            heatmap = sns.heatmap(
                self.heatmap,
                xticklabels=self.labels,
                yticklabels=self.labels,
                annot=True,
            )

            heatmap.figure.tight_layout()
            plt.savefig(f"{self.draws_path}/accuracy_heatmap.png")

    def calculate_heatmap_median(self, heatmap):
        heatmap_no_diag = heatmap[~np.eye(heatmap.shape[0], dtype=bool)].reshape(
            heatmap.shape[0], -1
        )
        heatmap_median = np.median(heatmap_no_diag)
        print("Median of Jensen Shannon Distance heatmap is", "%.3f" % heatmap_median)
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

        # return min(original_score, synthetic_score) / max(original_score, synthetic_score)
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

    def __compute_vs_columns(self, categ_columns: List[str]):
        valid_cols = self.original.columns
        mask = list(map(lambda col: col in categ_columns, valid_cols))
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

    def __normalize(self, dist):
        min_ = dist.min()
        max_ = dist.max()
        # std = (dist - min_) / (max_ - min_)
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
    def calculate_all(
        self, categ_columns: List[str], cont_columns: List[str], text_columns: List[str]
    ):
        self.original = text_to_continuous(self.original, text_columns)
        self.synthetic = text_to_continuous(self.synthetic, text_columns)
        self.original[categ_columns] = self.original[categ_columns].fillna("")
        self.synthetic[categ_columns] = self.synthetic[categ_columns].fillna("")
        for col in [i + "_word_count" for i in text_columns]:
            if self.original[col].nunique() < 50:
                categ_columns = categ_columns | {col}
            else:
                cont_columns = cont_columns | {col}
        cont_columns += [i + "_char_len" for i in text_columns]

        for col in categ_columns:
            map_dict = {
                k: i + 1
                for i, k in enumerate(
                    set(self.original[col]) | set(self.synthetic[col])
                )
            }
            self.original[col] = self.original[col].map(map_dict)
            self.synthetic[col] = self.synthetic[col].map(map_dict)

        self.original_heatmap = self.__calculate_correlations(
            self.original[categ_columns + cont_columns]
        )
        self.synthetic_heatmap = self.__calculate_correlations(
            self.synthetic[categ_columns + cont_columns]
        )

    def __calculate_correlations(self, data):
        return abs(data.corr())


class BivariateMetric(BaseMetric):
    def __init__(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        plot: bool,
        draws_path: str,
    ):
        super().__init__(original, synthetic)
        self.plot = plot
        self.draws_path = draws_path

    def calculate_all(
        self,
        cont_columns: List[str],
        categ_columns: List[str],
        num_not_na_cont_ticks: int = 10,
    ):
        self.num_not_na_ticks = num_not_na_cont_ticks
        all_columns = set(cont_columns) | set(categ_columns)
        column_pairs = list(combinations(all_columns, 2))
        for first_col, second_col in column_pairs:
            fig, self._axes = plt.subplots(1, 2, figsize=(30, 15))
            if first_col in cont_columns:
                if second_col in cont_columns:
                    (
                        heatmap_orig_data,
                        heatmap_synthetic_data,
                    ) = self._calculate_pair_continuous_vs_continuous(
                        y_col=first_col, x_col=second_col
                    )
                elif second_col in categ_columns:
                    (
                        heatmap_orig_data,
                        heatmap_synthetic_data,
                    ) = self._calculate_pair_continuous_vs_categ(
                        cont_col=first_col, categ_col=second_col
                    )
            elif first_col in categ_columns:
                if second_col in cont_columns:
                    (
                        heatmap_orig_data,
                        heatmap_synthetic_data,
                    ) = self._calculate_pair_continuous_vs_categ(
                        cont_col=second_col, categ_col=first_col
                    )
                elif second_col in categ_columns:
                    (
                        heatmap_orig_data,
                        heatmap_synthetic_data,
                    ) = self._calculate_pair_categ_vs_categ(
                        y_col=first_col, x_col=second_col
                    )

            heatmap_min, heatmap_max = self.get_common_min_max(
                heatmap_orig_data[0], heatmap_synthetic_data[0]
            )
            self._plot_heatmap(
                heatmap_orig_data,
                f"Original: {first_col} vs. {second_col}",
                0,
                heatmap_min,
                heatmap_max,
                cbar=False,
            )

            self._plot_heatmap(
                heatmap_synthetic_data,
                f"Synthetic: {first_col} vs. {second_col}",
                1,
                heatmap_min,
                heatmap_max,
                cbar=True,
            )
            # plt.show()
            print(f"{self.draws_path}/bivariate_{first_col}_{second_col}.png")
            plt.savefig(f"{self.draws_path}/bivariate_{first_col}_{second_col}.png")

    @staticmethod
    def get_common_min_max(original, synthetic):
        original_flat = [c for row in original for c in row]
        synthetic_flat = [c for row in synthetic for c in row]
        vmax = max(max(original_flat), max(synthetic_flat))
        vmin = min(min(original_flat), min(synthetic_flat))
        return vmin, vmax

    def _plot_heatmap(
        self,
        heatmap_data: List,
        title: str,
        plt_index: int,
        vmin: float,
        vmax: float,
        cbar=True,
    ):
        ax = self._axes.flat[plt_index]
        ax.tick_params(labelsize=14)
        heatmap, x_tick_labels, y_tick_labels = heatmap_data
        sns.heatmap(
            heatmap,
            xticklabels=x_tick_labels,
            yticklabels=y_tick_labels,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap="Blues",
            cbar=cbar,
        )
        ax.set_title(title, fontsize=18)

    def _get_continuous_ticks(self, col_name: str):
        original_col_values = self.original[col_name].dropna().values
        synthetic_col_values = self.original[col_name].dropna().values
        col_values = np.concatenate((original_col_values, synthetic_col_values))
        return np.percentile(col_values, list(range(0, 101, self.num_not_na_ticks)))

    def _get_categorical_ticks(self, col_name: str):
        categ_ticks = sorted(
            set(
                set(self.original[col_name].fillna("?"))
                | set(self.synthetic[col_name].fillna("?"))
            )
        )
        return categ_ticks

    def _format_categorical_labels(self, labels):
        return [
            str(label) if len(str(label)) < 15 else str(label[:12]) + "..."
            for label in labels
        ]

    def _smooth(self, dist):
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
                    count = df[df[y_col].eq(y_val) & df[x_col].eq(x_val)].shape[0]
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
        synthetic_heatmap = calc_heatmap_data(
            self.synthetic, y_col, x_col, xtick, ytick
        )
        return original_heatmap, synthetic_heatmap

    def _calculate_pair_continuous_vs_categ(self, cont_col: str, categ_col: str):
        def calc_heatmap_data(
            df: pd.DataFrame,
            cont_col: str,
            categ_col: str,
            xticks: List[float],
            yticks: List[str],
        ):
            heatmap = []
            cont_has_nan = df[cont_col].isnull().values.any()
            cont_ticks = (
                self.num_not_na_ticks + 1 if cont_has_nan else self.num_not_na_ticks
            )
            for i in range(cont_ticks):
                heatmap_row = []
                if i == 0:
                    x_index = (df[cont_col] >= xticks[i]) & (
                        df[cont_col] <= xticks[i + 1]
                    )
                elif i < self.num_not_na_ticks:
                    x_index = (df[cont_col] > xticks[i]) & (
                        df[cont_col] <= xticks[i + 1]
                    )
                else:
                    x_index = df[cont_col].isna()

                for val in yticks:
                    count = df[x_index & df[categ_col].eq(val)].shape[0]
                    heatmap_row.append(count)
                heatmap.append(heatmap_row)

            if cont_has_nan:
                xticks = np.append(xticks, np.nan)

            return heatmap, self._format_categorical_labels(yticks), xticks[1:]

        xtick = self._smooth(self._get_continuous_ticks(cont_col))
        ytick = self._get_categorical_ticks(categ_col)

        original_heatmap = calc_heatmap_data(
            self.original, cont_col, categ_col, xtick, ytick
        )
        synthetic_heatmap = calc_heatmap_data(
            self.synthetic, cont_col, categ_col, xtick, ytick
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
            y_ticks_num = (
                self.num_not_na_ticks + 1 if y_has_nan else self.num_not_na_ticks
            )
            x_ticks_num = (
                self.num_not_na_ticks + 1 if x_has_nan else self.num_not_na_ticks
            )

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
                        x_index = (df[x_col] >= xticks[j]) & (
                            df[x_col] <= xticks[j + 1]
                        )
                    elif j < self.num_not_na_ticks:
                        x_index = (df[x_col] > xticks[j]) & (df[x_col] <= xticks[j + 1])
                    else:
                        x_index = df[x_col].isna()

                    count = df[y_index & x_index].shape[0]
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
        synthetic_heatmap = calc_heatmap_data(
            self.synthetic, y_col, x_col, xtick, ytick
        )

        return original_heatmap, synthetic_heatmap


class UnivariateMetric(BaseMetric):
    def __init__(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        plot: bool,
        draws_path: str,
    ):
        super().__init__(original, synthetic)
        self.plot = plot
        self.draws_path = draws_path

    def _calculate(self, column):
        pass

    def calculate_all(
        self, cont_columns: List[str], categ_columns: List[str], print_nan: bool = False
    ):
        cont_columns = list(cont_columns)

        for col in cont_columns:
            self.__calc_continuous(col, print_nan)
        for col in categ_columns:
            self.__calc_categ(col)

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
            return re.sub("\$|\^", "", label)

        original_column = self.original[column].fillna("?")
        synthetic_column = self.synthetic[column].fillna("?")
        full_values_set = set(original_column.values) | set(synthetic_column.values)

        original_ratio_counts = plot_dist(original_column, True, full_values_set)
        original_labels = list(original_ratio_counts.keys())
        original_ratio = list(original_ratio_counts.values())

        synthetic_ratio_counts = plot_dist(synthetic_column, False, full_values_set)
        synthetic_ratio = [synthetic_ratio_counts[label] for label in original_labels]

        if self.plot:
            fig = plt.figure()

            width = 0.35
            x = np.arange(len(original_labels))
            plt.bar(
                x - width / 2,
                original_ratio,
                width=width,
                label="Original",
                color="#009DC4",
            )
            plt.bar(
                x + width / 2,
                synthetic_ratio,
                width=width,
                label="Synthetic",
                color="#F6748B",
            )
            ax = plt.gca()
            ax.set_xticks(x)
            ax.set_xticklabels(
                [
                    str(sanitize_labels(label[:30])) + "..." if len(str(label)) > 33 else sanitize_labels(str(label))
                    for label in original_labels
                ]
            )

            fig.autofmt_xdate()
            plt.xlabel("Category")
            plt.ylabel("Percents")
            plt.legend()
            plt.title(column)
            if self.draws_path:
                plt.savefig(f"{self.draws_path}/univariate_{column}.png")
                # pl.dump(fig, open(f"{self.draws_path}univariate_{column}.pickle", "wb"))

    def __calc_continuous(self, column: str, print_nan: bool = False):
        original_nan_count = self.original[column].isna().sum()
        synthetic_nan_count = self.synthetic[column].isna().sum()
        original_unique_count = self.original[column].nunique()
        synthetic_unique_count = self.synthetic[column].nunique()

        if self.plot and original_unique_count > 1 and synthetic_unique_count > 1:
            fig_handle = plt.figure()
            # Kernel Density Estimation plot
            self.original[column].plot(kind="density", color="#009DC4")
            self.synthetic[column].plot(kind="density", color="#F6748B")
            plt.xlabel("Value")
            plt.legend(["Original", "Synthetic"])
            plt.title(column)
            if self.draws_path:
                plt.savefig(f"{self.draws_path}/univariate_{column}.png")
                # pl.dump(
                #     fig_handle,
                #     open(f"{self.draws_path}univariate_{column}.pickle", "wb"),
                # )
        if print_nan:
            print(f"Number of original NaN values in {column}: {original_nan_count}")
            print(f"Number of synthetic NaN values in {column}: {synthetic_nan_count}")
