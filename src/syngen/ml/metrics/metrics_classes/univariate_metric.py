import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import pandas as pd
from typing import List

from ml.metrics import BaseMetric


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
                    str(label[:30]) + "..." if len(str(label)) > 33 else str(label)
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

