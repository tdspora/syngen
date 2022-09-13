import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
import pandas as pd
from typing import List
import seaborn as sns

from ml.metrics import BaseMetric


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
