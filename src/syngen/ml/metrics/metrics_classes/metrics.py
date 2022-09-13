from abc import ABC
import matplotlib.pyplot as plt
from typing import Union
import numpy as np
import pandas as pd
import scipy.stats as st
from typing import List
import seaborn as sns

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
