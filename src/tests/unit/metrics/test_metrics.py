import pandas as pd

from syngen.ml.metrics.metrics_classes.metrics import Clustering
from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME


def test_clustering_calculate_all(rp_logger):
    """
    Testing the clustering metric score
    """
    rp_logger.info(
        "Testing the clustering metric score"
    )

    path_to_original = f"{DIR_NAME}/unit/metrics/fixtures/clusters-2.csv"
    path_to_synthetic = f"{DIR_NAME}/unit/metrics/fixtures/synthetic_clusters-2_10k.csv"

    original = pd.read_csv(path_to_original)
    synthetic = pd.read_csv(path_to_synthetic)

    clustering = Clustering(original, synthetic, plot=False, reports_path="")

    categ_columns = []
    cont_columns = original.columns.to_list()

    mean_score = clustering.calculate_all(categ_columns, cont_columns)

    threshold = 0.95
    assert mean_score >= threshold, f"Mean score should not less than {threshold}"

    rp_logger.info(SUCCESSFUL_MESSAGE)
