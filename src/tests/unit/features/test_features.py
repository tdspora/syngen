import pytest

import pandas as pd
import numpy as np
import tensorflow as tf

from syngen.ml.vae.models.features import BaseFeature, CharBasedTextFeature
from tests.conftest import SUCCESSFUL_MESSAGE


@pytest.mark.parametrize(
    "name, expected_name",
    [
        ("base_feature", "base-feature"),
        ("baseFeature", "baseFeature"),

    ]
)
def test_init_base_feature(name, expected_name, rp_logger):
    rp_logger.info(
        "Testing the initialization of the class BaseFeature with different names"
    )
    feature = BaseFeature(name)
    assert feature.name == expected_name
    assert feature.original_name == name
    assert feature.weight == 1.0
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_inverse_transform_of_char_based_text_feature(rp_logger):
    rp_logger.info(
        "Testing the method 'inverse_transform' of the class CharBasedTextFeature"
    )
    feature = CharBasedTextFeature(
        name="text_column",
        text_max_len=4
    )
    data = pd.DataFrame(
        data=[
            "C433", "C794", "0786", "7393", "C983", "9898", "9736",
            "5819", "C472", "C646", "C749", "6698", "C777", "C367",
            "0434", "9889", "C779", "C936", "C379", "C879"
        ]
    )
    feature.fit(data=data)
    feature.tokenizer.inverse_dict = {
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "6",
        6: "5",
        7: "7",
        8: "0",
        9: "8",
        10: "9",
        11: "C",
        12: "M"
    }
    data = np.loadtxt(
        "tests/unit/features/fixtures/tensor.csv"
    ).reshape((20, 4, 12)).astype(np.float32)
    result = feature.inverse_transform(data=data)
    assert len(result) == 20
    assert np.all(np.array([len(i) for i in result]) == 4)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_top_k_filtering(rp_logger):
    rp_logger.info(
        "Testing the method 'top_k_top_p_filtering' "
        "of the class CharBasedTextFeature in top_k mode"
    )
    feature = CharBasedTextFeature(
        name="text_column",
        text_max_len=4
    )
    data = np.loadtxt(
        "tests/unit/features/fixtures/tensor.csv"
    ).reshape((20, 4, 12)).astype(np.float32)
    result = feature.top_k_top_p_filtering(data, top_k=3, filter_value=0.0)
    ethalon = np.loadtxt(
        "tests/unit/features/fixtures/top_k-tensor.csv"
    ).reshape((20, 4, 12)).astype(np.float32)
    np.testing.assert_array_equal(result, ethalon)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_top_p_filtering(rp_logger):
    rp_logger.info(
        "Testing the method 'top_k_top_p_filtering' "
        "of the class CharBasedTextFeature in top_p mode"
    )
    feature = CharBasedTextFeature(
        name="text_column",
        text_max_len=4
    )
    data = np.loadtxt(
        "tests/unit/features/fixtures/tensor.csv"
    ).reshape((20, 4, 12)).astype(np.float32)

    result = tf.nn.softmax(
        np.stack(
            [feature.top_k_top_p_filtering(matrix, top_p=0.7) for matrix in data],
            axis=0),
        axis=-1).numpy()
    ethalon = np.loadtxt(
        "tests/unit/features/fixtures/top_p-tensor.csv"
    ).reshape((20, 4, 12)).astype(np.float32)
    np.testing.assert_array_equal(result, ethalon)
    rp_logger.info(SUCCESSFUL_MESSAGE)
