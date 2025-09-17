import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from unittest.mock import patch, MagicMock
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    QuantileTransformer
)

from syngen.ml.vae.models.features import (
    BaseFeature,
    CharBasedTextFeature,
    ContinuousFeature,
    CategoricalFeature,
)
from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME


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
        f"{DIR_NAME}/unit/features/fixtures/tensor.csv"
    ).reshape((20, 4, 12)).astype(np.float32)
    result = feature.inverse_transform(data=data)
    assert len(result) == 20
    assert np.all(np.array([len(i) for i in result]) == 4)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_top_k_filtering(rp_logger):
    rp_logger.info(
        "Testing the method '_top_k_filtering' of the class CharBasedTextFeature"
    )
    feature = CharBasedTextFeature(
        name="text_column",
        text_max_len=4
    )
    data = np.loadtxt(
        f"{DIR_NAME}/unit/features/fixtures/tensor.csv"
    ).reshape((20, 4, 12))
    result = feature._top_k_filtering(data, top_k=3)
    ethalon = np.loadtxt(
        f"{DIR_NAME}/unit/features/fixtures/top_k-tensor.csv"
    ).reshape((20, 4, 12))
    np.testing.assert_array_equal(result, ethalon)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_top_p_filtering(rp_logger):
    rp_logger.info(
        "Testing the method '_top_p_filtering' of the class CharBasedTextFeature"
    )
    feature = CharBasedTextFeature(
        name="text_column",
        text_max_len=4
    )
    data = tf.nn.softmax(
        np.loadtxt(
            f"{DIR_NAME}/unit/features/fixtures/tensor.csv"
        ).reshape((20, 4, 12)).astype(np.float32),
        axis=-1
    )

    result = feature._top_p_filtering(data, top_p=0.7)
    result /= result.sum(axis=2, keepdims=True)

    ethalon = np.loadtxt(
        f"{DIR_NAME}/unit/features/fixtures/top_p-tensor.csv"
    ).reshape((20, 4, 12))
    np.testing.assert_allclose(result, ethalon, rtol=1e-6)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "mock_pvalue, mock_kurt, kurtosis_threshold, expected_scaler_type",
    [
        # Normal distribution case - use StandardScaler
        (0.1, 3.0, 50, StandardScaler),
        # Non-normal, low kurtosis case - use MinMaxScaler
        (0.01, 3.0, 50, MinMaxScaler),
        # Non-normal, high kurtosis case - use QuantileTransformer
        (0.01, 60.0, 50, QuantileTransformer),
        # Custom threshold - below threshold uses MinMaxScaler
        (0.01, 24.0, 25, MinMaxScaler),
        # Custom threshold - above threshold uses QuantileTransformer
        (0.01, 26.0, 25, QuantileTransformer),
    ]
)
def test_select_scaler(
    mock_pvalue, mock_kurt,
    kurtosis_threshold, expected_scaler_type,
    rp_logger
):
    rp_logger.info(
        "Testing _select_scaler method with different data distributions and thresholds"
    )
    feature = ContinuousFeature(name="test_feature")
    data = pd.DataFrame(np.random.uniform(0, 1, size=(1000, 1)))
    # Mock shapiro to return configured p-value
    # Mock kurtosis to return configured kurtosis value
    with patch('syngen.ml.vae.models.features.shapiro',
               return_value=MagicMock(pvalue=mock_pvalue)
               ), \
         patch('syngen.ml.vae.models.features.kurtosis',
               return_value=mock_kurt):
        # mock _get_quantile_transformer_params
        if expected_scaler_type == QuantileTransformer:
            with patch.object(
                feature,
                '_get_quantile_transformer_params',
                return_value={
                    'n_quantiles': 10000,
                    'subsample': None,
                    'output_distribution': 'normal'
                }
            ):
                scaler = feature._select_scaler(
                    data, kurtosis_threshold=kurtosis_threshold
                    )
                assert isinstance(scaler, expected_scaler_type)
        else:
            scaler = feature._select_scaler(
                data, kurtosis_threshold=kurtosis_threshold
            )
            assert isinstance(scaler, expected_scaler_type)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "n_samples, kurt, threshold, expected_factor",
    [
        (10000, 75, 50, 1.0),   # Just above threshold
        (10000, 150, 50, 1.2),  # > 2x threshold
        (10000, 250, 50, 1.5),  # > 4x threshold
    ]
)
def test_get_quantile_transformer_params(
    n_samples, kurt, threshold, expected_factor, rp_logger
):
    rp_logger.info(
        "Testing _get_quantile_transformer_params method with different kurtosis values"
    )

    feature = ContinuousFeature(name="test_feature")
    result = feature._get_quantile_transformer_params(
        n_samples=n_samples,
        kurt=kurt,
        kurtosis_threshold=threshold
    )

    expected_n_quantiles = min(int(n_samples * expected_factor), n_samples)

    assert result['n_quantiles'] == expected_n_quantiles
    assert result['subsample'] is None
    assert result['output_distribution'] == 'normal'

    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "input_data, expected_mapping, expected_inverse_mapping, expected_dimension",
    [
        (
            pd.DataFrame({'col': ['A', 'B', 'C']}),
            {'A': 0, 'B': 1, 'C': 2},
            {0: 'A', 1: 'B', 2: 'C'},
            3
        ),
        (
            pd.DataFrame({'col': [1, 2, 3]}),  # integers
            {'1': 0, '2': 1, '3': 2},
            {0: '1', 1: '2', 2: '3'},
            3
        ),

        (
            pd.DataFrame({'col': [1.0, 2.0, 3.0]}),  # floats
            {'1.0': 0, '2.0': 1, '3.0': 2},
            {0: '1.0', 1: '2.0', 2: '3.0'},
            3
        ),
        (
            pd.DataFrame({'col': ['A', 'A', 'A']}),  # unique category
            {'A': 0},
            {0: 'A'},
            1
        ),
        (
            pd.DataFrame({'col': [1, 2, '?']}),  # with filled NaN
            {'1': 0, '2': 1, '?': 2},
            {0: '1', 1: '2', 2: '?'},
            3
        ),
    ]
)
def test_categorical_feature_fit(
    input_data,
    expected_mapping,
    expected_inverse_mapping,
    expected_dimension,
    rp_logger
):
    rp_logger.info(
        "Testing the method 'fit' of the class CategoricalFeature"
    )
    feature = CategoricalFeature(name="categorical_feature")
    feature.fit(input_data)

    assert feature.mapping == expected_mapping
    assert feature.inverse_mapping == expected_inverse_mapping
    assert feature.input_dimension == expected_dimension
    assert callable(feature.inverse_vectorizer)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "input_data, expected_output",
    [
        (
            pd.DataFrame({'col': ['A', 'B', 'C']}),
            np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]).astype('float32')
        ),
        # Integers
        (
            pd.DataFrame({'col': [1, 2, 3]}),
            np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]).astype('float32')
        ),
        # Floats
        (
            pd.DataFrame({'col': [1.0, 2.0, 3.0]}),
            np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]).astype('float32')
        ),
        # Single category repeated
        (
            pd.DataFrame({'col': ['A', 'A', 'A']}),
            np.array([
                [1],
                [1],
                [1]
            ]).astype('float32')
        ),
        # With filled NaN
        (
            pd.DataFrame({'col': [1, 2, '?', 2]}),
            np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 1, 0]
            ]).astype('float32')
        )
    ]
)
def test_categorical_feature_transform(input_data, expected_output, rp_logger):
    rp_logger.info(
        "Testing the method 'transform' of the class CategoricalFeature"
    )
    feature = CategoricalFeature(name="categorical_feature")
    feature.fit(input_data)
    transformed = feature.transform(input_data)

    np.testing.assert_array_equal(transformed, expected_output)
    assert transformed.dtype == np.float32
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "fit_data, encoded_data, expected_output",
    [
        (
            pd.DataFrame({'col': ['A', 'B', 'C', 'A']}),
            np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0]
            ]),
            np.array(['A', 'B', 'C', 'A'])
        ),
        (
            pd.DataFrame({'col': [1, 2, 3]}),
            np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]),
            np.array(['1', '2', '3'])
        ),
        (
            pd.DataFrame({'col': [1.0, 2.0, 3.0]}),
            np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]),
            np.array(['1.0', '2.0', '3.0'])
        ),
        (
            pd.DataFrame({'col': ['A', 'B', 'C']}),
            np.array([
                [0.9, 0.1, 0.0],
                [0.1, 0.8, 0.1],
                [0.0, 0.1, 0.9]
            ]),
            np.array(['A', 'B', 'C'])
        ),
        (
            pd.DataFrame({'col': [1, 2, '?']}),
            np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]),
            np.array(['1', '2', None])
        )
    ]
)
def test_categorical_feature_inverse_transform(
            fit_data, encoded_data,
            expected_output, rp_logger
):
    rp_logger.info(
        "Testing the method 'inverse_transform' "
        "of the class CategoricalFeature"
    )
    feature = CategoricalFeature(name="categorical_feature")
    feature.fit(fit_data)
    inversed = feature.inverse_transform(encoded_data)

    np.testing.assert_array_equal(inversed, expected_output)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "input_data",
    [
        pd.DataFrame({'col': ['A', 'B', 'C', 'A', 'B']}),
        pd.DataFrame({'col': [1, 2, 3, 1, 2]}),
        pd.DataFrame({'col': [1.0, 2.0, 3.0, 1.0, 2.0]}),
        pd.DataFrame({'col': ['A', 'A', 'A']})
    ]
)
def test_categorical_feature_fit_transform_pipeline(input_data, rp_logger):
    rp_logger.info(
        "Testing full pipeline (fit->transform->inverse_transform) "
        "of the class CategoricalFeature"
    )
    feature = CategoricalFeature(name="categorical_feature")

    feature.fit(input_data)
    transformed = feature.transform(input_data)

    recovered = feature.inverse_transform(transformed)

    np.testing.assert_array_equal(
        recovered,
        input_data['col'].astype(str).values
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)
