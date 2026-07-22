import pickle

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
    DateFeature,
    EmailFeature,
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


def test_date_feature_fit_transform_with_numpy_datetime64(rp_logger):
    """EPMCTDM-7581: a datetime64[ns] column (as loaded from Parquet/Delta) must
    not poison the date feature. Previously 'datetime_to_timestamp' returned None
    for numpy.datetime64, making the fitted/transformed data entirely NaN, which
    diverged training to a NaN model and crashed at inference."""
    rp_logger.info(
        "Testing 'DateFeature.fit'/'transform' on a datetime64[ns] column"
    )
    dates = pd.Series(pd.date_range("2015-01-01", periods=300, freq="D"))
    data = pd.DataFrame({"created_dt": dates})
    assert str(data["created_dt"].dtype) == "datetime64[ns]"

    feature = DateFeature(name="created_dt")
    feature.fit(
        data,
        date_mapping={"created_dt": "%Y-%m-%d"},
        to_datetime_conversion={"created_dt": False},
    )
    transformed = np.asarray(feature.transform(data), dtype=float)
    assert np.isfinite(transformed).all()
    assert not np.isnan(transformed).any()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_date_feature_fit_transform_with_tz_aware_datetime64(rp_logger):
    """EPMCTDM-7581 (Avro counterpart): a tz-aware datetime64[ns, UTC] column (as
    produced by Avro 'timestamp' logical types via pandavro) must not poison the
    date feature. The timezone is stripped during conversion and every transformed
    value stays finite."""
    rp_logger.info(
        "Testing 'DateFeature.fit'/'transform' on a tz-aware datetime64[ns, UTC] column"
    )
    dates = pd.date_range("2015-01-01", periods=300, freq="D").tz_localize("UTC")
    data = pd.DataFrame({"created_ts": dates})
    assert str(data["created_ts"].dtype) == "datetime64[ns, UTC]"

    feature = DateFeature(name="created_ts")
    feature.fit(
        data,
        date_mapping={"created_ts": "%Y-%m-%d %H:%M:%S%z"},
        to_datetime_conversion={"created_ts": True},
    )
    transformed = np.asarray(feature.transform(data), dtype=float)
    assert np.isfinite(transformed).all()
    assert not np.isnan(transformed).any()
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


def _build_text_feature_model(feature):
    shared_decoder = tf.keras.layers.Dense(128)(feature.encoder)
    output = feature.create_decoder(shared_decoder)
    model = tf.keras.Model(feature.input, output)
    model.add_loss(feature.loss)
    return model


def test_char_feature_uses_integer_tokens_and_complete_vocabulary(rp_logger):
    data = pd.DataFrame({"text": ["ab", "caba"]})
    feature = CharBasedTextFeature(name="text", text_max_len=5)
    feature.fit(data)

    transformed = feature.transform(data)

    assert transformed.dtype == np.int32
    assert transformed.shape == (2, 5)
    assert feature.vocab_size == len(feature.tokenizer.word_index) + 1
    assert transformed.max() == len(feature.tokenizer.word_index)
    assert np.all(transformed[:, -1] == 0)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_char_feature_optimized_model_has_finite_sparse_loss(rp_logger):
    data = pd.DataFrame({"text": ["ABC123", "DEF456", "ABC456"]})
    feature = CharBasedTextFeature(name="text", text_max_len=7)
    feature.fit(data)
    model = _build_text_feature_model(feature)

    transformed = feature.transform(data)
    output = model(transformed, training=True)
    loss = tf.add_n(model.losses)
    expected_loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            transformed,
            output,
            from_logits=True,
        )
    )
    embeddings = [
        layer
        for layer in model.submodules
        if isinstance(layer, tf.keras.layers.Embedding)
    ]

    assert output.shape == (3, 7, feature.vocab_size)
    assert np.isfinite(float(loss.numpy()))
    np.testing.assert_allclose(loss.numpy(), expected_loss.numpy())
    assert len(embeddings) == 1
    assert embeddings[0].input_dim == feature.vocab_size
    assert any(isinstance(layer, tf.keras.layers.GRU) for layer in model.submodules)
    assert not any(isinstance(layer, tf.keras.layers.LSTM) for layer in model.submodules)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_char_feature_has_no_parameter_cliff_at_length_seven(rp_logger):
    parameter_counts = []
    for text_max_len in (6, 7):
        tf.keras.backend.clear_session()
        data = pd.DataFrame({"text": ["ABCDEF", "FEDCBA"]})
        feature = CharBasedTextFeature(name="text", text_max_len=text_max_len)
        feature.fit(data)
        parameter_counts.append(_build_text_feature_model(feature).count_params())

    assert parameter_counts[0] == parameter_counts[1]
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_char_feature_without_version_uses_legacy_lstm(rp_logger):
    data = pd.DataFrame({"text": ["ABCDEF", "FEDCBA"]})
    feature = CharBasedTextFeature(name="text", text_max_len=6, rnn_units=32)
    feature.architecture_version = 1
    feature.fit(data)
    del feature.architecture_version

    restored = pickle.loads(pickle.dumps(feature))
    transformed = restored.transform(data)
    model = _build_text_feature_model(restored)

    assert restored.vocab_size == len(restored.tokenizer.word_index)
    assert transformed.shape == (2, 6, restored.vocab_size)
    assert any(isinstance(layer, tf.keras.layers.LSTM) for layer in model.submodules)
    assert not any(isinstance(layer, tf.keras.layers.GRU) for layer in model.submodules)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_char_feature_optimized_pickle_round_trip(rp_logger):
    data = pd.DataFrame({"text": ["ABCDEF", "FEDCBA"]})
    feature = CharBasedTextFeature(name="text", text_max_len=6)
    feature.fit(data)

    restored = pickle.loads(pickle.dumps(feature))

    np.testing.assert_array_equal(restored.transform(data), feature.transform(data))
    assert restored.architecture_version == 2
    assert restored.tokenizer.word_index == feature.tokenizer.word_index
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_email_feature_optimizes_only_the_local_part(rp_logger):
    data = pd.DataFrame({"email": ["AA11@example.com", "BB22@example.com"]})
    feature = EmailFeature(name="email", text_max_len=15, rnn_units=32)

    feature.fit(data)
    transformed = feature.transform(data)

    assert set(feature.tokenizer.word_index) == {"A", "B", "1", "2"}
    assert feature.vocab_size == len(feature.tokenizer.word_index) + 1
    assert transformed.dtype == np.int32
    assert transformed.shape == (2, 15)
    assert all(value.endswith("@tdspora.ai") for value in feature.inverse_transform(
        np.zeros((2, 15, feature.vocab_size), dtype=np.float32)
    ))
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
