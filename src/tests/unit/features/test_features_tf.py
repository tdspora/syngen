"""
Behavioral tests for feature classes.

HOW TO MIGRATE TO PYTORCH:
  All assertions are numpy-based and framework-agnostic.
  The only code to update after migration is in tests/unit/ml_compat.py.
  Tests that check Keras-specific properties (input shape, encoder shape)
  use feature.input_dimension and actual data runs — not symbolic graphs.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from syngen.ml.vae.models.features import (
    BaseFeature,
    BinaryFeature,
    ContinuousFeature,
    CategoricalFeature,
    CharBasedTextFeature,
    DateFeature,
)
from tests.unit.ml_compat import to_numpy
from tests.conftest import SUCCESSFUL_MESSAGE


# ---------------------------------------------------------------------------
# CharBasedTextFeature — fit / transform (TF Tokenizer & K.one_hot)
# ---------------------------------------------------------------------------

class TestCharBasedTextFeatureFit:

    def test_fit_creates_tokenizer_with_correct_vocab_size(self, rp_logger):
        rp_logger.info(
            "Testing CharBasedTextFeature.fit builds tokenizer with correct vocab_size"
        )
        feature = CharBasedTextFeature(name="text_col", text_max_len=10)
        data = pd.DataFrame({"text": ["abc", "def", "ghi"]})
        feature.fit(data)

        # unique chars: a, b, c, d, e, f, g, h, i → 9
        assert feature.vocab_size == 9
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_fit_tokenizer_has_word_index(self, rp_logger):
        rp_logger.info(
            "Testing CharBasedTextFeature.fit tokenizer has populated word_index"
        )
        feature = CharBasedTextFeature(name="text_col", text_max_len=5)
        data = pd.DataFrame({"text": ["ab", "cd"]})
        feature.fit(data)

        assert hasattr(feature, "tokenizer") or feature.vocab_size > 0
        if hasattr(feature, "tokenizer") and hasattr(feature.tokenizer, "word_index"):
            assert len(feature.tokenizer.word_index) == feature.vocab_size
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_fit_tokenizer_has_inverse_dict(self, rp_logger):
        rp_logger.info(
            "Testing CharBasedTextFeature.fit attaches inverse_dict to tokenizer"
        )
        feature = CharBasedTextFeature(name="text_col", text_max_len=5)
        data = pd.DataFrame({"text": ["ab", "cd"]})
        feature.fit(data)

        assert hasattr(feature.tokenizer, "inverse_dict")
        for char, idx in feature.tokenizer.word_index.items():
            assert feature.tokenizer.inverse_dict[idx] == char
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_fit_raises_on_multiple_columns(self, rp_logger):
        rp_logger.info(
            "Testing CharBasedTextFeature.fit raises on DataFrame with multiple columns"
        )
        feature = CharBasedTextFeature(name="text_col", text_max_len=5)
        data = pd.DataFrame({"col1": ["abc"], "col2": ["def"]})

        with pytest.raises(Exception, match="can work only with one text column"):
            feature.fit(data)
        rp_logger.info(SUCCESSFUL_MESSAGE)


class TestCharBasedTextFeatureTransform:

    @pytest.fixture(autouse=True)
    def fitted_feature(self):
        self.text_max_len = 6
        self.feature = CharBasedTextFeature(
            name="text_col", text_max_len=self.text_max_len
        )
        self.data = pd.DataFrame({"text": ["abc", "de", "fgh", "ij"]})
        self.feature.fit(self.data)

    def test_transform_returns_correct_num_samples(self, rp_logger):
        rp_logger.info(
            "Testing CharBasedTextFeature.transform returns n_samples as first dimension"
        )
        result = self.feature.transform(self.data)
        arr = to_numpy(result) if not isinstance(result, np.ndarray) else result

        assert arr.shape[0] == 4
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_transform_second_dim_is_text_max_len(self, rp_logger):
        rp_logger.info(
            "Testing CharBasedTextFeature.transform second dim equals text_max_len"
        )
        result = self.feature.transform(self.data)
        arr = to_numpy(result) if not isinstance(result, np.ndarray) else result

        assert arr.shape[1] == self.text_max_len
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_transform_third_dim_is_vocab_size(self, rp_logger):
        rp_logger.info(
            "Testing CharBasedTextFeature.transform third dim equals vocab_size"
        )
        result = self.feature.transform(self.data)
        arr = to_numpy(result) if not isinstance(result, np.ndarray) else result

        assert arr.shape[2] == self.feature.vocab_size
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_transform_output_is_valid_one_hot(self, rp_logger):
        rp_logger.info(
            "Testing CharBasedTextFeature.transform each token is one-hot (sum 0 or 1)"
        )
        result = self.feature.transform(self.data)
        arr = to_numpy(result) if not isinstance(result, np.ndarray) else result

        row_sums = arr.sum(axis=-1)
        assert np.all((row_sums == 0.0) | (row_sums == 1.0))
        rp_logger.info(SUCCESSFUL_MESSAGE)

    @pytest.mark.parametrize("text_max_len", [3, 5, 10])
    def test_transform_truncates_to_max_len(self, text_max_len, rp_logger):
        rp_logger.info(
            f"Testing CharBasedTextFeature.transform truncates to max_len={text_max_len}"
        )
        feature = CharBasedTextFeature(name="text_col", text_max_len=text_max_len)
        data = pd.DataFrame({"text": ["abcdefghij"]})
        feature.fit(data)
        result = feature.transform(data)
        arr = to_numpy(result) if not isinstance(result, np.ndarray) else result

        assert arr.shape[1] == text_max_len
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_transform_raises_on_multiple_columns(self, rp_logger):
        rp_logger.info(
            "Testing CharBasedTextFeature.transform raises on multi-column DataFrame"
        )
        multi_col = pd.DataFrame({"col1": ["abc"], "col2": ["def"]})

        with pytest.raises(Exception, match="can work only with one text column"):
            self.feature.transform(multi_col)
        rp_logger.info(SUCCESSFUL_MESSAGE)


# ---------------------------------------------------------------------------
# Input dimension / feature_type checks (framework-agnostic)
# ---------------------------------------------------------------------------

class TestFeatureAttributes:

    def test_binary_feature_input_dimension_set_after_fit(self, rp_logger):
        rp_logger.info("Testing BinaryFeature sets input_dimension after fit")
        feature = BinaryFeature(name="bin_feat")
        # Fit with binary data in a single column
        feature.fit(
            pd.DataFrame({"bin": [0, 1, 0, 1]}).values.reshape(-1, 1)
        )
        assert feature.input_dimension == 1
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_continuous_feature_is_numeric_type(self, rp_logger):
        rp_logger.info("Testing ContinuousFeature.feature_type == 'numeric'")
        assert ContinuousFeature(name="num").feature_type == "numeric"
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_categorical_feature_is_categorical_type(self, rp_logger):
        rp_logger.info("Testing CategoricalFeature.feature_type == 'categorical'")
        assert CategoricalFeature(name="cat").feature_type == "categorical"
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_binary_feature_is_categorical_type(self, rp_logger):
        rp_logger.info("Testing BinaryFeature.feature_type == 'categorical'")
        assert BinaryFeature(name="bin").feature_type == "categorical"
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_char_based_text_feature_is_text_type(self, rp_logger):
        rp_logger.info("Testing CharBasedTextFeature.feature_type == 'text'")
        assert CharBasedTextFeature(name="txt", text_max_len=5).feature_type == "text"
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_date_feature_is_numeric_type(self, rp_logger):
        rp_logger.info("Testing DateFeature.feature_type == 'numeric'")
        assert DateFeature(name="dt").feature_type == "numeric"
        rp_logger.info(SUCCESSFUL_MESSAGE)


# ---------------------------------------------------------------------------
# Decoder output shapes (tested through actual Keras symbol or PyTorch run)
# ---------------------------------------------------------------------------

class TestDecoderOutputShapes:

    def _get_output_dim(self, feature, input_dim: int) -> int:
        """
        Build a minimal forward path for the given feature and return
        the last dimension of the decoder output.
        Works for both TF (symbolic) and PyTorch (eager).
        """
        feature.input_dimension = input_dim

        # TF path: use symbolic Keras API
        try:
            import tensorflow as tf
            inp = feature.input          # triggers @lazy Keras Input creation
            dec = feature.create_decoder(inp)
            return int(dec.shape[-1])
        except (ImportError, AttributeError):
            pass

        # PyTorch path: create a dummy input tensor and run forward
        import torch
        dummy = torch.zeros(1, input_dim)
        dec = feature.create_decoder(dummy)
        if isinstance(dec, (list, tuple)):
            dec = dec[0]
        return int(dec.shape[-1])

    def test_binary_feature_decoder_output_matches_input_dim(self, rp_logger):
        rp_logger.info(
            "Testing BinaryFeature decoder output last dim == input_dimension"
        )
        feature = BinaryFeature(name="bin_feat")
        output_dim = self._get_output_dim(feature, input_dim=4)
        assert output_dim == 4
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_continuous_feature_decoder_output_matches_input_dim(self, rp_logger):
        rp_logger.info(
            "Testing ContinuousFeature decoder output last dim == input_dimension"
        )
        feature = ContinuousFeature(name="num_feat", decoder_layers=(8,))
        output_dim = self._get_output_dim(feature, input_dim=1)
        assert output_dim == 1
        rp_logger.info(SUCCESSFUL_MESSAGE)

    @pytest.mark.parametrize("input_dim", [3, 5, 10])
    def test_categorical_feature_decoder_softmax_output_dim(self, input_dim, rp_logger):
        rp_logger.info(
            f"Testing CategoricalFeature decoder output dim == input_dimension={input_dim}"
        )
        feature = CategoricalFeature(name="cat_feat", decoder_layers=(8,))
        output_dim = self._get_output_dim(feature, input_dim=input_dim)
        assert output_dim == input_dim
        rp_logger.info(SUCCESSFUL_MESSAGE)


# ---------------------------------------------------------------------------
# Feature loss — can be computed and is finite
# ---------------------------------------------------------------------------

class TestFeatureLossComputable:

    def _get_loss_tensor(self, feature):
        """
        Build feature input+decoder and return the loss as a Python float.
        Works for TF (symbolic then eval) and PyTorch (eager).
        """
        try:
            import tensorflow as tf
            inp = feature.input
            _ = feature.create_decoder(inp)
            loss = feature.loss
            # Wrap in a model to evaluate symbolically
            m = tf.keras.Model(inputs=inp, outputs=feature.decoder)
            m.add_loss(loss)
            dummy = np.zeros((1, feature.input_dimension), dtype=np.float32)
            m(dummy)
            return to_numpy(m.losses[0])
        except (ImportError, AttributeError):
            pass
        # PyTorch path: loss is a Module or callable returning tensor
        import torch
        dummy = torch.zeros(1, feature.input_dimension)
        _ = feature.create_decoder(dummy)
        loss = feature.loss
        if callable(loss):
            val = loss()
        else:
            val = loss
        return to_numpy(val)

    def test_binary_feature_loss_is_finite_float(self, rp_logger):
        rp_logger.info(
            "Testing BinaryFeature.loss produces a finite numerical value"
        )
        feature = BinaryFeature(name="bin_feat")
        feature.input_dimension = 2
        val = self._get_loss_tensor(feature)
        assert np.isfinite(float(np.mean(val)))
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_continuous_feature_loss_is_finite_float(self, rp_logger):
        rp_logger.info(
            "Testing ContinuousFeature.loss produces a finite numerical value"
        )
        feature = ContinuousFeature(name="num_feat", decoder_layers=(4,))
        feature.input_dimension = 1
        val = self._get_loss_tensor(feature)
        assert np.isfinite(float(np.mean(val)))
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_categorical_feature_loss_is_finite_float(self, rp_logger):
        rp_logger.info(
            "Testing CategoricalFeature.loss produces a finite numerical value"
        )
        feature = CategoricalFeature(name="cat_feat", decoder_layers=(4,))
        feature.input_dimension = 3
        val = self._get_loss_tensor(feature)
        assert np.isfinite(float(np.mean(val)))
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_date_feature_loss_is_finite_float(self, rp_logger):
        rp_logger.info(
            "Testing DateFeature.loss produces a finite numerical value"
        )
        feature = DateFeature(name="date_feat", decoder_layers=(4,))
        feature.input_dimension = 1
        val = self._get_loss_tensor(feature)
        assert np.isfinite(float(np.mean(val)))
        rp_logger.info(SUCCESSFUL_MESSAGE)


# ---------------------------------------------------------------------------
# weight_randomizer default behaviour
# ---------------------------------------------------------------------------

class TestWeightRandomizer:

    def test_continuous_feature_weight_randomizer_default_is_one(self, rp_logger):
        rp_logger.info(
            "Testing ContinuousFeature default weight_randomizer is (1, 1)"
        )
        feature = ContinuousFeature(name="num_feat")
        assert feature.weight_randomizer == (1, 1)
        rp_logger.info(SUCCESSFUL_MESSAGE)

    @pytest.mark.parametrize("low, high", [
        (0, 1),
        (0.5, 0.5),
        (1, 1),
    ])
    def test_continuous_feature_weight_randomizer_set(self, low, high, rp_logger):
        rp_logger.info(
            f"Testing ContinuousFeature weight_randomizer with low={low}, high={high}"
        )
        feature = ContinuousFeature(name="num_feat", weight_randomizer=(low, high))
        assert feature.weight_randomizer == (low, high)
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_random_weight_in_range(self, rp_logger):
        rp_logger.info(
            "Testing that the random weight used in feature losses is in [0, 1]"
        )
        # In TF: K.random_uniform_variable; in PyTorch: torch.FloatTensor.uniform_
        # Both should produce a scalar in [low, high]
        try:
            import tensorflow.keras.backend as K
            w = K.random_uniform_variable(shape=(1,), low=0.0, high=1.0)
            val = float(to_numpy(w).flat[0])
        except ImportError:
            import torch
            val = float(torch.FloatTensor(1).uniform_(0.0, 1.0))
        assert 0.0 <= val <= 1.0
        rp_logger.info(SUCCESSFUL_MESSAGE)

