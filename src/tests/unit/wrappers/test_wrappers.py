import numpy as np
import pytest
import tensorflow as tf

from syngen.ml.vae.wrappers.wrappers import VAEWrapper
from tests.conftest import SUCCESSFUL_MESSAGE


def test_find_non_finite_features_detects_nan_and_inf(rp_logger):
    """EPMCTDM-7581 guardrail: features whose transformed (model-input) arrays
    contain NaN or inf are reported by name."""
    rp_logger.info("Test 'VAEWrapper._find_non_finite_features' detects NaN/inf")
    feature_names = ["clean_num", "bad_date", "clean_cat"]
    transformed_data = [
        np.array([[0.1], [0.2], [0.3]], dtype="float32"),
        np.array([[np.nan], [0.5], [0.6]], dtype="float32"),
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype="float32"),
    ]
    assert VAEWrapper._find_non_finite_features(feature_names, transformed_data) == [
        "bad_date"
    ]

    transformed_data[2] = np.array([[np.inf], [0.0], [0.0]], dtype="float32")
    assert set(
        VAEWrapper._find_non_finite_features(feature_names, transformed_data)
    ) == {"bad_date", "clean_cat"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_find_non_finite_features_passes_clean_data(rp_logger):
    rp_logger.info("Test 'VAEWrapper._find_non_finite_features' passes clean data")
    feature_names = ["a", "b"]
    transformed_data = [
        np.array([[0.1], [0.2]], dtype="float32"),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
    ]
    assert VAEWrapper._find_non_finite_features(feature_names, transformed_data) == []
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validate_transformed_data_raises_on_nan(rp_logger):
    """The guardrail must abort training with a ValueError naming the offending
    feature when NaN reaches the model input."""
    rp_logger.info("Test 'VAEWrapper._validate_transformed_data' raises on NaN")

    class _StubWrapper:
        dataset = type("_Dataset", (), {"features": {"good": None, "bad": None}})()
        _find_non_finite_features = staticmethod(VAEWrapper._find_non_finite_features)
        _validate_transformed_data = VAEWrapper._validate_transformed_data

    transformed_data = [
        np.array([[0.1], [0.2]], dtype="float32"),
        np.array([[np.nan], [0.2]], dtype="float32"),
    ]
    with pytest.raises(ValueError, match="bad"):
        _StubWrapper()._validate_transformed_data(transformed_data)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validate_transformed_data_passes_clean_data(rp_logger):
    rp_logger.info("Test 'VAEWrapper._validate_transformed_data' passes clean data")

    class _StubWrapper:
        dataset = type("_Dataset", (), {"features": {"good": None}})()
        _find_non_finite_features = staticmethod(VAEWrapper._find_non_finite_features)
        _validate_transformed_data = VAEWrapper._validate_transformed_data

    _StubWrapper()._validate_transformed_data([np.array([[0.1], [0.2]], dtype="float32")])
    rp_logger.info(SUCCESSFUL_MESSAGE)


class _FakeVaeModel:
    """Minimal stand-in for the real CVAE Keras model: exposes just enough
    (`losses`, `trainable_weights`, callability) for `_train_step_graph_impl`
    to run under `tf.function` without needing a full VAE."""

    def __init__(self):
        self.w = tf.Variable(2.0, dtype=tf.float32)
        self.losses = []

    def __call__(self, batch):
        x = batch[0]
        feature_loss = tf.reduce_mean(tf.square(x - self.w))
        kl_loss = tf.constant(0.1, dtype=tf.float32)
        self.losses = [feature_loss, kl_loss]

    @property
    def trainable_weights(self):
        return [self.w]


class _StubTrainStepWrapper:
    """Exercises the real (unmodified) `_train_step`/`_train_step_graph_impl`
    from `VAEWrapper` against a lightweight fake model, instead of a full VAE."""

    _train_step_graph_impl = VAEWrapper._train_step_graph_impl
    _train_step = VAEWrapper._train_step

    def __init__(self):
        self.model = _FakeVaeModel()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
        self.loss_metric = tf.keras.metrics.Mean()
        self.vae = type("_Vae", (), {"feature_losses": {"feat_a": None}})()


def _make_batch(value: float):
    return (tf.constant([[value]], dtype=tf.float32),)


def test_train_step_wraps_graph_impl_in_tf_function_once(rp_logger):
    """`_train_step` must lazily wrap `_train_step_graph_impl` in `tf.function`
    on first call, and reuse (not re-wrap/retrace) that same graph function
    on subsequent calls."""
    rp_logger.info(
        "Test 'VAEWrapper._train_step' wraps '_train_step_graph_impl' in "
        "tf.function exactly once and reuses it"
    )
    stub = _StubTrainStepWrapper()
    assert not hasattr(stub, "_train_step_graph")

    stub._train_step(_make_batch(1.0))
    assert hasattr(stub, "_train_step_graph")
    assert hasattr(stub._train_step_graph, "get_concrete_function"), (
        "_train_step_graph should be a tf.function-wrapped callable"
    )
    graph_fn_after_first_call = stub._train_step_graph

    stub._train_step(_make_batch(2.0))
    assert stub._train_step_graph is graph_fn_after_first_call, (
        "the same instance should reuse its cached graph function across "
        "calls, not re-wrap it every time"
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_step_return_values_match_pre_graph_contract(rp_logger):
    """Moving `.numpy()` calls out of the traced function must not change
    what `_train_step` returns to its caller - loss stays a tensor,
    kl_loss/feature_losses come back as plain numpy floats keyed by
    `vae.feature_losses` order."""
    rp_logger.info(
        "Test 'VAEWrapper._train_step' return values match the pre-graph-compile contract"
    )
    stub = _StubTrainStepWrapper()

    loss, kl_loss, feature_losses = stub._train_step(_make_batch(1.0))

    assert isinstance(loss, tf.Tensor)
    assert isinstance(kl_loss, np.floating)
    assert kl_loss == pytest.approx(0.1)
    assert set(feature_losses.keys()) == {"feat_a"}
    assert isinstance(feature_losses["feat_a"], np.floating)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_step_updates_model_weights_under_graph_mode(rp_logger):
    """The optimizer step and loss_metric update inside
    `_train_step_graph_impl` must still take effect correctly once compiled
    into a graph via tf.function."""
    rp_logger.info(
        "Test 'VAEWrapper._train_step' actually updates model weights under tf.function"
    )
    stub = _StubTrainStepWrapper()
    initial_weight = float(stub.model.w.numpy())

    for _ in range(5):
        stub._train_step(_make_batch(1.0))

    updated_weight = float(stub.model.w.numpy())
    assert updated_weight != initial_weight
    assert abs(updated_weight - 1.0) < abs(initial_weight - 1.0), (
        "weight should move closer to the batch target (1.0) after training steps"
    )
    assert stub.loss_metric.result().numpy() >= 0
    rp_logger.info(SUCCESSFUL_MESSAGE)
