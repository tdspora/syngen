import numpy as np
import pytest
import torch

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


class _FakeVaeModule(torch.nn.Module):
    """Minimal stand-in for the real `CVAEModule`: a single trainable weight
    plus the `(recons, mu, log_sigma)` triple `_train_step` unpacks, so the
    step can be exercised without building a full VAE."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, batch):
        x = batch[0]
        recon = self.w * torch.ones_like(x)
        # zeros give kl_divergence == 0 exactly, keeping the assertions exact
        mu = torch.zeros_like(x)
        log_sigma = torch.zeros_like(x)
        return [recon], mu, log_sigma


class _FakeFeature:
    @staticmethod
    def compute_loss(target, output):
        return torch.mean((target - output) ** 2)


class _StubTrainStepWrapper:
    """Exercises the real (unmodified) `_train_step` from `VAEWrapper`
    against a lightweight fake model, instead of a full VAE."""

    _train_step = VAEWrapper._train_step

    def __init__(self):
        self.model = _FakeVaeModule()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.vae = type("_Vae", (), {"feature_order": ["feat_a"]})()
        self.dataset = type("_Dataset", (), {"features": {"feat_a": _FakeFeature()}})()


def _make_batch(value: float):
    return (torch.tensor([[value]], dtype=torch.float32),)


def test_train_step_returns_plain_floats(rp_logger):
    """`_train_step` must detach its tensors and hand the caller plain floats -
    the epoch loop accumulates them with `+=` and feeds them to numpy/MLflow,
    so leaking graph-attached tensors would retain the autograd graph."""
    rp_logger.info("Test 'VAEWrapper._train_step' returns detached plain floats")
    stub = _StubTrainStepWrapper()

    loss, kl_loss, feature_losses = stub._train_step(_make_batch(1.0))

    assert isinstance(loss, float)
    assert isinstance(kl_loss, float)
    # target 1.0 vs initial weight 2.0 -> MSE of 1.0, KL is exactly 0 here
    assert loss == pytest.approx(1.0)
    assert kl_loss == pytest.approx(0.0)
    assert set(feature_losses.keys()) == {"feat_a"}
    assert isinstance(feature_losses["feat_a"], float)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_step_updates_model_weights(rp_logger):
    """The backward pass and optimizer step must actually move the weights
    toward the batch target."""
    rp_logger.info("Test 'VAEWrapper._train_step' updates model weights")
    stub = _StubTrainStepWrapper()
    initial_weight = float(stub.model.w.detach())

    for _ in range(5):
        stub._train_step(_make_batch(1.0))

    updated_weight = float(stub.model.w.detach())
    assert updated_weight != initial_weight
    assert abs(updated_weight - 1.0) < abs(initial_weight - 1.0), (
        "weight should move closer to the batch target (1.0) after training steps"
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_train_step_excludes_kl_from_optimized_loss(rp_logger):
    """KL is reported but multiplied by 0 in the optimized total, mirroring the
    TF graph's `add_loss(kl_loss * 0)` (models/model.py:129 on `main`). The
    returned total must therefore equal the reconstruction loss alone."""
    rp_logger.info("Test 'VAEWrapper._train_step' excludes KL from the optimized total")
    stub = _StubTrainStepWrapper()

    loss, kl_loss, feature_losses = stub._train_step(_make_batch(1.0))

    assert loss == pytest.approx(sum(feature_losses.values()))
    rp_logger.info(SUCCESSFUL_MESSAGE)
