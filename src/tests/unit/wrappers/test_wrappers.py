from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from syngen.ml.vae.wrappers.wrappers import VAEWrapper, collate_feature_batch
from tests.conftest import SUCCESSFUL_MESSAGE


@patch("syngen.ml.vae.wrappers.wrappers.enable_flush_denormal")
def test_post_init_enables_flush_denormal(mock_enable_flush_denormal, rp_logger):
    """EPMCTDM-7643: the entire 2.5-4x win rides on this single call, so pin the
    call site - otherwise deleting it keeps the suite green. Train and infer both
    run the char-level text LSTMs, hence it sits before the process branch."""
    rp_logger.info("Test 'VAEWrapper.__post_init__' enables FTZ/DAZ")
    # A mock `self` matches neither the "train" nor the "infer" branch, so only the
    # unconditional head of __post_init__ runs.
    VAEWrapper.__post_init__(MagicMock())

    mock_enable_flush_denormal.assert_called_once_with()
    rp_logger.info(SUCCESSFUL_MESSAGE)


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


class _BatchingStub:
    """Drives the real `_create_batched_dataset` without building a full VAE."""

    def __init__(self, n_rows: int, n_features: int, batch_size: int):
        # each feature's column encodes its own index, so a reordered tuple is visible
        self._features = [
            np.arange(n_rows, dtype="float32").reshape(n_rows, 1) + i * 10_000
            for i in range(n_features)
        ]
        self.batch_size = batch_size
        names = [f"f{i}" for i in range(n_features)]
        features = self._features

        class _Dataset:
            features_dict = {n: None for n in names}

            @staticmethod
            def transform(df):
                return features

        self.dataset = _Dataset()

    def _validate_transformed_data(self, transformed_data):
        return None

    _create_batched_dataset = VAEWrapper._create_batched_dataset


def test_batched_dataset_reports_row_length_for_opacus(rp_logger):
    """EPMCTDM-7630: `__len__` must stay the ROW count, not the batch count.

    On the enterprise DP path Opacus wraps this loader in a `DPDataLoader` whose
    `UniformWithReplacementSampler` derives the Poisson sample rate from
    `len(dataset)`. A batch-level dataset would hand it a rate wrong by a factor of
    `batch_size` and silently weaken the privacy guarantee - no exception raised.
    """
    rp_logger.info("Test '_create_batched_dataset' exposes row-level __len__")
    loader = _BatchingStub(1000, 5, 32)._create_batched_dataset(pd.DataFrame())

    assert len(loader.dataset) == 1000, "must be rows, not batches"
    assert isinstance(loader, torch.utils.data.DataLoader), "Opacus must be able to wrap it"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_batched_dataset_preserves_feature_order_and_shape(rp_logger):
    """EPMCTDM-7630 / guardrail G7: the batch must remain a *tuple* of per-feature
    tensors in `Dataset.transform` order.

    `_train_step` zips `vae.feature_order` against this tuple, so a reordering
    mis-assigns every feature's loss without raising. The container type matters too:
    the DataLoader's `default_convert` rebuilds sequences as lists unless an identity
    `collate_fn` is supplied.
    """
    rp_logger.info("Test '_create_batched_dataset' preserves feature order/shape")
    n_rows, n_features, batch_size = 1000, 5, 32
    loader = _BatchingStub(n_rows, n_features, batch_size)._create_batched_dataset(
        pd.DataFrame()
    )
    batches = list(loader)

    assert len(batches) == n_rows // batch_size, "partial trailing batch must be dropped"
    first = batches[0]
    assert isinstance(first, tuple), "batch container must stay a tuple"
    assert len(first) == n_features
    assert all(tuple(t.shape) == (batch_size, 1) for t in first)
    # feature i is offset by i*10_000, so this pins the ordering
    assert [t[0, 0].item() for t in first] == [i * 10_000 for i in range(n_features)]
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_batched_dataset_covers_every_row_in_order(rp_logger):
    """Batching must stay unshuffled and gap-free: the concatenated batches have to
    reproduce the original row sequence exactly (minus the dropped tail)."""
    rp_logger.info("Test '_create_batched_dataset' yields rows sequentially")
    n_rows, batch_size = 1000, 32
    loader = _BatchingStub(n_rows, 3, batch_size)._create_batched_dataset(pd.DataFrame())

    seen = torch.cat([b[0] for b in loader]).flatten().numpy()
    expected = np.arange(n_rows // batch_size * batch_size, dtype="float32")
    assert np.array_equal(seen, expected)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_create_optimizer_enables_foreach(rp_logger):
    """EPMCTDM-7630: torch only defaults `foreach=True` for CUDA params, so a CPU
    install otherwise falls back to `_single_tensor_adam` - a Python loop issuing ~6
    elementwise ops per parameter tensor on every step (~18% of train wall-clock on
    housing). Pin it so the CPU path cannot silently regress."""
    rp_logger.info("Test 'VAEWrapper._create_optimizer' pins foreach=True")
    model = torch.nn.Linear(4, 4)

    optimizer = VAEWrapper._create_optimizer(model, 1e-4)

    assert isinstance(optimizer, torch.optim.Adam)
    assert all(g["foreach"] is True for g in optimizer.param_groups)
    assert all(g["lr"] == pytest.approx(1e-4) for g in optimizer.param_groups)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def _consumer_loader(dataset, batch_size):
    """A loader arranged the way a consumer that substitutes its own sampler gets it.

    Putting the BatchSampler in the ``batch_sampler`` slot re-enables PyTorch's
    automatic batching, which is exactly what Opacus's ``DPDataLoader`` does when it
    swaps in ``UniformWithReplacementSampler``. Reproducing it with pure torch keeps
    `opacus` out of base's dependencies.
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(dataset),
            batch_size=batch_size,
            drop_last=True,
        ),
        collate_fn=collate_feature_batch,
    )


def test_collate_is_sampler_agnostic(rp_logger):
    """EPMCTDM-7630: the batch must come back as a tuple of *stacked* tensors under
    both batching arrangements.

    The previous identity `lambda batch: batch` was correct only for this loader's own
    arrangement. A consumer that installs its own `batch_sampler` turns automatic
    batching on, so the collate receives a list of per-row tuples; passing that through
    unstacked left the consumer with `list[B]` of `(Tensor(1,), ...)` instead of
    per-feature tensors.
    """
    rp_logger.info("Test 'collate_feature_batch' handles both batching arrangements")
    n_rows, n_features, batch_size = 320, 4, 32
    stub = _BatchingStub(n_rows, n_features, batch_size)
    base_loader = stub._create_batched_dataset(pd.DataFrame())
    consumer_loader = _consumer_loader(base_loader.dataset, batch_size)

    for label, loader in (("base", base_loader), ("consumer", consumer_loader)):
        batch = next(iter(loader))
        assert isinstance(batch, tuple), f"{label}: batch must be a tuple"
        assert len(batch) == n_features, f"{label}: one tensor per feature"
        assert all(torch.is_tensor(t) for t in batch), (
            f"{label}: elements must be stacked tensors, not per-row tuples"
        )
        assert all(t.shape[0] == batch_size for t in batch), f"{label}: wrong batch size"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_collate_preserves_feature_order_across_arrangements(rp_logger):
    """Feature order must match between the two arrangements. `_train_step` zips
    `vae.feature_order` against the batch, so a reordering mis-assigns every feature's
    loss *without raising* - which is why this is asserted explicitly."""
    rp_logger.info("Test 'collate_feature_batch' preserves feature order")
    n_rows, n_features, batch_size = 320, 4, 32
    stub = _BatchingStub(n_rows, n_features, batch_size)
    base_loader = stub._create_batched_dataset(pd.DataFrame())
    consumer_loader = _consumer_loader(base_loader.dataset, batch_size)

    base_batch = next(iter(base_loader))
    consumer_batch = next(iter(consumer_loader))

    # feature i is offset by i*10_000 in the stub, so this pins the ordering
    expected = [i * 10_000 for i in range(n_features)]
    assert [t[0, 0].item() for t in base_batch] == expected
    assert [t[0, 0].item() for t in consumer_batch] == expected
    for a, b in zip(base_batch, consumer_batch):
        assert torch.equal(a, b), "both arrangements must yield identical batches"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_loader_is_picklable_for_worker_processes(rp_logger):
    """EPMCTDM-7630: both the collate_fn *and* the dataset must be picklable for
    `num_workers > 0` under the `spawn` start method - the default on Windows and
    macOS, both of which this package claims support for.

    Two separate blockers existed: an identity lambda as `collate_fn`, and
    `_FeatureTuples` being declared inside `_create_batched_dataset` (a function-local
    class is equally unpicklable). Fixing only one leaves the loader unusable, so both
    are asserted here.
    """
    rp_logger.info("Test the batched loader is picklable for worker processes")
    import pickle

    assert pickle.loads(pickle.dumps(collate_feature_batch)) is collate_feature_batch

    loader = _BatchingStub(320, 3, 32)._create_batched_dataset(pd.DataFrame())
    pickle.dumps(loader.collate_fn)
    restored_dataset = pickle.loads(pickle.dumps(loader.dataset))

    assert len(restored_dataset) == len(loader.dataset)
    assert isinstance(restored_dataset[[0, 1]], tuple)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_loader_len_supports_sampling_rate_contract(rp_logger):
    """`len(loader) == rows // batch_size` and `len(dataset) == rows`. Consumers derive
    a sampling rate from these, so a change here alters that rate silently."""
    rp_logger.info("Test loader/dataset lengths honour the sampling-rate contract")
    n_rows, batch_size = 1000, 32
    loader = _BatchingStub(n_rows, 3, batch_size)._create_batched_dataset(pd.DataFrame())

    assert len(loader.dataset) == n_rows
    assert len(loader) == n_rows // batch_size
    assert 1 / len(loader) == pytest.approx(batch_size / (n_rows - n_rows % batch_size))
    rp_logger.info(SUCCESSFUL_MESSAGE)
