from unittest.mock import patch

import torch

from syngen.ml.utils.device import (
    assign_gpu_index,
    cuda_device_count,
    gpu_available,
    resolve_device,
)


def _mock_cuda(available: bool, device_count: int = 0):
    return (
        patch("torch.cuda.is_available", return_value=available),
        patch("torch.cuda.device_count", return_value=device_count),
    )


def test_cuda_device_count_zero_when_unavailable(rp_logger):
    rp_logger.info("Test 'cuda_device_count' returns 0 when CUDA is unavailable")
    with patch("torch.cuda.is_available", return_value=False):
        assert cuda_device_count() == 0


def test_cuda_device_count_reflects_device_count_when_available(rp_logger):
    rp_logger.info("Test 'cuda_device_count' reflects torch.cuda.device_count() when available")
    is_available, device_count = _mock_cuda(available=True, device_count=3)
    with is_available, device_count:
        assert cuda_device_count() == 3


def test_gpu_available_false_when_no_devices(rp_logger):
    rp_logger.info("Test 'gpu_available' is False when no CUDA devices are visible")
    with patch("torch.cuda.is_available", return_value=False):
        assert gpu_available() is False


def test_gpu_available_true_when_devices_present(rp_logger):
    rp_logger.info("Test 'gpu_available' is True when at least one CUDA device is visible")
    is_available, device_count = _mock_cuda(available=True, device_count=1)
    with is_available, device_count:
        assert gpu_available() is True


def test_assign_gpu_index_none_when_no_gpu(rp_logger):
    rp_logger.info("Test 'assign_gpu_index' returns None when no GPU is visible")
    with patch("torch.cuda.is_available", return_value=False):
        assert assign_gpu_index(0) is None
        assert assign_gpu_index(5) is None


def test_assign_gpu_index_round_robins_across_devices(rp_logger):
    rp_logger.info("Test 'assign_gpu_index' round-robins across the visible device count")
    is_available, device_count = _mock_cuda(available=True, device_count=3)
    with is_available, device_count:
        assert [assign_gpu_index(i) for i in range(7)] == [0, 1, 2, 0, 1, 2, 0]


def test_resolve_device_none_index_is_cpu(rp_logger):
    rp_logger.info("Test 'resolve_device' maps a None index to cpu regardless of CUDA state")
    with patch("torch.cuda.is_available", return_value=True):
        assert resolve_device(None) == torch.device("cpu")


def test_resolve_device_with_index_and_cuda_available(rp_logger):
    rp_logger.info("Test 'resolve_device' maps an index to cuda:<index> when CUDA is available")
    with patch("torch.cuda.is_available", return_value=True):
        assert resolve_device(2) == torch.device("cuda:2")


def test_resolve_device_falls_back_to_cpu_when_cuda_unavailable(rp_logger):
    """Defensive fallback: a stale gpu_index surviving into a process where CUDA
    is not (or no longer) available must resolve to cpu, not raise."""
    rp_logger.info("Test 'resolve_device' falls back to cpu when CUDA is unavailable")
    with patch("torch.cuda.is_available", return_value=False):
        assert resolve_device(2) == torch.device("cpu")
