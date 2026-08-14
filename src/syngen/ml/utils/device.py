from typing import Optional

import torch


def cuda_device_count() -> int:
    """Single choke point every other function in this module goes through -
    tests only ever need to mock this one function (or torch.cuda.is_available /
    torch.cuda.device_count directly)."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def gpu_available() -> bool:
    return cuda_device_count() > 0


def assign_gpu_index(item_index: int) -> Optional[int]:
    """Round-robin GPU index for the item_index'th table/batch/job.

    Returns None when no GPU is visible - callers must branch on None rather
    than build torch.device(f"cuda:{None}").
    """
    n = cuda_device_count()
    return None if n == 0 else item_index % n


def resolve_device(gpu_index: Optional[int] = None) -> torch.device:
    """Turn a GPU index into a torch.device, falling back to CPU whenever
    CUDA is unavailable - including when a stale index is passed but CUDA is
    not (or is no longer) available in the current process."""
    if gpu_index is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_index}")
    return torch.device("cpu")
