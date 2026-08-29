"""Pick the best device available and name it for the record.

Every model runs on the accelerator when there is one. Records store which device
produced them, because floating-point reductions associate differently and a results
directory should not mix them silently.
"""

from __future__ import annotations

import platform


def device(preferred: str | None = None):
    """The best device available, or the one asked for."""
    import torch

    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe(target) -> str:
    """A one-line label for the results record."""
    import torch

    if target.type == "cuda":
        index = target.index or 0
        return f"cuda:{index} ({torch.cuda.get_device_name(index)})"
    return f"cpu ({platform.processor() or platform.machine()})"
