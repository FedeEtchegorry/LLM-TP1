"""Device selection and deterministic execution."""

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


_DETERMINISTIC = False


def deterministic() -> None:
    """Enable deterministic PyTorch algorithms once."""
    global _DETERMINISTIC
    if _DETERMINISTIC:
        return
    import torch

    torch.use_deterministic_algorithms(True)
    _DETERMINISTIC = True


def describe(target) -> str:
    """A one-line label for the results record."""
    import torch

    if target.type == "cuda":
        index = target.index or 0
        return f"cuda:{index} ({torch.cuda.get_device_name(index)})"
    return f"cpu ({platform.processor() or platform.machine()})"
