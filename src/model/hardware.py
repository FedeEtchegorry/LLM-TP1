"""Where the pretrained models run, and how the write-up is allowed to say it.

Our own Transformer is small enough that the CPU is not the constraint, and it stays
there so its numbers do not depend on which machine produced them. The two pretrained
regimes are a different size -- 22M parameters -- so they take the accelerator when
there is one, and the run records which one it used.
"""

from __future__ import annotations

import platform


def device(preferred: str | None = None):
    """The best device available, or the one asked for.

    Imported lazily so the modules that never touch torch keep loading without it.
    """
    import torch

    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe(target) -> str:
    """A one-line label for the results record and for the printed table."""
    import torch

    if target.type == "cuda":
        index = target.index or 0
        return f"cuda:{index} ({torch.cuda.get_device_name(index)})"
    return f"cpu ({platform.processor() or platform.machine()})"
