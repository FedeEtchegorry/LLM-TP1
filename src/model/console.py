"""Console encoding helpers."""

from __future__ import annotations

import sys


def utf8_console() -> None:
    """Use UTF-8 when the output stream supports reconfiguration."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
