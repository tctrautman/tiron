"""Tiron public API."""

from __future__ import annotations

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "TironEngine":
        from .engine import TironEngine
        return TironEngine
    raise AttributeError(name)


def transcribe(audio, **kwargs):
    """Transcribe audio using a newly created engine."""
    from .engine import TironEngine
    return TironEngine().transcribe(audio, **kwargs)


__all__ = ["TironEngine", "transcribe", "__version__"]
