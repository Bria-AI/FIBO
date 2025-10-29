"""Convenience exports for the fibo_inference package."""

import sys
from importlib import import_module


def _expose_module(alias: str) -> None:
    """Make `alias` importable as a top-level module."""
    module = import_module(f".{alias}", __name__)
    globals()[alias] = module
    sys.modules[alias] = module


_expose_module("fibo_pipeline")

__all__ = [
    "fibo_pipeline",
]
