"""Fake review detection: from-scratch Adaline benchmarked against sklearn.

Public surface::

    from fakereview import AdalineGD, load_split, build_pipeline

See the README for the measured results and the reasoning behind the four
pipeline configurations.
"""

from __future__ import annotations

from .adaline import AdalineDivergedError, AdalineGD
from .data import CLASS_NAMES, Dataset, load_reviews, load_split
from .features import extract, feature_names
from .pipelines import DEFAULT_PIPELINE, PIPELINES, build_pipeline, describe

__version__ = "1.0.0"

__all__ = [
    "AdalineGD",
    "AdalineDivergedError",
    "CLASS_NAMES",
    "Dataset",
    "load_reviews",
    "load_split",
    "extract",
    "feature_names",
    "PIPELINES",
    "DEFAULT_PIPELINE",
    "build_pipeline",
    "describe",
]
