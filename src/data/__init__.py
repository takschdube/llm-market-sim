# src/data/__init__.py
"""
Dataset generation and export for Kaggle/HuggingFace.

This module provides tools for exporting simulation results to
standard dataset formats for publication and sharing.

Example:
    from src.data import DatasetExporter

    exporter = DatasetExporter("my_experiment")
    for trial_id, simulation in enumerate(simulations):
        exporter.add_trial(simulation, trial_id)

    exporter.to_parquet("output/")
    exporter.to_huggingface("username/my-dataset")
"""

from .schemas import DECISION_COLUMNS, TRADE_COLUMNS, ROUND_COLUMNS
from .exporter import DatasetExporter

__all__ = [
    "DECISION_COLUMNS",
    "TRADE_COLUMNS",
    "ROUND_COLUMNS",
    "DatasetExporter",
]
