"""Canonical filesystem paths for the project."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SUBMISSIONS_DATA_DIR = DATA_DIR / "submissions"
MODELS_DIR = ROOT / "models"
MLE_TREES_DIR = MODELS_DIR / "mle_trees"
NOTEBOOKS_DIR = ROOT / "notebooks"
