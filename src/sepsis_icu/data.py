"""Data loading, per-patient aggregation, and the train/test split.

The raw PhysioNet 2019 dataset is a time-series: one row per hour per patient.
Tabular classifiers expect one row per patient, so :func:`aggregate_patients`
collapses each patient's stay into summary statistics (min/max/mean/std/range)
per vital and lab, plus demographics and total ICU length of stay.

CAVEAT (see README "Limitations"): aggregations span the patient's *entire*
stay, including hours after sepsis onset, and the target is "ever septic". This
is a retrospective whole-stay classifier, not a real-time early-warning system.
Nothing here uses the target to transform features, so running it on the full
dataset does not risk train/test leakage.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the raw hourly Sepsis dataset (one row per patient-hour)."""
    path = path or config.RAW_CSV
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Locally, keep '{config.RAW_CSV_NAME}' "
            "in the repo root; on Colab, set SEPSIS_DATA_DIR or place it in your "
            "Drive folder. The file is the PhysioNet/CinC Challenge 2019 records "
            f"concatenated with a '{config.PATIENT_ID_COL}' column."
        )
    return pd.read_csv(path)


def aggregate_patients(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the hourly time-series into one row per patient.

    For each of the 34 vitals/labs: min, max, mean, std, and range (max - min).
    Plus Age/Gender (first value), ICULOS_max (total hours in ICU), and the
    per-patient Sepsis target (1 if SepsisLabel was ever 1).

    Returns a frame indexed by ``Patient_ID`` with 173 feature columns + target.
    """
    pid = config.PATIENT_ID_COL
    vl_cols = [c for c in config.VITAL_AND_LAB_COLS if c in raw_df.columns]

    demographics = raw_df.groupby(pid)[config.DEMOGRAPHIC_COLS].first()

    vitals_labs = raw_df.groupby(pid)[vl_cols].agg(config.AGG_FUNCS)
    # Flatten multi-level column names: ('HR', 'min') -> 'HR_min'
    vitals_labs.columns = ["_".join(col) for col in vitals_labs.columns]
    for col in vl_cols:
        vitals_labs[f"{col}_range"] = (
            vitals_labs[f"{col}_max"] - vitals_labs[f"{col}_min"]
        )

    iculos = raw_df.groupby(pid)["ICULOS"].max().rename("ICULOS_max")
    target = raw_df.groupby(pid)[config.RAW_TARGET_COL].max().rename(config.TARGET_COL)

    return demographics.join(vitals_labs).join(iculos).join(target)


def split_features_target(patient_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X feature matrix, y Sepsis target series)."""
    X = patient_df.drop(columns=[config.TARGET_COL])
    y = patient_df[config.TARGET_COL]
    return X.copy(), y.copy()


def make_split(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split with a fixed seed for reproducibility."""
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_SEED,
    )
