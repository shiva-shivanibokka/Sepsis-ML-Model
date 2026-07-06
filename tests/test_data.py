"""Tests for the hourly-record -> one-row-per-patient aggregation."""

import numpy as np
import pandas as pd

from sepsis_icu import config, data


def _toy_raw():
    """Two patients, a few hours each, with one lab fully missing for patient 2."""
    rows = []
    # patient 1: becomes septic on its last hour
    for h, (hr, lac, sep) in enumerate([(80, 1.0, 0), (90, 2.0, 0), (110, 4.0, 1)]):
        rows.append({"Patient_ID": 1, "Age": 60, "Gender": 1, "HR": hr,
                     "Lactate": lac, "ICULOS": h + 1, "SepsisLabel": sep})
    # patient 2: never septic, Lactate missing throughout
    for h, hr in enumerate([70, 72]):
        rows.append({"Patient_ID": 2, "Age": 45, "Gender": 0, "HR": hr,
                     "Lactate": np.nan, "ICULOS": h + 1, "SepsisLabel": 0})
    return pd.DataFrame(rows)


def test_aggregate_one_row_per_patient():
    out = data.aggregate_patients(_toy_raw())
    assert list(out.index) == [1, 2]
    assert out.index.name == config.PATIENT_ID_COL


def test_target_is_ever_septic():
    out = data.aggregate_patients(_toy_raw())
    assert out.loc[1, config.TARGET_COL] == 1   # septic on last hour
    assert out.loc[2, config.TARGET_COL] == 0


def test_range_and_iculos_features():
    out = data.aggregate_patients(_toy_raw())
    # HR range for patient 1 = 110 - 80 = 30
    assert out.loc[1, "HR_range"] == 30
    # ICULOS_max = total hours in ICU
    assert out.loc[1, "ICULOS_max"] == 3
    assert out.loc[2, "ICULOS_max"] == 2


def test_min_max_mean_std_columns_exist():
    out = data.aggregate_patients(_toy_raw())
    for agg in ("min", "max", "mean", "std", "range"):
        assert f"HR_{agg}" in out.columns


def test_split_features_target_removes_target():
    out = data.aggregate_patients(_toy_raw())
    X, y = data.split_features_target(out)
    assert config.TARGET_COL not in X.columns
    assert y.name == config.TARGET_COL
    assert len(X) == len(y) == 2
