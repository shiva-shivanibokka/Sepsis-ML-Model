"""Early sepsis prediction from PhysioNet 2019 ICU vital-sign / lab time-series.

A small, reproducible ML package: hourly-record aggregation into one row per
patient, leakage-free feature selection, two tuned classifiers (Random Forest,
XGBoost) with SMOTE for the class imbalance, threshold calibration on a
validation split, evaluation, and a FastAPI serving layer. The four teaching
notebooks import from these modules so there is one implementation of each step.
"""

from __future__ import annotations

__version__ = "1.0.0"

from . import config, data, evaluate, features, models  # noqa: F401
