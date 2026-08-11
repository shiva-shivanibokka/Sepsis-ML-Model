"""Early sepsis prediction from PhysioNet 2019 ICU vital-sign / lab time-series.

A small, reproducible ML package: hourly-record aggregation into one row per
patient, leakage-free feature selection, two tuned classifiers (Random Forest,
XGBoost) with SMOTE for the class imbalance, threshold calibration on a
validation split, evaluation, and a FastAPI serving layer. The four teaching
notebooks import from these modules so there is one implementation of each step.
"""

from __future__ import annotations

__version__ = "1.1.0"

# Submodules are resolved on first access rather than imported here.
#
# `data`, `features` and `models` pull in pandas, scikit-learn and
# imbalanced-learn. Importing them from the package root meant that merely
# importing `sepsis_icu.serve` dragged the entire training stack in — which the
# deployed service does not install at all, so it would have failed on import,
# and which costs seconds of cold start even where it is installed.
#
# `from sepsis_icu import features` still works: PEP 562 sends the lookup here.
_SUBMODULES = ("config", "data", "evaluate", "features", "models", "serve")


def __getattr__(name: str):
    if name in _SUBMODULES:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SUBMODULES))
