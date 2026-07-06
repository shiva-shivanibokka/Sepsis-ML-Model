"""Feature reduction.

Two kinds of reduction, and the distinction is the whole point:

1. **Label-free (unsupervised) cleaning** — zero-variance filter, high-null
   filter + median impute, and a variance filter. None look at the target, so
   fitting them once on the training data leaks nothing and does not inflate
   cross-validation scores. Notebook 02 does this and saves a smaller matrix.

2. **Supervised selection (RFE)** — this DOES use the target. Running it once on
   the full training set and then cross-validating the model on the result is a
   classic *selection leakage* trap: every CV fold's model was built on features
   chosen with that fold's own labels, so CV scores come out optimistically
   high. The notebooks selected ``k`` this way for teaching clarity; the package
   improves on it by putting RFE *inside* a ``Pipeline`` (:func:`build_model_pipeline`)
   so it is re-fit within each CV fold, and by tuning ``k`` as a hyperparameter.

Note on ordering: unlike a scale-then-threshold pipeline (where StandardScaler
forces every column to variance 1.0 and makes any threshold < 1 a silent no-op),
the variance filter here runs on the *raw imputed* values, before scaling — so
it actually removes near-constant features. Scaling lives inside the model
pipeline, after selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.preprocessing import StandardScaler

from . import config


# --- 1. Label-free cleaning (fit on train, apply to test) --------------------
def drop_zero_variance(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Drop features that are constant across all training patients."""
    var = X_train.var()
    zero_cols = var[var == 0].index.tolist()
    return X_train.drop(columns=zero_cols), X_test.drop(columns=zero_cols), zero_cols


def drop_high_null_and_impute(
    X_train: pd.DataFrame, X_test: pd.DataFrame, null_thresh: float = 0.90
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Drop features missing in >``null_thresh`` of training patients; median-impute the rest.

    Median (not mean) is used because clinical outliers are common. Imputation
    happens *after* the high-null drop so we never fill 90% of a column with one
    guessed value.
    """
    null_rate = X_train.isnull().mean()
    high_null = null_rate[null_rate > null_thresh].index.tolist()
    X_train = X_train.drop(columns=high_null)
    X_test = X_test.drop(columns=high_null)

    if X_train.isnull().values.any() or X_test.isnull().values.any():
        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)
    return X_train, X_test, high_null


def variance_filter(
    X_train: pd.DataFrame, X_test: pd.DataFrame, threshold: float = 0.01
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Remove near-constant features using variance on the raw imputed data.

    Fitted on training data only. Returns the reduced frames and kept columns.
    Runs before scaling (see the module docstring for why that matters).
    """
    vt = VarianceThreshold(threshold=threshold).fit(X_train)
    keep = X_train.columns[vt.get_support()]
    return X_train[keep].copy(), X_test[keep].copy(), np.asarray(keep)


# --- 2. Supervised selection lives INSIDE the model pipeline -----------------
def build_model_pipeline(estimator) -> ImbPipeline:
    """Leakage-free pipeline: scale -> RFE select -> SMOTE -> ``estimator``.

    All steps are re-fit inside each CV fold when this is wrapped by
    GridSearchCV / BayesSearchCV. SMOTE is an imblearn step, so it resamples
    only the training portion of each fold — never the validation portion — which
    is exactly why the whole pipeline must use ``imblearn.pipeline.Pipeline``.
    RFE runs before SMOTE (ranking on real, not synthetic, patients); its RF
    ranker uses ``class_weight='balanced'`` to cope with the imbalance while
    ranking.
    """
    return ImbPipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rfe",
                RFE(
                    estimator=RandomForestClassifier(
                        n_estimators=20,
                        max_depth=5,
                        class_weight="balanced",
                        random_state=config.RANDOM_SEED,
                        # single-threaded on purpose: the outer search
                        # parallelizes across folds, so a parallel RF here would
                        # oversubscribe every core and run *slower*.
                        n_jobs=1,
                    ),
                    n_features_to_select=50,  # tuned by the search; just a default
                    step=0.5,
                ),
            ),
            ("smote", SMOTE(random_state=config.RANDOM_SEED)),
            ("clf", estimator),
        ]
    )


def selected_feature_names(fitted_pipeline: ImbPipeline, input_columns) -> np.ndarray:
    """Recover the feature names the fitted pipeline's RFE step actually kept."""
    cols = np.asarray(input_columns)
    return cols[fitted_pipeline.named_steps["rfe"].get_support()]
