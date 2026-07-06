"""Model definitions and hyperparameter tuning.

Both models are tuned through the leakage-free pipeline from
:mod:`sepsis_icu.features` (scale -> RFE -> SMOTE -> classifier), so the reported
cross-validation scores are honest: scaling, feature selection, and oversampling
are all re-fit inside every CV fold, and the number of selected features is tuned
alongside the model's own hyperparameters.

The deployed model is compact: a plain ``StandardScaler -> classifier`` trained
on only the selected features (SMOTE applied once to that training data), so the
serving API takes a handful of feature values rather than all 173.
"""

from __future__ import annotations

from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config
from .features import build_model_pipeline

# Number of features RFE keeps. The notebooks' dense sweep peaked at k=50 and
# both models in the full run re-confirmed it, so we fix k here rather than
# sweeping it — selecting inside CV for *every* candidate multiplies the (already
# expensive) fit count with little payoff. Pass a list of >1 value to re-enable
# tuning it as a pipeline hyperparameter.
RFE_CANDIDATES = [50]


def _cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)


def clone_best_classifier(template_estimator, params):
    """Return a fresh classifier of the same type as ``template_estimator`` with ``params``."""
    est = clone(template_estimator)
    # cast numpy scalars from the search to plain python for a clean estimator repr
    clean = {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}
    est.set_params(**clean)
    return est


def deployable_model_from_search(search, X_train_reduced, y_train, selected_features):
    """Fit the compact model that actually gets deployed.

    Tuning finds the best hyperparameters and the features RFE selected; the
    model we ship is a plain ``StandardScaler -> classifier`` trained on just
    those features, with SMOTE applied once to balance the training data (no
    additional class weighting — SMOTE already balances). This is the standard
    "select via CV, then refit on all training data" pattern; the test set is
    still never touched.
    """
    params = {
        k.replace("clf__", ""): v
        for k, v in search.best_params_.items()
        if k.startswith("clf__")
    }
    estimator = clone_best_classifier(search.best_estimator_.named_steps["clf"], params)

    X_sel = X_train_reduced[list(selected_features)]
    X_res, y_res = SMOTE(random_state=config.RANDOM_SEED).fit_resample(X_sel, y_train)

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    pipe.fit(X_res, y_res)
    return pipe


# --- Random Forest -----------------------------------------------------------
def tune_random_forest(X_train, y_train, verbose: int = 0) -> GridSearchCV:
    """Grid-search Random Forest hyperparameters *and* the RFE feature count.

    No ``class_weight`` here: SMOTE inside the pipeline balances the classes, so
    adding class weighting on top would double-count the imbalance correction.
    """
    pipe = build_model_pipeline(
        RandomForestClassifier(random_state=config.RANDOM_SEED)
    ).set_params(rfe__n_features_to_select=RFE_CANDIDATES[0])
    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 20],
        "clf__min_samples_split": [2, 5],
        "clf__max_features": ["sqrt"],
    }
    if len(RFE_CANDIDATES) > 1:
        param_grid["rfe__n_features_to_select"] = RFE_CANDIDATES
    search = GridSearchCV(
        pipe, param_grid, cv=_cv(), scoring="f1", n_jobs=-1, verbose=verbose
    )
    search.fit(X_train, y_train)
    return search


# --- XGBoost -----------------------------------------------------------------
def tune_xgboost(X_train, y_train, n_iter: int = 25, verbose: int = 0):
    """Bayesian-search XGBoost hyperparameters *and* the RFE feature count.

    Uses scikit-optimize's ``BayesSearchCV`` when available; falls back to
    ``RandomizedSearchCV`` (equivalent budget) so the pipeline never hard-fails
    on a missing optional dependency. ``scale_pos_weight`` is left at its default
    of 1 because SMOTE balances the classes inside the pipeline.
    """
    pipe = build_model_pipeline(
        XGBClassifier(
            eval_metric="logloss", random_state=config.RANDOM_SEED, verbosity=0
        )
    ).set_params(rfe__n_features_to_select=RFE_CANDIDATES[0])
    try:
        from skopt import BayesSearchCV
        from skopt.space import Categorical, Integer, Real

        search_space = {
            "clf__n_estimators": Integer(50, 400),
            "clf__max_depth": Integer(2, 8),
            "clf__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "clf__subsample": Real(0.5, 1.0),
            "clf__colsample_bytree": Real(0.5, 1.0),
        }
        if len(RFE_CANDIDATES) > 1:
            search_space["rfe__n_features_to_select"] = Categorical(RFE_CANDIDATES)
        search = BayesSearchCV(
            pipe,
            search_space,
            n_iter=n_iter,
            cv=_cv(),
            scoring="f1",
            n_jobs=-1,
            random_state=config.RANDOM_SEED,
            verbose=verbose,
        )
    except ImportError:
        from scipy.stats import loguniform, randint, uniform
        from sklearn.model_selection import RandomizedSearchCV

        param_dist = {
            "clf__n_estimators": randint(50, 400),
            "clf__max_depth": randint(2, 9),
            "clf__learning_rate": loguniform(0.01, 0.3),
            "clf__subsample": uniform(0.5, 0.5),
            "clf__colsample_bytree": uniform(0.5, 0.5),
        }
        if len(RFE_CANDIDATES) > 1:
            param_dist["rfe__n_features_to_select"] = RFE_CANDIDATES
        search = RandomizedSearchCV(
            pipe,
            param_dist,
            n_iter=n_iter,
            cv=_cv(),
            scoring="f1",
            n_jobs=-1,
            random_state=config.RANDOM_SEED,
            verbose=verbose,
        )

    search.fit(X_train, y_train)
    return search
