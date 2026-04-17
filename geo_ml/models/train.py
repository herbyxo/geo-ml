"""
Prospectivity model training with Positive-Unlabeled (PU) bagging.

Approach:
  We have confirmed deposit locations (positive) and millions of unlabeled pixels.
  Standard binary classification treats unlabeled as negative — wrong, because
  undiscovered deposits exist in the unlabeled set.

  Bagging-PU: train N classifiers, each on all positives + a random unlabeled
  subset as pseudo-negatives, then average predictions.

References:
  Mordelet & Vert (2014) — PU bagging framework
  Liu et al. (2021) — Bagging-PU with Bayesian HPO for 3D mineral potential
  Zhang et al. (2025) — geospatially dissimilar negative selection
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import shap
from sklearn.base import clone
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

log = logging.getLogger(__name__)


def make_base_estimator(
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
        verbosity=0,
    )


def bagging_pu_train(
    X_pos: np.ndarray,
    X_unlabeled: np.ndarray,
    base_estimator: Any | None = None,
    n_bags: int = 50,
    unlabeled_ratio: float = 1.0,
    random_state: int = 42,
) -> list[Any]:
    """
    Bagging-PU learning.

    Args:
        X_pos: Feature matrix for known deposits.
        X_unlabeled: Feature matrix for unlabeled pixels.
        base_estimator: sklearn-compatible classifier. Defaults to XGBClassifier.
        n_bags: Number of bagged models.
        unlabeled_ratio: Ratio of unlabeled samples to positive per bag.
        random_state: Seed.

    Returns:
        List of fitted estimators.
    """
    if base_estimator is None:
        base_estimator = make_base_estimator(random_state=random_state)

    rng = np.random.default_rng(random_state)
    n_neg = max(1, int(len(X_pos) * unlabeled_ratio))
    n_neg = min(n_neg, len(X_unlabeled))

    estimators = []
    for i in range(n_bags):
        idx = rng.choice(len(X_unlabeled), size=n_neg, replace=False)
        X_neg = X_unlabeled[idx]

        X_train = np.vstack([X_pos, X_neg])
        y_train = np.array([1] * len(X_pos) + [0] * len(X_neg))

        est = clone(base_estimator)
        est.set_params(random_state=random_state + i)
        est.fit(X_train, y_train)
        estimators.append(est)

        if (i + 1) % 10 == 0:
            log.info(f"  Trained bag {i + 1}/{n_bags}")

    return estimators


def pu_predict_proba(estimators: list[Any], X: np.ndarray) -> np.ndarray:
    """Average predicted probability (positive class) across all bags."""
    probs = np.stack([est.predict_proba(X)[:, 1] for est in estimators], axis=0)
    return probs.mean(axis=0)


def spatial_cross_validate(
    X_pos: np.ndarray,
    X_unlabeled: np.ndarray,
    pos_coords: np.ndarray,
    n_folds: int = 5,
    n_bags: int = 20,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Spatial cross-validation using geographic blocking on positive samples.

    Splits positive samples into spatial folds based on latitude bands,
    trains on (n-1) folds, evaluates on held-out fold.

    Args:
        X_pos: Features for positive samples.
        X_unlabeled: Features for unlabeled pixels.
        pos_coords: (N, 2) array of (lon, lat) for positive samples.
        n_folds: Number of spatial folds.
        n_bags: Bags per fold (fewer for CV speed).
        random_state: Seed.

    Returns:
        Dict with fold-level metrics and summary.
    """
    from sklearn.metrics import roc_auc_score

    lats = pos_coords[:, 1]
    lat_sorted = np.argsort(lats)
    fold_size = len(lat_sorted) // n_folds

    fold_results = []
    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else len(lat_sorted)
        test_idx = lat_sorted[start:end]
        train_idx = np.concatenate([lat_sorted[:start], lat_sorted[end:]])

        X_train_pos = X_pos[train_idx]
        X_test_pos = X_pos[test_idx]

        estimators = bagging_pu_train(
            X_train_pos, X_unlabeled,
            n_bags=n_bags, random_state=random_state + fold
        )

        # Evaluate: positive test samples + random unlabeled as pseudo-test-negatives
        rng = np.random.default_rng(random_state + fold + 1000)
        n_test_neg = min(len(X_test_pos) * 5, len(X_unlabeled))
        test_neg_idx = rng.choice(len(X_unlabeled), size=n_test_neg, replace=False)
        X_test_neg = X_unlabeled[test_neg_idx]

        X_test = np.vstack([X_test_pos, X_test_neg])
        y_test = np.array([1] * len(X_test_pos) + [0] * len(X_test_neg))

        proba = pu_predict_proba(estimators, X_test)
        auc = roc_auc_score(y_test, proba)
        fold_results.append({"fold": fold, "auc": auc, "n_test_pos": len(X_test_pos)})
        log.info(f"  Fold {fold}: AUC={auc:.3f} (n_pos={len(X_test_pos)})")

    mean_auc = np.mean([r["auc"] for r in fold_results])
    std_auc = np.std([r["auc"] for r in fold_results])
    log.info(f"Spatial CV: AUC = {mean_auc:.3f} +/- {std_auc:.3f}")

    return {"folds": fold_results, "mean_auc": mean_auc, "std_auc": std_auc}


def compute_shap_importance(
    estimator: Any,
    X: np.ndarray,
    feature_names: list[str],
    max_samples: int = 500,
) -> tuple[dict[str, float], np.ndarray]:
    """
    Compute SHAP feature importance using TreeExplainer.

    Returns:
        (importance_dict, shap_values_array)
    """
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(max_samples, len(X)), replace=False)
    X_sample = X[idx]

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = dict(zip(feature_names, mean_abs))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    return importance, shap_values


def predict_map(
    estimators: list[Any],
    X_full: np.ndarray,
    mask: np.ndarray,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """
    Generate a 2D prospectivity probability map.

    Returns:
        2D float32 array, NaN where masked.
    """
    proba = pu_predict_proba(estimators, X_full)
    result = np.full(grid_shape, np.nan, dtype=np.float32)
    result[mask] = proba
    return result
