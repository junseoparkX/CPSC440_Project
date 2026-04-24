# svm_l1_nz.py
# Usage:
#   python svm_l1_nz.py
#   python svm_l1_nz.py --scoring roc_auc
#
# Required files in SVM/data:
#   X_train.csv, y_train.csv, X_test.csv

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold


# Base directory = SVM folder (one level above this file)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "nz"   # shorter folder name


def load_data():
    """Load train/test split from CSV files."""
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()  # Series
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    return X_train, y_train, X_test


def svm_l1_select_nonzero_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    C_grid,
    eps: float = 1e-6,          # default changed to 1e-6
    scoring: str = "roc_auc",
    random_state: int = 42,
):
    """
    Feature selection using an L1-regularized Linear SVM (binary).
    Steps:
      1) Standardize
      2) CV-tune C by GridSearchCV
      3) Fit best model on full train
      4) Select features with |coef| > eps

    Returns:
      selected_idx (np.ndarray), coefs (np.ndarray), best_params (dict), best_score (float)
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", LinearSVC(
            penalty="l1",
            dual=False,            # L1 => primal
            max_iter=20000,
            random_state=random_state,
        )),
    ])

    param_grid = {"svc__C": list(C_grid)}

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )

    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    best_score = float(grid.best_score_)
    best_pipe = grid.best_estimator_

    best_svc = best_pipe.named_steps["svc"]
    coefs = best_svc.coef_.ravel()
    abs_coefs = np.abs(coefs)

    selected_idx = np.where(abs_coefs > eps)[0]

    if selected_idx.size == 0:
        raise RuntimeError(
            f"All coefficients are <= eps={eps}. "
            f"Try increasing C grid (larger C) or check convergence (max_iter/tol). "
            f"Current best params: {best_params}, best CV score: {best_score:.4f}"
        )

    return selected_idx, coefs, best_params, best_score


def main(scoring: str, random_state: int):
    eps = 1e-6  # fixed to 1e-6 as requested
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test = load_data()

    # Drop all-NaN columns from train and align test columns
    X_train = X_train.dropna(axis="columns", how="all")
    X_test = X_test[X_train.columns]

    # Replace inf -> NaN -> 0 to be safe
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    # C grid (expandable)
    C_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 30.0, 100.0]

    idx, coefs, best_params, best_score = svm_l1_select_nonzero_features(
        X_train=X_train,
        y_train=y_train,
        C_grid=C_grid,
        eps=eps,
        scoring=scoring,
        random_state=random_state,
    )

    selected_cols = X_train.columns[idx]

    # extra print: reduced to how many (and from how many)
    n_before = X_train.shape[1]
    n_after = len(selected_cols)

    print(f"[SVM-L1 GridSearch] Best params: {best_params}")
    print(f"[SVM-L1 GridSearch] Best CV {scoring}: {best_score:.4f}")
    print(f"[SVM-L1 FS] eps={eps} | features: {n_before} -> {n_after}")

    # Save reduced train/test (new names)
    X_train_sel = X_train[selected_cols].copy()
    X_test_sel = X_test[selected_cols].copy()

    out_train = OUT_DIR / "Xtr_nz.csv"
    out_test = OUT_DIR / "Xte_nz.csv"

    X_train_sel.to_csv(out_train, index=False)
    X_test_sel.to_csv(out_test, index=False)

    print(f"[SAVE] train: {out_train}")
    print(f"[SAVE] test : {out_test}")

    # Save selected feature names + coefficients
    coef_df = pd.DataFrame({
        "feature": X_train.columns,
        "coef": coefs,
        "abs_coef": np.abs(coefs),
        "selected": np.abs(coefs) > eps,
    })

    coef_selected = (
        coef_df.loc[idx]
        .sort_values("abs_coef", ascending=False)
        .reset_index(drop=True)
    )

    out_coef = OUT_DIR / "nz_coef.csv"
    coef_selected.to_csv(out_coef, index=False)
    print(f"[SAVE] coef : {out_coef}")

    # Save just feature list
    out_list = OUT_DIR / "nz_feat.txt"
    with open(out_list, "w", encoding="utf-8") as f:
        for name in selected_cols:
            f.write(f"{name}\n")
    print(f"[SAVE] feats: {out_list}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoring", type=str, default="roc_auc", help="GridSearchCV scoring (e.g., roc_auc).")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    main(scoring=args.scoring, random_state=args.random_state)
