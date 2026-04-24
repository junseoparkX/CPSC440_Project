# svm_l1_selector.py
# Example: python svm_l1_selector.py --target_k 25
# Required files in SVM/data: X_train.csv, y_train.csv, X_test.csv

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


def load_data():
    """Load train/test split from CSV files (original ~700 features)."""
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()  # convert to Series
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    return X_train, y_train, X_test


def svm_l1_select_features(X_train, y_train, target_k=25):
    """
    Perform feature selection using an L1-regularized Linear SVM.
    - L1 penalty → sparse solution (many coefficients become exactly zero)
    - Among non-zero coefficients, keep up to target_k features with the largest |coef|
    """
    # Pipeline: standardize features + L1-regularized LinearSVC
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", LinearSVC(
            penalty="l1",
            dual=False,          # for L1 we use the primal: dual=False
            max_iter=10000,
            random_state=42,
        )),
    ])

    # Tune only the C parameter (keep the grid small to avoid overkill)
    param_grid = {
        "svc__C": [0.001, 0.01, 0.1, 1.0, 10.0],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )

    grid.fit(X_train, y_train)

    print(f"[SVM-L1 GridSearch] Best params: {grid.best_params_}")
    print(f"[SVM-L1 GridSearch] Best CV AUC: {grid.best_score_:.4f}")

    # Final SVM model (LinearSVC after the scaler in the pipeline)
    best_svc = grid.best_estimator_.named_steps["svc"]

    # coef_: shape (1, n_features) for binary classification
    coefs = best_svc.coef_.ravel()
    abs_coefs = np.abs(coefs)

    # Keep only non-zero weights
    nonzero_mask = abs_coefs > 0
    nonzero_idx = np.where(nonzero_mask)[0]

    if len(nonzero_idx) == 0:
        # If all coefficients are zero, fall back to top-k by |coef| or by index order
        print("[WARN] All coefficients are zero. Falling back to raw top-k indices.")
        sorted_idx = np.argsort(abs_coefs)[::-1]
        effective_k = min(target_k, len(sorted_idx))
        idx = sorted_idx[:effective_k]
    else:
        # Sort non-zero coefficients by descending |coef|
        sorted_nonzero_idx = nonzero_idx[np.argsort(abs_coefs[nonzero_idx])[::-1]]

        effective_k = min(target_k, len(sorted_nonzero_idx))
        if effective_k < target_k:
            print(
                f"[INFO] Requested target_k={target_k}, but only {len(sorted_nonzero_idx)} "
                f"non-zero features available. Using target_k={effective_k} instead."
            )

        idx = sorted_nonzero_idx[:effective_k]

    return idx, coefs


def main(target_k):
    X_train, y_train, X_test = load_data()

    # Optionally: drop all-NaN columns from train and align test columns
    X_train = X_train.dropna(axis="columns", how="all")
    X_test = X_test[X_train.columns]

    # Feature selection with L1-SVM
    idx, coefs = svm_l1_select_features(X_train, y_train, target_k=target_k)

    selected_cols = X_train.columns[idx]
    print(f"[SVM-L1 FS] selected {len(selected_cols)} features (requested target_k={target_k})")

    # Save reduced train/test with only selected features
    X_train_sel = X_train[selected_cols].copy()
    X_test_sel = X_test[selected_cols].copy()

    out_train = DATA_DIR / "X_train_svm25.csv"
    out_test = DATA_DIR / "X_test_svm25.csv"

    X_train_sel.to_csv(out_train, index=False)
    X_test_sel.to_csv(out_test, index=False)

    print(f"[SAVE] Reduced train saved to: {out_train}")
    print(f"[SAVE] Reduced test  saved to: {out_test}")

    # Save feature names + coefficients for interpretation
    coef_df = pd.DataFrame({
        "feature": X_train.columns,
        "coef": coefs,
        "abs_coef": np.abs(coefs),
    })
    coef_df = coef_df.loc[idx].sort_values("abs_coef", ascending=False)
    coef_df.to_csv(DATA_DIR / "svm_l1_feature_coef.csv", index=False)
    print(f"[SAVE] SVM-L1 feature coefficients saved to: {DATA_DIR / 'svm_l1_feature_coef.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_k",
        type=int,
        default=25,
        help="Number of features to keep (by |coef| among non-zero L1-SVM coefficients).",
    )
    args = parser.parse_args()
    main(args.target_k)
