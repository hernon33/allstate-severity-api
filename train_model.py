"""
train_model.py — Allstate Claims Severity
==========================================
Trains LightGBM regressor and classifier, saves all four model artifacts
to the output directory, and logs results to MLflow.

Usage:
    python train_model.py --data_dir ./data/raw --output_dir ./models

Outputs:
    models/lgbm_regressor.joblib
    models/lgbm_classifier.joblib
    models/label_encoders.joblib
    models/metadata.joblib
"""

import argparse
import os
import time
import warnings

import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train Allstate severity models")
    p.add_argument("--data_dir",   default="./data/raw", help="Folder containing train.csv and test.csv")
    p.add_argument("--output_dir", default="./models",   help="Folder to write model artifacts")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--cv_folds",   type=int, default=5)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading and feature setup
# ---------------------------------------------------------------------------

def load_data(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "test.csv"))
    print(f"Train: {train.shape}  |  Test: {test.shape}")
    return train, test


def get_feature_lists(train):
    cat_cols  = [c for c in train.columns if c.startswith("cat")]
    num_cols  = [c for c in train.columns if c.startswith("cont")]
    feat_cols = cat_cols + num_cols
    return feat_cols, cat_cols, num_cols


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def label_encode(train, test, feat_cols, cat_cols):
    """
    Fit one LabelEncoder per categorical column on the union of train and test
    values. This guarantees that any category seen at inference time has a valid
    integer mapping, which is the same guarantee the deployed API relies on.
    """
    combined = pd.concat([train[feat_cols], test[feat_cols]], axis=0, ignore_index=True)

    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(combined[col].astype(str))
        label_encoders[col] = le
        combined[col] = le.transform(combined[col].astype(str))

    X_train_full = combined.iloc[:len(train)].copy()
    X_test       = combined.iloc[len(train):].copy()

    return X_train_full, X_test, label_encoders


# ---------------------------------------------------------------------------
# Regression model
# ---------------------------------------------------------------------------

def train_regressor(X_train, y_train, X_valid, y_valid, seed):
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="mae",
        callbacks=[
            # Early stopping: halt if validation MAE doesn't improve for 50 rounds
            __import__("lightgbm").early_stopping(stopping_rounds=50, verbose=False),
            __import__("lightgbm").log_evaluation(period=-1),
        ],
    )
    return model


def evaluate_regressor(model, X_valid, y_valid_log):
    pred_log = model.predict(X_valid)
    pred     = np.expm1(pred_log)
    actual   = np.expm1(y_valid_log)

    mae  = mean_absolute_error(actual, pred)
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    r2   = r2_score(actual, pred)

    print(f"  Regressor  |  MAE: ${mae:,.2f}  RMSE: ${rmse:,.2f}  R²: {r2:.4f}")
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4)}


def run_cross_validation(X, y_log, seed, n_folds):
    print(f"\nRunning {n_folds}-fold cross-validation...")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_maes, fold_r2s = [], []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        t0 = time.time()
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_log.iloc[tr_idx], y_log.iloc[val_idx]

        m = LGBMRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, verbose=-1,
        )
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            callbacks=[
                __import__("lightgbm").early_stopping(stopping_rounds=50, verbose=False),
                __import__("lightgbm").log_evaluation(period=-1),
            ],
        )

        preds  = np.expm1(m.predict(X_val))
        actual = np.expm1(y_val)
        mae    = mean_absolute_error(actual, preds)
        r2     = r2_score(actual, preds)

        fold_maes.append(mae)
        fold_r2s.append(r2)
        elapsed = time.time() - t0
        print(f"  Fold {fold + 1}  |  MAE: ${mae:,.2f}  R²: {r2:.4f}  ({elapsed:.1f}s)")

    mean_mae = float(np.mean(fold_maes))
    std_mae  = float(np.std(fold_maes))
    mean_r2  = float(np.mean(fold_r2s))
    print(f"\n  CV Summary  |  Mean MAE: ${mean_mae:,.2f} ± ${std_mae:.2f}  Mean R²: {mean_r2:.4f}")
    return {"cv_mean_mae": round(mean_mae, 2), "cv_std_mae": round(std_mae, 2), "cv_mean_r2": round(mean_r2, 4)}


# ---------------------------------------------------------------------------
# Classification model
# ---------------------------------------------------------------------------

def train_classifier(X_train, y_train_cls, X_valid, y_valid_cls, seed):
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train_cls)
    return model


def evaluate_classifier(model, X_valid, y_valid_cls):
    probs = model.predict_proba(X_valid)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc  = roc_auc_score(y_valid_cls, probs)
    prec = precision_score(y_valid_cls, preds, zero_division=0)
    rec  = recall_score(y_valid_cls, preds, zero_division=0)
    f1   = f1_score(y_valid_cls, preds, zero_division=0)

    print(f"  Classifier  |  AUC: {auc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    return {"auc_roc": round(auc, 4), "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load data ---
    print("Loading data...")
    train, test = load_data(args.data_dir)
    feat_cols, cat_cols, num_cols = get_feature_lists(train)

    # --- Encode ---
    print("\nEncoding categorical features...")
    X_full, X_test, label_encoders = label_encode(train, test, feat_cols, cat_cols)

    # --- Target ---
    y_log = np.log1p(train["loss"])
    HIGH_SEV_THRESHOLD = float(train["loss"].quantile(0.90))
    y_cls = (train["loss"] >= HIGH_SEV_THRESHOLD).astype(int)

    # --- Split ---
    X_train, X_valid, y_train_log, y_valid_log, y_train_cls, y_valid_cls = train_test_split(
        X_full, y_log, y_cls, test_size=0.2, random_state=args.seed
    )
    print(f"\nTrain: {X_train.shape[0]} rows  |  Valid: {X_valid.shape[0]} rows")
    print(f"High-severity threshold: ${HIGH_SEV_THRESHOLD:,.2f}  |  Positive rate: {y_train_cls.mean():.3f}")

    # --- MLflow setup ---
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("allstate_claims_severity")

    # --- Train regressor ---
    print("\nTraining LightGBM regressor...")
    with mlflow.start_run(run_name="lgbm_regressor"):
        reg_model = train_regressor(X_train, y_train_log, X_valid, y_valid_log, args.seed)
        reg_metrics = evaluate_regressor(reg_model, X_valid, y_valid_log)
        cv_metrics = run_cross_validation(X_full, y_log, args.seed, args.cv_folds)

        mlflow.log_params({
            "model_type": "LGBMRegressor", "n_estimators": 1000,
            "learning_rate": 0.05, "max_depth": 6, "num_leaves": 63,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "target_transform": "log1p", "encoding": "label_encoding",
        })
        mlflow.log_metrics({**reg_metrics, **cv_metrics})

    # --- Train classifier ---
    print("\nTraining LightGBM classifier...")
    with mlflow.start_run(run_name="lgbm_classifier"):
        clf_model = train_classifier(X_train, y_train_cls, X_valid, y_valid_cls, args.seed)
        clf_metrics = evaluate_classifier(clf_model, X_valid, y_valid_cls)

        mlflow.log_params({
            "model_type": "LGBMClassifier", "n_estimators": 500,
            "learning_rate": 0.05, "max_depth": 6,
            "severity_threshold": HIGH_SEV_THRESHOLD,
        })
        mlflow.log_metrics(clf_metrics)

    # --- Save artifacts ---
    print("\nSaving artifacts...")
    tier_boundaries = {
        "low_max":      1000.0,
        "moderate_max": 3000.0,
        "high_max":     HIGH_SEV_THRESHOLD,
    }
    metadata = {
        "feat_cols":          feat_cols,
        "cat_cols":           cat_cols,
        "num_cols":           num_cols,
        "high_sev_threshold": HIGH_SEV_THRESHOLD,
        "tier_boundaries":    tier_boundaries,
        "val_metrics": {
            **reg_metrics,
            **cv_metrics,
            **clf_metrics,
        },
        "dataset": {
            "train_rows": len(train),
            "test_rows":  len(test),
            "n_features": len(feat_cols),
        },
    }

    joblib.dump(reg_model,       os.path.join(args.output_dir, "lgbm_regressor.joblib"))
    joblib.dump(clf_model,       os.path.join(args.output_dir, "lgbm_classifier.joblib"))
    joblib.dump(label_encoders,  os.path.join(args.output_dir, "label_encoders.joblib"))
    joblib.dump(metadata,        os.path.join(args.output_dir, "metadata.joblib"))

    print(f"\nAll artifacts saved to {args.output_dir}/")
    for fname in ["lgbm_regressor.joblib", "lgbm_classifier.joblib", "label_encoders.joblib", "metadata.joblib"]:
        path = os.path.join(args.output_dir, fname)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {fname} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
