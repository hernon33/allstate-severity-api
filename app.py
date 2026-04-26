import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model artifacts once at startup — not on every request
# ---------------------------------------------------------------------------
MODEL_DIR = os.environ.get("MODEL_DIR", "./models")

regressor      = joblib.load(os.path.join(MODEL_DIR, "lgbm_regressor.joblib"))
classifier     = joblib.load(os.path.join(MODEL_DIR, "lgbm_classifier.joblib"))
label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.joblib"))
metadata       = joblib.load(os.path.join(MODEL_DIR, "metadata.joblib"))

FEAT_COLS          = metadata["feat_cols"]
CAT_COLS           = metadata["cat_cols"]
NUM_COLS           = metadata["num_cols"]
HIGH_SEV_THRESHOLD = metadata["high_sev_threshold"]
TIER_BOUNDARIES    = metadata["tier_boundaries"]

# Continuous features in the Allstate dataset are already scaled to [0, 1]
CONT_VALID_RANGE = (0.0, 1.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_severity_tier(loss):
    if loss < TIER_BOUNDARIES["low_max"]:
        return "Low"
    elif loss < TIER_BOUNDARIES["moderate_max"]:
        return "Moderate"
    elif loss < TIER_BOUNDARIES["high_max"]:
        return "High"
    return "Extreme"


def encode_input(raw_dict):
    """
    Turn an arbitrary incoming JSON dict into a fully aligned feature DataFrame
    ready for model inference.

    - Categorical features: encoded using the fitted LabelEncoders from training.
      Any unseen category value falls back to 0 (the most common safe default).
    - Continuous features: passed through as-is, defaulting to 0.0 if missing.
    - All 130 features are present in training order regardless of what the
      caller sent.

    Returns (df, warnings) where warnings is a list of strings describing any
    input values that fell outside expected ranges.
    """
    warnings = []
    row = {}

    for col in CAT_COLS:
        raw_val = str(raw_dict.get(col, ""))
        le = label_encoders[col]
        if raw_val in le.classes_:
            row[col] = int(le.transform([raw_val])[0])
        else:
            row[col] = 0
            if raw_val:
                warnings.append(f"{col}: unknown category '{raw_val}', defaulted to 0")

    for col in NUM_COLS:
        val = raw_dict.get(col, 0.0)
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = 0.0
            warnings.append(f"{col}: non-numeric value, defaulted to 0.0")
        lo, hi = CONT_VALID_RANGE
        if not (lo <= val <= hi):
            warnings.append(f"{col}: value {val:.4f} outside expected range [{lo}, {hi}]")
        row[col] = val

    df = pd.DataFrame([row], columns=FEAT_COLS)
    return df, warnings


def encode_batch(raw_list):
    """
    Same as encode_input but processes a list of claim dicts in a single pass,
    which is substantially faster than calling encode_input in a loop.
    """
    all_warnings = []
    rows = []

    for i, raw_dict in enumerate(raw_list):
        row = {}
        claim_warnings = []

        for col in CAT_COLS:
            raw_val = str(raw_dict.get(col, ""))
            le = label_encoders[col]
            if raw_val in le.classes_:
                row[col] = int(le.transform([raw_val])[0])
            else:
                row[col] = 0
                if raw_val:
                    claim_warnings.append(f"{col}: unknown category '{raw_val}'")

        for col in NUM_COLS:
            val = raw_dict.get(col, 0.0)
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = 0.0
                claim_warnings.append(f"{col}: non-numeric value")
            lo, hi = CONT_VALID_RANGE
            if not (lo <= val <= hi):
                claim_warnings.append(f"{col}: {val:.4f} outside [{lo}, {hi}]")
            row[col] = val

        rows.append(row)
        all_warnings.append(claim_warnings)

    df = pd.DataFrame(rows, columns=FEAT_COLS)
    return df, all_warnings


def run_inference(df):
    """
    Run both models on an already-encoded feature DataFrame. Returns
    predicted losses (dollar scale) and high-severity probabilities.
    """
    log_preds  = regressor.predict(df)
    losses     = np.expm1(log_preds).astype(float)
    sev_probs  = classifier.predict_proba(df)[:, 1].astype(float)
    return losses, sev_probs


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model":  "LightGBM Claim Severity",
        "n_features": len(FEAT_COLS),
    })


@app.route("/model_info", methods=["GET"])
def model_info():
    m = metadata.get("val_metrics", {})
    ds = metadata.get("dataset", {})
    return jsonify({
        "model": "LightGBM (LGBMRegressor + LGBMClassifier)",
        "n_features": len(FEAT_COLS),
        "n_categorical": len(CAT_COLS),
        "n_continuous": len(NUM_COLS),
        "high_severity_threshold": HIGH_SEV_THRESHOLD,
        "severity_tiers": TIER_BOUNDARIES,
        "validation_metrics": {
            "regression": {
                "mae":         m.get("mae"),
                "rmse":        m.get("rmse"),
                "r2":          m.get("r2"),
                "cv_mean_mae": m.get("cv_mean_mae"),
                "cv_std_mae":  m.get("cv_std_mae"),
            },
            "classification": {
                "auc_roc":   m.get("auc_roc"),
                "precision": m.get("precision"),
                "recall":    m.get("recall"),
                "f1":        m.get("f1"),
            },
        },
        "dataset": {
            "train_rows": ds.get("train_rows"),
            "test_rows":  ds.get("test_rows"),
        },
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be a JSON object"}), 400

    try:
        df, warnings = encode_input(data)
        losses, probs = run_inference(df)

        predicted_loss = round(losses[0], 2)
        high_sev_prob  = round(probs[0], 4)

        return jsonify({
            "predicted_loss":          predicted_loss,
            "severity_tier":           get_severity_tier(predicted_loss),
            "high_severity_flag":      bool(predicted_loss >= HIGH_SEV_THRESHOLD),
            "high_severity_probability": high_sev_prob,
            "warnings":                warnings,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = request.get_json(silent=True)
    if not data or "claims" not in data:
        return jsonify({"error": "Expected JSON body with key 'claims' containing a list"}), 400

    claims = data["claims"]
    if not isinstance(claims, list) or len(claims) == 0:
        return jsonify({"error": "'claims' must be a non-empty list"}), 400
    if len(claims) > 500:
        return jsonify({"error": "Batch size limit is 500 claims per request"}), 400

    try:
        df, all_warnings = encode_batch(claims)
        losses, probs = run_inference(df)

        results = []
        for i, (loss, prob) in enumerate(zip(losses, probs)):
            loss = round(float(loss), 2)
            results.append({
                "index":                   i,
                "predicted_loss":          loss,
                "severity_tier":           get_severity_tier(loss),
                "high_severity_flag":      bool(loss >= HIGH_SEV_THRESHOLD),
                "high_severity_probability": round(float(prob), 4),
                "warnings":                all_warnings[i],
            })

        return jsonify({
            "total_claims": len(results),
            "predictions":  results,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
