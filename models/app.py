"""
app.py
------
Allstate Claims Severity — Prediction API
Endpoints:
    GET  /health          — liveness check
    POST /predict         — single claim prediction
    POST /predict_batch   — batch claim predictions
    GET  /model_info      — model metadata and performance summary
"""

import os
import logging
import numpy as np
import joblib
from flask import Flask, request, jsonify

# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# App Initialization & Artifact Loading

app = Flask(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "./models")

log.info("Loading model artifacts from: %s", MODEL_DIR)

try:
    regressor      = joblib.load(os.path.join(MODEL_DIR, "lgbm_regressor.joblib"))
    classifier     = joblib.load(os.path.join(MODEL_DIR, "lgbm_classifier.joblib"))
    label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.joblib"))
    metadata       = joblib.load(os.path.join(MODEL_DIR, "metadata.joblib"))
    log.info("All artifacts loaded successfully.")
except FileNotFoundError as e:
    log.error("Model artifact not found: %s", e)
    log.error("Run train_model.py first to generate model files.")
    raise

FEAT_COLS   = metadata["feat_cols"]
CAT_COLS    = metadata["cat_cols"]
NUM_COLS    = metadata["num_cols"]
TIERS       = metadata["tier_boundaries"]
HS_THRESH   = metadata["high_sev_threshold"]

# Helper Functions

def assign_severity_tier(predicted_loss: float) -> str:
    """Map a dollar-scale predicted loss to a severity tier label."""
    if predicted_loss <= TIERS["low_max"]:
        return "Low"
    elif predicted_loss <= TIERS["moderate_max"]:
        return "Moderate"
    elif predicted_loss <= TIERS["high_max"]:
        return "High"
    else:
        return "Extreme"


def encode_claim(claim: dict) -> np.ndarray:
    """
    Encode a single claim dictionary into a feature vector.

    Categorical features are label-encoded using the fitted encoders.
    Unseen categories are mapped to 0 (the most common fallback).
    Continuous features are passed through as-is.
    Missing features default to 0.
    """
    row = []
    for col in FEAT_COLS:
        val = claim.get(col, None)

        if col in CAT_COLS:
            le = label_encoders[col]
            str_val = str(val) if val is not None else "A"
            if str_val in le.classes_:
                encoded = int(le.transform([str_val])[0])
            else:
                # Unseen category — default to index 0 (most frequent proxy)
                encoded = 0
            row.append(encoded)
        else:
            # Continuous feature — use 0.0 as default for missing values
            row.append(float(val) if val is not None else 0.0)

    return np.array(row).reshape(1, -1)


def predict_claim(claim: dict) -> dict:
    """Run regressor + classifier on a single encoded claim dict."""
    X = encode_claim(claim)

    # Regression: predict log loss → inverse transform to dollars
    log_pred     = regressor.predict(X)[0]
    dollar_pred  = float(np.expm1(log_pred))

    # Classification: high-severity probability
    hs_prob      = float(classifier.predict_proba(X)[0][1])
    hs_flag      = bool(hs_prob >= 0.5)

    return {
        "predicted_loss"          : round(dollar_pred, 2),
        "severity_tier"           : assign_severity_tier(dollar_pred),
        "high_severity_flag"      : hs_flag,
        "high_severity_probability": round(hs_prob, 4),
    }


def validate_continuous_features(claim: dict) -> list:
    """Return a list of warnings for out-of-range continuous features."""
    warnings = []
    for col in NUM_COLS:
        val = claim.get(col)
        if val is not None:
            try:
                fval = float(val)
                if not (0.0 <= fval <= 1.0):
                    warnings.append(
                        f"{col} value {fval} is outside the expected [0, 1] range."
                    )
            except (TypeError, ValueError):
                warnings.append(f"{col} could not be parsed as a number.")
    return warnings


# Routes

@app.route("/health", methods=["GET"])
def health():
    """Liveness check — returns 200 if the service is up and models are loaded."""
    return jsonify({
        "status" : "ok",
        "model"  : "LightGBM Claims Severity v1.0",
        "features": len(FEAT_COLS),
    }), 200


@app.route("/model_info", methods=["GET"])
def model_info():
    """Return model metadata and validation performance."""
    return jsonify({
        "model_version"       : "1.0",
        "dataset"             : "Allstate Claims Severity (Kaggle)",
        "training_rows"       : 188318,
        "features"            : {"categorical": len(CAT_COLS), "continuous": len(NUM_COLS)},
        "high_severity_threshold_usd": round(HS_THRESH, 2),
        "severity_tiers"      : {
            "Low"     : f"$0 – ${TIERS['low_max']:,.2f}",
            "Moderate": f"${TIERS['low_max']:,.2f} – ${TIERS['moderate_max']:,.2f}",
            "High"    : f"${TIERS['moderate_max']:,.2f} – ${TIERS['high_max']:,.2f}",
            "Extreme" : f"> ${TIERS['high_max']:,.2f}",
        },
        "validation_performance": metadata.get("validation_metrics", {}),
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict loss severity for a single insurance claim.

    Request body (JSON):
        A flat object with claim feature values.
        Categorical features: cat1 – cat116 (string values, e.g. "A", "B")
        Continuous features:  cont1 – cont14 (float values in [0, 1])
        All features are optional; missing values use safe defaults.

    Response (JSON):
        predicted_loss            (float)  — estimated claim loss in USD
        severity_tier             (string) — Low / Moderate / High / Extreme
        high_severity_flag        (bool)   — true if predicted top-10% claim
        high_severity_probability (float)  — classifier confidence score
        warnings                  (list)   — data quality notes, if any
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON (Content-Type: application/json)"}), 400

    claim = request.get_json()
    if not isinstance(claim, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        result   = predict_claim(claim)
        warnings = validate_continuous_features(claim)
        result["warnings"] = warnings

        log.info("Prediction — Loss: $%.2f | Tier: %s | HS Prob: %.3f",
                 result["predicted_loss"],
                 result["severity_tier"],
                 result["high_severity_probability"])

        return jsonify(result), 200

    except Exception as e:
        log.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    Predict loss severity for a list of insurance claims.

    Request body (JSON):
        { "claims": [ <claim_object>, <claim_object>, ... ] }
        Maximum 500 claims per request.

    Response (JSON):
        { "predictions": [ <result_object>, ... ], "count": N }
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON (Content-Type: application/json)"}), 400

    body = request.get_json()

    if not isinstance(body, dict) or "claims" not in body:
        return jsonify({"error": "Body must be { \"claims\": [...] }"}), 400

    claims = body["claims"]
    if not isinstance(claims, list) or len(claims) == 0:
        return jsonify({"error": "\"claims\" must be a non-empty list."}), 400

    if len(claims) > 500:
        return jsonify({"error": "Maximum 500 claims per batch request."}), 400

    try:
        results = []
        for i, claim in enumerate(claims):
            if not isinstance(claim, dict):
                results.append({"index": i, "error": "Claim must be a JSON object."})
                continue
            result           = predict_claim(claim)
            result["index"]  = i
            result["warnings"] = validate_continuous_features(claim)
            results.append(result)

        log.info("Batch prediction — %d claims processed.", len(results))
        return jsonify({"predictions": results, "count": len(results)}), 200

    except Exception as e:
        log.exception("Batch prediction failed")
        return jsonify({"error": str(e)}), 500


# Entry Point

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
