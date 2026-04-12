
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

regressor = joblib.load("/content/drive/MyDrive/Colab Notebooks/DATA6545/allstate_project/models/lgbm_regressor.joblib")
classifier = joblib.load("/content/drive/MyDrive/Colab Notebooks/DATA6545/allstate_project/models/lgbm_classifier.joblib")

HIGH_SEVERITY_THRESHOLD = 6401.74

SEVERITY_TIERS = [
    (0, 1000, "Low"),
    (1000, 3000, "Moderate"),
    (3000, 6401.74, "High"),
    (6401.74, float("inf"), "Extreme")
]

def get_severity_tier(predicted_loss):
    for lower, upper, label in SEVERITY_TIERS:
        if lower <= predicted_loss < upper:
            return label
    return "Extreme"

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "LightGBM Claim Severity"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        input_df = pd.DataFrame([data])
        cat_cols = [c for c in input_df.columns if c.startswith("cat")]
        for col in cat_cols:
            le = LabelEncoder()
            input_df[col] = le.fit_transform(input_df[col].astype(str))
        pred_log = regressor.predict(input_df)[0]
        predicted_loss = float(np.expm1(pred_log))
        high_sev_prob = float(classifier.predict_proba(input_df)[0][1])
        high_sev_flag = bool(predicted_loss >= HIGH_SEVERITY_THRESHOLD)
        severity_tier = get_severity_tier(predicted_loss)
        return jsonify({
            "predicted_loss": round(predicted_loss, 2),
            "severity_tier": severity_tier,
            "high_severity_flag": high_sev_flag,
            "high_severity_probability": round(high_sev_prob, 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    try:
        data = request.get_json()
        if not data or "claims" not in data:
            return jsonify({"error": "Expected JSON with key claims containing a list"}), 400
        input_df = pd.DataFrame(data["claims"])
        cat_cols = [c for c in input_df.columns if c.startswith("cat")]
        for col in cat_cols:
            le = LabelEncoder()
            input_df[col] = le.fit_transform(input_df[col].astype(str))
        pred_logs = regressor.predict(input_df)
        predicted_losses = np.expm1(pred_logs)
        high_sev_probs = classifier.predict_proba(input_df)[:, 1]
        results = []
        for i, (loss, prob) in enumerate(zip(predicted_losses, high_sev_probs)):
            results.append({
                "claim_index": i,
                "predicted_loss": round(float(loss), 2),
                "severity_tier": get_severity_tier(float(loss)),
                "high_severity_flag": bool(float(loss) >= HIGH_SEVERITY_THRESHOLD),
                "high_severity_probability": round(float(prob), 4)
            })
        return jsonify({"predictions": results, "total_claims": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
