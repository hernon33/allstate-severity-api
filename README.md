# Allstate Claims Severity — Prediction API

**DATA 6545 Final Project | Spring 2026 | Connor Hernon**

A production-grade machine learning system that predicts the financial severity of insurance claims using the [Allstate Claims Severity](https://www.kaggle.com/c/allstate-claims-severity) dataset. The system provides a REST API for scoring new claims with predicted dollar loss, severity tier, and high-severity risk flag.

---

## Live API

> Base URL: `https://allstate-severity-api.onrender.com`

| Method | Endpoint         | Description                          |
|--------|-----------------|--------------------------------------|
| GET    | `/health`        | Liveness check                       |
| GET    | `/model_info`    | Model metadata and performance       |
| POST   | `/predict`       | Single claim prediction              |
| POST   | `/predict_batch` | Batch claim predictions (max 500)    |

### Example Request

```bash
curl -X POST https://allstate-severity-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cat1": "A", "cat2": "B", "cat80": "D",
    "cont1": 0.72, "cont14": 0.71
  }'
```

### Example Response

```json
{
  "predicted_loss": 2847.35,
  "severity_tier": "Moderate",
  "high_severity_flag": false,
  "high_severity_probability": 0.0821,
  "warnings": []
}
```

---

## Project Structure

```
allstate_project/
├── data/
│   ├── raw/                  # train.csv, test.csv (not committed)
│   └── processed/            # test_predictions.csv
├── models/
│   ├── lgbm_regressor.joblib
│   ├── lgbm_classifier.joblib
│   ├── label_encoders.joblib
│   └── metadata.joblib
├── notebooks/
│   ├── notebook_1.ipynb      # EDA, preprocessing, Ridge, Random Forest
│   └── notebook_2.ipynb      # LightGBM, CV, SHAP, error analysis
├── figures/
│   ├── shap_summary_dot.png
│   ├── shap_summary_bar.png
│   └── ...
├── app.py                    # Flask API
├── train_model.py            # Model training script
├── requirements.txt
├── Dockerfile
├── render.yaml
└── README.md
```

---

## Model Performance

### Regression (Primary Task — Predict Loss in USD)

| Model            | MAE        | RMSE       | R²     |
|-----------------|-----------|-----------|--------|
| Ridge Regression | $1,244.68 | $2,185.06 | 0.4149 |
| Random Forest    | $1,223.44 | $2,055.23 | 0.4823 |
| **LightGBM**     | **$1,135.16** | **$1,895.83** | **0.5595** |

5-fold cross-validation: Mean MAE **$1,145.91** ± $7.95 (stable, no overfitting)

### Classification (Secondary Task — Flag High-Severity Claims)

High-severity threshold: **$6,401.74** (top 10% of training loss distribution)

| Metric              | Value  |
|--------------------|--------|
| AUC-ROC             | 0.9315 |
| Precision (High)    | 0.72   |
| Recall (High)       | 0.47   |
| F1 (High)           | 0.57   |

### Severity Tier Distribution (Test Set)

| Tier     | Claims  | % of Total |
|---------|---------|-----------|
| Low      | 9,784   | 7.8%      |
| Moderate | 80,756  | 64.3%     |
| High     | 28,049  | 22.4%     |
| Extreme  | 6,957   | 5.5%      |

---

## Key Findings

- **cat80** is the single most predictive feature by a significant margin (SHAP analysis)
- **cont14** shows a highly nonlinear relationship with loss — explaining why linear models underperform
- The model performs well on Low and Moderate claims but **systematically underpredicts Extreme claims** (mean error −$3,237 for claims above the high-severity threshold)
- 5-fold CV standard deviation of $7.95 confirms the model is stable and not overfitting

---

## How to Reproduce

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/allstate-severity-api.git
cd allstate-severity-api
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the data

Download `train.csv` and `test.csv` from [Kaggle](https://www.kaggle.com/c/allstate-claims-severity/data) and place them in `data/raw/`.

### 4. Train the models

```bash
python train_model.py --data_dir ./data/raw --output_dir ./models
```

This saves all four model artifacts to `./models/`.

### 5. Run the API locally

```bash
python app.py
# or with gunicorn:
gunicorn --bind 0.0.0.0:5000 app:app
```

### 6. Run with Docker

```bash
docker build -t allstate-api .
docker run -p 8080:8080 allstate-api
```

---

## Deploying to Render

1. Push this repository to GitHub (include the `models/` directory)
2. Log in to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repository
4. Render will detect `render.yaml` automatically
5. Click **Deploy** — the `/health` endpoint confirms a successful deployment

---

## Limitations

- The dataset is fully anonymized — feature names like `cat80` and `cont14` have no real-world label, limiting business interpretability
- The model systematically underpredicts extreme claims (the most financially important segment)
- Recall on high-severity claims is 0.47 — roughly half of truly extreme claims are not flagged by the classifier
- No time-based features are available, so claim lifecycle dynamics cannot be modeled
- Fairness cannot be fully assessed because protected attributes are not available in the anonymized dataset

---

## Ethical Considerations

- **Potential harms**: The system could be used to deprioritize review of low-scored claims even when claimant need is high
- **Bias**: Anonymized categorical features may encode proxies for protected characteristics; bias auditing is recommended before production use
- **Privacy**: The Kaggle dataset contains no PII; no claimant data is stored or transmitted
- **Transparency**: SHAP values provide per-prediction explanations; affected parties should have access to explanations for automated decisions

---

## Dependencies

- Python 3.11
- `lightgbm==4.3.0`
- `scikit-learn==1.4.2`
- `flask==3.0.3`
- `gunicorn==22.0.0`
- `numpy`, `pandas`, `joblib`

See `requirements.txt` for pinned versions.
