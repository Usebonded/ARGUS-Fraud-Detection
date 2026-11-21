
import pandas as pd
import joblib
import json

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "saved"

# Load optimized model
model = joblib.load(
    MODEL_PATH / "random_forest_threshold.pkl"
)

# Load scaler
scaler = joblib.load(
    MODEL_PATH / "transaction_scaler.joblib"
)

# Load optimized threshold
with open(MODEL_PATH / "threshold.json") as f:
    threshold = json.load(f)["threshold"]


def predict_transaction(transaction):

    # Convert transaction into DataFrame
    data = pd.DataFrame([transaction])

    # Match training feature order
    feature_columns = list(model.feature_names_in_)
    data = data[feature_columns]

    # Scale Time and Amount
    columns_to_scale = ["Time", "Amount"]

    data[columns_to_scale] = scaler.transform(
        data[columns_to_scale]
    )

    # Predict fraud probability
    probability = model.predict_proba(data)[0][1]

    # Apply optimized threshold
    prediction = int(probability >= threshold)

    # Determine risk level
    if probability >= 0.80:
        risk = "HIGH"
    elif probability >= threshold:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "prediction": prediction,
        "status": "FRAUD" if prediction else "LEGITIMATE",
        "fraud_probability": round(float(probability), 4),
        "risk_level": risk,
        "threshold": threshold
    }


if __name__ == "__main__":
    print("A.R.G.U.S. Fraud Detection Engine")
    print("Model loaded successfully!")
    print("Threshold:", threshold)
