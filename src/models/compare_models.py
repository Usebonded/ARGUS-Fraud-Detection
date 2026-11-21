
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "saved"
REPORT_PATH = PROJECT_ROOT / "reports"

REPORT_PATH.mkdir(parents=True, exist_ok=True)

# Load test data
test_df = pd.read_csv(TEST_PATH)

X_test = test_df.drop(columns=["Class"])
y_test = test_df["Class"]

# Available models
models = {
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib"
}

results = []

print("\n========== A.R.G.U.S. MODEL COMPARISON ==========\n")

for name, filename in models.items():

    print(f"Evaluating {name}...")

    model = joblib.load(MODEL_PATH / filename)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "PR-AUC": average_precision_score(y_test, y_prob),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

# Create comparison table
comparison = pd.DataFrame(results)

print("\nMODEL PERFORMANCE COMPARISON:\n")
print(comparison.round(4).to_string(index=False))

# Save results
output_path = REPORT_PATH / "model_comparison.csv"

comparison.to_csv(output_path, index=False)

print(f"\nComparison saved to: {output_path}")

print("\n========== COMPARISON COMPLETE ==========\n")
