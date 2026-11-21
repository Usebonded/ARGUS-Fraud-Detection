
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)
import matplotlib.pyplot as plt

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "saved"
REPORT_PATH = PROJECT_ROOT / "reports"

MODEL_PATH.mkdir(parents=True, exist_ok=True)
REPORT_PATH.mkdir(parents=True, exist_ok=True)


print("\n========== A.R.G.U.S. THRESHOLD OPTIMIZATION ==========\n")

# Load datasets
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X = train_df.drop(columns=["Class"])
y = train_df["Class"]

X_test = test_df.drop(columns=["Class"])
y_test = test_df["Class"]

# Create training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

# Initialize Random Forest
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

# Train model
print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# Validation probabilities
val_prob = model.predict_proba(X_val)[:, 1]

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(
    y_val, val_prob
)

# Calculate F1 score for each threshold
f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-10)
)

# Select best threshold
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

print("\nOPTIMAL THRESHOLD RESULTS")
print("-------------------------")
print(f"Best Threshold: {best_threshold:.4f}")
print(f"Validation Precision: {precision[best_index]:.4f}")
print(f"Validation Recall: {recall[best_index]:.4f}")
print(f"Validation F1 Score: {f1_scores[best_index]:.4f}")

# Final evaluation on untouched test set
test_prob = model.predict_proba(X_test)[:, 1]

y_pred = (test_prob >= best_threshold).astype(int)

print("\nFINAL TEST PERFORMANCE")
print("-------------------------")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save optimized model
joblib.dump(
    model,
    MODEL_PATH / "random_forest_threshold.pkl"
)

# Save threshold results
results = pd.DataFrame({
    "Threshold": thresholds,
    "Precision": precision[:-1],
    "Recall": recall[:-1],
    "F1 Score": f1_scores
})

results.to_csv(
    REPORT_PATH / "threshold_results.csv",
    index=False
)

# Plot F1 against threshold
plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_scores, label="F1 Score")
plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Best Threshold: {best_threshold:.4f}"
)

plt.xlabel("Classification Threshold")
plt.ylabel("F1 Score")
plt.title("A.R.G.U.S. - Threshold Optimization")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    REPORT_PATH / "threshold_optimization.png",
    dpi=300
)

plt.close()

print("\nOptimized model saved successfully!")
print("Threshold report saved successfully!")
print("\n========== OPTIMIZATION COMPLETE ==========")

import json

with open(MODEL_PATH / "threshold.json", "w") as f:
    json.dump(
        {"threshold": float(best_threshold)},
        f,
        indent=4
    )

print("Threshold configuration saved!")