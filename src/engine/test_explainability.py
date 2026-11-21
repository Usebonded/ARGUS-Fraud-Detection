
import pandas as pd

from src.engine.explainability import explain_transaction

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
)

# Load one transaction
df = pd.read_csv(DATA_PATH, nrows=1)

# Remove actual fraud label
transaction = df.drop(
    columns=["Class"],
    errors="ignore"
).iloc[0].to_dict()

# Generate explanation
explanation = explain_transaction(transaction)

print("\n========== A.R.G.U.S. SHAP ANALYSIS ==========\n")

print("Top 10 Important Features:\n")

print(explanation.head(10).to_string(index=False))

print("\n========== EXPLANATION COMPLETE ==========")

import matplotlib.pyplot as plt

# Select top 10 features
top_features = explanation.head(10).copy()

# Reverse order for horizontal chart
top_features = top_features.iloc[::-1]

# Create visualization
plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["SHAP Value"]
)

plt.axvline(x=0, linestyle="--", linewidth=1)

plt.title("A.R.G.U.S. - SHAP Feature Contributions")
plt.xlabel("SHAP Value")
plt.ylabel("Transaction Features")

plt.tight_layout()

# Save chart
PROJECT_ROOT.joinpath("reports").mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    PROJECT_ROOT / "reports" / "shap_contributions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("SHAP visualization saved successfully!")
