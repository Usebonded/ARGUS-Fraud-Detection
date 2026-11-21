
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "saved"
REPORT_PATH = PROJECT_ROOT / "reports"

REPORT_PATH.mkdir(parents=True, exist_ok=True)

# Load feature names
df = pd.read_csv(DATA_PATH)
features = df.drop(columns=["Class"]).columns

# Models
models = {
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib"
}

for name, filename in models.items():

    print(f"\nAnalyzing {name}...")

    model = joblib.load(MODEL_PATH / filename)

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop 15 Important Features:")
    print(importance_df.head(15).to_string(index=False))

    # Plot
    top_features = importance_df.head(15).sort_values(
        by="Importance"
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.title(f"A.R.G.U.S. - {name} Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Transaction Features")
    plt.tight_layout()

    output = REPORT_PATH / f"{name.lower().replace(' ', '_')}_importance.png"

    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Chart saved: {output}")

print("\nFEATURE IMPORTANCE ANALYSIS COMPLETE")
