
import pandas as pd

from fraud_detector import predict_transaction
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"

print("\n========== A.R.G.U.S. ENGINE TEST ==========\n")

# Load one original transaction
df = pd.read_csv(DATA_PATH, nrows=1)

# Remove actual class label
transaction = df.drop(columns=["Class"]).iloc[0].to_dict()

# Run prediction
result = predict_transaction(transaction)

print("TRANSACTION ANALYSIS")
print("-------------------------")

for key, value in result.items():
    print(f"{key}: {value}")

print("\n========== TEST COMPLETE ==========")
