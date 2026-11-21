
import pandas as pd
from pathlib import Path

# Locate the project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset location
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"


def inspect_dataset():

    print("\n========== A.R.G.U.S. DATA INSPECTION ==========\n")

    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    # Dataset dimensions
    print("Dataset Shape:", df.shape)

    # Column information
    print("\nDataset Columns:")
    print(df.columns.tolist())

    # First five records
    print("\nFirst Five Transactions:")
    print(df.head())

    # Missing values
    print("\nMissing Values:")
    print(df.isnull().sum().sum())

    # Fraud distribution
    print("\nTransaction Classification:")
    print(df["Class"].value_counts())

    # Fraud percentage
    fraud_percentage = (df["Class"].mean()) * 100

    print(f"\nFraud Percentage: {fraud_percentage:.4f}%")

    print("\n========== INSPECTION COMPLETE ==========\n")


if __name__ == "__main__":
    inspect_dataset()
