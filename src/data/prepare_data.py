
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "saved"

# Create output directories
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)


def prepare_data():

    print("\n========== A.R.G.U.S. DATA PREPROCESSING ==========\n")

    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    print("Original Dataset:", df.shape)

    # Separate features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining Samples:", len(X_train))
    print("Testing Samples:", len(X_test))

    # Scale Time and Amount only
    scaler = StandardScaler()

    columns_to_scale = ["Time", "Amount"]

    X_train[columns_to_scale] = scaler.fit_transform(
        X_train[columns_to_scale]
    )

    X_test[columns_to_scale] = scaler.transform(
        X_test[columns_to_scale]
    )

    # Save scaler
    joblib.dump(
        scaler,
        MODEL_PATH / "transaction_scaler.joblib"
    )

    # Combine features and target
    train_data = X_train.copy()
    train_data["Class"] = y_train

    test_data = X_test.copy()
    test_data["Class"] = y_test

    # Save processed datasets
    train_data.to_csv(
        PROCESSED_PATH / "train.csv",
        index=False
    )

    test_data.to_csv(
        PROCESSED_PATH / "test.csv",
        index=False
    )

    # Display class distribution
    print("\nTraining Class Distribution:")
    print(y_train.value_counts())

    print("\nTesting Class Distribution:")
    print(y_test.value_counts())

    print("\nSaved Files:")
    print("- train.csv")
    print("- test.csv")
    print("- transaction_scaler.joblib")

    print("\n========== PREPROCESSING COMPLETE ==========\n")


if __name__ == "__main__":
    prepare_data()
