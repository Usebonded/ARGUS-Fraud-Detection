
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset path
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Configure visualization
sns.set_theme(style="whitegrid")

# Count transactions
class_counts = df["Class"].value_counts().sort_index()

# Create figure
plt.figure(figsize=(8, 5))

ax = sns.countplot(
    data=df,
    x="Class",
    hue="Class",
    palette=["#2196F3", "#E53935"],
    legend=False
)

# Add transaction counts
for container in ax.containers:
    ax.bar_label(container, fmt="%.0f", padding=3)

plt.title("Fraud vs Legitimate Transactions")
plt.xlabel("Transaction Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")

plt.tight_layout()

# Save visualization
output_path = PROJECT_ROOT / "reports" / "class_distribution.png"
plt.savefig(output_path, dpi=300)

plt.show()

print("Visualization saved successfully!")
print("Output:", output_path)


# Transaction Amount Distribution

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="Amount",
    bins=100,
    color="royalblue"
)

plt.title("Distribution of Transaction Amounts")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.tight_layout()

output_path = PROJECT_ROOT / "reports" / "amount_distribution.png"
plt.savefig(output_path, dpi=300)

plt.show()

print("Amount distribution saved successfully!")


# Fraud Transaction Amount Distribution

fraud_df = df[df["Class"] == 1].copy()

# Log transformation for better visualization
fraud_df["Log_Amount"] = np.log1p(fraud_df["Amount"])

plt.figure(figsize=(10, 5))

sns.histplot(
    data=fraud_df,
    x="Log_Amount",
    bins=40,
    color="crimson"
)

plt.title("Fraud Transaction Amount Distribution")
plt.xlabel("Log Transformed Transaction Amount")
plt.ylabel("Number of Fraud Transactions")

plt.tight_layout()

output_path = PROJECT_ROOT / "reports" / "fraud_amount_distribution.png"
plt.savefig(output_path, dpi=300)

plt.show()

print("Fraud amount distribution saved successfully!")


# Correlation Heatmap

correlation_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(18, 14))

sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0,
    linewidths=0.2,
    cbar=True
)

plt.title("A.R.G.U.S. - Feature Correlation Heatmap")
plt.tight_layout()

output_path = PROJECT_ROOT / "reports" / "correlation_heatmap.png"

plt.savefig(output_path, dpi=300)

plt.show()

# Display strongest correlations with fraud
print("\nTop Features Correlated with Fraud:")

fraud_correlation = (
    correlation_matrix["Class"]
    .drop("Class")
    .sort_values(key=abs, ascending=False)
)

print(fraud_correlation.head(10))

print("\nCorrelation heatmap saved successfully!")
