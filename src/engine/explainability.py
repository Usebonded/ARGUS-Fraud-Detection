
import pandas as pd
import numpy as np
import shap

from src.engine.fraud_detector import model, scaler


# ==========================================
# A.R.G.U.S. - EXPLAINABLE AI ENGINE
# ==========================================

# Initialize SHAP Tree Explainer
explainer = shap.TreeExplainer(model)


def explain_transaction(transaction):

    # Convert transaction into DataFrame
    data = pd.DataFrame([transaction])

    # Match model feature order
    feature_columns = list(model.feature_names_in_)

    missing_columns = [
        col for col in feature_columns if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing transaction features: {missing_columns}"
        )

    data = data[feature_columns].copy()

    # Apply the same preprocessing as fraud_detector.py
    columns_to_scale = ["Time", "Amount"]

    data[columns_to_scale] = scaler.transform(
        data[columns_to_scale]
    )

    # Calculate SHAP values
    shap_values = explainer.shap_values(data)

    # Identify fraud class
    fraud_class_index = list(model.classes_).index(1)

    # Extract fraud contributions
    if isinstance(shap_values, list):

        values = np.asarray(
            shap_values[fraud_class_index]
        )[0]

    else:

        shap_values = np.asarray(shap_values)

        if shap_values.ndim == 3:

            values = shap_values[
                0, :, fraud_class_index
            ]

        elif shap_values.ndim == 2:

            values = shap_values[0]

        else:
            raise ValueError(
                f"Unexpected SHAP output shape: {shap_values.shape}"
            )

    # Create feature contribution table
    explanation = pd.DataFrame({
        "Feature": feature_columns,
        "SHAP Value": values,
        "Impact": np.abs(values)
    })

    # Sort by strongest contribution
    explanation = explanation.sort_values(
        by="Impact",
        ascending=False
    )

    return explanation.reset_index(drop=True)


# ==========================================
# TEST ENGINE
# ==========================================

if __name__ == "__main__":

    print("A.R.G.U.S. Explainable AI Engine")
    print("SHAP initialized successfully!")
