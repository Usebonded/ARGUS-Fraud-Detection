import streamlit as st
import pandas as pd
from pathlib import Path

from src.engine.fraud_detector import predict_transaction


# ==========================================
# A.R.G.U.S. - FINANCIAL FRAUD DETECTION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"

MODEL_THRESHOLD = 0.4083
MAX_TRANSACTIONS = 1000


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="A.R.G.U.S. | Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)


# ==========================================
# CUSTOM STYLING
# ==========================================

st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #58A6FF;
    }

    .subtitle {
        font-size: 18px;
        color: #AAB4C3;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(80, 120, 180, 0.08);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(120, 150, 190, 0.2);
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# SESSION STATE
# ==========================================

if "last_results" not in st.session_state:
    st.session_state.last_results = pd.DataFrame()

if "transaction_data" not in st.session_state:
    st.session_state.transaction_data = None

if "analyzed_transactions" not in st.session_state:
    st.session_state.analyzed_transactions = pd.DataFrame()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🛡️ A.R.G.U.S.")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Fraud Detection",
        "Model Information"
    ]
)

st.sidebar.divider()
st.sidebar.success("Detection Engine Connected")
st.sidebar.caption(
    f"Random Forest | Threshold: {MODEL_THRESHOLD:.4f}"
)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🛡️ A.R.G.U.S.</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Financial Fraud Detection System'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# OVERVIEW - FINANCIAL INTELLIGENCE DASHBOARD
# ==========================================

if page == "Overview":

    st.subheader("Financial Intelligence Dashboard")

    results = st.session_state.last_results

    if results.empty:

        st.info(
            "A.R.G.U.S. is operational. "
            "Analyze transactions to generate financial insights."
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Transactions Analyzed", 0)
        col2.metric("Fraud Detected", 0)
        col3.metric("Legitimate", 0)
        col4.metric("Fraud Rate", "0%")

    else:

        # Normalize status values for consistent filtering.
        status = results["Status"].astype(str).str.upper()

        total = len(results)
        fraud = int((status == "FRAUD").sum())
        legitimate = total - fraud
        fraud_rate = (fraud / total) * 100

        avg_probability = (
            pd.to_numeric(
                results["Fraud Probability"],
                errors="coerce"
            ).mean() * 100
        )

        # ======================================
        # KEY PERFORMANCE INDICATORS
        # ======================================

        st.markdown("### System Performance")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Transactions Analyzed",
            f"{total:,}"
        )

        col2.metric(
            "Fraud Detected",
            f"{fraud:,}"
        )

        col3.metric(
            "Legitimate",
            f"{legitimate:,}"
        )

        col4.metric(
            "Fraud Rate",
            f"{fraud_rate:.2f}%"
        )

        st.divider()

        # ======================================
        # FRAUD INTELLIGENCE
        # ======================================

        st.markdown("### Fraud Intelligence")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Average Fraud Probability",
                f"{avg_probability:.4f}%"
            )

        with col2:

            if "Amount" in results.columns:

                fraud_amount = pd.to_numeric(
                    results.loc[
                        status == "FRAUD",
                        "Amount"
                    ],
                    errors="coerce"
                ).sum()

                st.metric(
                    "Total Flagged Amount",
                    f"${fraud_amount:,.2f}"
                )

            else:

                st.metric(
                    "Total Flagged Amount",
                    "Not available"
                )

        st.divider()

        # ======================================
        # TRANSACTION DISTRIBUTION
        # ======================================

        st.markdown("### Transaction Analytics")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("**Fraud vs Legitimate**")

            counts = status.value_counts().reindex(
                ["FRAUD", "LEGITIMATE"],
                fill_value=0
            )

            st.bar_chart(counts)

        with col2:

            st.markdown("**Risk Level Distribution**")

            if "Risk Level" in results.columns:

                risk_status = (
                    results["Risk Level"]
                    .astype(str)
                    .str.upper()
                )

                risk_counts = risk_status.value_counts().reindex(
                    ["HIGH", "MEDIUM", "LOW"],
                    fill_value=0
                )

                st.bar_chart(risk_counts)

            else:

                st.info("Risk-level data is not available.")

        st.divider()

        # ======================================
        # FRAUD INVESTIGATION
        # ======================================

        st.markdown("### Fraud Investigation")

        fraud_cases = results.loc[status == "FRAUD"].copy()

        if not fraud_cases.empty:

            fraud_cases["Fraud Probability"] = pd.to_numeric(
                fraud_cases["Fraud Probability"],
                errors="coerce"
            )

            fraud_cases = fraud_cases.sort_values(
                by="Fraud Probability",
                ascending=False
            )

            st.error(
                f"{len(fraud_cases)} fraudulent transactions identified."
            )

            st.dataframe(
                fraud_cases,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                label="Download Fraud Investigation Report",
                data=fraud_cases.to_csv(index=False),
                file_name="argus_fraud_investigation.csv",
                mime="text/csv"
            )

        else:

            st.success(
                "No fraudulent transactions identified."
            )

        st.divider()

        # ======================================
        # HIGH-RISK TRANSACTIONS
        # ======================================

        st.markdown("### High-Risk Transactions")

        if "Risk Level" in results.columns:

            risk_status = (
                results["Risk Level"]
                .astype(str)
                .str.upper()
            )

            high_risk = results.loc[
                risk_status == "HIGH"
            ].copy()

            if not high_risk.empty:

                high_risk = high_risk.sort_values(
                    by="Fraud Probability",
                    ascending=False
                )

                st.warning(
                    f"{len(high_risk)} high-risk transactions identified."
                )

                st.dataframe(
                    high_risk,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.success("No high-risk transactions identified.")

        else:

            st.info("Risk-level data is not available.")

        st.divider()

        # ======================================
        # RECENT TRANSACTION ACTIVITY
        # ======================================

        st.markdown("### Recent Transaction Activity")

        st.dataframe(
            results.tail(10),
            use_container_width=True,
            hide_index=True
        )


# ==========================================
# FRAUD DETECTION
# ==========================================

elif page == "Fraud Detection":

    st.subheader("Transaction Analysis")

    st.write(
        "Upload transaction data or run a sample "
        "transaction through the A.R.G.U.S. detection engine."
    )

    uploaded_file = st.file_uploader(
        "Upload Transaction CSV",
        type=["csv"]
    )

    col1, col2 = st.columns(2)

    with col1:

        run_sample = st.button(
            "Test Sample Transaction",
            type="primary"
        )

    with col2:

        clear_data = st.button("Clear Transaction")

    # ======================================
    # CLEAR TRANSACTION DATA
    # ======================================

    if clear_data:

        st.session_state.transaction_data = None
        st.session_state.last_results = pd.DataFrame()
        st.session_state.analyzed_transactions = pd.DataFrame()
        st.rerun()

    # ======================================
    # LOAD UPLOADED CSV
    # ======================================

    if uploaded_file is not None:

        try:

            st.session_state.transaction_data = pd.read_csv(
                uploaded_file
            ).drop(
                columns=["Class"],
                errors="ignore"
            )

        except Exception as e:

            st.error(f"Unable to read CSV: {e}")

    # ======================================
    # LOAD SAMPLE TRANSACTIONS
    # ======================================

    elif run_sample:

        if DATA_PATH.exists():

            try:

                st.session_state.transaction_data = pd.read_csv(
                    DATA_PATH,
                    nrows=MAX_TRANSACTIONS
                ).drop(
                    columns=["Class"],
                    errors="ignore"
                )

                st.success(
                    "Sample transactions loaded successfully!"
                )

            except Exception as e:

                st.error(f"Unable to load sample data: {e}")

        else:

            st.error("Original dataset not found.")

    # ======================================
    # RETRIEVE PERSISTENT TRANSACTION DATA
    # ======================================

    transaction_data = st.session_state.transaction_data

    if transaction_data is not None:

        if transaction_data.empty:

            st.warning("No transactions found.")

        elif len(transaction_data) > MAX_TRANSACTIONS:

            st.warning(
                f"Please upload a maximum of {MAX_TRANSACTIONS} transactions."
            )

        else:

            st.write(
                f"Transactions loaded: {len(transaction_data):,}"
            )

            st.dataframe(
                transaction_data.head(),
                use_container_width=True
            )

            # ==================================
            # ANALYZE TRANSACTIONS
            # ==================================

            if st.button(
                "Analyze Transactions",
                type="primary"
            ):

                results = []

                with st.spinner("Analyzing transactions..."):

                    try:

                        for _, row in transaction_data.iterrows():

                            result = predict_transaction(
                                row.to_dict()
                            )

                            results.append({
                                "Status": result["status"],
                                "Amount": float(row.get("Amount", 0)),
                                "Fraud Probability":
                                    result["fraud_probability"],
                                "Risk Level":
                                    result["risk_level"],
                                "Threshold":
                                    result["threshold"]
                            })

                        result_df = pd.DataFrame(results)

                        # Keep the original feature rows aligned with predictions.
                        # SHAP needs these raw transaction features to explain a result.
                        st.session_state.last_results = result_df
                        st.session_state.analyzed_transactions = (
                            transaction_data.reset_index(drop=True).copy()
                        )

                        result_status = (
                            result_df["Status"]
                            .astype(str)
                            .str.upper()
                        )

                        fraud_count = int(
                            (result_status == "FRAUD").sum()
                        )

                        legitimate_count = (
                            len(result_df) - fraud_count
                        )

                        # ==================================
                        # ANALYSIS SUMMARY
                        # ==================================

                        st.success(
                            "Analysis completed successfully!"
                        )

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Total Analyzed",
                            len(result_df)
                        )

                        col2.metric(
                            "Fraud Detected",
                            fraud_count
                        )

                        col3.metric(
                            "Legitimate",
                            legitimate_count
                        )

                        # ==================================
                        # FRAUD INVESTIGATION RESULTS
                        # ==================================

                        st.subheader("Fraud Investigation")

                        fraud_cases = result_df.loc[
                            result_status == "FRAUD"
                        ].copy()

                        if not fraud_cases.empty:

                            fraud_cases = fraud_cases.sort_values(
                                by="Fraud Probability",
                                ascending=False
                            )

                            st.error(
                                f"{len(fraud_cases)} fraud cases detected."
                            )

                            st.dataframe(
                                fraud_cases,
                                use_container_width=True,
                                hide_index=True
                            )

                            st.download_button(
                                "Download Fraud Investigation Report",
                                data=fraud_cases.to_csv(index=False),
                                file_name="argus_fraud_investigation.csv",
                                mime="text/csv"
                            )

                        else:

                            st.success(
                                "No fraudulent transactions detected."
                            )

                        # ==================================
                        # COMPLETE DETECTION RESULTS
                        # ==================================

                        st.subheader("Detection Results")

                        st.dataframe(
                            result_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        st.download_button(
                            "Download Results",
                            data=result_df.to_csv(index=False),
                            file_name="argus_detection_results.csv",
                            mime="text/csv"
                        )

                    except Exception as e:

                        st.error(
                            f"Transaction analysis failed: {e}"
                        )

            # ==================================
            # EXPLAINABLE AI - SHAP ANALYSIS
            # ==================================

            analyzed_transactions = st.session_state.analyzed_transactions
            saved_results = st.session_state.last_results

            if (
                not analyzed_transactions.empty
                and not saved_results.empty
                and len(analyzed_transactions) == len(saved_results)
            ):

                st.divider()
                st.subheader("Transaction Explainability")
                st.write(
                    "Understand which transaction features influenced "
                    "the Random Forest model's fraud prediction."
                )

                def transaction_label(index):
                    row = saved_results.iloc[index]
                    return (
                        f"Transaction {index + 1} | "
                        f"{row['Status']} | "
                        f"Fraud probability: {row['Fraud Probability']:.4f}"
                    )

                selected_index = st.selectbox(
                    "Select a transaction to explain",
                    options=list(range(len(saved_results))),
                    format_func=transaction_label,
                    key="shap_transaction_selector"
                )

                selected_result = saved_results.iloc[selected_index]

                metric1, metric2, metric3 = st.columns(3)
                metric1.metric("Prediction", selected_result["Status"])
                metric2.metric(
                    "Fraud Probability",
                    f"{selected_result['Fraud Probability'] * 100:.2f}%"
                )
                metric3.metric("Risk Level", selected_result["Risk Level"])

                if st.button(
                    "Generate SHAP Explanation",
                    type="primary",
                    key="generate_shap_explanation"
                ):

                    with st.spinner("Generating feature contributions..."):

                        try:
                            # Import only when requested so the main dashboard
                            # remains usable if the optional SHAP package is absent.
                            from src.engine.explainability import explain_transaction

                            original_transaction = (
                                analyzed_transactions.iloc[selected_index].to_dict()
                            )

                            explanation = explain_transaction(original_transaction)

                            if explanation.empty:
                                st.warning(
                                    "No feature contributions were returned."
                                )
                            else:
                                st.success("SHAP explanation generated successfully!")

                                top_features = explanation.head(10).copy()

                                st.markdown("#### Top 10 Feature Contributions")

                                # Positive SHAP values push toward the fraud class;
                                # negative values push away from the fraud class.
                                try:
                                    import matplotlib.pyplot as plt

                                    figure, axis = plt.subplots(figsize=(10, 6))
                                    bar_colors = [
                                        "#F85149" if value > 0 else "#58A6FF"
                                        for value in top_features["SHAP Value"]
                                    ]

                                    axis.barh(
                                        top_features["Feature"],
                                        top_features["SHAP Value"],
                                        color=bar_colors
                                    )
                                    axis.axvline(0, color="white", linewidth=0.8)
                                    axis.set_xlabel("SHAP Value")
                                    axis.set_ylabel("Transaction Features")
                                    axis.set_title(
                                        "A.R.G.U.S. - SHAP Feature Contributions"
                                    )
                                    axis.invert_yaxis()
                                    figure.tight_layout()
                                    st.pyplot(figure)
                                    plt.close(figure)

                                except ImportError:
                                    st.warning(
                                        "Matplotlib is not installed. "
                                        "Showing the feature contribution table instead."
                                    )

                                st.dataframe(
                                    top_features,
                                    use_container_width=True,
                                    hide_index=True
                                )

                                st.caption(
                                    "Red bars indicate positive contributions toward "
                                    "the fraud class; blue bars indicate negative "
                                    "contributions. SHAP values explain this individual "
                                    "prediction and do not imply that a feature always "
                                    "has the same effect on every transaction."
                                )
                                # ==================================
                                # DYNAMIC SHAP INTERPRETATION
                                # ==================================

                                positive_features = (
                                    top_features[
                                        top_features["SHAP Value"] > 0
                                    ]
                                    .sort_values(
                                        "SHAP Value",
                                        ascending=False
                                    )
                                    .head(3)
                                )

                                negative_features = (
                                    top_features[
                                        top_features["SHAP Value"] < 0
                                    ]
                                    .sort_values(
                                        "SHAP Value",
                                        ascending=True
                                    )
                                    .head(3)
                                )

                                st.markdown("#### A.R.G.U.S. AI Analysis")

                                st.write(
                                    f"**Final Prediction:** "
                                    f"{selected_result['Status']}"
                                )

                                st.write(
                                    f"**Fraud Probability:** "
                                    f"{selected_result['Fraud Probability'] * 100:.2f}%"
                                )

                                if not positive_features.empty:
                                    positive_names = ", ".join(
                                        positive_features["Feature"].tolist()
                                    )
                                    st.error(
                                        f"Features pushing towards fraud: "
                                        f"{positive_names}"
                                    )
                                else:
                                    st.success(
                                        "No positive fraud contributions "
                                        "were found among the top 10 features."
                                    )

                                if not negative_features.empty:
                                    negative_names = ", ".join(
                                        negative_features["Feature"].tolist()
                                    )
                                    st.info(
                                        f"Features pushing away from fraud: "
                                        f"{negative_names}"
                                    )

                                st.caption(
                                    "This interpretation is generated from "
                                    "the selected transaction's SHAP values. "
                                    "It describes feature contributions, "
                                    "not proof of fraudulent activity."
                                )

                        except Exception as e:
                            st.error(f"Unable to generate SHAP explanation: {e}")

            elif not saved_results.empty:
                st.warning(
                    "The original transaction features are unavailable for the "
                    "saved predictions. Analyze the transactions again to enable SHAP."
                )


# ==========================================
# MODEL INFORMATION
# ==========================================

elif page == "Model Information":

    st.subheader("Machine Learning Model")

    st.write(
        "A.R.G.U.S. currently uses an optimized "
        "Random Forest classifier."
    )

    col1, col2 = st.columns(2)

    col1.metric("Model", "Random Forest")

    col2.metric(
        "Classification Threshold",
        f"{MODEL_THRESHOLD:.4f}"
    )

    st.divider()

    # ======================================
    # MODEL PERFORMANCE
    # ======================================

    st.subheader("Model Performance")

    performance = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "PR-AUC",
            "ROC-AUC"
        ],
        "Score": [
            0.9996,
            0.9213,
            0.8367,
            0.8770,
            0.8408,
            0.9620
        ]
    })

    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Performance values are from the earlier model comparison "
        "and threshold optimization test runs. "
        "The dashboard displays them as recorded results."
    )

    st.info(
        "The model analyzes 30 transaction features: "
        "Time, V1-V28 and Amount."
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "A.R.G.U.S. | Intelligent Financial Fraud Detection System"
)
