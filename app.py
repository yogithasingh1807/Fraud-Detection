import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model = joblib.load("fraud_detection_model.pkl")

st.set_page_config(layout="wide")
st.title("🔍 Credit Card Fraud Detection App")
st.markdown("This app predicts whether a transaction is **Fraudulent** or **Legitimate** based on input features.")

# Sidebar input
st.sidebar.header("Enter Transaction Manually")
features = ['V'+str(i) for i in range(1, 29)] + ['Amount']
input_data = {f: st.sidebar.number_input(f, value=0.00, format="%.4f") for f in features}
input_df = pd.DataFrame([input_data])

# Single prediction
if st.sidebar.button("🔎 Predict"):
    prediction = model.predict(input_df)[0]
    st.subheader("Entered Transaction Features")
    st.write(input_df)
    st.success("✅ Legitimate Transaction" if prediction == 0 else "❌ Fraudulent Transaction")

# --- CSV Upload Section ---
st.sidebar.header("Or Upload CSV for Batch Prediction")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        # Validate columns
        expected_cols = features
        if all(col in data.columns for col in expected_cols):
            preds = model.predict(data)
            data["Prediction"] = np.where(preds == 0, "Legitimate", "Fraudulent")
            st.subheader("📄 Batch Prediction Results")
            st.write(data.head(10))

            # Download option
            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")
        else:
            st.error("❌ Make sure your CSV has exactly these columns: " + ", ".join(expected_cols))
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")

# --- Evaluation Section ---
st.sidebar.header("Model Evaluation")
if st.sidebar.checkbox("Show Evaluation Metrics"):
    # Load test data
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv").values.ravel()

    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = model.predict(X_test)

    # Confusion matrix
    st.subheader("🧮 Confusion Matrix")
    cm = confusion_matrix(y_test, y_preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ROC Curve
    st.subheader("📈 ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
    ax2.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(loc="lower right")
    st.pyplot(fig2)

    # Precision-Recall Curve
    st.subheader("📊 Precision-Recall Curve")
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    fig3, ax3 = plt.subplots()
    ax3.plot(recall, precision, marker='.')
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    st.pyplot(fig3)