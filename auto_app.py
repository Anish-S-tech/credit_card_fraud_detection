import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

st.title("Credit Card Fraud Detection using Autoencoder")

# Constants: Path to saved models
SCALER_FILE = "models/scaler.pkl"
AUTOENCODER_MODEL_FILE = "models/autoencoder_model.h5"

# Load scaler and autoencoder model
scaler = joblib.load(SCALER_FILE)

autoencoder = keras.models.load_model(
    AUTOENCODER_MODEL_FILE,
    custom_objects={"mse": keras.losses.MeanSquaredError()}
)

# Fixed reconstruction error threshold from training
THRESHOLD = 0.588152202950144

# Helper function to detect fraud
def detect_fraud(data):
    data = data.fillna(0)
    data[["Time", "Amount"]] = scaler.transform(data[["Time", "Amount"]])

    reconstructions = autoencoder.predict(data)
    mse = np.mean(np.power(data - reconstructions, 2), axis=1)

    predictions = (mse > THRESHOLD).astype(int)
    return mse, predictions

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file (must contain V1-V28, Time, Amount)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(df.head())

    if st.button("Detect Fraud"):
        try:
            mse, predictions = detect_fraud(df)

            df["Reconstruction Error"] = mse
            df["Fraud Prediction"] = predictions

            st.write("Prediction Results:")
            st.dataframe(df)

            fraud_count = predictions.sum()
            st.success(f"Fraudulent transactions detected: {fraud_count}")

            # If ground truth 'Class' column is present
            if "Class" in df.columns:
                cm = confusion_matrix(df["Class"], predictions)
                st.write("### Confusion Matrix")
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                st.pyplot()

                st.write("\n### Classification Report:")
                st.text(classification_report(df["Class"], predictions, digits=4))

                roc_auc = roc_auc_score(df["Class"], mse)
                st.write(f"ROC-AUC (using MSE): {roc_auc:.4f}")

            # Download predictions
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error during prediction: {e}")

else:
    st.info("Please upload a CSV file to begin fraud detection.")
