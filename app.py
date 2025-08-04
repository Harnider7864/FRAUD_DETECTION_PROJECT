import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load trained models
model = joblib.load('models/isolation_forest_model.pkl')
pca = joblib.load('models/pca_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("🔍 Credit Card Fraud Detection Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your credit card transaction CSV file", type=["csv"], key="file_uploader")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Required columns
    required_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
                        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 
                        'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 
                        'V27', 'V28', 'Amount']

    if all(col in data.columns for col in required_columns):
        # Feature selection for scaler and PCA (exclude 'Time' if model wasn't trained with it)

       # Feature selection for scaler and PCA — KEEP 'Time' since the model expects it
        feature_columns = [col for col in required_columns]  # Do not remove 'Time'

        # Preprocess
        X = data[feature_columns]
        X_scaled = scaler.transform(X)  # Ensure same order and columns as used during fit
        X_pca = pca.transform(X_scaled)

        # Predict
        preds = model.predict(X_pca)
        data['Prediction'] = ['Fraudulent' if x == -1 else 'Not Fraudulent' for x in preds]
 
       
    
        # Show filter option
        if st.checkbox("🔍 Show only fraudulent transactions"):
            data = data[data['Prediction'] == 'Fraudulent']

        st.success("✅ Analysis complete. Below is the prediction summary:")

        # Display the first few rows
        st.dataframe(data.head(100), use_container_width=True)

        # Download button
        st.download_button("📥 Download Results CSV", data.to_csv(index=False).encode('utf-8'), "predictions.csv", "text/csv")

        # Summary stats
        fraud_count = data['Prediction'].value_counts().get('Fraudulent', 0)
        total = len(data)
        fraud_pct = (fraud_count / total) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", f"{total:,}")
        col2.metric("Fraudulent Transactions", f"{fraud_count:,}")
        col3.metric("Fraud Rate", f"{fraud_pct:.2f}%")

        st.markdown("---")

        # Pie Chart
        fig1 = px.pie(
            data_frame=data,
            names='Prediction',
            title='📊 Distribution of Fraudulent vs Non-Fraudulent Transactions',
            color_discrete_map={'Fraudulent': 'red', 'Not Fraudulent': 'green'}
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Histogram of Amount
        st.subheader("💰 Transaction Amount Distribution")
        fig2 = px.histogram(
            data, x='Amount', color='Prediction',
            nbins=50,
            title='Transaction Amount by Fraud Status',
            color_discrete_map={'Fraudulent': 'red', 'Not Fraudulent': 'green'},
            marginal="box"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Line Plot over Time
        st.subheader("⏱️ Fraudulent Transactions Over Time")
        if 'Time' in data.columns:
            time_df = data[data['Prediction'] == 'Fraudulent'][['Time']]
            if not time_df.empty:
                fig3 = px.histogram(time_df, x='Time', nbins=50, title="Fraudulent Transaction Count Over Time")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("No fraudulent transactions found for time-series plot.")

        # Optional: PCA Scatter Plot
        st.subheader("🧠 PCA Visualization")
        try:
            if X_pca.shape[1] >= 2:
                pca_df = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
                pca_df['Prediction'] = data['Prediction']
                fig4 = px.scatter(
                    pca_df, x='PC1', y='PC2', color='Prediction',
                    title="PCA Component Space",
                    color_discrete_map={'Fraudulent': 'red', 'Not Fraudulent': 'green'}
                )
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.warning("PCA doesn't have at least 2 components to plot.")
        except Exception as e:
            st.error(f"Error generating PCA plot: {e}")

        
        # Evaluation metrics if ground truth available
        y_true = None
        if 'Class' in data.columns:
            # Map 1 to 'Fraudulent', 0 to 'Not Fraudulent'
            y_true = data['Class'].map({1: 'Fraudulent', 0: 'Not Fraudulent'})
        if y_true is not None:
            st.subheader("📊 Evaluation Metrics")
            st.text("Classification Report:")
            from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
            st.text(classification_report(y_true, data['Prediction']))
            st.text("Confusion Matrix:")
            st.write(pd.DataFrame(confusion_matrix(y_true, data['Prediction']),
                                  index=['Actual Not Fraudulent', 'Actual Fraudulent'],
                                  columns=['Predicted Not Fraudulent', 'Predicted Fraudulent']))
            try:
                auc_score = roc_auc_score((y_true == 'Fraudulent').astype(int), (data['Prediction'] == 'Fraudulent').astype(int))
                st.metric("ROC AUC Score", f"{auc_score:.4f}")
            except Exception as e:
                st.warning(f"Could not compute ROC AUC Score: {e}")


        # Explanation
        with st.expander("ℹ️ How are transactions marked as fraud?"):
            st.markdown("""
        The model uses **Isolation Forest**, an unsupervised anomaly detection algorithm.

        - It isolates observations by randomly selecting features and splitting values.
        - Anomalies require fewer splits and are thus easier to isolate.
        - Based on this logic, transactions far from the normal pattern are marked as **fraudulent**.

        ⚙️ Preprocessing includes:
        - **Standard Scaling** of features
        - **PCA** (Principal Component Analysis) to reduce dimensionality

        **Note**: The model was trained on imbalanced real-world data using **SMOTE** to balance it.
        """)

    else:
        st.error("❌ Uploaded CSV does not contain all required columns.")