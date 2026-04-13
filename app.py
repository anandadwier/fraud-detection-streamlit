import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

# ===========================================
# KONFIGURASI STREAMLIT
# ===========================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

st.title("💳 Deteksi Penipuan Kartu Kredit Berbasis Machine Learning & Streamlit")


# ===========================================
# SIDEBAR MENU
# ===========================================
menu = st.sidebar.selectbox(
    "Navigasi",
    ["Home", "Exploratory Data Analysis (EDA)", "Training Model", "Model Comparison", "Predict New Transaction"]
)

uploaded = st.sidebar.file_uploader("Upload Dataset Kaggle (creditcard.csv)", type=["csv"])


# ===========================================
# LOAD DATASET
# ===========================================
@st.cache_data
def load_data(file):
    return pd.read_csv(file, nrows=50000)  # ambil sebagian data saja

if uploaded is not None:
    df = load_data(uploaded)
else:
    st.warning("Upload dataset terlebih dahulu.")
    st.stop()


# ===========================================
# HOME
# ===========================================
if menu == "Home":
    st.subheader("📌 Deskripsi Aplikasi")
    st.write("""
    Aplikasi ini merupakan implementasi akhir dari deteksi penipuan kartu kredit menggunakan algoritma 
    **Decision Tree** dan **Random Forest**, dilengkapi visualisasi interaktif Streamlit.

    Fitur Utama:
    - Upload dataset Kaggle
    - Visualisasi EDA (Histogram, Correlation Heatmap)
    - Training model ML (DT & RF)
    - Visualisasi Confusion Matrix & ROC Curve
    - Perbandingan model lengkap (Accuracy, Precision, Recall, F1, ROC-AUC)
    - Prediksi transaksi baru secara realtime
    """)

    st.info("Silakan upload dataset melalui sidebar untuk mulai menggunakan aplikasi.")


# ===========================================
# EDA
# ===========================================
elif menu == "Exploratory Data Analysis (EDA)":
    if df is None:
        st.warning("Upload dataset terlebih dahulu.")
    else:
        st.subheader("📊 Exploratory Data Analysis (EDA)")
        st.write("Dataset 5 baris pertama:")
        st.dataframe(df.head())

        # Drop Time jika ada
        if "Time" in df.columns:
            df = df.drop(columns=["Time"])

        # Histogram Amount
        st.subheader("📈 Distribusi Amount")
        fig, ax = plt.subplots()
        sns.histplot(df["Amount"], kde=True, ax=ax)
        st.pyplot(fig)

        # Correlation heatmap
        st.subheader("🔍 Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)


# ===========================================
# TRAINING MODEL
# ===========================================
elif menu == "Training Model":
    if df is None:
        st.warning("Upload dataset terlebih dahulu.")
    else:
        st.subheader("🧠 Training Model Machine Learning")

        # Preprocessing
        df = df.drop_duplicates()
        if "Time" in df.columns:
            df = df.drop(columns=["Time"])

        df["Amount"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()

        X = df.drop(columns=["Class"])
        y = df["Class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        model_choice = st.selectbox("Pilih Algoritma", ["Decision Tree", "Random Forest"])

        if model_choice == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        st.success("Model berhasil dilatih!")

        # Classification report
        st.subheader("📄 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report))

        # Confusion Matrix
        st.subheader("📉 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        st.pyplot(fig_cm)

        # ROC Curve
        st.subheader("📈 ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend()
        st.pyplot(fig_roc)

        # Feature Importance
        st.subheader("🔍 Feature Importance")
        importance = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        st.bar_chart(importance.set_index("feature"))


# ===========================================
# MODEL COMPARISON (Fitur Jawaban Rumusan Masalah 2)
# ===========================================
elif menu == "Model Comparison":
    if df is None:
        st.warning("Upload dataset terlebih dahulu.")
    else:
        st.subheader("⚖️ Perbandingan Kinerja Decision Tree vs Random Forest")

        # Preprocessing
        df = df.drop_duplicates()
        if "Time" in df.columns:
            df = df.drop(columns=["Time"])

        df["Amount"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()

        X = df.drop(columns=["Class"])
        y = df["Class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Train kedua model
        dt = DecisionTreeClassifier(random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)

        dt.fit(X_train, y_train)
        rf.fit(X_train, y_train)

        dt_pred = dt.predict(X_test)
        rf_pred = rf.predict(X_test)

        dt_prob = dt.predict_proba(X_test)[:, 1]
        rf_prob = rf.predict_proba(X_test)[:, 1]

        # ================================
        # METRICS PERBANDINGAN
        # ================================
        results = pd.DataFrame({
            "Model": ["Decision Tree", "Random Forest"],
            "Accuracy": [
                accuracy_score(y_test, dt_pred),
                accuracy_score(y_test, rf_pred)
            ],
            "Precision": [
                precision_score(y_test, dt_pred),
                precision_score(y_test, rf_pred)
            ],
            "Recall": [
                recall_score(y_test, dt_pred),
                recall_score(y_test, rf_pred)
            ],
            "F1-score": [
                f1_score(y_test, dt_pred),
                f1_score(y_test, rf_pred)
            ],
            "ROC-AUC": [
                roc_auc_score(y_test, dt_prob),
                roc_auc_score(y_test, rf_prob)
            ]
        })

        st.subheader("📄 Tabel Perbandingan Model")
        st.dataframe(results)

        best_model_name = results.loc[results["ROC-AUC"].idxmax(), "Model"]
        st.success(f"🏆 **Model Terbaik Berdasarkan Evaluasi adalah: {best_model_name}**")

        st.subheader("📊 Visualisasi Perbandingan")
        st.bar_chart(results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]])

        # ROC Curve comparison
        st.subheader("📈 ROC Curve Comparison")
        fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_prob)
        fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)

        fig, ax = plt.subplots()
        ax.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC = {roc_auc_score(y_test, dt_prob):.4f})")
        ax.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_score(y_test, rf_prob):.4f})")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)


# ===========================================
# PREDIKSI TRANSAKSI BARU
# ===========================================
elif menu == "Predict New Transaction":
    if df is None:
        st.warning("Upload dataset terlebih dahulu.")
    else:
        st.subheader("🟦 Prediksi Transaksi Baru")

        df = df.drop_duplicates()
        if "Time" in df.columns:
            df = df.drop(columns=["Time"])

        df["Amount"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()

        X = df.drop(columns=["Class"])
        y = df["Class"]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        st.info("Masukkan nilai fitur transaksi untuk diprediksi:")

        input_data = {}
        for col in X.columns:
            input_data[col] = st.number_input(col, value=float(df[col].mean()))

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        if st.button("Prediksi"):
            if prediction == 1:
                st.error("⚠️ Transaksi Ini Diduga Penipuan (FRAUD)")
            else:
                st.success("✔️ Transaksi Aman / Normal")
