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

# ===========================================
# PILIH SUMBER DATA
# ===========================================
data_option = st.sidebar.radio(
    "Pilih Sumber Data",
    ["Upload File (Max 200MB)", "Gunakan Link Google Drive"]
)

# ===========================================
# LOAD DATASET
# ===========================================
@st.cache_data
def load_uploaded(file):
    return pd.read_csv(file)

@st.cache_data
def load_from_drive():
    url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
    return pd.read_csv(url)

df = None

if data_option == "Upload File (Max 200MB)":
    uploaded = st.sidebar.file_uploader("Upload Dataset CSV", type=["csv"])

    if uploaded is not None:
        try:
            df = load_uploaded(uploaded)
            st.sidebar.success("Dataset dari upload berhasil dimuat ✅")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

elif data_option == "Gunakan Link Google Drive":
    try:
        df = load_from_drive()
        st.sidebar.success("Dataset dari Google Drive berhasil dimuat ✅")
    except Exception as e:
        st.sidebar.error(f"Gagal load data: {e}")


# ===========================================
# HOME
# ===========================================
if menu == "Home":
    st.subheader("📌 Deskripsi Aplikasi")
    st.write("""
    Aplikasi ini merupakan implementasi akhir dari deteksi penipuan kartu kredit menggunakan algoritma 
    **Decision Tree** dan **Random Forest**, dilengkapi visualisasi interaktif Streamlit.

    Fitur Utama:
    - Upload dataset (≤ 200MB)
    - Load dataset dari Google Drive (full data)
    - Visualisasi EDA
    - Training model ML
    - Evaluasi model
    - Prediksi transaksi
    """)


# ===========================================
# EDA
# ===========================================
elif menu == "Exploratory Data Analysis (EDA)":
    if df is None:
        st.warning("Dataset belum tersedia.")
    else:
        st.subheader("📊 Exploratory Data Analysis (EDA)")
        st.dataframe(df.head())

        if "Time" in df.columns:
            df = df.drop(columns=["Time"])

        st.subheader("📈 Distribusi Amount")
        fig, ax = plt.subplots()
        sns.histplot(df["Amount"], kde=True, ax=ax)
        st.pyplot(fig)

        st.subheader("🔍 Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)


# ===========================================
# TRAINING MODEL
# ===========================================
elif menu == "Training Model":
    if df is None:
        st.warning("Dataset belum tersedia.")
    else:
        st.subheader("🧠 Training Model Machine Learning")

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

        model = DecisionTreeClassifier(random_state=42) if model_choice == "Decision Tree" else RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        st.success("Model berhasil dilatih!")

        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        st.pyplot(fig)


# ===========================================
# MODEL COMPARISON
# ===========================================
elif menu == "Model Comparison":
    if df is None:
        st.warning("Dataset belum tersedia.")
    else:
        st.subheader("⚖️ Perbandingan Model")

        df = df.drop_duplicates()
        if "Time" in df.columns:
            df = df.drop(columns=["Time"])

        df["Amount"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()

        X = df.drop(columns=["Class"])
        y = df["Class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        dt = DecisionTreeClassifier(random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)

        dt.fit(X_train, y_train)
        rf.fit(X_train, y_train)

        results = pd.DataFrame({
            "Model": ["Decision Tree", "Random Forest"],
            "Accuracy": [
                accuracy_score(y_test, dt.predict(X_test)),
                accuracy_score(y_test, rf.predict(X_test))
            ],
            "ROC-AUC": [
                roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1]),
                roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
            ]
        })

        st.dataframe(results)
        st.bar_chart(results.set_index("Model"))


# ===========================================
# PREDICT
# ===========================================
elif menu == "Predict New Transaction":
    if df is None:
        st.warning("Dataset belum tersedia.")
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

        input_data = {}
        for col in X.columns:
            input_data[col] = st.number_input(col, value=float(df[col].mean()))

        input_df = pd.DataFrame([input_data])

        if st.button("Prediksi"):
            prediction = model.predict(input_df)[0]
            st.error("⚠️ FRAUD") if prediction == 1 else st.success("✔️ NORMAL")
