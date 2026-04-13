import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

# ===========================================
# KONFIGURASI
# ===========================================
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Deteksi Penipuan Kartu Kredit Berbasis Machine Learning & Streamlit")

# ===========================================
# SIDEBAR
# ===========================================
menu = st.sidebar.selectbox(
    "Navigasi",
    ["Home", "EDA", "Training Model", "Model Comparison", "Predict"]
)

data_source = st.sidebar.radio("Sumber Data", ["Upload", "Link"])
rows = st.sidebar.slider("Jumlah data yang dibaca", 1000, 100000, 50000)

# ===========================================
# LOAD DATA (OPTIMIZED)
# ===========================================
@st.cache_data
def load_data(source, rows):
    return pd.read_csv(source, nrows=rows)

df = None

if data_source == "Upload":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = load_data(uploaded, rows)

else:
    url = st.sidebar.text_input("Masukkan URL CSV")
    if url:
        df = load_data(url, rows)

if df is None:
    st.warning("Silakan upload atau masukkan link dataset.")
    st.stop()

# ===========================================
# HOME
# ===========================================
if menu == "Home":
    st.subheader("📌 Deskripsi Aplikasi")
    st.write("Aplikasi deteksi fraud menggunakan Decision Tree & Random Forest.")

# ===========================================
# EDA
# ===========================================
elif menu == "EDA":
    st.subheader("📊 Exploratory Data Analysis")
    st.dataframe(df.head())

    if "Time" in df.columns:
        df = df.drop(columns=["Time"])

    fig, ax = plt.subplots()
    sns.histplot(df["Amount"], kde=True, ax=ax)
    st.pyplot(fig)

# ===========================================
# TRAINING
# ===========================================
elif menu == "Training Model":
    df = df.drop_duplicates()

    if "Time" in df.columns:
        df = df.drop(columns=["Time"])

    df["Amount"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model_choice = st.selectbox("Model", ["Decision Tree", "Random Forest"])

    if model_choice == "Decision Tree":
        model = DecisionTreeClassifier()
    else:
        model = RandomForestClassifier(n_estimators=50)

    with st.spinner("Training model..."):
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    st.success("Model selesai dilatih")

    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    st.pyplot(fig)

# ===========================================
# COMPARISON
# ===========================================
elif menu == "Model Comparison":
    df = df.drop_duplicates()

    if "Time" in df.columns:
        df = df.drop(columns=["Time"])

    df["Amount"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier(n_estimators=50)

    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    dt_pred = dt.predict(X_test)
    rf_pred = rf.predict(X_test)

    results = pd.DataFrame({
        "Model": ["Decision Tree", "Random Forest"],
        "Accuracy": [
            accuracy_score(y_test, dt_pred),
            accuracy_score(y_test, rf_pred)
        ]
    })

    st.dataframe(results)
    st.bar_chart(results.set_index("Model"))

# ===========================================
# PREDICT
# ===========================================
elif menu == "Predict":
    df = df.drop_duplicates()

    if "Time" in df.columns:
        df = df.drop(columns=["Time"])

    df["Amount"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()

    X = df.drop(columns=["Class"])
    y = df["Class"]

    model = RandomForestClassifier(n_estimators=50)
    model.fit(X, y)

    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(col, value=float(df[col].mean()))

    input_df = pd.DataFrame([input_data])

    if st.button("Prediksi"):
        pred = model.predict(input_df)[0]
        if pred == 1:
            st.error("⚠️ Fraud")
        else:
            st.success("✔️ Normal")
