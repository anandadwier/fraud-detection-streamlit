import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ============================================
#   STREAMLIT — TTU 1 VERSION (SESUAI SKRIPSI)
# ============================================
st.set_page_config(page_title="Fraud Detection TTU 1", layout="wide")

# -------- STYLE --------
st.markdown(
    """
    <style>
        .stApp { background-color: #f6f7fa; }
        h1, h2, h3 { color: #0b4a83; }
        .stButton>button { background-color: #0b4a83 !important; color: white !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------- SIDEBAR --------
st.sidebar.title("Navigasi")
menu = st.sidebar.radio(
    "Pilih Halaman",
    ["Upload Dataset", "Preprocessing & Training", "Evaluasi", "Perbandingan Model", "Feature Importance", "Tentang"],
)

# ============================================
#   1. UPLOAD DATASET
# ============================================
if menu == "Upload Dataset":
    st.title("📂 Upload Dataset Fraud")

    data_file = st.file_uploader("Upload file CSV Fraud Kaggle", type=["csv"])

    if data_file:
        df = pd.read_csv(data_file)
        st.session_state["df"] = df

        st.success("Dataset berhasil dimuat!")
        st.dataframe(df.head())

        st.subheader("Statistik Data")
        st.dataframe(df.describe().T)

# ============================================
#   2. PREPROCESSING & TRAINING
# ============================================
if menu == "Preprocessing & Training":
    st.title("🛠 Preprocessing & Training Model")

    if "df" not in st.session_state:
        st.warning("Upload dataset terlebih dahulu.")
        st.stop()

    df = st.session_state["df"].copy()

    st.subheader("Preprocessing otomatis sesuai TTU 1:")
    st.markdown(
        """
        - Menghapus kolom **Time** (tidak digunakan dalam modeling)
        - Normalisasi **Amount** menggunakan Z-Score
        - Target tetap: **Class**
        - Semua fitur numerik digunakan otomatis
        """
    )

    # Drop Time jika ada
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])

    # Normalisasi Amount
    if "Amount" in df.columns:
        scaler_amount = StandardScaler()
        df["Amount"] = scaler_amount.fit_transform(df[["Amount"]])
        st.session_state["scaler_amount"] = scaler_amount

    # Fitur & Target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling seluruh fitur
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test
    st.session_state["scaler_features"] = scaler

    st.success("Preprocessing selesai!")

    st.subheader("Pilih Model untuk Training")
    model_name = st.selectbox("Model", ["Decision Tree", "Random Forest"])

    if st.button("🚀 Train Model"):
        if model_name == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=150, random_state=42)

        model.fit(X_train, y_train)

        st.session_state["model"] = model
        st.success("Model berhasil dilatih!")

# ============================================
#   3. EVALUASI MODEL
# ============================================
if menu == "Evaluasi":
    st.title("📊 Evaluasi Model Fraud Detection")

    if "model" not in st.session_state:
        st.warning("Model belum dilatih.")
        st.stop()

    model = st.session_state["model"]
    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    st.subheader("Akurasi & Classification Report")
    st.write("Accuracy:", accuracy_score(y_test, preds))
    st.text(classification_report(y_test, preds))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.subheader("ROC Curve & AUC Score")
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax2.plot([0, 1], [0, 1], "r--")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend()

    st.pyplot(fig2)

# ============================================
#   4. PERBANDINGAN MODEL
# ============================================
if menu == "Perbandingan Model":
    st.title("📊 Perbandingan Decision Tree vs Random Forest")

    if "df" not in st.session_state:
        st.warning("Upload dataset terlebih dahulu.")
        st.stop()

    df = st.session_state["df"].copy()

    # Preprocessing minimal
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])
    if "Amount" in df.columns:
        scaler_amount = StandardScaler()
        df["Amount"] = scaler_amount.fit_transform(df[["Amount"]])

    X = df.drop(columns=["Class"]).select_dtypes(include=[np.number])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train kedua model
    dt = DecisionTreeClassifier(random_state=42)
    rf = RandomForestClassifier(n_estimators=150, random_state=42)

    dt.fit(X_train_s, y_train)
    rf.fit(X_train_s, y_train)

    dt_pred = dt.predict(X_test_s)
    rf_pred = rf.predict(X_test_s)

    dt_prob = dt.predict_proba(X_test_s)[:,1]
    rf_prob = rf.predict_proba(X_test_s)[:,1]

    # Tabel perbandingan
    comparison = pd.DataFrame({
        "Model": ["Decision Tree", "Random Forest"],
        "Accuracy": [accuracy_score(y_test, dt_pred), accuracy_score(y_test, rf_pred)],
        "Precision": [classification_report(y_test, dt_pred, output_dict=True)["1"]["precision"],
                       classification_report(y_test, rf_pred, output_dict=True)["1"]["precision"]],
        "Recall": [classification_report(y_test, dt_pred, output_dict=True)["1"]["recall"],
                    classification_report(y_test, rf_pred, output_dict=True)["1"]["recall"]],
        "F1-Score": [classification_report(y_test, dt_pred, output_dict=True)["1"]["f1-score"],
                      classification_report(y_test, rf_pred, output_dict=True)["1"]["f1-score"]],
        "ROC-AUC": [roc_auc_score(y_test, dt_prob), roc_auc_score(y_test, rf_prob)]
    })

    st.subheader("📌 Hasil Perbandingan Model")
    st.dataframe(comparison)

    st.subheader("📈 Kesimpulan Analisis")
    st.markdown("""
    **Random Forest** hampir selalu lebih unggul karena:
    - Menggunakan ratusan pohon (bagging) → lebih stabil
    - Risiko overfitting lebih kecil daripada Decision Tree
    - Bias lebih rendah & akurasi umumnya lebih tinggi

    **Decision Tree** tetap berguna bila:
    - Butuh model cepat dan mudah dipahami
    - Ingin interpretasi sederhana
    """)

# ============================================
#   5. FEATURE IMPORTANCE
# ============================================
if menu == "Feature Importance":
    st.title("🌲 Feature Importance (RF / DT)")

    if "model" not in st.session_state:
        st.warning("Model belum dilatih.")
        st.stop()

    model = st.session_state["model"]
    df = st.session_state["df"].copy()

    X_cols = df.drop(columns=["Class"]).columns

    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        fi_df = pd.DataFrame({"Feature": X_cols, "Importance": fi}).sort_values(
            "Importance", ascending=False
        )

        st.dataframe(fi_df.head(20))

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=fi_df.head(15), x="Importance", y="Feature", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Model tidak mendukung feature importance.")

# ============================================
#   6. TENTANG
# ============================================
if menu == "Tentang":
    st.title("Tentang Aplikasi")
    st.markdown(
        """
        Aplikasi ini dibuat **khusus untuk skripsi TTU 1** menggunakan dataset Credit Card Fraud Kaggle.

        Fitur utama:
        - Upload dataset
        - Preprocessing otomatis sesuai metodologi TTU 1
        - Training Decision Tree & Random Forest
        - Evaluasi lengkap (Accuracy, Report, Confusion Matrix, ROC AUC)
        - Feature Importance
        """
    )
