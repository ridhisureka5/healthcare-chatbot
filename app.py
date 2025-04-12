import streamlit as st
import pandas as pd
from pandasai.llm.local_llm import LocalLLM
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pandasai import SmartDataframe

st.set_page_config(page_title="Smart AutoML Analyst (Python 3.10)", layout="wide")
st.title("🤖 Smart Analyst (Python 3.10 Compatible)")

# Load LLM
try:
    model = LocalLLM(api_base="http://localhost:11434/v1", model="llama3")
    st.success("🔋 LLM Model loaded.")
except Exception as e:
    st.error(f"Failed to load LLM: {e}")
    st.stop()

# Upload
uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Preview")
    st.write(data.head())

    # Initialize SmartDataframe
    try:
        df_smart = SmartDataframe(data, config={"llm": model})
        st.success("🧠 SmartDataframe ready")
    except Exception as e:
        st.error(f"SmartDataframe init failed: {e}")

    st.markdown("---")

    # LLM Chat
    st.subheader("💬 Ask Your Data (LLM)")
    prompt = st.text_area("Ask a question about your data:")
    if st.button("Generate"):
        with st.spinner("Generating..."):
            try:
                response = df_smart.chat(prompt)
                st.write(response)
            except Exception as e:
                st.error(f"Error: {e}")

    # EDA Summary (Custom)
    st.markdown("---")
    st.subheader("📊 Basic Data Summary")
    if st.button("Show Summary"):
        st.write("🧮 Shape:", data.shape)
        st.write("🧠 Columns & Types:")
        st.write(data.dtypes)
        st.write("🔍 Missing Values:")
        st.write(data.isnull().sum())
        st.write("📈 Describe Numeric Columns:")
        st.write(data.describe())

    # AutoML via LazyPredict
    st.markdown("---")
    st.subheader("🤖 AutoML (LazyPredict)")
    target = st.selectbox("Select target column:", options=data.columns)
    if st.button("Run AutoML"):
        try:
            X = data.drop(columns=[target])
            y = data[target]
            X = pd.get_dummies(X)  # Handle categoricals
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            clf = LazyClassifier(verbose=0, ignore_warnings=True)
            models, predictions = clf.fit(X_train, X_test, y_train, y_test)

            st.write("📋 Model Comparison:")
            st.dataframe(models)
        except Exception as e:
            st.error(f"AutoML failed: {e}")

    # Anomaly Detection
    st.markdown("---")
    st.subheader("🚨 Anomaly Detection")
    if st.button("Detect Anomalies"):
        try:
            numeric = data.select_dtypes(include='number').dropna()
            iso = IsolationForest(contamination=0.05)
            data["Anomaly"] = iso.fit_predict(numeric)
            outliers = data[data["Anomaly"] == -1]
            st.warning(f"{len(outliers)} anomalies found.")
            st.dataframe(outliers)
        except Exception as e:
            st.error(f"Anomaly detection error: {e}")

    # PCA Visualization
    st.markdown("---")
    st.subheader("📉 PCA Visualization")
    numeric_data = data.select_dtypes(include='number').dropna()
    if not numeric_data.empty:
        try:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(numeric_data)
            fig, ax = plt.subplots()
            ax.scatter(pca_result[:, 0], pca_result[:, 1])
            ax.set_title("PCA Scatter Plot")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"PCA error: {e}")
    else:
        st.info("No numeric data available for PCA.")