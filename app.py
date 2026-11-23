
import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import plotly.express as px
from pathlib import Path

# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Cloud Failure Diagnosis",
    page_icon="☁️",
    layout="wide"
)

st.title("☁️ AI-Based Cloud Application Failure Diagnosis")
st.caption("Intelligent cloud log monitoring and failure analysis")

# ----------------------------------
# LOAD TRAINED MODELS
# ----------------------------------

BASE = Path(__file__).parent
ARTIFACTS = BASE / "artifacts"

@st.cache_resource
def load_models():
    classifier = joblib.load(
        ARTIFACTS / "failure_classifier.pkl"
    )

    vectorizer = joblib.load(
        ARTIFACTS / "anomaly_vectorizer.pkl"
    )

    anomaly_model = joblib.load(
        ARTIFACTS / "anomaly_model.pkl"
    )

    guide = joblib.load(
        ARTIFACTS / "diagnosis_guide.pkl"
    )

    return classifier, vectorizer, anomaly_model, guide

try:
    classifier, vectorizer, anomaly_model, guide = load_models()
except Exception as e:
    st.error(f"Could not load trained models: {e}")
    st.stop()

# ----------------------------------
# LOG PROCESSING
# ----------------------------------

def process_logs(messages):

    results = []

    for original in messages:

        original = str(original).strip()

        if not original:
            continue

        severity_match = re.search(
            r"\b(INFO|WARNING|WARN|ERROR|CRITICAL|DEBUG)\b",
            original,
            re.IGNORECASE
        )

        severity = (
            severity_match.group(1).upper()
            if severity_match else "UNKNOWN"
        )

        message = re.sub(
            r"^\s*\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\s*",
            "",
            original
        )

        message = re.sub(
            r"^\s*\d{2}:\d{2}:\d{2}\s*",
            "",
            message
        )

        message = re.sub(
            r"^\s*(INFO|WARNING|WARN|ERROR|CRITICAL|DEBUG)\s*:?\s*",
            "",
            message,
            flags=re.IGNORECASE
        )

        if not message:
            continue

        category = classifier.predict([message])[0]

        confidence = max(
            classifier.predict_proba([message])[0]
        ) * 100

        features = vectorizer.transform([message])

        anomaly = anomaly_model.predict(features)[0]

        details = guide[category]

        results.append({
            "Original Log": original,
            "Severity": severity,
            "Predicted Category": category.title(),
            "Confidence (%)": round(confidence, 2),
            "Anomaly Status": (
                "Anomaly Detected"
                if anomaly == -1 else "Normal Pattern"
            ),
            "Possible Cause": details["cause"],
            "Recommended Action": details["solution"]
        })

    return pd.DataFrame(results)

# ----------------------------------
# SIDEBAR
# ----------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard",
        "Log Analysis",
        "Failure Diagnosis",
        "About Project"
    ]
)

# ----------------------------------
# FILE UPLOAD
# ----------------------------------

uploaded_file = st.sidebar.file_uploader(
    "Upload application logs",
    type=["log", "txt", "csv"]
)

results = pd.DataFrame()

if uploaded_file is not None:

    try:
        if uploaded_file.name.endswith(".csv"):

            raw = pd.read_csv(uploaded_file)

            possible = [
                "message", "log_message", "log", "content"
            ]

            column = next(
                (
                    c for c in raw.columns
                    if c.lower() in possible
                ),
                None
            )

            if column is None:
                st.sidebar.error(
                    "CSV needs a message/log_message/log/content column."
                )
            else:
                messages = raw[column].dropna().astype(str).tolist()

                results = process_logs(messages)

        else:
            content = uploaded_file.read().decode(
                "utf-8", errors="replace"
            )

            messages = content.splitlines()

            results = process_logs(messages)

    except Exception as e:
        st.sidebar.error(f"File processing error: {e}")

# ----------------------------------
# SAMPLE DATA OPTION
# ----------------------------------

if results.empty:

    st.info(
        "Upload a log file using the sidebar, "
        "or load sample data to explore the dashboard."
    )

    if st.button("Load Sample Logs"):

        sample_logs = [
            "2026-11-22 10:00:01 INFO Application started successfully",
            "2026-11-22 10:00:05 INFO Database connection established",
            "2026-11-22 10:00:10 ERROR Database connection timeout",
            "2026-11-22 10:00:15 WARNING High memory usage detected",
            "2026-11-22 10:00:20 CRITICAL Server memory limit exceeded",
            "2026-11-22 10:00:25 ERROR Network connection refused",
            "2026-11-22 10:00:30 ERROR Invalid login credentials",
            "2026-11-22 10:00:35 ERROR HTTP 500 internal server error"
        ]

        results = process_logs(sample_logs)

# ----------------------------------
# DASHBOARD
# ----------------------------------

if page == "Dashboard":

    st.header("Application Monitoring Dashboard")

    if not results.empty:

        total = len(results)

        errors = results["Severity"].isin(
            ["ERROR", "CRITICAL"]
        ).sum()

        anomalies = (
            results["Anomaly Status"] == "Anomaly Detected"
        ).sum()

        categories = results[
            results["Predicted Category"] != "Normal"
        ].shape[0]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Logs", total)
        c2.metric("Error Logs", int(errors))
        c3.metric("Anomalies", int(anomalies))
        c4.metric("Non-Normal Predictions", int(categories))

        st.subheader("Failure Category Distribution")

        category_counts = results[
            "Predicted Category"
        ].value_counts().reset_index()

        category_counts.columns = [
            "Category", "Count"
        ]

        fig = px.bar(
            category_counts,
            x="Category",
            y="Count",
            color="Category",
            title="Detected Failure Categories"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Log Severity Distribution")

        severity_counts = results[
            "Severity"
        ].value_counts().reset_index()

        severity_counts.columns = [
            "Severity", "Count"
        ]

        fig2 = px.pie(
            severity_counts,
            names="Severity",
            values="Count",
            title="Severity Overview"
        )

        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.warning("Upload logs or load sample data first.")

# ----------------------------------
# LOG ANALYSIS
# ----------------------------------

elif page == "Log Analysis":

    st.header("AI Log Analysis")

    if not results.empty:

        st.dataframe(
            results[
                [
                    "Original Log",
                    "Severity",
                    "Predicted Category",
                    "Confidence (%)",
                    "Anomaly Status"
                ]
            ],
            use_container_width=True
        )

    else:
        st.info("No log data available.")

# ----------------------------------
# FAILURE DIAGNOSIS
# ----------------------------------

elif page == "Failure Diagnosis":

    st.header("Failure Diagnosis and Recommendations")

    if not results.empty:

        issues = results[
            results["Predicted Category"] != "Normal"
        ]

        if not issues.empty:

            for _, row in issues.iterrows():

                with st.expander(
                    f"{row['Predicted Category']} - {row['Severity']}"
                ):

                    st.write("**Original Log:**")
                    st.code(row["Original Log"])

                    st.write("**Model Confidence:**")
                    st.write(f"{row['Confidence (%)']}%")

                    st.write("**Anomaly Status:**")
                    st.write(row["Anomaly Status"])

                    st.write("**Possible Cause:**")
                    st.write(row["Possible Cause"])

                    st.success(
                        row["Recommended Action"]
                    )

        else:
            st.success("No known failure categories predicted.")

    else:
        st.info("Upload logs or load sample data first.")

# ----------------------------------
# ABOUT PROJECT
# ----------------------------------

elif page == "About Project":

    st.header("About the Project")

    st.write("""
    This project uses machine learning to analyze
    cloud application logs and assist in identifying
    possible application failures.

    Machine Learning Components:

    1. TF-IDF text feature extraction
    2. Logistic Regression failure classification
    3. Isolation Forest anomaly detection

    The system generates possible causes and
    troubleshooting recommendations for detected issues.
    """)

    st.subheader("Technology Stack")

    st.write("""
    - Python
    - Google Colab
    - Scikit-learn
    - Pandas
    - Streamlit
    - Plotly
    """)

# ----------------------------------
# DOWNLOAD REPORT
# ----------------------------------

if not results.empty:

    st.sidebar.divider()

    csv_data = results.to_csv(
        index=False
    ).encode("utf-8")

    st.sidebar.download_button(
        "Download Analysis Report",
        data=csv_data,
        file_name="cloud_failure_report.csv",
        mime="text/csv"
    )

st.sidebar.caption("AI Cloud Failure Diagnosis System")
