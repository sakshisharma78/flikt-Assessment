# app/streamlit_app.py
import streamlit as st
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from scripts.summarize_feedback import summarize_transformer, extractive_summary
from scripts.insights import recurring_issues, forecast_satisfaction
import matplotlib.pyplot as plt
import os

# ---------- App Configuration ----------
st.set_page_config(
    page_title="AI Customer Feedback Analysis",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Intelligent Customer Feedback Analysis System")
st.markdown("Analyze, summarize, and visualize customer sentiment using AI.")

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    model_path = "models/sentiment_model"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

def predict_sentiment(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.nn.functional.softmax(outputs.logits, dim=1)
        labels = preds.argmax(axis=1).numpy()
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return [label_map[l] for l in labels]

# ---------- Upload Section ----------
st.sidebar.header("📤 Upload Feedback Data")
uploaded = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success(f"✅ Uploaded: {uploaded.name} ({len(df)} records)")

    if 'text_clean' not in df.columns:
        st.warning("⚠️ No preprocessed text found — using 'text' column directly.")
        df['text_clean'] = df['text']

    # ---------- Run Sentiment Analysis ----------
    with st.spinner("Analyzing Sentiment..."):
        df['sentiment'] = predict_sentiment(df['text_clean'].tolist())

    st.subheader("📈 Sentiment Distribution")
    sentiment_counts = df['sentiment'].value_counts()
    st.bar_chart(sentiment_counts)

    # ---------- Display Sample Analysis ----------
    st.subheader("🧠 Sample Summaries")
    sample_text = st.selectbox("Select a feedback to summarize:", df['text_clean'].head(10))
    if st.button("Generate Summaries"):
        with st.spinner("Generating summaries..."):
            st.write("**Short Summary:**", summarize_transformer(sample_text, "short"))
            st.write("**Detailed Summary:**", extractive_summary(sample_text))

    # ---------- Recurring Issues ----------
    st.subheader("🔍 Recurring Issues / Complaint Themes")
    issues = recurring_issues(df, n_clusters=4)
    st.dataframe(issues)

    # ---------- Forecast Trends ----------
    st.subheader("📅 Customer Satisfaction Forecast")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    forecast = forecast_satisfaction(df)
    st.image("AI_insights_report.png")

    # ---------- Download Processed Data ----------
    st.subheader("📥 Download Results")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Analyzed Data as CSV",
        data=csv,
        file_name='analyzed_feedback.csv',
        mime='text/csv',
    )
else:
    st.info("👆 Upload a CSV file to get started.")
