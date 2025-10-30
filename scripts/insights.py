# scripts/insights.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from prophet import Prophet
import matplotlib.pyplot as plt
import os


def recurring_issues(df, n_clusters=5):
    """Cluster feedbacks to identify common complaint themes."""
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df['text_clean'])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(X)
    df['cluster'] = km.labels_
    top_terms = []
    for i in range(n_clusters):
        terms = vectorizer.get_feature_names_out()
        centroid = km.cluster_centers_[i]
        top_idx = centroid.argsort()[-8:][::-1]
        top_words = [terms[j] for j in top_idx]
        top_terms.append({"Cluster": i, "Top_Words": top_words})
    return pd.DataFrame(top_terms)


def forecast_satisfaction(df, output_path="AI_insights_report.pdf"):
    """Predict customer satisfaction trend using Prophet."""
    # Convert sentiment to numeric: pos=1, neu=0, neg=-1
    mapping = {"Positive": 1, "Neutral": 0, "Negative": -1}
    if 'sentiment' not in df.columns:
        df['sentiment'] = df['pred_label'].map({2: "Positive", 1: "Neutral", 0: "Negative"})
    df['score'] = df['sentiment'].map(mapping)
    daily = df.groupby('date')['score'].mean().reset_index()
    daily.columns = ['ds', 'y']

    m = Prophet()
    m.fit(daily)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)

    plt.figure(figsize=(10, 5))
    plt.plot(daily['ds'], daily['y'], label='Actual', linewidth=2)
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', linestyle='--')
    plt.title('Customer Satisfaction Trend Forecast')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path.replace('.pdf', '.png'))
    print(f"Forecast plot saved to {output_path.replace('.pdf', '.png')}")
    return forecast


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_feedback.csv")
    print("Recurring issues:")
    print(recurring_issues(df))
    print("Generating forecast...")
    forecast_satisfaction(df)
