# scripts/summarize_feedback.py
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk

nltk.download('punkt')

# Load model once (lazy init)
_summarizer = None

def summarize_transformer(text, mode="short"):
    """Use pretrained T5 or BART summarizer."""
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model="t5-small")
    max_len = 60 if mode == "short" else 150
    summary = _summarizer(text, max_length=max_len, min_length=15, do_sample=False)
    return summary[0]['summary_text']


def extractive_summary(text, top_k=3):
    """Simple extractive summary via TF-IDF + cosine similarity."""
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
    if len(sentences) <= top_k:
        return text
    vectorizer = TfidfVectorizer().fit_transform(sentences)
    sim_matrix = cosine_similarity(vectorizer)
    scores = sim_matrix.sum(axis=1)
    ranked_sentences = [sentences[i] for i in np.argsort(scores)[::-1]]
    summary = " ".join(ranked_sentences[:top_k])
    return summary


if __name__ == "__main__":
    sample_text = (
        "The product works great, but the packaging was damaged. "
        "Customer service helped quickly and replaced the item. "
        "Overall, I’m satisfied but shipping could be faster."
    )
    print("--- Short Summary ---")
    print(summarize_transformer(sample_text, "short"))
    print("--- Extractive Summary ---")
    print(extractive_summary(sample_text))
