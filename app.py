"""
Real-Time Customer Sentiment Analysis Web App
---------------------------------------------
Install : pip install streamlit scikit-learn pandas
Run     : streamlit run sentiment_app.py

Pipeline: text cleaning -> TF-IDF (1-2 grams) -> Logistic Regression / Naive Bayes
Features: live single-message analysis, simulated real-time stream dashboard,
          batch CSV analysis + download, model evaluation, custom training CSV.
"""
import random
import re
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import Tfidfvectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ----------------------------------------------------------------- DATA
POS = [
    "I love this product, it works perfectly", "Excellent customer service, very helpful staff",
    "Fast delivery and great packaging", "Amazing quality, totally worth the price",
    "The app is easy to use and very smooth", "Highly recommend, best purchase I have made this year",
    "Support resolved my issue in minutes, thank you", "Absolutely fantastic experience from start to finish",
    "Great value for money and works as described", "I am very happy with my order",
    "The staff were friendly and professional", "Superb performance, exceeded my expectations",
    "Really impressed with the new update", "Smooth checkout and quick refund, great job",
    "Wonderful design and the battery lasts all day",
]
NEG = [
    "Terrible service, I waited an hour for a reply", "The product broke after two days",
    "Worst purchase ever, complete waste of money", "Delivery was late and the box was damaged",
    "The app keeps crashing and is very slow", "Customer support was rude and unhelpful",
    "I am very disappointed, it does not work as advertised", "Overpriced and poor quality",
    "They refused to refund me, awful experience", "Billing error again, this is unacceptable",
    "Not happy at all, would not recommend", "The website is confusing and checkout failed twice",
    "Horrible packaging and missing items", "Very frustrating, nobody answers my calls",
    "Cheap material, it started falling apart immediately",
]
NEU = [
    "I received the package today", "The product is available in three colors", "I ordered it on Monday",
    "The store opens at 9 am", "It is an average product, nothing special",
    "I contacted support about my account", "The delivery was scheduled for Friday",
    "The device has a 6 inch screen", "I changed my shipping address yesterday",
    "The item arrived in a box", "Please send me the invoice for order 1234",
    "It works okay, neither good nor bad", "I am checking the warranty details",
    "The package contains a charger and a manual", "I will try it and see",
]
SAMPLE_DF = pd.DataFrame(
    [(t, "Positive") for t in POS] + [(t, "Negative") for t in NEG] + [(t, "Neutral") for t in NEU],
    columns=["text", "label"],
)

# Messages used by the simulated live stream
STREAM = POS[:6] + NEG[:6] + NEU[:4] + [
    "Loving the new feature!", "Why is my order still not here?", "Where can I find my receipt?",
    "Support was quick and polite", "The payment page is broken again",
]
EMOJI = {"Positive": "😊", "Neutral": "😐", "Negative": "😠"}


# ---------------------------------------------------------------- MODEL
def clean(text: str) -> str:
    text = re.sub(r"http\S+|@\w+", " ", str(text).lower())
    return re.sub(r"[^a-z' ]+", " ", text).strip()


@st.cache_resource(show_spinner="Training model...")
def train(df: pd.DataFrame, algo: str):
    X, y = df["text"].map(clean), df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=1000) if algo == "Logistic Regression" else MultinomialNB()
    pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2))), ("clf", clf)])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    labels = list(pipe.classes_)
    report = pd.DataFrame(classification_report(y_te, pred, output_dict=True, zero_division=0)).T
    cm = pd.DataFrame(confusion_matrix(y_te, pred, labels=labels), index=labels, columns=labels)
    acc = accuracy_score(y_te, pred)
    pipe.fit(X, y)  # refit on all data for the deployed model
    return pipe, acc, report, cm


def analyze(pipe, text: str):
    probs = dict(zip(pipe.classes_, pipe.predict_proba([clean(text)])[0]))
    return max(probs, key=probs.get), probs


def score(probs: dict) -> float:
    """Sentiment score from -1 (negative) to +1 (positive)."""
    return probs.get("Positive", 0) - probs.get("Negative", 0)


def log(text, label, probs):
    st.session_state.history.append(
        {"time": datetime.now().strftime("%H:%M:%S"), "text": text,
         "sentiment": label, "score": round(score(probs), 3)}
    )


def dashboard(box):
    h = pd.DataFrame(st.session_state.history)
    with box.container():
        if h.empty:
            st.info("No messages analysed yet.")
            return
        c1, c2, c3 = st.columns(3)
        c1.metric("Messages", len(h))
        c2.metric("Avg sentiment", f"{h['score'].mean():+.2f}")
        c3.metric("Negative share", f"{(h['sentiment'] == 'Negative').mean():.0%}")
        st.caption("Rolling sentiment score (last 5 messages)")
        st.line_chart(h["score"].rolling(5, min_periods=1).mean())
        st.bar_chart(h["sentiment"].value_counts())
        st.dataframe(h.tail(10).iloc[::-1], hide_index=True)


# ------------------------------------------------------------------- UI
st.set_page_config(page_title="Customer Sentiment Analyzer", page_icon="💬", layout="wide")
st.title("💬 Real-Time Customer Sentiment Analysis")
st.session_state.setdefault("history", [])

with st.sidebar:
    st.header("Settings")
    algo = st.selectbox("Model", ["Logistic Regression", "Naive Bayes"])
    upload = st.file_uploader("Custom training data (CSV: text,label)", type="csv")
    if st.button("Clear history"):
        st.session_state.history = []

data = SAMPLE_DF
if upload is not None:
    custom = pd.read_csv(upload)
    if {"text", "label"} <= set(custom.columns) and custom["label"].value_counts().min() >= 4:
        data = custom.dropna(subset=["text", "label"])
        st.sidebar.success(f"Using {len(data)} custom rows")
    else:
        st.sidebar.error("Need 'text' and 'label' columns, 4+ rows per label. Using sample data.")

model, acc, report, cm = train(data, algo)

tab1, tab2, tab3, tab4 = st.tabs(["Analyze", "Live stream", "Batch CSV", "Model performance"])

# ---- Tab 1: single message
with tab1:
    msg = st.text_area("Customer message", placeholder="e.g. The delivery was late and support never replied")
    if st.button("Analyze", type="primary") and msg.strip():
        label, probs = analyze(model, msg)
        log(msg, label, probs)
        st.subheader(f"{EMOJI.get(label, '')} {label}")
        st.progress(float(max(probs.values())), text=f"Confidence: {max(probs.values()):.0%}")
        st.bar_chart(pd.Series(probs, name="probability"))

# ---- Tab 2: simulated real-time stream
with tab2:
    c1, c2 = st.columns(2)
    n = c1.slider("Messages to stream", 5, 50, 15)
    delay = c2.slider("Seconds between messages", 0.2, 3.0, 1.0)
    feed, box = st.empty(), st.empty()
    if st.button("▶ Start stream"):
        for _ in range(n):
            text = random.choice(STREAM)
            label, probs = analyze(model, text)
            log(text, label, probs)
            feed.markdown(f"**Incoming:** {text}  \n→ {EMOJI.get(label, '')} **{label}**")
            dashboard(box)
            time.sleep(delay)
    else:
        dashboard(box)

# ---- Tab 3: batch CSV
with tab3:
    batch = st.file_uploader("Upload reviews CSV", type="csv", key="batch")
    if batch is not None:
        bdf = pd.read_csv(batch)
        col = st.selectbox("Column containing the text", bdf.columns)
        if st.button("Run batch analysis"):
            probs = model.predict_proba(bdf[col].astype(str).map(clean))
            P = pd.DataFrame(probs, columns=model.classes_)
            bdf["sentiment"] = P.idxmax(axis=1)
            bdf["score"] = (P.get("Positive", 0) - P.get("Negative", 0)).round(3)
            st.bar_chart(bdf["sentiment"].value_counts())
            st.dataframe(bdf)
            st.download_button("Download results", bdf.to_csv(index=False), "sentiment_results.csv")

# ---- Tab 4: evaluation
with tab4:
    st.metric("Test accuracy", f"{acc:.0%}")
    st.caption(f"Trained on {len(data)} labelled messages (25% held out for testing).")
    st.subheader("Classification report")
    st.dataframe(report.round(2))
    st.subheader("Confusion matrix (rows = actual, columns = predicted)")
    st.dataframe(cm)
