import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon for sentiment analysis
@st.cache_resource
def download_vader():
    nltk.download('vader_lexicon')

download_vader()

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Streamlit UI design
st.title("🛍️ Customer Sentiment Analysis App")
st.write("Type a product review or customer feedback below to analyze its sentiment in real-time!")

# Text input box for user
user_input = st.text_area("Enter customer review here:", "I absolutely love this product! It works wonderfully and exceeded my expectations.")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Get sentiment scores
        scores = sia.polarity_scores(user_input)
        compound = scores['compound']
        
        # Classify based on compound score
        if compound >= 0.05:
            sentiment = "Positive 😊"
            st.success(f"Sentiment: **{sentiment}**")
        elif compound <= -0.05:
            sentiment = "Negative 😞"
            st.error(f"Sentiment: **{sentiment}**")
        else:
            sentiment = "Neutral 😐"
            st.info(f"Sentiment: **{sentiment}**")
            
        # Display breakdown of scores
        with st.expander("See detailed scores"):
            st.write(f"Compound Score: {compound}")
            st.write(f"Positive Score: {scores['pos']}")
            st.write(f"Neutral Score: {scores['neu']}")
            st.write(f"Negative Score: {scores['neg']}")
