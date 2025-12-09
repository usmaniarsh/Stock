import streamlit as st
import pandas as pd
import requests
import time
import plotly.express as px
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ----------------------------
# Load Sentiment Model
# ----------------------------
@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "finiteautomata/bertweet-base-sentiment-analysis",
        use_fast=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "finiteautomata/bertweet-base-sentiment-analysis"
    )
    return tokenizer, model


# ----------------------------
# Sentiment Analysis Function
# ----------------------------
def analyze_sentiment(text, tokenizer, model):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)

        negative, neutral, positive = scores[0].tolist()
        compound = positive - negative
        return compound


# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="Social Media Sentiment Analyzer",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------
# Bootstrap UI
# ----------------------------
def set_bootstrap():
    st.markdown(
        """
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------
# Twitter API Key
# ----------------------------
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAPLN0QEAAAAALMI1KHrXVIn9h%2FmUDV5fexcvr90%3DdwZGbwuA47X0QxCTaVE9By2YxYEtD8zD5DNTAzPkLTQBFfeoQa"


# ----------------------------
# Fetch Tweets Function
# ----------------------------
def fetch_tweets(word_query, number_of_tweets=10):
    tokenizer, model = load_sentiment_model()

    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    params = {
        "query": word_query,
        "max_results": min(max(number_of_tweets, 10), 100),
        "tweet.fields": "created_at,text",
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        st.error(f"Error fetching tweets: {response.status_code}")
        return pd.DataFrame()

    data = response.json()
    tweets = data.get("data", [])

    tweets_df = pd.DataFrame(tweets)

    if tweets_df.empty:
        return tweets_df

    tweets_df["created_at"] = pd.to_datetime(tweets_df["created_at"])
    tweets_df["sentiment_score"] = tweets_df["text"].apply(
        lambda x: analyze_sentiment(x, tokenizer, model)
    )

    return tweets_df


# ----------------------------
# MAIN STREAMLIT APP
# ----------------------------
def app():
    set_bootstrap()
    st.sidebar.header("📊 Analytics Dashboard")

    menu_options = ["Overview", "Statistics", "Sentiment Analysis"]
    choice = st.sidebar.radio("Select a section:", menu_options)

    # ----------------------------
    # OVERVIEW PAGE
    # ----------------------------
    if choice == "Overview":
        st.markdown("<h2 class='text-center'>📈 Social Media Analytics Overview</h2>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tweets Analyzed", "2,450", "+8.5%")
        col2.metric("Positive Sentiments", "1,220", "+12.3%")
        col3.metric("Negative Sentiments", "650", "-5.2%")
        col4.metric("Neutral Sentiments", "580", "+3.1%")

        st.markdown("---")

        data = pd.DataFrame({
            "Days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "Positive": [500, 600, 700, 550, 750, 620, 500],
            "Negative": [200, 150, 180, 170, 140, 160, 190],
            "Neutral": [300, 320, 310, 290, 305, 310, 315],
        })

        fig = px.line(
            data, x="Days", y=["Positive", "Negative", "Neutral"],
            title="Sentiment Trends Over the Week",
            labels={"value": "Tweet Count", "Days": "Day of the Week"},
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # STATISTICS PAGE
    # ----------------------------
    elif choice == "Statistics":
        st.markdown("<h2 class='text-center'>📊 Sentiment Statistics</h2>", unsafe_allow_html=True)

        stats_data = pd.DataFrame({
            "Category": ["Positive Tweets", "Negative Tweets", "Neutral Tweets"],
            "Count": [120, 45, 60],
        })

        st.write("### Tweet Sentiment Breakdown")
        st.table(stats_data)

        fig_stats = px.bar(stats_data, x="Category", y="Count", title="Sentiment Analysis Statistics")
        st.plotly_chart(fig_stats, use_container_width=True)

    # ----------------------------
    # SENTIMENT ANALYSIS PAGE
    # ----------------------------
    elif choice == "Sentiment Analysis":
        word_query = st.text_input("🔍 Enter a hashtag or keyword:", placeholder="#example")
        number_of_tweets = st.slider("Number of Tweets to Analyze:", min_value=10, max_value=100, step=10)

        if st.button("Analyze Sentiment"):
            if not word_query:
                st.warning("Please enter a hashtag or keyword.")
                return

            with st.spinner("Fetching tweets and analyzing sentiment..."):
                data = fetch_tweets(word_query, number_of_tweets)

                if data.empty:
                    st.warning("No tweets found.")
                    return

                st.subheader("📄 Extracted Dataset")
                st.dataframe(data)

                positive = len(data[data["sentiment_score"] > 0.3])
                neutral = len(data[(data["sentiment_score"] >= -0.3) & (data["sentiment_score"] <= 0.3)])
                negative = len(data[data["sentiment_score"] < -0.3])

                sentiment_df = pd.DataFrame({
                    "Sentiment": ["Positive", "Neutral", "Negative"],
                    "Count": [positive, neutral, negative]
                })

                fig_sentiment = px.bar(sentiment_df, x="Sentiment", y="Count", title="Sentiment Summary")
                st.plotly_chart(fig_sentiment, use_container_width=True)

                fig_hist = px.histogram(data, x="sentiment_score", nbins=10, title="Sentiment Score Distribution")
                st.plotly_chart(fig_hist, use_container_width=True)


# ----------------------------
# RUN APP
# ----------------------------
if __name__ == "__main__":
    app()
