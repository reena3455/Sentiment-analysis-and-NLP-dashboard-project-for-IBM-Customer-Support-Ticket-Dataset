# =====================================================
# NLP CUSTOMER SUPPORT DASHBOARD (PRODUCTION READY)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import nltk
from nltk.corpus import stopwords

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="NLP Support Dashboard", layout="wide")

st.title("📊 NLP-Powered Customer Support Insights Dashboard")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/customer_support_tickets.csv")
    return df

df = load_data()

# -------------------------------
# TEXT CLEANING
# -------------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['clean_text'] = df['Ticket Description'].apply(clean_text)

# -------------------------------
# SENTIMENT (RULE-BASED SIMPLE)
# -------------------------------
def sentiment_label(text):
    if any(word in text for word in ['error','fail','issue','problem','not working']):
        return "Negative"
    elif any(word in text for word in ['thanks','good','resolved']):
        return "Positive"
    else:
        return "Neutral"

df['sentiment'] = df['clean_text'].apply(sentiment_label)

# -------------------------------
# URGENCY DETECTION
# -------------------------------
urgent_keywords = ['urgent','asap','immediately','critical','failure']

df['urgent'] = df['clean_text'].apply(
    lambda x: any(word in x for word in urgent_keywords)
)

# ===============================
# 📊 ROW 1
# ===============================
col1, col2 = st.columns(2)

# Sentiment Pie
with col1:
    st.subheader("Sentiment Distribution")
    fig = px.pie(df, names='sentiment')
    st.plotly_chart(fig, use_container_width=True)

# Urgent Tickets Table
with col2:
    st.subheader("Urgent Tickets")
    st.dataframe(df[df['urgent'] == True][['Ticket Description']].head(5))

# ===============================
# 📊 ROW 2
# ===============================
col3, col4 = st.columns(2)

# Top Issues
with col3:
    st.subheader("Top Customer Issues")

    vectorizer = CountVectorizer(max_features=10)
    X = vectorizer.fit_transform(df['clean_text'])
    words = vectorizer.get_feature_names_out()
    counts = np.asarray(X.sum(axis=0)).flatten()

    top_df = pd.DataFrame({'word': words, 'count': counts})
    fig = px.bar(top_df.sort_values(by='count', ascending=False),
                 x='count', y='word', orientation='h')
    st.plotly_chart(fig, use_container_width=True)

# WordCloud
with col4:
    st.subheader("Keywords WordCloud")

    text = " ".join(df['clean_text'])
    wc = WordCloud(width=800, height=400).generate(text)

    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis('off')
    st.pyplot(fig)

# ===============================
# 📊 ROW 3
# ===============================
col5, col6 = st.columns(2)

# Topic Clusters (LDA)
with col5:
    st.subheader("Topic Clusters")

    vectorizer = CountVectorizer(max_df=0.9, min_df=5)
    X = vectorizer.fit_transform(df['clean_text'])

    lda = LatentDirichletAllocation(n_components=4, random_state=42)
    lda.fit(X)

    topics = []
    words = vectorizer.get_feature_names_out()

    for i, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[-5:]]
        topics.append(f"Topic {i}: {', '.join(top_words)}")

    for t in topics:
        st.write(t)

# Recent Urgent Tickets
with col6:
    st.subheader("Recent Urgent Tickets")
    st.dataframe(df[df['urgent'] == True][['Ticket Description']].tail(5))

# ===============================
# 📊 ROW 4
# ===============================
col7, col8 = st.columns(2)

# Sentiment Trends (if date exists)
with col7:
    st.subheader("Sentiment Trends Over Time")

    if 'Date of Purchase' in df.columns:
        df['date'] = pd.to_datetime(df['Date of Purchase'])
        trend = df.groupby(['date','sentiment']).size().reset_index(name='count')
        fig = px.line(trend, x='date', y='count', color='sentiment')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No date column found")

# Summary Metrics
with col8:
    st.subheader("Tickets Summary")

    total = len(df)
    urgent = df['urgent'].sum()
    negative = (df['sentiment'] == 'Negative').sum()

    st.metric("Total Tickets", total)
    st.metric("Urgent Tickets", urgent)
    st.metric("Negative Tickets", negative)

# ===============================
# SAMPLE DATA
# ===============================
st.subheader("Sample Tickets")
st.dataframe(df[['Ticket Description','sentiment']].sample(10))