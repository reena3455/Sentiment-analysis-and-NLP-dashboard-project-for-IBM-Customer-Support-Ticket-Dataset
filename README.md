# Sentiment-analysis-and-NLP-dashboard-project-for-IBM-Customer-Support-Ticket-Dataset

## Overview

This project is NLP-powered analytics system designed to transform unstructured customer support tickets into actionable business insights.

This project leverages Natural Language Processing (NLP) techniques to analyze customer issues, detect sentiment, identify urgent tickets, and uncover hidden patterns through topic modeling and keyword extraction.

---

## Objectives

* Analyze unstructured support tickets
* Detect customer sentiment (Positive, Negative, Neutral)
* Identify urgent and critical issues
* Discover recurring problems using topic modeling
* Provide actionable insights for business decision-making

---

##  Features

###  NLP Processing

* Text cleaning and preprocessing
* Stopword removal and normalization

###  Sentiment Analysis

* Rule-based sentiment classification
* Categorizes tickets into Positive, Negative, Neutral

###  Urgency Detection

* Detects high-priority tickets using keyword matching
* Helps identify SLA risks

###  Keyword Extraction

* Identifies top recurring issues using CountVectorizer
* Displays most frequent problem keywords

###  WordCloud Visualization

* Visual representation of customer pain points

###  Topic Modeling

* LDA (Latent Dirichlet Allocation) for issue clustering
* Extracts hidden themes in support tickets

###  Trend Analysis

* Tracks sentiment trends over time (if date available)

###  Interactive Dashboard

* Built using Streamlit
* Includes filters, charts, KPIs, and tables

---

##  Tech Stack

* **Language:** Python
* **Frontend:** Streamlit
* **Libraries:**

  * pandas, numpy
  * nltk
  * scikit-learn
  * matplotlib, seaborn
  * plotly
  * wordcloud

---

##  Dashboard Sections in Streamlit

*  KPI Metrics (Total, Urgent, Negative %)
*  Sentiment Distribution (Pie Chart)
*  Urgent Ticket Detection
*  Top Customer Issues
*  WordCloud Visualization
*  Topic Clusters (LDA)
*  Sentiment Trends Over Time

##  Business Insights

*  High negative sentiment → potential product/service issues
*  Frequent keywords → recurring bugs or complaints
*  Urgent tickets → SLA risk and escalation needs
*  Trends over time → service quality monitoring


