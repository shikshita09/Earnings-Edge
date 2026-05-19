# EarningsEdge 
### An Intelligent Earnings Call Analysis & Stock Movement Prediction System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

## Overview
EarningsEdge is an end-to-end data science project that analyzes earnings 
call transcripts using EDA, Machine Learning, and LLMs to predict post-earnings 
stock movement and extract actionable insights.

## Features
- 🔍 **EDA** — tone trends, keyword frequency, sector wordclouds
- 🤖 **ML** — XGBoost model predicting post-earnings stock direction (up/down)
- 💬 **NLP** — FinBERT sentiment analysis + LDA topic modelling
- 🧠 **LLM** — GPT-powered call summarization + RAG chatbot over transcripts
- 📊 **Dashboard** — interactive Streamlit app with 4 tabs

## Tech Stack
Python · Pandas · FinBERT · XGBoost · SHAP · LangChain · FAISS · Streamlit · yfinance

## Project Structure
(paste the folder structure above)

## Setup
pip install -r requirements.txt

## Roadmap
- [x] Repo setup
- [ ] Data collection & EDA
- [ ] ML model
- [ ] NLP pipeline
- [ ] LLM summarization + RAG
- [ ] Streamlit dashboard
- [ ] Deploy to Hugging Face Spaces

## Dataset
- Earnings transcripts: Kaggle / Motley Fool
- Stock prices: yfinance
- Financial metadata: Yahoo Finance / SEC EDGAR