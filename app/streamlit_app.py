import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings('ignore')

# ── Absolute paths ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="EarningsEdge",
    page_icon="📈",
    layout="wide"
)

# ── Load data (cached) ─────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_pickle(os.path.join(DATA_DIR, 'full_dataset.pkl'))
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['transcript_length'] = df['transcript'].str.split().str.len()
    return df

@st.cache_data
def load_sentiment():
    return pd.read_pickle(os.path.join(DATA_DIR, 'sentiment_scores.pkl'))

@st.cache_resource
def load_rag():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from groq import Groq
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, '.env'))

    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )
    vs = FAISS.load_local(
        os.path.join(DATA_DIR, 'faiss_tiny'),
        embeddings,
        allow_dangerous_deserialization=True
    )
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    return vs, client

@st.cache_resource
def load_ml_model():
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import train_test_split

    df = load_data()
    signal_words = ['growth','revenue','guidance','margin','beat','miss',
                    'raised','lowered','uncertainty','headwinds','record',
                    'strong','weak','declined','exceeded','profit','loss',
                    'outlook','forecast','demand']

    df['avg_word_length'] = df['transcript'].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if pd.notna(x) else 0)
    df['question_count'] = df['transcript'].str.count(r'\?')
    df['quarter_num']    = df['q'].str.extract(r'Q(\d)').astype(float)

    for word in signal_words:
        df[f'word_{word}'] = df['transcript'].str.lower().str.count(word)

    feature_cols = (['transcript_length','avg_word_length',
                     'question_count','quarter_num','year']
                    + [f'word_{w}' for w in signal_words])

    df_ml = df[feature_cols + ['direction']].dropna()
    X = df_ml[feature_cols]
    y = df_ml['direction']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = LGBMClassifier(n_estimators=200, max_depth=4,
                           learning_rate=0.05, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    return model, feature_cols, signal_words

# ── Header ─────────────────────────────────────────────────
st.title("📈 EarningsEdge")
st.caption("An Intelligent Earnings Call Analysis & Stock Movement Prediction System")
st.divider()

# ── Load data ──────────────────────────────────────────────
with st.spinner("Loading data..."):
    df = load_data()

# ── Tabs ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Explore", "🤖 Predict", "💬 Summarize", "🧠 Chat"
])

# ══════════════════════════════════════════════════════════
# TAB 1 — EXPLORE
# ══════════════════════════════════════════════════════════
with tab1:
    st.header("📊 Earnings Call Explorer")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transcripts", f"{len(df):,}")
    col2.metric("Companies", f"{df['ticker'].nunique():,}")
    col3.metric("Date Range", "2019 – 2023")
    col4.metric("Avg Transcript", f"{df['transcript_length'].mean():.0f} words")

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Price change distribution")
        fig = px.histogram(df, x='pct_change', nbins=50,
                           color_discrete_sequence=['#3b82f6'],
                           labels={'pct_change': '% change (3 days after call)'})
        fig.add_vline(x=0, line_dash='dash', line_color='red')
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Up vs Down after earnings")
        counts = df['direction'].value_counts()
        fig = px.pie(values=counts.values,
                     names=['Down', 'Up'],
                     color_discrete_sequence=['#ef4444','#22c55e'],
                     hole=0.4)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average price change by year")
    year_stats = df.groupby('year')['pct_change'].mean().reset_index()
    fig = px.bar(year_stats, x='year', y='pct_change',
                 color='pct_change',
                 color_continuous_scale='RdYlGn',
                 labels={'pct_change': 'Avg % change', 'year': 'Year'})
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Filter by company")
    ticker = st.selectbox("Select ticker", sorted(df['ticker'].unique()))
    ticker_df = df[df['ticker'] == ticker].sort_values('date')
    if not ticker_df.empty:
        st.dataframe(
            ticker_df[['date','q','pct_change','direction','transcript_length']]
            .reset_index(drop=True),
            use_container_width=True
        )

# ══════════════════════════════════════════════════════════
# TAB 2 — PREDICT
# ══════════════════════════════════════════════════════════
with tab2:
    st.header("🤖 Stock Movement Predictor")
    st.write("Select a company and quarter to predict post-earnings stock movement.")

    col1, col2 = st.columns(2)
    with col1:
        pred_ticker = st.selectbox("Ticker", sorted(df['ticker'].unique()), key='pred_ticker')
    with col2:
        ticker_quarters = df[df['ticker']==pred_ticker]['q'].unique()
        pred_quarter = st.selectbox("Quarter", sorted(ticker_quarters, reverse=True))

    if st.button("🔮 Predict", type='primary'):
        row = df[(df['ticker']==pred_ticker) & (df['q']==pred_quarter)]
        if row.empty:
            st.error("No data found for this ticker/quarter combination.")
        else:
            with st.spinner("Running model..."):
                model, feature_cols, signal_words = load_ml_model()
                row = row.iloc[0]
                features = {
                    'transcript_length': len(str(row['transcript']).split()),
                    'avg_word_length':   np.mean([len(w) for w in str(row['transcript']).split()]),
                    'question_count':    str(row['transcript']).count('?'),
                    'quarter_num':       float(pred_quarter[-1]),
                    'year':              float(row['year'])
                }
                for word in signal_words:
                    features[f'word_{word}'] = str(row['transcript']).lower().count(word)

                X_pred = pd.DataFrame([features])[feature_cols]
                pred   = model.predict(X_pred)[0]
                prob   = model.predict_proba(X_pred)[0]

            col1, col2, col3 = st.columns(3)
            if pred == 1:
                col1.success("📈 Predicted: UP")
            else:
                col1.error("📉 Predicted: DOWN")
            col2.metric("Confidence", f"{max(prob)*100:.1f}%")
            col3.metric("Actual result",
                        f"{'▲' if row['direction']==1 else '▼'} {row['pct_change']:+.2f}%")

            st.subheader("Top signal words in this transcript")
            word_counts = {w: str(row['transcript']).lower().count(w) for w in signal_words}
            word_df = (pd.DataFrame(list(word_counts.items()), columns=['word','count'])
                       .sort_values('count', ascending=False).head(10))
            fig = px.bar(word_df, x='count', y='word', orientation='h',
                         color='count', color_continuous_scale='Blues')
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 3 — SUMMARIZE
# ══════════════════════════════════════════════════════════
with tab3:
    st.header("💬 Sentiment Analyzer")

    try:
        sent_df = load_sentiment()
        sent_df['date'] = pd.to_datetime(sent_df['date'])

        col1, col2 = st.columns(2)
        with col1:
            sum_ticker = st.selectbox("Ticker", sorted(sent_df['ticker'].unique()), key='sum_ticker')
        with col2:
            ticker_sent = sent_df[sent_df['ticker']==sum_ticker]
            sum_quarter = st.selectbox("Quarter", sorted(ticker_sent['q'].unique(), reverse=True), key='sum_quarter')

        if st.button("📊 Analyze Sentiment", type='primary'):
            row = sent_df[(sent_df['ticker']==sum_ticker) & (sent_df['q']==sum_quarter)]
            if row.empty:
                st.error("No sentiment data found.")
            else:
                row = row.iloc[0]
                col1, col2, col3 = st.columns(3)
                col1.metric("😊 Positive", f"{row['positive']:.1%}")
                col2.metric("😐 Neutral",  f"{row['neutral']:.1%}")
                col3.metric("😟 Negative", f"{row['negative']:.1%}")

                fig = go.Figure(go.Bar(
                    x=['Positive','Neutral','Negative'],
                    y=[row['positive'], row['neutral'], row['negative']],
                    marker_color=['#22c55e','#f59e0b','#ef4444']
                ))
                fig.update_layout(
                    title=f"FinBERT Sentiment — {sum_ticker} {sum_quarter}",
                    height=350, yaxis_tickformat='.0%'
                )
                st.plotly_chart(fig, use_container_width=True)

                dominant = max(['positive','neutral','negative'], key=lambda x: row[x])
                st.info(f"**Dominant sentiment:** {dominant.capitalize()} | "
                        f"**Actual price move:** {'▲' if row['direction']==1 else '▼'} "
                        f"{row['pct_change']:+.2f}%")

    except Exception as e:
        st.error(f"Sentiment data not found: {e}")

# ══════════════════════════════════════════════════════════
# TAB 4 — CHAT
# ══════════════════════════════════════════════════════════
with tab4:
    st.header("🧠 Earnings Call Chatbot")
    st.write("Ask anything about earnings calls from 2019–2023.")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    if prompt := st.chat_input("Ask about any company or trend..."):
        st.session_state.messages.append({'role':'user','content':prompt})
        with st.chat_message('user'):
            st.write(prompt)

        with st.chat_message('assistant'):
            with st.spinner("Searching transcripts..."):
                try:
                    vs, groq_client = load_rag()
                    docs = vs.similarity_search(prompt, k=3)
                    context = '\n'.join([
                        f"[{d.metadata['ticker']} | {d.metadata['date']} | {d.metadata['quarter']}]\n{d.page_content}"
                        for d in docs
                    ])
                    r = groq_client.chat.completions.create(
                        model='llama-3.3-70b-versatile',
                        messages=[{'role':'user','content':
                            f"Using these earnings calls:\n{context}\n\nQuestion: {prompt}\nAnswer citing ticker and date:"}],
                        max_tokens=500
                    )
                    answer = r.choices[0].message.content
                    st.write(answer)

                    with st.expander("📄 Sources"):
                        for d in docs:
                            st.write(f"→ **{d.metadata['ticker']}** | {d.metadata['date']} | {d.metadata['quarter']}")

                    st.session_state.messages.append({'role':'assistant','content':answer})

                except Exception as e:
                    st.error(f"Error: {e}")
