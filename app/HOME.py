import streamlit as st
import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from chatbot import ask_agent

st.set_page_config(page_title="IPL Intelligence", page_icon="🏏", layout="wide")

# ---------------------------------------------------------
# Shared data loader (har page isko import karega)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'data', 'IPL_cleaned.csv')
    matches_path = os.path.join(base_dir, '..', 'data', 'matches_with_features.csv')
    df = pd.read_csv(data_path, parse_dates=['date'], low_memory=False)
    matches_df = pd.read_csv(matches_path, parse_dates=['date'])
    return df, matches_df

df, matches_df = load_data()


# Custom CSS (achievable styling - metric cards, spacing, subtle borders)

st.markdown("""
<style>
    .hero-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #A78BFA, #F472B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-subtitle {
        text-align: center;
        color: #9CA3AF;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background-color: #161622;
        border: 1px solid #2D2D3A;
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Hero section
st.markdown('<div class="hero-title">🏏 IPL INTELLIGENCE</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Cricket × Data × AI — 2008 to 2025</div>', unsafe_allow_html=True)

hero_query = st.text_input("", placeholder="🔍 Ask anything about IPL...", label_visibility="collapsed")

if hero_query:
    with st.spinner("Analyzing IPL data..."):
        answer = ask_agent(hero_query, "home-quick-search")
    st.info(answer)
    

st.write("")


# Stat cards

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Matches", str(matches_df['match_id'].nunique()) + "+")
with col2:
    st.metric("Players", str(df['batter'].nunique()) + "+")
with col3:
    st.metric("Seasons", str(df['season'].nunique()))
with col4:
    st.metric("Venues", str(df['venue'].nunique()))

st.write("")
st.write("")



st.caption("👈 Use the sidebar to explore Analytics, Players, Teams, Compare, Predictor, and full AI Chat.")