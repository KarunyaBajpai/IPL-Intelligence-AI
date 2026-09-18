import streamlit as st
import sys
import os
import pandas as pd


# =========================================================
# PATH / IMPORTS
# =========================================================

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from chatbot import ask_agent


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="IPL Intelligence",
    page_icon="🏏",
    layout="wide"
)


# =========================================================
# DATA LOADER
# =========================================================

@st.cache_data
def load_data():

    base_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(
        base_dir,
        '..',
        'data',
        'IPL_cleaned.csv'
    )

    matches_path = os.path.join(
        base_dir,
        '..',
        'data',
        'matches_with_features.csv'
    )

    df = pd.read_csv(
        data_path,
        parse_dates=['date'],
        low_memory=False
    )

    matches_df = pd.read_csv(
        matches_path,
        parse_dates=['date']
    )

    return df, matches_df


df, matches_df = load_data()


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* =========================
   HERO
========================= */

.hero-title {
    text-align: center;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #A78BFA, #F472B6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 0.5rem;
    margin-bottom: 0.2rem;
}

.hero-subtitle {
    text-align: center;
    color: #A1A1AA;
    font-size: 1.15rem;
    margin-bottom: 1.6rem;
}

.intro-text {
    text-align: center;
    color: #D1D5DB;
    font-size: 1rem;
    margin-bottom: 1.7rem;
}


/* =========================
   STAT CARDS
========================= */

div[data-testid="stMetric"] {
    background-color: #161622;
    border: 1px solid #2D2D3A;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    min-height: 105px;
}

div[data-testid="stMetricLabel"] {
    color: #D1D5DB;
}

div[data-testid="stMetricValue"] {
    color: #F5F5F5;
}


/* =========================
   SECTION
========================= */

.quick-heading {
    text-align: center;
    font-size: 1.7rem;
    font-weight: 700;
    color: #F5F5F5;
    margin-top: 2rem;
    margin-bottom: 0.25rem;
}

.quick-subheading {
    text-align: center;
    color: #9CA3AF;
    font-size: 0.95rem;
    margin-bottom: 1.4rem;
}


/* =========================
   QUICK ACCESS CARDS
========================= */

div[data-testid="column"] {
    padding: 0.2rem;
}


/* Card content */
.quick-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #F5F5F5;
}

.quick-description {
    font-size: 0.88rem;
    color: #9CA3AF;
}


/* =========================
   BUTTONS
========================= */

div.stButton > button {
    width: 100%;
    height: 2.5rem;
    border-radius: 9px;
    border: 1px solid #35354A;
    background-color: #111827;
    color: #F5F5F5;
    font-size: 0.88rem;
    font-weight: 600;
}

div.stButton > button:hover {
    border-color: #A78BFA;
    color: #C084FC;
}


/* =========================
   FOOTER
========================= */

.footer {
    text-align: center;
    color: #71717A;
    font-size: 0.8rem;
    margin-top: 2rem;
    padding-bottom: 1rem;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HERO SECTION
# =========================================================

st.markdown(
    '<div class="hero-title">🏏 IPL INTELLIGENCE</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">Cricket × Data × AI — 2008 to 2025</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="intro-text">Explore IPL history through data-driven analytics, player insights, team intelligence and AI-powered predictions.</div>',
    unsafe_allow_html=True
)


# =========================================================
# STAT CARDS
# =========================================================

col1, col2, col3, col4 = st.columns(4, gap="medium")

with col1:
    st.metric(
        "Matches",
        str(matches_df["match_id"].nunique()) + "+"
    )

with col2:
    st.metric(
        "Players",
        str(df["batter"].nunique()) + "+"
    )

with col3:
    st.metric(
        "Seasons",
        str(df["season"].nunique())
    )

with col4:
    st.metric(
        "Venues",
        str(df["venue"].nunique())
    )


# =========================================================
# QUICK ACCESS HEADING
# =========================================================

st.markdown(
    '<div class="quick-heading">Explore IPL Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="quick-subheading">Access the key features of the platform</div>',
    unsafe_allow_html=True
)


# =========================================================
# QUICK ACCESS — ROW 1
# =========================================================

q1, q2 = st.columns(2, gap="medium")


with q1:

    st.markdown("## 🤖 AI Cricket Analyst")

    st.markdown(
        '<div class="quick-description">Ask questions and get AI-powered cricket insights.</div>',
        unsafe_allow_html=True
    )

    st.write("")

    if st.button(
        "Open AI Analyst →",
        key="quick_ai",
        use_container_width=True
    ):
        st.switch_page("pages/AI_CRICKET_ANALYST.py")


with q2:

    st.markdown("## 📊 IPL Analytics")

    st.markdown(
        '<div class="quick-description">Explore batting, bowling, team and season trends.</div>',
        unsafe_allow_html=True
    )

    st.write("")

    if st.button(
        "Open Analytics →",
        key="quick_analytics",
        use_container_width=True
    ):
        st.switch_page("pages/IPL_Analytics.py")


# =========================================================
# SPACE
# =========================================================

st.write("")


# =========================================================
# QUICK ACCESS — ROW 2
# =========================================================

q3, q4 = st.columns(2, gap="medium")


with q3:

    st.markdown("## ⚔️ Compare Players")

    st.markdown(
        '<div class="quick-description">Compare player performance across key metrics.</div>',
        unsafe_allow_html=True
    )

    st.write("")

    if st.button(
        "Compare Players →",
        key="quick_compare",
        use_container_width=True
    ):
        st.switch_page("pages/Compare.py")


with q4:

    st.markdown("## 🎯 Match Predictor")

    st.markdown(
        '<div class="quick-description">Explore ML-powered IPL match predictions.</div>',
        unsafe_allow_html=True
    )

    st.write("")

    if st.button(
        "Open Predictor →",
        key="quick_predictor",
        use_container_width=True
    ):
        st.switch_page("pages/Match_Predictor.py")


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">IPL Intelligence • Cricket × Data × AI • 2008–2025</div>',
    unsafe_allow_html=True
)