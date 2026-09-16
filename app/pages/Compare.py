import streamlit as st
import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from tools import resolve_player_name

st.set_page_config(page_title="Compare - IPL Intelligence", page_icon="⚔️", layout="wide")

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', '..', 'data', 'IPL_cleaned.csv')
    df = pd.read_csv(data_path, parse_dates=['date'], low_memory=False)
    return df

df = load_data()


def get_full_batting_stats(name):
    player_df = df[df['batter'] == name]
    total_runs = player_df['runs_batter'].sum()
    total_matches = player_df['match_id'].nunique()
    total_balls = player_df['balls_faced'].sum()
    out_count = df[df['player_out'] == name]['match_id'].nunique()
    average = total_runs / out_count if out_count > 0 else total_runs
    strike_rate = (total_runs / total_balls * 100) if total_balls > 0 else 0

    boundaries = player_df[player_df['runs_not_boundary'] == False]
    sixes = (boundaries['runs_batter'] == 6).sum()
    fours = (boundaries['runs_batter'] == 4).sum()
    match_scores = player_df.groupby('match_id')['runs_batter'].sum()
    fifties = ((match_scores >= 50) & (match_scores < 100)).sum()
    hundreds = (match_scores >= 100).sum()

    return {
        'Matches': total_matches,
        'Runs': int(total_runs),
        'Average': round(average, 2),
        'Strike Rate': round(strike_rate, 2),
        'Fours': int(fours),
        'Sixes': int(sixes),
        'Fifties': int(fifties),
        'Hundreds': int(hundreds),
    }


st.title("⚔️ Player Comparison")

all_players = sorted(df['batter'].unique().tolist())

col1, col2 = st.columns(2)
with col1:
    p1_input = st.text_input("Player 1", value="Virat Kohli")
with col2:
    p2_input = st.text_input("Player 2", value="Rohit Sharma")

if p1_input and p2_input:
    p1 = resolve_player_name(p1_input)
    p2 = resolve_player_name(p2_input)

    if p1 is None or p2 is None:
        st.error("Ek ya dono players nahi mile. Sahi naam ya nickname try karo.")
    else:
        stats1 = get_full_batting_stats(p1)
        stats2 = get_full_batting_stats(p2)

        st.divider()

        # ---- Headline metrics side by side ----
        col1, colvs, col2 = st.columns([2, 0.5, 2])
        with col1:
            st.markdown("### " + p1)
        with colvs:
            st.markdown("### VS")
        with col2:
            st.markdown("### " + p2)

        metrics_to_show = ['Runs', 'Average', 'Strike Rate', 'Hundreds', 'Fifties', 'Sixes']
        for metric in metrics_to_show:
            c1, cmid, c2 = st.columns([2, 0.5, 2])
            v1, v2 = stats1[metric], stats2[metric]
            with c1:
                st.metric(metric, v1, delta=(round(v1 - v2, 2) if v1 >= v2 else None))
            with cmid:
                st.write("")
            with c2:
                st.metric(metric, v2, delta=(round(v2 - v1, 2) if v2 >= v1 else None))

        st.divider()

        # Full comparison table 
        st.markdown("**Full Comparison Table**")
        compare_df = pd.DataFrame({p1: stats1, p2: stats2})
        st.dataframe(compare_df, use_container_width=True)

        # ---- Simple radar-style chart (using columns of normalized bars)
        st.divider()
        st.markdown("**Visual Comparison (normalized)**")
        radar_metrics = ['Runs', 'Average', 'Strike Rate', 'Sixes', 'Hundreds']
        radar_data = pd.DataFrame({
            p1: [stats1[m] for m in radar_metrics],
            p2: [stats2[m] for m in radar_metrics]
        }, index=radar_metrics)
        st.bar_chart(radar_data)