import streamlit as st
import sys
import os
import pandas as pd
 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from tools import resolve_player_name
 
st.set_page_config(page_title="Players - IPL Intelligence", page_icon="🧑‍🏏", layout="wide")
 
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', '..', 'data', 'IPL_cleaned.csv')
    df = pd.read_csv(data_path, parse_dates=['date'], low_memory=False)
    return df
 
df = load_data()
 
st.markdown("""
<style>
    .player-card {
        background: linear-gradient(145deg, #1A1A26, #12121C);
        border: 1px solid #2D2D3A;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .player-name {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F5F5F7;
    }
</style>
""", unsafe_allow_html=True)
 
st.title("🧑‍🏏 Player Profiles")
 
search_input = st.text_input("Search a player", placeholder="Virat Kohli,Rohit Sharma, Bumrah...")
 
if search_input:
    resolved_name = resolve_player_name(search_input)
 
    if resolved_name is None:
        st.error("Player not Found. Please write the correct name and spelling.")
    else:
        player_df = df[df['batter'] == resolved_name]
        is_batsman = len(player_df) > 0 and player_df['balls_faced'].sum() >= df[df['bowler'] == resolved_name].shape[0]
 
        col1, col2 = st.columns([1, 2])
 
        with col1:
            st.markdown('<div class="player-card">', unsafe_allow_html=True)
            st.markdown('<div class="player-name">🏏 ' + resolved_name + '</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
 
        with col2:
            if len(player_df) > 0:
                st.subheader("Batting")
                total_runs = player_df['runs_batter'].sum()
                total_matches = player_df['match_id'].nunique()
                total_balls = player_df['balls_faced'].sum()
                out_count = df[df['player_out'] == resolved_name]['match_id'].nunique()
                average = total_runs / out_count if out_count > 0 else total_runs
                strike_rate = (total_runs / total_balls * 100) if total_balls > 0 else 0
 
                boundaries = player_df[player_df['runs_not_boundary'] == False]
                fours = (boundaries['runs_batter'] == 4).sum()
                sixes = (boundaries['runs_batter'] == 6).sum()
                match_scores = player_df.groupby('match_id')['runs_batter'].sum()
                fifties = ((match_scores >= 50) & (match_scores < 100)).sum()
                hundreds = (match_scores >= 100).sum()
 
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Matches", total_matches)
                m2.metric("Runs", total_runs)
                m3.metric("Average", round(average, 2))
                m4.metric("Strike Rate", round(strike_rate, 2))
 
                m5, m6, m7, m8 = st.columns(4)
                m5.metric("Fours", int(fours))
                m6.metric("Sixes", int(sixes))
                m7.metric("Fifties", int(fifties))
                m8.metric("Hundreds", int(hundreds))
 
                highest_score = match_scores.max()
                best_score_match_id = match_scores.idxmax()
                best_score_vs = player_df[player_df['match_id'] == best_score_match_id]['bowling_team'].iloc[0]
                st.metric("Highest Score", str(int(highest_score)) + " (vs " + best_score_vs + ")")
 
            bowler_df = df[df['bowler'] == resolved_name]
            if len(bowler_df) > 0:
                st.subheader("Bowling")
                NOT_BOWLER_WICKET = ['run out', 'retired hurt', 'retired out', 'obstructing the field']
                is_illegal = bowler_df['extra_type'].str.contains('wides|noballs', na=False)
                valid_balls = bowler_df[~is_illegal]
                total_balls_bowled = len(valid_balls)
                runs_conceded = bowler_df['runs_total'].sum()
                wickets_df = bowler_df[bowler_df['wicket_kind'].notna() & (~bowler_df['wicket_kind'].isin(NOT_BOWLER_WICKET))]
                total_wickets = len(wickets_df)
                overs = total_balls_bowled / 6
                economy = runs_conceded / overs if overs > 0 else 0
 
                b1, b2, b3 = st.columns(3)
                b1.metric("Bowling Matches", bowler_df['match_id'].nunique())
                b2.metric("Wickets", total_wickets)
                b3.metric("Economy", round(economy, 2))
 
                # Best bowling figures - har match mein wickets aur runs conceded nikalo
                match_wickets = wickets_df.groupby('match_id').size()
                match_runs = bowler_df.groupby('match_id')['runs_total'].sum()
                match_bowling = pd.DataFrame({
                    'wickets': match_wickets,
                    'runs': match_runs
                }).fillna(0)
                match_bowling['wickets'] = match_bowling['wickets'].astype(int)
 
                if len(match_bowling) > 0:
                    # Best figures = sabse zyada wickets, tie hone par kam runs wala
                    best = match_bowling.sort_values(['wickets', 'runs'], ascending=[False, True]).iloc[0]
                    best_figures = str(int(best['wickets'])) + "/" + str(int(best['runs']))
                    st.metric("Best Bowling Figures", best_figures)
 
        # Season-wise trend chart
        if len(player_df) > 0:
            st.divider()
            st.markdown("**Season-wise Runs**")
            season_runs = player_df.groupby('season')['runs_batter'].sum()
            st.bar_chart(season_runs)

 



















