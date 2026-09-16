import streamlit as st
import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

st.set_page_config(page_title="Analytics - IPL Intelligence", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', '..', 'data', 'IPL_cleaned.csv')
    matches_path = os.path.join(base_dir, '..', '..', 'data', 'matches_with_features.csv')
    df = pd.read_csv(data_path, parse_dates=['date'], low_memory=False)
    matches_df = pd.read_csv(matches_path, parse_dates=['date'])
    return df, matches_df

df, matches_df = load_data()

st.title("📊 IPL Analytics")

# Filters 
col1, col2 = st.columns(2)
with col1:
    seasons = ["All Time"] + sorted(df['season'].unique().tolist(), reverse=True)
    selected_season = st.selectbox("Season", seasons)
with col2:
    top_n = st.slider("Select, How many players to display :-", 5, 20, 10)

filtered_df = df if selected_season == "All Time" else df[df['season'] == selected_season]

st.divider()

#  Row 1: Top run scorers + Top wicket takers 
col1, col2 = st.columns(2)

with col1:
    st.markdown("**🏏 Top Run Scorers**")
    top_runs = filtered_df.groupby('batter')['runs_batter'].sum().sort_values(ascending=False).head(top_n)
    st.bar_chart(top_runs)

with col2:
    st.markdown("**🎯 Top Wicket Takers**")
    NOT_BOWLER_WICKET = ['run out', 'retired hurt', 'retired out', 'obstructing the field']
    wickets_df = filtered_df[filtered_df['wicket_kind'].notna() & (~filtered_df['wicket_kind'].isin(NOT_BOWLER_WICKET))]
    top_wickets = wickets_df.groupby('bowler').size().sort_values(ascending=False).head(top_n)
    st.bar_chart(top_wickets)

st.divider()

#  Row 2: Season-wise total runs trend 
st.markdown("**📈 Season-wise Total Runs (IPL Trend)**")
season_runs = df.groupby('season')['runs_batter'].sum()
st.line_chart(season_runs)

st.divider()

#  Row 3: Team win percentages 
st.markdown("**🏆 Team-wise Win %**")
team_stats = []
all_teams = pd.concat([matches_df['team1'], matches_df['team2']]).unique()
for team in all_teams:
    team_matches = matches_df[(matches_df['team1'] == team) | (matches_df['team2'] == team)]
    wins = team_matches[team_matches['match_won_by'] == team].shape[0]
    win_pct = (wins / len(team_matches) * 100) if len(team_matches) > 0 else 0
    team_stats.append({'Team': team, 'Win %': round(win_pct, 1), 'Matches': len(team_matches)})

team_stats_df = pd.DataFrame(team_stats).sort_values('Win %', ascending=False)
st.bar_chart(team_stats_df.set_index('Team')['Win %'])

st.divider()

# Row 4: Venue-wise average first innings score 
st.markdown("**🏟️ Venue-wise Average 1st Innings Score (Top 10 by matches played)**")
first_innings = df[df['innings'] == 1]
venue_avg = first_innings.groupby('match_id').agg(
    venue=('venue', 'first'),
    score=('team_runs', 'max')
)
venue_summary = venue_avg.groupby('venue').agg(
    avg_score=('score', 'mean'),
    matches=('score', 'count')
).sort_values('matches', ascending=False).head(10)
st.bar_chart(venue_summary['avg_score'])

st.caption("Note: Only those venues are shown where atleast a few matches have played")