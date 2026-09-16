import streamlit as st
import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

st.set_page_config(page_title="Teams - IPL Intelligence", page_icon="🏆", layout="wide")

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', '..', 'data', 'IPL_cleaned.csv')
    matches_path = os.path.join(base_dir, '..', '..', 'data', 'matches_with_features.csv')
    df = pd.read_csv(data_path, parse_dates=['date'], low_memory=False)
    matches_df = pd.read_csv(matches_path, parse_dates=['date'])
    return df, matches_df

df, matches_df = load_data()

# ---------------------------------------------------------
# Har season ka winner nikaalo (titles count karne ke liye)
# ---------------------------------------------------------
@st.cache_data
def get_season_winners(matches_df):
    winners = {}
    for s in sorted(matches_df['season'].unique()):
        season_matches = matches_df[matches_df['season'] == s]
        final_match = season_matches.sort_values('date').iloc[-1]
        winners[s] = final_match['match_won_by']
    return winners

season_winners = get_season_winners(matches_df)
titles_count = pd.Series(list(season_winners.values())).value_counts()

st.title("🏆 IPL Teams")

all_teams = sorted(pd.concat([matches_df['team1'], matches_df['team2']]).unique().tolist())

# ---------------------------------------------------------
# Team cards grid
# ---------------------------------------------------------
cols = st.columns(4)
for i, team in enumerate(all_teams):
    with cols[i % 4]:
        team_matches = matches_df[(matches_df['team1'] == team) | (matches_df['team2'] == team)]
        wins = team_matches[team_matches['match_won_by'] == team].shape[0]
        win_pct = round((wins / len(team_matches) * 100), 1) if len(team_matches) > 0 else 0
        titles = int(titles_count.get(team, 0))

        st.markdown("**" + team + "**")
        st.caption(str(titles) + " title(s) | " + str(win_pct) + "% win rate")

st.divider()


# Detailed team view

st.subheader("Team Deep-dive")
selected_team = st.selectbox("Select a team", all_teams)

if selected_team:
    team_matches = matches_df[(matches_df['team1'] == selected_team) | (matches_df['team2'] == selected_team)]
    wins = team_matches[team_matches['match_won_by'] == selected_team].shape[0]
    win_pct = round((wins / len(team_matches) * 100), 1) if len(team_matches) > 0 else 0
    titles = int(titles_count.get(selected_team, 0))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Matches", len(team_matches))
    c2.metric("Wins", wins)
    c3.metric("Win %", str(win_pct) + "%")
    c4.metric("Titles", titles)

    st.markdown("**Season-wise appearances and results**")
    season_summary = team_matches.groupby('season').agg(
        matches=('match_id', 'count'),
        wins=('match_won_by', lambda x: (x == selected_team).sum())
    )
    st.bar_chart(season_summary['wins'])

    st.markdown("**Best players for " + selected_team + " (by runs)**")
    team_batting = df[df['batting_team'] == selected_team]
    top_players = team_batting.groupby('batter')['runs_batter'].sum().sort_values(ascending=False).head(5)
    st.bar_chart(top_players)

    st.markdown("**Head-to-head record**")
    other_teams = [t for t in all_teams if t != selected_team]
    h2h_data = []
    for opp in other_teams:
        h2h = team_matches[
            ((team_matches['team1'] == opp) | (team_matches['team2'] == opp))
        ]
        if len(h2h) > 0:
            h2h_wins = h2h[h2h['match_won_by'] == selected_team].shape[0]
            h2h_data.append({'Opponent': opp, 'Matches': len(h2h), 'Wins': h2h_wins, 'Losses': len(h2h) - h2h_wins})

    if h2h_data:
        h2h_df = pd.DataFrame(h2h_data).sort_values('Matches', ascending=False)
        st.dataframe(h2h_df, use_container_width=True, hide_index=True)