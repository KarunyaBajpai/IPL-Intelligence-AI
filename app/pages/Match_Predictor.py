import streamlit as st
import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

st.set_page_config(page_title="Predictor - IPL Intelligence", page_icon="🔮", layout="wide")

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', '..', 'data', 'IPL_cleaned.csv')
    matches_path = os.path.join(base_dir, '..', '..', 'data', 'matches_with_features.csv')
    df = pd.read_csv(data_path, parse_dates=['date'], low_memory=False)
    matches_df = pd.read_csv(matches_path, parse_dates=['date'])
    return df, matches_df

df, matches_df = load_data()

st.title("🔮 Match Predictor")
st.write("Select 2 teams and see which one is favorite, or why")

all_teams = sorted(pd.concat([matches_df['team1'], matches_df['team2']]).unique().tolist())

col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox("Team A", all_teams, index=all_teams.index("Mumbai Indians") if "Mumbai Indians" in all_teams else 0)
with col2:
    team_b_options = [t for t in all_teams if t != team_a]
    default_b = "Chennai Super Kings" if "Chennai Super Kings" in team_b_options else team_b_options[0]
    team_b = st.selectbox("Team B", team_b_options, index=team_b_options.index(default_b))

venues = ["Not specified"] + sorted(df['venue'].unique().tolist())
venue = st.selectbox("Venue (optional)", venues)

if st.button("Analyze Match", type="primary"):
    a_matches = matches_df[(matches_df['team1'] == team_a) | (matches_df['team2'] == team_a)]
    b_matches = matches_df[(matches_df['team1'] == team_b) | (matches_df['team2'] == team_b)]
    a_win_pct = (a_matches[a_matches['match_won_by'] == team_a].shape[0] / len(a_matches) * 100) if len(a_matches) > 0 else 50
    b_win_pct = (b_matches[b_matches['match_won_by'] == team_b].shape[0] / len(b_matches) * 100) if len(b_matches) > 0 else 50

    h2h_overall = matches_df[
        ((matches_df['team1'] == team_a) & (matches_df['team2'] == team_b)) |
        ((matches_df['team1'] == team_b) & (matches_df['team2'] == team_a))
    ]

    using_venue_specific = False
    if venue != "Not specified":
        h2h_venue = h2h_overall[h2h_overall['venue'] == venue]
        if len(h2h_venue) >= 3:
            h2h = h2h_venue
            using_venue_specific = True
        else:
            h2h = h2h_overall
    else:
        h2h = h2h_overall

    a_h2h_wins = h2h[h2h['match_won_by'] == team_a].shape[0]
    b_h2h_wins = h2h[h2h['match_won_by'] == team_b].shape[0]
    total_h2h = len(h2h)

    a_recent = a_matches.sort_values('date').tail(5)
    b_recent = b_matches.sort_values('date').tail(5)
    a_recent_wins = a_recent[a_recent['match_won_by'] == team_a].shape[0]
    b_recent_wins = b_recent[b_recent['match_won_by'] == team_b].shape[0]

    a_score = (a_win_pct * 0.4) + ((a_h2h_wins / total_h2h * 100 if total_h2h > 0 else 50) * 0.3) + ((a_recent_wins / 5 * 100) * 0.3)
    b_score = (b_win_pct * 0.4) + ((b_h2h_wins / total_h2h * 100 if total_h2h > 0 else 50) * 0.3) + ((b_recent_wins / 5 * 100) * 0.3)
    total_score = a_score + b_score
    a_prob = round((a_score / total_score) * 100, 1) if total_score > 0 else 50
    b_prob = round(100 - a_prob, 1)

    st.divider()
    st.markdown("### Win Probability")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(team_a, str(a_prob) + "%")
        st.progress(a_prob / 100)
    with col2:
        st.metric(team_b, str(b_prob) + "%")
        st.progress(b_prob / 100)

    st.divider()
    st.markdown("### Why " + (team_a if a_prob >= b_prob else team_b) + "?")

    favorite, underdog = (team_a, team_b) if a_prob >= b_prob else (team_b, team_a)
    fav_win_pct, dog_win_pct = (a_win_pct, b_win_pct) if a_prob >= b_prob else (b_win_pct, a_win_pct)
    fav_recent, dog_recent = (a_recent_wins, b_recent_wins) if a_prob >= b_prob else (b_recent_wins, a_recent_wins)
    fav_h2h, dog_h2h = (a_h2h_wins, b_h2h_wins) if a_prob >= b_prob else (b_h2h_wins, a_h2h_wins)

    reasons = []
    if fav_win_pct > dog_win_pct:
        reasons.append("Better overall win % (" + str(round(fav_win_pct, 1)) + "% vs " + str(round(dog_win_pct, 1)) + "%)")
    if fav_recent > dog_recent:
        reasons.append("Stronger recent form (" + str(fav_recent) + "/5 vs " + str(dog_recent) + "/5 wins)")
    if fav_h2h > dog_h2h:
        h2h_label = "at " + venue if using_venue_specific else "overall"
        reasons.append("Better head-to-head record " + h2h_label + " (" + str(fav_h2h) + " vs " + str(dog_h2h) + ")")

    if not reasons:
        reasons.append("Very close matchup - marginal statistical edge only")

    for r in reasons:
        st.write("- " + r)

    st.divider()
    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.write("**" + team_a + "**")
        st.write("Overall win %: " + str(round(a_win_pct, 1)) + "%")
        st.write("Recent form: " + str(a_recent_wins) + "/5 wins")
    with detail_col2:
        st.write("**" + team_b + "**")
        st.write("Overall win %: " + str(round(b_win_pct, 1)) + "%")
        st.write("Recent form: " + str(b_recent_wins) + "/5 wins")

    if using_venue_specific:
        st.write("**Head-to-head at " + venue + "** (" + str(total_h2h) + " matches): " + team_a + " won " + str(a_h2h_wins) + ", " + team_b + " won " + str(b_h2h_wins))
    elif total_h2h > 0:
        note = " (overall, all venues - not enough matches at " + venue + ")" if venue != "Not specified" else " (overall, all venues)"
        st.write("**Head-to-head**" + note + " (" + str(total_h2h) + " matches): " + team_a + " won " + str(a_h2h_wins) + ", " + team_b + " won " + str(b_h2h_wins))
    else:
        st.write("**Head-to-head**: In dono teams ka koi record nahi mila.")

    
    