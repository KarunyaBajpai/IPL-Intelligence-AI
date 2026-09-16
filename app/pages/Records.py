import streamlit as st
import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

st.set_page_config(page_title="Records - IPL Intelligence", page_icon="🏅", layout="wide")

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', '..', 'data', 'IPL_cleaned.csv')
    matches_path = os.path.join(base_dir, '..', '..', 'data', 'matches_with_features.csv')
    df = pd.read_csv(data_path, parse_dates=['date'], low_memory=False)
    matches_df = pd.read_csv(matches_path, parse_dates=['date'])
    return df, matches_df

df, matches_df = load_data()

NOT_BOWLER_WICKET = ['run out', 'retired hurt', 'retired out', 'obstructing the field']

st.title("🏅 IPL Records")

tab1, tab2, tab3, tab4 = st.tabs(["🏏 Batting", "🎯 Bowling", "🏆 Team", "📅 Season"])


# BATTING RECORDS

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Most Runs**")
        most_runs = df.groupby('batter')['runs_batter'].sum().sort_values(ascending=False).head(10)
        st.dataframe(most_runs.reset_index().rename(columns={'batter': 'Player', 'runs_batter': 'Runs'}), hide_index=True, use_container_width=True)

    with col2:
        st.markdown("**Most Sixes**")
        boundaries = df[df['runs_not_boundary'] == False]
        sixes_df = boundaries[boundaries['runs_batter'] == 6]
        most_sixes = sixes_df.groupby('batter').size().sort_values(ascending=False).head(10)
        st.dataframe(most_sixes.reset_index().rename(columns={'batter': 'Player', 0: 'Sixes'}), hide_index=True, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Most Fours**")
        fours_df = boundaries[boundaries['runs_batter'] == 4]
        most_fours = fours_df.groupby('batter').size().sort_values(ascending=False).head(10)
        st.dataframe(most_fours.reset_index().rename(columns={'batter': 'Player', 0: 'Fours'}), hide_index=True, use_container_width=True)

    with col4:
        st.markdown("**Most Fifty+ Scores**")
        match_scores = df.groupby(['batter', 'match_id'])['runs_batter'].sum().reset_index()
        fifty_plus = match_scores[match_scores['runs_batter'] >= 50]
        most_fifties = fifty_plus.groupby('batter').size().sort_values(ascending=False).head(10)
        st.dataframe(most_fifties.reset_index().rename(columns={'batter': 'Player', 0: '50+ scores'}), hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("**Highest Individual Score**")
    highest = match_scores.sort_values('runs_batter', ascending=False).head(10)
    highest = highest.merge(
        df[['match_id', 'batter', 'bowling_team', 'season']].drop_duplicates(subset=['match_id', 'batter']),
        on=['match_id', 'batter'], how='left'
    )
    highest_display = highest[['batter', 'runs_batter', 'bowling_team', 'season']].rename(
        columns={'batter': 'Player', 'runs_batter': 'Score', 'bowling_team': 'Against', 'season': 'Season'}
    )
    st.dataframe(highest_display, hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("**Best Strike Rate** (minimum 500 balls faced)")
    sr_stats = df.groupby('batter').agg(runs=('runs_batter', 'sum'), balls=('balls_faced', 'sum')).reset_index()
    sr_stats = sr_stats[sr_stats['balls'] >= 500]
    sr_stats['strike_rate'] = round(sr_stats['runs'] / sr_stats['balls'] * 100, 2)
    best_sr = sr_stats.sort_values('strike_rate', ascending=False).head(10)
    st.dataframe(best_sr[['batter', 'strike_rate', 'runs', 'balls']].rename(
        columns={'batter': 'Player', 'strike_rate': 'Strike Rate', 'runs': 'Runs', 'balls': 'Balls'}
    ), hide_index=True, use_container_width=True)


# BOWLING RECORDS

with tab2:
    wickets_df = df[df['wicket_kind'].notna() & (~df['wicket_kind'].isin(NOT_BOWLER_WICKET))]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Most Wickets**")
        most_wickets = wickets_df.groupby('bowler').size().sort_values(ascending=False).head(10)
        st.dataframe(most_wickets.reset_index().rename(columns={'bowler': 'Player', 0: 'Wickets'}), hide_index=True, use_container_width=True)

    with col2:
        st.markdown("**Best Economy** (minimum 300 balls bowled)")
        is_illegal = df['extra_type'].str.contains('wides|noballs', na=False)
        valid_balls = df[~is_illegal]
        bowl_stats = valid_balls.groupby('bowler').size().reset_index(name='balls')
        runs_conceded = df.groupby('bowler')['runs_total'].sum().reset_index(name='runs')
        bowl_stats = bowl_stats.merge(runs_conceded, on='bowler')
        bowl_stats = bowl_stats[bowl_stats['balls'] >= 300]
        bowl_stats['economy'] = round(bowl_stats['runs'] / (bowl_stats['balls'] / 6), 2)
        best_economy = bowl_stats.sort_values('economy').head(10)
        st.dataframe(best_economy[['bowler', 'economy', 'balls']].rename(
            columns={'bowler': 'Player', 'economy': 'Economy', 'balls': 'Balls Bowled'}
        ), hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("**Best Bowling Figures in a Match**")
    match_wickets = wickets_df.groupby(['bowler', 'match_id']).size().reset_index(name='wickets')
    match_runs = df.groupby(['bowler', 'match_id'])['runs_total'].sum().reset_index(name='runs')
    match_bowling = match_wickets.merge(match_runs, on=['bowler', 'match_id'])
    best_figures = match_bowling.sort_values(['wickets', 'runs'], ascending=[False, True]).head(10)
    best_figures = best_figures.merge(
        df[['match_id', 'batting_team', 'season']].drop_duplicates(subset='match_id'),
        on='match_id', how='left'
    )
    best_figures['Figures'] = best_figures['wickets'].astype(str) + "/" + best_figures['runs'].astype(str)
    st.dataframe(best_figures[['bowler', 'Figures', 'batting_team', 'season']].rename(
        columns={'bowler': 'Player', 'batting_team': 'Against', 'season': 'Season'}
    ), hide_index=True, use_container_width=True)


# TEAM RECORDS

with tab3:
    all_teams = sorted(pd.concat([matches_df['team1'], matches_df['team2']]).unique().tolist())

    @st.cache_data
    def get_season_winners_local(matches_df):
        winners = {}
        for s in sorted(matches_df['season'].unique()):
            season_matches = matches_df[matches_df['season'] == s]
            final_match = season_matches.sort_values('date').iloc[-1]
            winners[s] = final_match['match_won_by']
        return winners

    season_winners = get_season_winners_local(matches_df)
    titles_count = pd.Series(list(season_winners.values())).value_counts()

    team_records = []
    for team in all_teams:
        team_matches = matches_df[(matches_df['team1'] == team) | (matches_df['team2'] == team)]
        wins = team_matches[team_matches['match_won_by'] == team].shape[0]
        win_pct = round((wins / len(team_matches) * 100), 1) if len(team_matches) > 0 else 0
        team_records.append({
            'Team': team,
            'Titles': int(titles_count.get(team, 0)),
            'Matches': len(team_matches),
            'Wins': wins,
            'Win %': win_pct
        })

    team_records_df = pd.DataFrame(team_records)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Most Successful (by Titles)**")
        st.dataframe(team_records_df.sort_values('Titles', ascending=False), hide_index=True, use_container_width=True)
    with col2:
        st.markdown("**Best Win %** (minimum 30 matches)")
        qualified = team_records_df[team_records_df['Matches'] >= 30]
        st.dataframe(qualified.sort_values('Win %', ascending=False), hide_index=True, use_container_width=True)

# SEASON RECORDS (Orange Cap / Purple Cap style)

with tab4:
    st.markdown("**Season-wise Champions, Orange Cap & Purple Cap**")

    season_records = []
    for s in sorted(df['season'].unique()):
        season_df = df[df['season'] == s]
        season_matches = matches_df[matches_df['season'] == s]

        champion = season_matches.sort_values('date').iloc[-1]['match_won_by'] if len(season_matches) > 0 else "N/A"

        orange_cap_series = season_df.groupby('batter')['runs_batter'].sum().sort_values(ascending=False)
        orange_cap = orange_cap_series.index[0] if len(orange_cap_series) > 0 else "N/A"
        orange_runs = int(orange_cap_series.iloc[0]) if len(orange_cap_series) > 0 else 0

        season_wickets = season_df[season_df['wicket_kind'].notna() & (~season_df['wicket_kind'].isin(NOT_BOWLER_WICKET))]
        purple_cap_series = season_wickets.groupby('bowler').size().sort_values(ascending=False)
        purple_cap = purple_cap_series.index[0] if len(purple_cap_series) > 0 else "N/A"
        purple_wickets = int(purple_cap_series.iloc[0]) if len(purple_cap_series) > 0 else 0

        season_records.append({
            'Season': s,
            'Champion': champion,
            'Orange Cap': orange_cap + " (" + str(orange_runs) + ")",
            'Purple Cap': purple_cap + " (" + str(purple_wickets) + ")"
        })

    season_records_df = pd.DataFrame(season_records).sort_values('Season', ascending=False)
    st.dataframe(season_records_df, hide_index=True, use_container_width=True)