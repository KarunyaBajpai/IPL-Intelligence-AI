import pandas as pd
import os
from langchain.tools import tool
from rapidfuzz import process
import chromadb
from chromadb.utils import embedding_functions


# Paths 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'IPL_cleaned.csv')
MATCHES_PATH = os.path.join(BASE_DIR, '..', 'data', 'matches_with_features.csv')
CHROMA_PATH = os.path.join(BASE_DIR, '..', 'data', 'chroma_db')
NAME_MAPPING_PATH = os.path.join(BASE_DIR, '..', 'data', 'player_name_mapping.csv')


# Data load

df = pd.read_csv(DATA_PATH, parse_dates=['date'], low_memory=False)
ALL_PLAYERS = df['batter'].unique().tolist()



# Vector DB (RAG) setup

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
player_collection = chroma_client.get_or_create_collection(
    name="player_summaries",
    embedding_function=embedding_fn
)


# Verified full-name -> dataset-format registry

name_mapping_df = pd.read_csv(NAME_MAPPING_PATH)
FULL_NAME_TO_UNIQUE = dict(zip(name_mapping_df['full_name_lower'], name_mapping_df['unique_name']))



# Famous nicknames -> full name 
NICKNAME_MAP = {
    # Virat Kohli
    "king": "virat kohli", "king kohli": "virat kohli", "cheeku": "virat kohli",
    "chiku": "virat kohli", "virat": "virat kohli", "kohli": "virat kohli",

    # MS Dhoni
    "thala": "ms dhoni", "thalaiva": "ms dhoni", "captain cool": "ms dhoni",
    "mahi": "ms dhoni", "msd": "ms dhoni", "dhoni": "ms dhoni",

    # Rohit Sharma
    "hitman": "rohit sharma", "shana": "rohit sharma", "ro": "rohit sharma",
    "rohit": "rohit sharma",

    # Hardik Pandya
    "kungfu pandya": "hardik pandya", "kung fu pandya": "hardik pandya",
    "hardik": "hardik pandya",

    # Shubman Gill
    "prince": "shubman gill", "gill": "shubman gill",

    # Shikhar Dhawan
    "gabbar": "shikhar dhawan", "jat": "shikhar dhawan", "dhawan": "shikhar dhawan",

    # Jasprit Bumrah
    "boom boom bumrah": "jasprit bumrah", "bumrah": "jasprit bumrah",

    # AB de Villiers
    "mr 360": "ab de villiers", "mr. 360": "ab de villiers", "abd": "ab de villiers",

    # Chris Gayle
    "universe boss": "chris gayle", "gayle storm": "chris gayle", "gayle": "chris gayle",

    # Suresh Raina
    "mr ipl": "suresh raina", "raina": "suresh raina",

    # David Warner
    "the bull": "david warner", "warner": "david warner",

    # Yuvraj Singh
    "yuvi": "yuvraj singh", "sixer king": "yuvraj singh",

    # Ravindra Jadeja
    "sir jadeja": "ravindra jadeja", "sir": "ravindra jadeja",
    "rockstar": "ravindra jadeja", "jaddu": "ravindra jadeja",

    # Andre Russell
    "dre russ": "andre russell", "russell": "andre russell",

    # KL Rahul
    "rahul": "kl rahul",

    # Ajinkya Rahane
    "rahane": "ajinkya rahane",

    # Glenn Maxwell
    "the big show": "glenn maxwell", "maxwell": "glenn maxwell",

    # Sachin Tendulkar
    "master blaster": "sachin tendulkar", "little master": "sachin tendulkar",
    "god of cricket": "sachin tendulkar", "sachin": "sachin tendulkar",

    # Sourav Ganguly
    "dada": "sourav ganguly", "prince of kolkata": "sourav ganguly",
    "bengal tiger": "sourav ganguly", "ganguly": "sourav ganguly",

    # Rahul Dravid
    "the wall": "rahul dravid", "jammy": "rahul dravid",
    "mr dependable": "rahul dravid", "dravid": "rahul dravid",

    # Harbhajan Singh
    "bhajji": "harbhajan singh", "turbanator": "harbhajan singh",

    # Yuzvendra Chahal
    "chahal": "yuzvendra chahal", "yuzi": "yuzvendra chahal",

    # Rishabh Pant
    "pant": "rishabh pant",

    # Sanju Samson
    "samson": "sanju samson",

    # Jos Buttler
    "buttler": "jos buttler",

    # Faf du Plessis
    "faf": "faf du plessis",

    # Quinton de Kock
    "qdk": "quinton de kock",

    # Shreyas Iyer
    "sarpanch saab": "shreyas iyer", "iyer": "shreyas iyer",

    # Kagiso Rabada
    "kg": "kagiso rabada", "rabada": "kagiso rabada",

    # Rashid Khan
    "rashid": "rashid khan",

    # Mohammed Shami
    "shami": "mohammed shami",

    # Mohammed Siraj
    "siraj": "mohammed siraj",

    # Shakib Al Hasan
    "shakib": "shakib al hasan",

    # Shahid Afridi
    "boom boom afridi": "shahid afridi", "boom boom": "shahid afridi",
    "afridi": "shahid afridi",

    # Dale Steyn
    "steyn gun": "dale steyn", "steyn": "dale steyn",

    # Lasith Malinga
    "slinga malinga": "lasith malinga", "malinga": "lasith malinga",

    # Kane Williamson
    "kane train": "kane williamson", "williamson": "kane williamson",

    # Trent Boult
    "boult": "trent boult",

    # Mitchell Starc
    "starc": "mitchell starc",

    # Kieron Pollard
    "pollard": "kieron pollard",

    # Dwayne Bravo
    "champion": "dwayne bravo", "dj bravo": "dwayne bravo", "bravo": "dwayne bravo",
}

# Registry automatic not matched players mapping
MANUAL_OVERRIDES = {
    "krunal pandya": "KH Pandya",
    "hardik pandya": "HH Pandya",  


}


KNOWN_WICKETKEEPERS = [
    "MS Dhoni", "RR Pant", "SV Samson", "Q de Kock", "KD Karthik",
    "WP Saha", "AT Rayudu", "JC Buttler", "N Pooran", "PA Patel",
    "RV Uthappa", "MS Bisla", "KC Sangakkara", "AC Gilchrist", "BJ Haddin"
]




def resolve_player_name(input_name):
    lower_input = input_name.lower().strip()

    if lower_input in NICKNAME_MAP:
        lower_input = NICKNAME_MAP[lower_input]

  
    if lower_input in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[lower_input]

    if lower_input in FULL_NAME_TO_UNIQUE:
        return FULL_NAME_TO_UNIQUE[lower_input]

    if input_name in ALL_PLAYERS:
        return input_name

    match = process.extractOne(input_name, ALL_PLAYERS)
    if match and match[1] >= 70:
        return match[0]

    return None





# wickets which is not taken by bowlers

NOT_BOWLER_WICKET = ['run out', 'retired hurt', 'retired out', 'obstructing the field']



# TOOLS


@tool
def get_player_stats(player_name):
    """IPL player ke career batting stats deta hai - runs, matches, average, strike rate, fours, sixes, fifties, hundreds. Player ka naam kisi bhi format mein de sakte ho."""
    resolved_name = resolve_player_name(player_name)
    if resolved_name is None:
        return "Player nahi mila. Sahi naam ya spelling try karo."

    player_df = df[df['batter'] == resolved_name]

    total_runs = player_df['runs_batter'].sum()
    total_matches = player_df['match_id'].nunique()
    total_balls = player_df['balls_faced'].sum()

    out_count = df[df['player_out'] == resolved_name]['match_id'].nunique()
    average = total_runs / out_count if out_count > 0 else total_runs
    strike_rate = (total_runs / total_balls * 100) if total_balls > 0 else 0

    # Fours aur sixes - runs_not_boundary False 
    boundaries = player_df[player_df['runs_not_boundary'] == False]
    fours = (boundaries['runs_batter'] == 4).sum()
    sixes = (boundaries['runs_batter'] == 6).sum()

    # 50s aur 100s 
    match_scores = player_df.groupby('match_id')['runs_batter'].sum()
    fifties = ((match_scores >= 50) & (match_scores < 100)).sum()
    hundreds = (match_scores >= 100).sum()

    balls_bowled = df[df['bowler'] == resolved_name].shape[0]
    balls_batted = total_balls

    result = resolved_name + " ke batting stats:\n"
    result = result + "Matches: " + str(total_matches) + "\n"
    result = result + "Runs: " + str(total_runs) + "\n"
    result = result + "Average: " + str(round(average, 2)) + "\n"
    result = result + "Strike rate: " + str(round(strike_rate, 2)) + "\n"
    result = result + "Fours: " + str(int(fours)) + "\n"
    result = result + "Sixes: " + str(int(sixes)) + "\n"
    result = result + "Fifties: " + str(int(fifties)) + "\n"
    result = result + "Hundreds: " + str(int(hundreds))

    if balls_bowled > balls_batted:
        result = result + "\n\nNote: " + resolved_name + " primarily ek bowler hai - batting stats limited hain."

    return result


@tool
def get_bowler_stats(player_name):
    """IPL bowler ke career stats deta hai - total wickets, matches, economy rate, average. Player ka naam kisi bhi format mein de sakte ho (full name, nickname, ya scorecard format)."""
    resolved_name = resolve_player_name(player_name)
    if resolved_name is None:
        return "Player nahi mila. Sahi naam ya spelling try karo."

    bowler_df = df[df['bowler'] == resolved_name]

    if len(bowler_df) == 0:
        return resolved_name + " ne koi ball bowl nahi ki hai dataset mein."

    total_matches = bowler_df['match_id'].nunique()

    is_illegal_delivery = bowler_df['extra_type'].str.contains('wides|noballs', na=False)
    valid_balls_df = bowler_df[~is_illegal_delivery]
    total_balls = len(valid_balls_df)

    total_runs_conceded = bowler_df['runs_total'].sum()

    wickets_df = bowler_df[bowler_df['wicket_kind'].notna() & (~bowler_df['wicket_kind'].isin(NOT_BOWLER_WICKET))]
    total_wickets = len(wickets_df)

    total_overs = total_balls / 6

    economy = total_runs_conceded / total_overs if total_overs > 0 else 0
    average = total_runs_conceded / total_wickets if total_wickets > 0 else 0
    strike_rate = total_balls / total_wickets if total_wickets > 0 else 0

    result = resolved_name + " ke bowling stats:\n"
    result = result + "Matches: " + str(total_matches) + "\n"
    result = result + "Wickets: " + str(total_wickets) + "\n"
    result = result + "Economy: " + str(round(economy, 2)) + "\n"
    result = result + "Bowling average: " + str(round(average, 2)) + "\n"
    result = result + "Bowling strike rate: " + str(round(strike_rate, 2))

    return result


@tool
def predict_match(team1, team2):
    """Do IPL teams ke beech match ka analysis deta hai - overall win %, head-to-head record. Team ka pura naam do jaise Mumbai Indians, Chennai Super Kings."""
    matches_df = pd.read_csv(MATCHES_PATH, parse_dates=['date'])

    team1_matches = matches_df[(matches_df['team1'] == team1) | (matches_df['team2'] == team1)]
    team2_matches = matches_df[(matches_df['team1'] == team2) | (matches_df['team2'] == team2)]

    if len(team1_matches) == 0 or len(team2_matches) == 0:
        return "Team ka naam sahi se check karo. Jaise: Mumbai Indians, Chennai Super Kings"

    team1_wins = team1_matches[team1_matches['match_won_by'] == team1].shape[0]
    team1_win_pct = team1_wins / len(team1_matches) * 100

    team2_wins = team2_matches[team2_matches['match_won_by'] == team2].shape[0]
    team2_win_pct = team2_wins / len(team2_matches) * 100

    h2h = matches_df[
        ((matches_df['team1'] == team1) & (matches_df['team2'] == team2)) |
        ((matches_df['team1'] == team2) & (matches_df['team2'] == team1))
    ]

    team1_h2h_wins = h2h[h2h['match_won_by'] == team1].shape[0]
    total_h2h = len(h2h)

    result = team1 + " vs " + team2 + "\n"
    result = result + team1 + " overall win %: " + str(round(team1_win_pct, 1)) + "\n"
    result = result + team2 + " overall win %: " + str(round(team2_win_pct, 1))

    if total_h2h > 0:
        result = result + "\nHead to head: " + str(total_h2h) + " match hue, " + team1 + " ne " + str(team1_h2h_wins) + " jeete"
    else:
        result = result + "\nIn dono teams ka koi head-to-head record nahi mila"

    return result


@tool
def get_player_form_trend(player_name):
    """Player ka recent BATTING form trend batata hai (runs ke basis pe) - upward, downward, ya stable. Sirf regular batsmen ke liye kaam karta hai, bowlers ke liye nahi."""
    resolved_name = resolve_player_name(player_name)
    if resolved_name is None:
        return "Player nahi mila. Sahi naam ya spelling try karo."

    balls_bowled = df[df['bowler'] == resolved_name].shape[0]
    balls_batted = df[df['batter'] == resolved_name]['balls_faced'].sum()

    # Bowler check 
    if balls_bowled > balls_batted:
        return resolved_name + " primarily ek BOWLER hai. Ye tool sirf batting form track karta hai, isliye is player ke liye applicable nahi hai."

    player_matches = df[df['batter'] == resolved_name].groupby('match_id')['runs_batter'].sum()

    if len(player_matches) < 10:
        return resolved_name + " ke paas itna batting data nahi hai (kam matches khele hain) form trend nikalne ke liye."

    last_10 = player_matches.tail(10).values
    recent_5_avg = last_10[-5:].mean()
    previous_5_avg = last_10[:5].mean()
    diff = recent_5_avg - previous_5_avg

    if diff > 5:
        trend = "upward"
    elif diff < -5:
        trend = "downward"
    else:
        trend = "stable"

    result = resolved_name + " ka BATTING form trend: " + trend + "\n"
    result = result + "Pichle 5 match ka batting average: " + str(round(recent_5_avg, 1)) + " runs\n"
    result = result + "Usse pehle 5 match ka batting average: " + str(round(previous_5_avg, 1)) + " runs"

    return result


@tool
def search_player_career(player_name):
    """Player ke career ki descriptive summary deta hai (career span, teams, highest score, fifties/hundreds). Player ka naam kisi bhi format mein de sakte ho (nickname, full name, scorecard format)."""
    resolved_name = resolve_player_name(player_name)
    if resolved_name is None:
        return "Player nahi mila. Sahi naam ya spelling try karo."

    result = player_collection.get(ids=[resolved_name])

    if not result['documents']:
        return resolved_name + " ka career summary available nahi hai."

    return result['documents'][0]

@tool
def get_top_players(category, season=None, top_n=5):
    """Top players ki leaderboard deta hai. category 'runs' ya 'wickets' ho sakti hai. season optional hai (jaise 2023) - agar nahi diya toh poore career ka data. top_n batata hai kitne players dikhane hain (default 5)."""
    data = df.copy()
    if season is not None:
        data = data[data['season'] == int(season)]

    if len(data) == 0:
        return "Is season ka koi data nahi mila."

    if category == "runs":
        leaderboard = data.groupby('batter')['runs_batter'].sum().sort_values(ascending=False).head(top_n)
        result = "Top " + str(top_n) + " run scorers"
        if season:
            result = result + " (Season " + str(season) + ")"
        result = result + ":\n"
        for i, (player, runs) in enumerate(leaderboard.items(), 1):
            result = result + str(i) + ". " + player + " - " + str(runs) + " runs\n"
        return result

    elif category == "wickets":
        wickets_df = data[data['wicket_kind'].notna() & (~data['wicket_kind'].isin(NOT_BOWLER_WICKET))]
        leaderboard = wickets_df.groupby('bowler').size().sort_values(ascending=False).head(top_n)
        result = "Top " + str(top_n) + " wicket takers"
        if season:
            result = result + " (Season " + str(season) + ")"
        result = result + ":\n"
        for i, (player, wickets) in enumerate(leaderboard.items(), 1):
            result = result + str(i) + ". " + player + " - " + str(wickets) + " wickets\n"
        return result

    else:
        return "category sirf 'runs' ya 'wickets' ho sakti hai."


@tool
def compare_players(player1, player2):
    """Do batsmen ke career stats compare karta hai side by side. Player naam kisi bhi format mein de sakte ho (full name, nickname, ya scorecard format)."""
    resolved1 = resolve_player_name(player1)
    resolved2 = resolve_player_name(player2)

    if resolved1 is None or resolved2 is None:
        return "Ek ya dono players nahi mile. Sahi naam check karo."

    def get_stats(name):
        pdf = df[df['batter'] == name]
        runs = pdf['runs_batter'].sum()
        matches = pdf['match_id'].nunique()
        balls = pdf['balls_faced'].sum()
        outs = df[df['player_out'] == name]['match_id'].nunique()
        avg = runs / outs if outs > 0 else runs
        sr = (runs / balls * 100) if balls > 0 else 0
        return matches, runs, avg, sr

    m1, r1, a1, s1 = get_stats(resolved1)
    m2, r2, a2, s2 = get_stats(resolved2)

    result = resolved1 + " vs " + resolved2 + ":\n\n"
    result = result + resolved1 + " - Matches: " + str(m1) + ", Runs: " + str(r1) + ", Average: " + str(round(a1, 2)) + ", Strike Rate: " + str(round(s1, 2)) + "\n"
    result = result + resolved2 + " - Matches: " + str(m2) + ", Runs: " + str(r2) + ", Average: " + str(round(a2, 2)) + ", Strike Rate: " + str(round(s2, 2))

    return result


@tool
def get_team_stats(team_name):
    """Kisi IPL team ka overall performance deta hai - total matches, wins, win %. Team ka pura naam do jaise Mumbai Indians, Chennai Super Kings."""
    matches_df = pd.read_csv(MATCHES_PATH, parse_dates=['date'])

    team_matches = matches_df[(matches_df['team1'] == team_name) | (matches_df['team2'] == team_name)]

    if len(team_matches) == 0:
        return "Team ka naam sahi se check karo. Jaise: Mumbai Indians, Chennai Super Kings"

    total_matches = len(team_matches)
    total_wins = team_matches[team_matches['match_won_by'] == team_name].shape[0]
    win_pct = (total_wins / total_matches) * 100
    seasons_played = team_matches['season'].nunique()

    result = team_name + " ka overall record:\n"
    result = result + "Total matches: " + str(total_matches) + "\n"
    result = result + "Wins: " + str(total_wins) + "\n"
    result = result + "Win %: " + str(round(win_pct, 1)) + "\n"
    result = result + "Seasons played: " + str(seasons_played)

    return result


@tool
def get_season_winner(season=None):
    """IPL season ka champion (winner) team batata hai. season diya jaye (jaise 2023) toh us season ka winner, agar nahi diya toh saare seasons ke winners ki list."""
    matches_df = pd.read_csv(MATCHES_PATH, parse_dates=['date'])
    
    if season is not None:
        season = int(season)
        # Us season ka aakhri match (final) dhoondo - sabse last date wala
        season_matches = matches_df[matches_df['season'] == season]
        if len(season_matches) == 0:
            return "Is season ka data nahi mila."
        final_match = season_matches.sort_values('date').iloc[-1]
        return "IPL " + str(season) + " ka winner: " + final_match['match_won_by']
    else:
        # Har season ka final match (winner) nikaalo
        result = "IPL Season-wise Winners:\n"
        for s in sorted(matches_df['season'].unique()):
            season_matches = matches_df[matches_df['season'] == s]
            final_match = season_matches.sort_values('date').iloc[-1]
            result = result + str(s) + ": " + final_match['match_won_by'] + "\n"
        return result


@tool
def get_venue_stats(venue_name):
    """Kisi venue/stadium ka overall record deta hai - kitne matches hue, batting-first win %, average score. Venue ka naam batao jaise Wankhede, Eden Gardens, Chinnaswamy."""
    matches_df = pd.read_csv(MATCHES_PATH, parse_dates=['date'])

    # Fuzzy match venue name bhi (kyunki user shayad poora naam na de)
    all_venues = matches_df['venue'].unique().tolist()
    venue_match = process.extractOne(venue_name, all_venues)

    if venue_match is None or venue_match[1] < 60:
        return "Venue nahi mila. Sahi naam try karo."

    matched_venue = venue_match[0]
    venue_matches = matches_df[matches_df['venue'] == matched_venue]

    total_matches = len(venue_matches)
    batting_first_wins = (venue_matches['batting_team'] == venue_matches['match_won_by']).sum()
    batting_first_win_pct = (batting_first_wins / total_matches * 100) if total_matches > 0 else 0

    # Average total score is venue pe (pehli innings)
    venue_match_ids = venue_matches['match_id'].tolist()
    first_innings = df[(df['match_id'].isin(venue_match_ids)) & (df['innings'] == 1)]
    avg_score = first_innings.groupby('match_id')['team_runs'].max().mean()

    result = matched_venue + " ka record:\n"
    result = result + "Total matches: " + str(total_matches) + "\n"
    result = result + "Batting first win %: " + str(round(batting_first_win_pct, 1)) + "\n"
    result = result + "Average 1st innings score: " + str(round(avg_score, 1))

    return result



@tool
def get_player_stats_filtered(player_name, season=None, against_team=None):
    """Player ke batting stats deta hai, optionally kisi specific season ke liye ya kisi specific team ke against. season jaise 2016, against_team jaise 'Mumbai Indians'. Dono optional hain - agar dono None hain, poora career ka data milega (get_player_stats jaisa)."""
    resolved_name = resolve_player_name(player_name)
    if resolved_name is None:
        return "Player nahi mila. Sahi naam ya spelling try karo."

    player_df = df[df['batter'] == resolved_name]

    filter_desc = resolved_name
    if season is not None:
        player_df = player_df[player_df['season'] == int(season)]
        filter_desc = filter_desc + " (Season " + str(season) + ")"
    if against_team is not None:
        player_df = player_df[player_df['bowling_team'] == against_team]
        filter_desc = filter_desc + " (vs " + against_team + ")"

    if len(player_df) == 0:
        return filter_desc + " ke liye is filter ke sath koi data nahi mila."

    total_runs = player_df['runs_batter'].sum()
    total_matches = player_df['match_id'].nunique()
    total_balls = player_df['balls_faced'].sum()

    out_count = df[
        (df['player_out'] == resolved_name) &
        (df['match_id'].isin(player_df['match_id'].unique()))
    ]['match_id'].nunique()
    average = total_runs / out_count if out_count > 0 else total_runs
    strike_rate = (total_runs / total_balls * 100) if total_balls > 0 else 0

    result = filter_desc + " ke stats:\n"
    result = result + "Matches: " + str(total_matches) + "\n"
    result = result + "Runs: " + str(total_runs) + "\n"
    result = result + "Average: " + str(round(average, 2)) + "\n"
    result = result + "Strike rate: " + str(round(strike_rate, 2))

    return result


@tool
def get_best_xi(season=None):
    """Best playing XI generate karta hai stats ke basis pe - openers, middle order, wicketkeeper, all-rounders, bowlers. season diya jaye (jaise 2023) toh us season ka best XI, nahi toh all-time best XI."""
    
    data = df.copy()
    if season is not None:
        data = data[data['season'] == int(season)]
        if len(data) == 0:
            return "Is season ka data nahi mila."

    # Har player ka batting summary
    batting = data.groupby('batter').agg(
        runs=('runs_batter', 'sum'),
        matches=('match_id', 'nunique'),
        balls=('balls_faced', 'sum'),
        avg_pos=('bat_pos', 'mean')
    ).reset_index()
    batting['strike_rate'] = (batting['runs'] / batting['balls'] * 100).fillna(0)
    batting = batting[batting['matches'] >= 5]  # minimum matches filter

    # Har player ka bowling summary
    NOT_BOWLER_WICKET = ['run out', 'retired hurt', 'retired out', 'obstructing the field']
    wickets_data = data[data['wicket_kind'].notna() & (~data['wicket_kind'].isin(NOT_BOWLER_WICKET))]
    bowling = data.groupby('bowler').agg(matches=('match_id', 'nunique')).reset_index()
    wicket_counts = wickets_data.groupby('bowler').size().reset_index(name='wickets')
    bowling = bowling.merge(wicket_counts, on='bowler', how='left')
    bowling['wickets'] = bowling['wickets'].fillna(0)
    bowling = bowling[bowling['matches'] >= 5]

    selected = []
    result_lines = []

    # 1. Wicketkeeper - best batting average among known keepers
    keeper_candidates = batting[batting['batter'].isin(KNOWN_WICKETKEEPERS)].sort_values('runs', ascending=False)
    if len(keeper_candidates) > 0:
        keeper = keeper_candidates.iloc[0]['batter']
        selected.append(keeper)
        result_lines.append("Wicketkeeper: " + keeper)

    # 2. Openers (bat_pos <= 2) - top 2 by runs
    openers = batting[(batting['avg_pos'] <= 2) & (~batting['batter'].isin(selected))].sort_values('runs', ascending=False).head(2)
    for _, row in openers.iterrows():
        selected.append(row['batter'])
        result_lines.append("Opener: " + row['batter'])

    # 3. Middle order (bat_pos 3-6) - top 3 by runs
    middle = batting[(batting['avg_pos'] > 2) & (batting['avg_pos'] <= 6) & (~batting['batter'].isin(selected))].sort_values('runs', ascending=False).head(3)
    for _, row in middle.iterrows():
        selected.append(row['batter'])
        result_lines.append("Middle order: " + row['batter'])

    # 4. All-rounders -  runs AND wickets, not already selected
    combined = batting.merge(bowling, left_on='batter', right_on='bowler', how='inner')
    all_rounders = combined[(combined['runs'] >= 200) & (combined['wickets'] >= 10) & (~combined['batter'].isin(selected))]
    all_rounders = all_rounders.sort_values(['runs', 'wickets'], ascending=False).head(2)
    for _, row in all_rounders.iterrows():
        selected.append(row['batter'])
        result_lines.append("All-rounder: " + row['batter'] + " (" + str(int(row['runs'])) + " runs, " + str(int(row['wickets'])) + " wickets)")

    # 5. Bowlers - top by wickets, not already selected
    remaining_slots = 11 - len(selected)
    bowlers = bowling[~bowling['bowler'].isin(selected)].sort_values('wickets', ascending=False).head(remaining_slots)
    for _, row in bowlers.iterrows():
        selected.append(row['bowler'])
        result_lines.append("Bowler: " + row['bowler'] + " (" + str(int(row['wickets'])) + " wickets)")

    header = "Best All-Time XI:" if season is None else "Best XI for Season " + str(season) + ":"
    result = header + "\n" + "\n".join(result_lines)
    result = result + "\n\nNote: Wicketkeeper ek chhoti curated list se select kiya gaya hai (dataset mein role explicitly nahi diya hota)."

    return result