import pandas as pd
import numpy as np
from scipy.stats import poisson

# Load  full modern era dataset to evaluate relative team strengths
df = pd.read_csv("all_matches.csv")
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Gauge baseline era stats (Last 8 years)
era_df = df[df['date'].dt.year >= 2018]
avg_home = era_df['home_score'].mean()
avg_away = era_df['away_score'].mean()

# Construct strength indices
home_stats = era_df.groupby('home_team').agg({'home_score': 'mean', 'away_score': 'mean'}).rename(
    columns={'home_score': 'att_home', 'away_score': 'def_home'}
)
away_stats = era_df.groupby('away_team').agg({'away_score': 'mean', 'home_score': 'mean'}).rename(
    columns={'away_score': 'att_away', 'home_score': 'def_away'}
)

def run_final_prediction(home, away):
    # Expected goals formula based on global baselines
    lamb_home = home_stats.loc[home, 'att_home'] * away_stats.loc[away, 'def_away'] / avg_away
    lamb_away = away_stats.loc[away, 'att_away'] * home_stats.loc[home, 'def_home'] / avg_home
    
    matrix = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            matrix[i, j] = poisson.pmf(i, lamb_home) * poisson.pmf(j, lamb_away)
            
    home_prob = np.sum(np.tril(matrix, -1))
    away_prob = np.sum(np.triu(matrix, 1))
    draw_prob = np.sum(np.diag(matrix))
    
    print("🏆 2026 WORLD CUP FINAL PROBABILITIES (90 Mins) 🏆")
    print(f"🇪🇸 Spain Regulation Win: {home_prob * 100:.1f}%")
    print(f"🇦🇷 Argentina Regulation Win: {away_prob * 100:.1f}%")
    print(f"🤝 Draw / Heading to Extra Time: {draw_prob * 100:.1f}%")

run_final_prediction('Spain', 'Argentina')