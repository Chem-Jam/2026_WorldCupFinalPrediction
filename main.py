# importing dependences
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as stats

# 1 Load  filtered dataset
df = pd.read_csv("spain_argentina_competitive.csv")
df['date'] = pd.to_datetime(df['date'])
#print(df['date']) 

# Filter out matches where tournament type is 'Friendly'
df = df[df['tournament'] != 'Friendly']
# Save cleaned dataset
df.to_csv("spain_argentina_competitive.csv", index=False)

# Drop any rows with missing scores just in case
df = df.dropna(subset=['home_score', 'away_score'])
#print (df)
#print(f"Removed friendlies, New dataset has {len(df)} highly competitive matches.")

# 2. Define the Likelihood function for the Poisson model
def calculate_nll(xi, data):
    """ Calculates Negative Log-Likelihood for a given xi decay factor.
    We want to minimize this value. """
    # Find the latest date to measure time distance
    max_date = data['date'].max()
    data['weeks_ago'] = (max_date - data['date']).dt.days / 7.0
    data['weight'] = np.exp(-xi * data['weeks_ago'])
    # Calculate weighted average goals across the dataset
    # This acts as the base rate for our lambda calculation
    weighted_avg_home = np.average(data['home_score'], weights=data['weight'])
    weighted_avg_away = np.average(data['away_score'], weights=data['weight'])
    # Target values to calculate error against
    home_goals = data['home_score'].values
    away_goals = data['away_score'].values
    weights = data['weight'].values
    
    # Calculate the log-likelihood of observing these goals given the averages
    # (Using basic averages here for grid search simplicity)
    home_log_pmf = stats.poisson.logpmf(home_goals, weighted_avg_home)
    away_log_pmf = stats.poisson.logpmf(away_goals, weighted_avg_away)
    
    # Multiply the log-likelihood by the match weights
    weighted_nll = -np.sum(weights * (home_log_pmf + away_log_pmf))
    
    return weighted_nll

# 3. Execution of the Expanded Grid Search
# We expand the upper bound from 0.015 to 0.060 to find the true mathematical minimum
xi_candidates = np.linspace(0.001, 0.060, 60)
best_xi = None
lowest_nll = float('inf')

print("Starting Expanded Grid Search for optimal time-decay parameter (xi)...")
print("-" * 50)

for xi in xi_candidates:
    current_nll = calculate_nll(xi, df.copy())
    print(f"Testing xi: {xi:.4f} -> Negative Log-Likelihood: {current_nll:.2f}")
    
    if current_nll < lowest_nll:
        lowest_nll = current_nll
        best_xi = xi

print("-" * 50)
print(f" Grid Search is Complete")
print(f"The mathematically optimal xi decay factor is: **{best_xi:.4f}**")
print("-" * 50)

# 4. Apply the optimal xi factor (or override with empirical baseline 0.0070 if desired)
# XI_FINAL = best_xi
XI_FINAL = 0.0070  # Lock to international empirical benchmark to prevent infinite decay slide

max_date = df['date'].max()
df['weeks_ago'] = (max_date - df['date']).dt.days / 7.0
df['weight'] = np.exp(-XI_FINAL * df['weeks_ago'])

# Calculate baseline metrics for the competitive ecosystem
global_avg_home_goals = np.average(df['home_score'], weights=df['weight'])
global_avg_away_goals = np.average(df['away_score'], weights=df['weight'])
global_base = (global_avg_home_goals + global_avg_away_goals) / 2

# Isolate individual team records to compute strengths
spain_home = df[df['home_team'] == 'Spain']
spain_away = df[df['away_team'] == 'Spain']
arg_home = df[df['home_team'] == 'Argentina']
arg_away = df[df['away_team'] == 'Argentina']

# Calculate Weighted Attacking Averages
spain_att = (np.sum(spain_home['home_score'] * spain_home['weight']) + np.sum(spain_away['away_score'] * spain_away['weight'])) / (np.sum(spain_home['weight']) + np.sum(spain_away['weight']))
arg_att = (np.sum(arg_home['home_score'] * arg_home['weight']) + np.sum(arg_away['away_score'] * arg_away['weight'])) / (np.sum(arg_home['weight']) + np.sum(arg_away['weight']))

# Calculate Weighted Defensive Averages (Goals Conceded)
spain_def = (np.sum(spain_home['away_score'] * spain_home['weight']) + np.sum(spain_away['home_score'] * spain_away['weight'])) / (np.sum(spain_home['weight']) + np.sum(spain_away['weight']))
arg_def = (np.sum(arg_home['away_score'] * arg_home['weight']) + np.sum(arg_away['home_score'] * arg_away['weight'])) / (np.sum(arg_home['weight']) + np.sum(arg_away['weight']))

# 5. Compute Expected Goals (Lambdas) for a Neutral Ground Match
lambda_spain = (spain_att / global_base) * (arg_def / global_base) * global_base
lambda_argentina = (arg_att / global_base) * (spain_def / global_base) * global_base

print(f"Expected Goals - Spain: {lambda_spain:.2f}")
print(f"Expected Goals - Argentina: {lambda_argentina:.2f}")
print("-" * 50)

# 6. Generate Bivariate Score Matrix & Match Outcomes
max_goals = 6
score_matrix = np.zeros((max_goals, max_goals))

for i in range(max_goals):  # Spain goals
    for j in range(max_goals):  # Argentina goals
        score_matrix[i, j] = stats.poisson.pmf(i, lambda_spain) * stats.poisson.pmf(j, lambda_argentina)

spain_win = np.sum(np.tril(score_matrix, -1))
argentina_win = np.sum(np.triu(score_matrix, 1))
draw = np.sum(np.diag(score_matrix))

print(f" 2026 World Cup Simulation Results:")
print(f" Spain Win Probability: {spain_win * 100:.2f}%")
print(f" Argentina Win Probability: {argentina_win * 100:.2f}%")
print(f" Draw Probability (Going to Extra Time): {draw * 100:.2f}%")