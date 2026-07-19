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
competitive_matches_df = df[df['tournament'] != 'Friendly']
# Save cleaned dataset
competitive_matches_df.to_csv("spain_argentina_competitive.csv", index=False)

# Drop any rows with missing scores just in case
df = df.dropna(subset=['home_score', 'away_score'])
#print (df)
#print(f"Removed friendlies, New dataset has {len(competitive_matches_df)} highly competitive matches.")

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

# 3. Execution of the Grid Search
# We will test xi values ranging from 0.001 to 0.015
xi_candidates = np.linspace(0.001, 0.015, 50)
best_xi = None
lowest_nll = float('inf')

print("Starting Grid Search for optimal time-decay parameter (xi)...")
print("-" * 50)

for xi in xi_candidates:
    current_nll = calculate_nll(xi, df.copy())
    print(f"Testing xi: {xi:.4f} -> Negative Log-Likelihood: {current_nll:.2f}")
    
    if current_nll < lowest_nll:
        lowest_nll = current_nll
        best_xi = xi

print("-" * 50)
print(f"🎉 Grid Search Complete!")
print(f"The mathematically optimal xi decay factor is: **{best_xi:.4f}**")