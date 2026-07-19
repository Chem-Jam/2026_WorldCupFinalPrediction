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
    