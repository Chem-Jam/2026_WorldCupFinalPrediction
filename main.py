# importing dependences
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as stats

# 1 Load  filtered dataset
df = pd.read_csv("spain_argentina_competitive.csv")
df['date'] = pd.to_datetime(df['date'])
print(df['date'])

# Filter out matches where tournament type is 'Friendly'
competitive_matches_df = df[df['tournament'] != 'Friendly']
# Save cleaned dataset
competitive_matches_df.to_csv("spain_argentina_competitive.csv", index=False)

# Drop any rows with missing scores just in case
df = df.dropna(subset=['home_score', 'away_score'])
print (df)
print(f"Removed friendlies, New dataset has {len(competitive_matches_df)} highly competitive matches.")
