import pandas as pd

# 1. Load the core dataset file (pointing to the archive folder)
df = pd.read_csv("all_matches.csv")

# 2. Make sure the date column is handled as proper dates
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# 3. Filter for matches from the last 8 years (2018 to 2026)
recent_era_df = df[df['date'].dt.year >= 2018]

# 4. Match if Spain OR Argentina played as home OR away
spain_or_argentina = recent_era_df[
    (recent_era_df['home_team'] == 'Spain') | 
    (recent_era_df['away_team'] == 'Spain') | 
    (recent_era_df['home_team'] == 'Argentina') | 
    (recent_era_df['away_team'] == 'Argentina')
]

# 5. Export this comprehensive dataset into your brand new file
spain_or_argentina.to_csv("spain_argentina_8years.csv", index=False)

print(f"Extraction successful! Created 'spain_argentina_8years.csv' with {len(spain_or_argentina)} total matches.")
