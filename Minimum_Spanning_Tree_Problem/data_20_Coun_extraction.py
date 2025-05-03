import pandas as pd
import os

# 1. Top‐N list exactly as they appear in your CSV’s COUNTRY/COUNTERPART_COUNTRY columns
#    (Note: Taiwan was not found in the dataset, so we omit it here.)
top_countries = [
    'United States',
    "China, People's Republic of",
    'Germany',
    'Netherlands, The',
    'Japan',
    'United Kingdom',
    'France',
    'Korea, Republic of',
    'India',
    'Italy',
    'Belgium',
    'Singapore',
    "Hong Kong Special Administrative Region, People's Republic of China",
    'Canada',
    'Mexico',
    'United Arab Emirates',
    'Spain',
    'Ireland',
    'Switzerland'
]

# 2. Correct path to your IMF IMTS CSV
FILE_PATH = r'C:\Users\joonc\My_github\My_projects\Minimum_Spanning_Tree_Problem\Data\dataset_2025-05-02T15_45_55.505889336Z_DEFAULT_INTEGRATION_IMF.STA_IMTS_1.0.0.csv'
assert os.path.exists(FILE_PATH), f"File not found: {FILE_PATH}"

# 3. Load only the needed columns to save memory
use_cols = ['COUNTRY', 'COUNTERPART_COUNTRY', 'FREQUENCY', 'INDICATOR', '2022']
df = pd.read_csv(FILE_PATH, usecols=use_cols, low_memory=False)

# 4. Filter for 2022 annual FOB exports between our selected countries
df_filtered = df[
    (df['FREQUENCY'] == 'Annual') &
    (df['INDICATOR'].str.contains('Exports of goods', na=False)) &
    (df['COUNTRY'].isin(top_countries)) &
    (df['COUNTERPART_COUNTRY'].isin(top_countries))
]

# 5. Pivot into an N×N matrix on the '2022' column
matrix_2022 = df_filtered.pivot_table(
    index='COUNTRY',
    columns='COUNTERPART_COUNTRY',
    values='2022',
    aggfunc='sum'
).reindex(index=top_countries, columns=top_countries).fillna(0)

# 6. Display and save
print(matrix_2022)
OUTPUT_PATH = r'C:\Users\joonc\My_github\My_projects\Minimum_Spanning_Tree_Problem\Figures\bilateral_trade_19x19_2022.csv'
matrix_2022.to_csv(OUTPUT_PATH)
print(f"Saved {len(top_countries)}×{len(top_countries)} matrix to {OUTPUT_PATH}")
