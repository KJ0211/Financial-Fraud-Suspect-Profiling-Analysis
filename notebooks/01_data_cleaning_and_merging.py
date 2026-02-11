import pandas as pd
import numpy as np

# --- STEP 1: DATA LOADING & PREPARATION ---
# Load datasets [cite: 1, 5]
df_suspects = pd.read_csv('suspects.csv')
df_transactions = pd.read_csv('transactions.csv')

# Convert date column to datetime objects [cite: 1]
df_transactions['date'] = pd.to_datetime(df_transactions['date'])

# Check for missing values [cite: 2]
print(f"Missing values in transactions:\n{df_transactions.isnull().sum()}")

# --- STEP 2: STATISTICAL OUTLIER DETECTION ---
# Using the IQR (Interquartile Range) method [cite: 2]
Q1 = df_transactions['amount'].quantile(0.25)
Q3 = df_transactions['amount'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_transactions[(df_transactions['amount'] < lower_bound) | 
                           (df_transactions['amount'] > upper_bound)]

print(f"Outlier Limits: {lower_bound:.2f} to {upper_bound:.2f}")
print(f"Statistical outliers found: {len(outliers)}") [cite: 2, 3]

# --- STEP 3: FEATURE ENGINEERING & MERGING ---
# Aggregate financial behavior per suspect [cite: 3]
suspect_financials = df_transactions.groupby('suspect_id').agg({
    'amount': 'sum',            # Total spending
    'transaction_id': 'count'   # Number of transactions
}).rename(columns={'amount': 'total_spent', 'transaction_id': 'num_transactions'})

# Create master table and handle suspects with no history [cite: 3, 4]
df_master = pd.merge(df_suspects, suspect_financials, on='suspect_id', how='left')
df_master['total_spent'] = df_master['total_spent'].fillna(0)
df_master['num_transactions'] = df_master['num_transactions'].fillna(0)

# Calculate Spending Intensity (Average cost per transaction) [cite: 6]
df_master['spending_intensity'] = df_master['total_spent'] / df_master['num_transactions']

# --- STEP 4: MOTIVE ANALYSIS & PIVOTING ---
# Pivot data to see spending categories per suspect 
pivot_transactions = df_transactions.pivot_table(
    index='suspect_id',
    columns='category',
    values='amount',
    aggfunc='sum'
).fillna(0)

# Identify suspects with high-value transactions (> 2000) 
high_value_ids = df_transactions[df_transactions['amount'] > 2000]['suspect_id'].unique().tolist()

# Identify suspects with abnormal intensity (> 100) [cite: 6]
abnormal_intensity_ids = df_master[df_master['spending_intensity'] > 100].index.tolist()

# Combine lists for a comprehensive motive shortlist [cite: 6]
final_motive_ids = list(set(high_value_ids + abnormal_intensity_ids))

# --- STEP 5: FINAL EXPORT ---
# Filter for final suspects and save results 
df_shortlist = df_master[df_master['num_transactions'] > 20]
motive_suspects = df_master[df_master.index.isin(final_motive_ids)]

df_master.to_csv('master_suspects.csv', index=False)
motive_suspects.to_csv('motive_suspects.csv', index=False)

print(f"Analysis Complete. Found {len(motive_suspects)} suspects with strong motives.")
