import pandas as pd
import numpy as np

# --- 1. Data Loading & Initial Exploration ---
df_ms = pd.read_csv('master_suspects.csv')
df_transactions = pd.read_csv('transactions.csv')

print('---- Master Suspects Metadata ----')
print(df_ms.head())

print('\n---- Transaction Amount Distribution ----')
print(df_transactions['amount'].describe())

# --- 2. Statistical Outlier Detection ---
# A high mean relative to the median suggests a right-skewed distribution 
# often caused by extreme outliers (potential fraud or data entry errors).
median_total_spent = df_ms['total_spent'].median()
mean_total_spent = df_ms['total_spent'].mean()

print(f'Median Total Spent: {median_total_spent}')
print(f'Mean Total Spent: {mean_total_spent}')

# --- 3. High-Value Transaction Filtering ---
# Identifying individual transactions that exceed a specific risk threshold ($2000)
top_10_transactions = df_transactions.sort_values(by='amount', ascending=False)
suspects_transactions = top_10_transactions[top_10_transactions['amount'] > 2000]

print('\n---- High-Value Suspect Transactions (>$2000) ----')
print(suspects_transactions.head(10))

# Extract unique suspect IDs from the top high-value transactions
suspects_with_motive = suspects_transactions.head(10)['suspect_id'].unique().tolist()

# --- 4. Behavioral Analysis via Pivot Tables ---
# Reshaping data to see spending habits across different categories per suspect
pivot_transactions = df_transactions.pivot_table(
    index='suspect_id',
    columns='category',
    values='amount',
    aggfunc='sum'
).fillna(0)

print('\n---- Category Spending per Suspect (Pivot) ----')
print(pivot_transactions.head())

# --- 5. Feature Engineering: Spending Intensity ---
# Calculating average value per transaction to identify "efficient" or high-burn spenders
df_ms['spending_intensity'] = df_ms['total_spent'] / df_ms['num_transactions']

# Identify suspects with an abnormally high spending intensity (>100 per transaction)
abnormal_values = df_ms[df_ms['spending_intensity'] > 100]

print('\n---- Abnormal Spending Intensity Detected ----')
print(abnormal_values.head())

# --- 6. Consolidating Findings & Export ---
# Merge the IDs from high-value transactions and high-intensity behavior
new_ids = abnormal_values.index.tolist()
suspects_with_motive = list(set(suspects_with_motive + new_ids))

print(f"\nTotal unique suspects identified for further investigation: {len(suspects_with_motive)}")

# Filter the master
