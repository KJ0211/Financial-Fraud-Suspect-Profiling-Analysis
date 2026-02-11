import pandas as pd
import numpy as np
df_ms = pd.read_csv('master_suspects.csv')
df_transactions = pd.read_csv('transactions.csv')
print('----master suspects info----')
print(df_ms.head())
# Assuming your transaction data is called df_transactions
print(df_transactions['amount'].describe())

median_total_spent = df_ms['total_spent'].median()
#this is a calculation so we dont need to filter
mean_total_spent = df_ms['total_spent'].mean()
print(f'median total spent: {median_total_spent}')
print(f'mean total spent: {mean_total_spent}')
#mean is significantly higher than median, which means there is a big fraud or data input wrongly
top_10_transactions = df_transactions.sort_values(by='amount', ascending=False)

# Use .head(10) to display only the top 10
suspects_transactions = top_10_transactions[top_10_transactions['amount'] > 2000]#this is a filter so we need to specific
print(suspects_transactions.head(10))

suspects_with_motive = suspects_transactions.head(10)['suspect_id'].unique().tolist()

print("suspects list：", suspects_with_motive)
#%%
pivot_transactions = df_transactions.pivot_table(
    index='suspect_id',
    columns='category',
    values='amount',
    aggfunc='sum'
)

pivot_transactions = pivot_transactions.fillna(0)

print(pivot_transactions.head())

df_ms['spending_intensity'] = df_ms['total_spent'] / df_ms['num_transactions']
abnormal_values = df_ms[df_ms['spending_intensity'] > 100]
print('\n---abnormal values---')
print(abnormal_values.head())

new_ids = abnormal_values.index.tolist()

suspects_with_motive = list(set(suspects_with_motive + new_ids))

print(f"the numbers of suspects: {len(suspects_with_motive)}")
motive_suspects = df_ms[df_ms.index.isin(suspects_with_motive)]
motive_suspects.to_csv('motive_suspects.csv',index=False)
print("Master DataFrame successfully saved as 'motive_suspects.csv'.")
