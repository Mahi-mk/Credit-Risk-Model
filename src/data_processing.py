import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

# ==========================================
# 1. DYNAMIC PATH SETUP
# ==========================================
# This finds the root "Credit-Risk-Model" folder automatically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'data.csv')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Create the processed directory if it doesn't exist
if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)
    print(f"--- Created folder: {PROCESSED_DIR} ---")

# Check if raw data exists
if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(f"CRITICAL ERROR: data.csv not found at {RAW_DATA_PATH}")

# Load the dataset
df = pd.read_csv(RAW_DATA_PATH)
print(f"--- Successfully loaded {len(df)} rows ---")

# ==========================================
# 2. TASK 3: FEATURE ENGINEERING
# ==========================================
print("--- Starting Feature Engineering ---")

# A. Aggregate Features per Customer (Frequency and Monetary)
customer_agg = df.groupby('CustomerId').agg({
    'Amount': ['sum', 'mean', 'std', 'count'],
    'Value': ['sum', 'mean']
})
customer_agg.columns = ['_'.join(col).strip() for col in customer_agg.columns.values]
customer_agg.reset_index(inplace=True)
customer_agg.fillna(0, inplace=True)

# B. Extract Time-Based Features
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
df['TransactionHour'] = df['TransactionStartTime'].dt.hour
df['TransactionDay'] = df['TransactionStartTime'].dt.day
df['TransactionMonth'] = df['TransactionStartTime'].dt.month

# C. One-Hot Encoding for Categorical Variables
categorical_cols = ['ProductCategory', 'ChannelId']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cats = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))
df = pd.concat([df, encoded_df], axis=1)

# D. Standardization
scaler = StandardScaler()
df[['Amount_scaled', 'Value_scaled']] = scaler.fit_transform(df[['Amount', 'Value']])

# Merge Aggregates back
df = df.merge(customer_agg, on='CustomerId', how='left')

# ==========================================
# 3. TASK 4: RISK LABELING (RFM PROXY)
# ==========================================
print("--- Creating Risk Proxy (RFM Clustering) ---")

# Recency Calculation
reference_date = df['TransactionStartTime'].max()
rfm = df.groupby('CustomerId').agg({
    'TransactionStartTime': lambda x: (reference_date - x.max()).days,
    'TransactionId': 'count',
    'Amount': 'sum'
}).rename(columns={
    'TransactionStartTime': 'Recency',
    'TransactionId': 'Frequency',
    'Amount': 'Monetary'
})

# K-Means Clustering to identify High-Risk customers
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm[['Recency', 'Frequency', 'Monetary']])

# Identify the "High Risk" cluster (lowest average Monetary/Frequency)
cluster_stats = rfm.groupby('Cluster')[['Frequency', 'Monetary']].mean()
high_risk_cluster_id = cluster_stats.sort_values(by='Monetary').index[0]
rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster_id).astype(int)

# Merge proxy label into main dataframe
df = df.merge(rfm[['is_high_risk']], on='CustomerId', how='left')

# ==========================================
# 4. SAVE FINAL DATA
# ==========================================
output_file = os.path.join(PROCESSED_DIR, 'final_model_data.csv')
df.to_csv(output_file, index=False)

print(f"--- SUCCESS ---")
print(f"Final dataset saved to: {output_file}")
print(f"High risk labels generated: {df['is_high_risk'].sum()} samples flagged.")