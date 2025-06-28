import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def load_data(file_path):
    """Load and preprocess raw dataset."""
    df = pd.read_csv(file_path)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

def calculate_rfm(df, snapshot_date=None):
    """Calculate RFM metrics for each customer."""
    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max()
    
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    })
    return rfm

def create_proxy_variable(rfm, n_clusters=3, random_state=42):
    """Create proxy variable using K-Means clustering on RFM."""
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    high_risk_cluster = cluster_summary['Monetary'].idxmin()
    rfm['is_high_risk'] = rfm['Cluster'].apply(lambda x: 1 if x == high_risk_cluster else 0)
    return rfm

def engineer_features(df):
    """Engineer features for modeling using a pipeline."""
    # Aggregate features
    agg_features = df.groupby('CustomerId').agg({
        'Amount': ['sum', 'mean', 'std', 'count'],
        'TransactionStartTime': ['min', 'max']
    }).reset_index()
    agg_features.columns = ['CustomerId', 'TotalAmount', 'AvgAmount', 'StdAmount', 'TransactionCount', 'FirstTransaction', 'LastTransaction']
    
    # Time-based features
    agg_features['TransactionSpan'] = (agg_features['LastTransaction'] - agg_features['FirstTransaction']).dt.days
    agg_features.drop(['FirstTransaction', 'LastTransaction'], axis=1, inplace=True)
    
    # Merge with RFM
    rfm = calculate_rfm(df)
    rfm = create_proxy_variable(rfm)
    features = agg_features.merge(rfm[['Recency', 'Frequency', 'Monetary', 'is_high_risk']], on='CustomerId')
    
    # Categorical features
    categorical = df.groupby('CustomerId').agg({
        'ProductCategory': lambda x: x.mode()[0],
        'ChannelId': lambda x: x.mode()[0]
    }).reset_index()
    features = features.merge(categorical, on='CustomerId')
    
    # Define pipeline
    numeric_features = ['TotalAmount', 'AvgAmount', 'StdAmount', 'TransactionCount', 'TransactionSpan', 'Recency', 'Frequency', 'Monetary']
    categorical_features = ['ProductCategory', 'ChannelId']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])
    
    return features, preprocessor

if __name__ == "__main__":
    # Example usage
    df = load_data('data/raw/TrainingData.csv')
    features, preprocessor = engineer_features(df)
    features.to_csv('data/processed/processed_features.csv', index=False)