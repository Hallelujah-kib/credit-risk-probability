import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import silhouette_score

def load_data(file_path):
    """Load and preprocess raw dataset."""
    df = pd.read_csv(file_path)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

def calculate_rfm(df, snapshot_date=None):
    """Calculate RFM metrics for each customer, handling negative amounts."""
    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max()
    
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': lambda x: x[x > 0].sum() if x[x > 0].sum() > 0 else -x[x < 0].sum()
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    })
    return rfm

def create_proxy_variable(rfm, n_clusters=5, risk_percentile=0.25, random_state=42):
    """Create proxy variable using K-Means clustering and percentile-based threshold."""
    # Scale features for clustering
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    # Determine optimal number of clusters using silhouette score
    silhouette_scores = []
    for n in range(2, 6):
        kmeans = KMeans(n_clusters=n, random_state=random_state)
        kmeans.fit(rfm_scaled)
        silhouette_scores.append(silhouette_score(rfm_scaled, kmeans.labels_))
    optimal_clusters = range(2, 6)[np.argmax(silhouette_scores)] + 1  # Add 1 to get the best n
    
    # Apply K-Means with optimal clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=random_state)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Use percentile-based threshold for high-risk
    monetary_threshold = rfm['Monetary'].quantile(risk_percentile)
    rfm['is_high_risk'] = (rfm['Monetary'] <= monetary_threshold).astype(int)
    
    # Validate balance
    high_risk_count = rfm['is_high_risk'].sum()
    total_customers = len(rfm)
    print(f"High-risk customers: {high_risk_count} ({high_risk_count/total_customers:.2%} of {total_customers})")
    
    return rfm

def handle_outliers(df, column):
    """Cap outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

def engineer_features(df):
    """Engineer features for modeling using a pipeline."""
    # Aggregate features
    agg_features = df.groupby('CustomerId').agg({
        'Amount': ['sum', 'mean', 'std', 'count'],
        'TransactionStartTime': ['min', 'max'],
        'ProductCategory': lambda x: x.mode()[0] if not x.mode().empty else 'unknown',
        'ChannelId': lambda x: x.mode()[0] if not x.mode().empty else 'unknown'
    }).reset_index()
    agg_features.columns = ['CustomerId', 'TotalAmount', 'AvgAmount', 'StdAmount', 'TransactionCount',
                           'FirstTransaction', 'LastTransaction', 'ProductCategory', 'ChannelId']
    
    # Time-based features
    agg_features['TransactionSpan'] = (agg_features['LastTransaction'] - agg_features['FirstTransaction']).dt.days
    agg_features.drop(['FirstTransaction', 'LastTransaction'], axis=1, inplace=True)
    
    # Handle outliers
    agg_features = handle_outliers(agg_features, 'TotalAmount')
    agg_features = handle_outliers(agg_features, 'AvgAmount')
    agg_features['StdAmount'] = agg_features['StdAmount'].fillna(0)
    
    # Merge with RFM
    rfm = calculate_rfm(df)
    rfm = create_proxy_variable(rfm, risk_percentile=0.25)  # Increased to 25% for better balance
    features = agg_features.merge(rfm[['Recency', 'Frequency', 'Monetary', 'is_high_risk']], on='CustomerId')
    
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
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features)
        ])
    
    return features, preprocessor

if __name__ == "__main__":
    df = load_data('data/raw/TrainingData.csv')
    features, preprocessor = engineer_features(df)
    features.to_csv('data/processed/processed_features.csv', index=False)
    print(f"Processed features saved to ../data/processed/processed_features.csv")