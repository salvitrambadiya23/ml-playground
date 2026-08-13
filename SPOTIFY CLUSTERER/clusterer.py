import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def load_data(filepath=None):
    """Load the Spotify tracks dataset."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), 'data', 'tracks.csv')
    df = pd.read_csv(filepath)
    return df


def select_features(df):
    """Keep only the numeric audio-feature columns relevant to clustering by sound."""
    feature_cols = [
        'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'time_signature'
    ]
    features = df[feature_cols].dropna()
    return features


def scale_features(features):
    """Scale all features to mean 0, std 1, so no single feature dominates distance."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, scaler


def find_elbow(scaled_features, max_k=10, sample_size=20000):
    """Test different K values on a sample and print inertia for each, to find the elbow point."""
    import numpy as np
    effective_sample_size = min(len(scaled_features), sample_size)
    sample_idx = np.random.choice(len(scaled_features), effective_sample_size, replace=False)
    sample = scaled_features[sample_idx]

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(sample)
        print(f"K={k}, inertia={km.inertia_:.0f}")


def fit_kmeans(scaled_features, k=6):
    """Fit K-Means on the full dataset with the chosen number of clusters."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = km.fit_predict(scaled_features)
    return km, clusters


if __name__ == "__main__":
    df = load_data()
    features = select_features(df)
    scaled, scaler = scale_features(features)

    print("\nFitting K-Means with K=6 on full dataset (this may take a minute)...")
    km, clusters = fit_kmeans(scaled, k=6)

    features['cluster'] = clusters
    print("\nSongs per cluster:\n", features['cluster'].value_counts())

    print("\nAverage feature values per cluster (this tells us what each cluster represents):")
    print(features.groupby('cluster').mean())