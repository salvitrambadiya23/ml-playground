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


if __name__ == "__main__":
    df = load_data()
    features = select_features(df)
    scaled, scaler = scale_features(features)

    print("\nFinding elbow point (this samples data, may take ~30-60 sec)...")
    find_elbow(scaled)