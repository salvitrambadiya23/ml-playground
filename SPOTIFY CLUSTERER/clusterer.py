import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


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


if __name__ == "__main__":
    df = load_data()
    print(df.shape)

    features = select_features(df)
    print("\nFeatures shape:", features.shape)
    print(features.head(3))

    scaled, scaler = scale_features(features)
    print("\nFirst scaled row:", scaled[0])