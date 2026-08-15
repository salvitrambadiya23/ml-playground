import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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


def visualize_clusters(scaled_features, clusters, sample_size=5000):
    """Reduce features to 2D with PCA and plot clusters as a scatter chart."""
    idx = np.random.choice(len(scaled_features), sample_size, replace=False)
    sample_features = scaled_features[idx]
    sample_clusters = clusters[idx]

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(sample_features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced[:, 0], reduced[:, 1],
        c=sample_clusters, cmap='tab10', alpha=0.5, s=10
    )
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Spotify Song Clusters (PCA 2D projection)')
    plt.legend(*scatter.legend_elements(), title='Cluster')
    plt.savefig('cluster_visualization.png')
    print("\nSaved chart to cluster_visualization.png")


def predict_cluster(song_features, scaler, km):
    """Given a dict of raw audio features, predict which cluster the song belongs to."""
    feature_cols = [
        'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'time_signature'
    ]
    ordered_values = [song_features[col] for col in feature_cols]
    df_temp = pd.DataFrame([ordered_values], columns=feature_cols)
    scaled = scaler.transform(df_temp)
    cluster = km.predict(scaled)[0]
    return cluster


if __name__ == "__main__":
    df = load_data()
    features = select_features(df)
    scaled, scaler = scale_features(features)

    print("\nFitting K-Means with K=6 on full dataset (this may take a minute)...")
    km, clusters = fit_kmeans(scaled, k=6)

    features['cluster'] = clusters
    print("\nSongs per cluster:\n", features['cluster'].value_counts())

    summary_cols = ['danceability', 'energy', 'valence', 'acousticness', 'tempo', 'loudness']
    print("\nAverage feature values per cluster:")
    print(features.groupby('cluster')[summary_cols].mean().round(2))

    visualize_clusters(scaled, clusters)

    # test: predict cluster for a made-up upbeat dance song
    test_song = {
        'danceability': 0.8, 'energy': 0.85, 'key': 5, 'loudness': -5.0,
        'mode': 1, 'speechiness': 0.05, 'acousticness': 0.02,
        'instrumentalness': 0.0, 'liveness': 0.1, 'valence': 0.9,
        'tempo': 128, 'time_signature': 4
    }
    predicted = predict_cluster(test_song, scaler, km)
    print(f"\nTest song predicted cluster: {predicted}")