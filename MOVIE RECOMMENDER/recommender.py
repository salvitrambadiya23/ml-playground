import pandas as pd


def load_data():
    """Load and merge the two TMDB CSVs into one dataframe."""
    movies = pd.read_csv('data/tmdb_5000_movies.csv')
    credits = pd.read_csv('data/tmdb_5000_credits.csv')

    print("movies shape:", movies.shape)
    print("credits shape:", credits.shape)

    merged = movies.merge(credits, on='title')
    print("merged shape:", merged.shape)

    return merged


if __name__ == "__main__":
    movies = load_data()
    print("\nColumns:\n", movies.columns.tolist())
    print("\nFirst row:\n", movies.iloc[0])