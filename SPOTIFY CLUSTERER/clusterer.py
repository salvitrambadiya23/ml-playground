import pandas as pd


def load_data():
    """Load the Spotify tracks dataset."""
    df = pd.read_csv('data/tracks.csv')
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.shape)
    print(df.columns.tolist())
    print(df.head(3))