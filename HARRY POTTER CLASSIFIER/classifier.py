import pandas as pd


def load_data():
    """Load the Harry Potter dialogue script into a dataframe."""
    df = pd.read_csv('data/hp_script.csv', encoding='latin-1')
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.shape)
    print(df.columns.tolist())
    print(df.head(10))
    print("\nCharacter line counts:\n", df['character_name'].value_counts().head(15))