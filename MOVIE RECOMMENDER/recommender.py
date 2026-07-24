import ast
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors


def load_data():
    """Load and merge the two TMDB CSVs into one dataframe."""
    movies = pd.read_csv('data/tmdb_5000_movies.csv')
    credits = pd.read_csv('data/tmdb_5000_credits.csv')
    merged = movies.merge(credits, on='title')
    return merged


def convert(text):
    """Extract 'name' field from a JSON-like string list, e.g. genres/keywords."""
    result = []
    real_list = ast.literal_eval(text)

    for item in real_list:
        result.append(item['name'])

    return result


def get_top_cast(text, limit=3):
    """Get only the top N actors, since minor cast doesn't matter much."""
    result = []
    real_list = ast.literal_eval(text)

    for i, item in enumerate(real_list):
        if i < limit:
            result.append(item['name'])

    return result


def get_director(text):
    """Find the crew member whose job is 'Director'."""
    real_list = ast.literal_eval(text)

    for item in real_list:
        if item['job'] == 'Director':
            return [item['name']]

    return []


def remove_spaces(lst):
    """Remove spaces inside names so 'Chris Evans' stays one token, not two."""
    return [i.replace(" ", "") for i in lst]


def engineer_features(movies):
    """Turn raw JSON-like columns into one clean 'tags' string per movie."""
    movies = movies.dropna(subset=['overview', 'genres', 'keywords', 'cast', 'crew']).copy()

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(get_top_cast)
    movies['crew'] = movies['crew'].apply(get_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    movies['genres'] = movies['genres'].apply(remove_spaces)
    movies['keywords'] = movies['keywords'].apply(remove_spaces)
    movies['cast'] = movies['cast'].apply(remove_spaces)
    movies['crew'] = movies['crew'].apply(remove_spaces)

    movies['tags'] = (
        movies['overview'] + movies['genres'] +
        movies['keywords'] + movies['cast'] + movies['crew']
    )

    new_df = movies[['movie_id', 'title', 'tags']].copy()
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())
    return new_df


def vectorize_tags(new_df):
    """Convert each movie's tags into a numeric vector using bag-of-words counts."""
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    return vectors, cv


def build_model(vectors):
    """Fit a KNN model that finds each movie's nearest neighbors by cosine distance."""
    knn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
    knn.fit(vectors)
    return knn


def recommend(movie_title, new_df, vectors, knn):
    """Print the 5 most similar movies to the given title."""
    matches = new_df[new_df['title'].str.lower() == movie_title.lower()]

    if matches.empty:
        print(f"'{movie_title}' not found in dataset.")
        return

    idx = matches.index[0]
    movie_vector = vectors[idx].reshape(1, -1)
    distances, indices = knn.kneighbors(movie_vector)

    print(f"\nBecause you liked '{movie_title}', you might like:\n")
    for i in indices[0][1:]:   # skip index 0 — that's the movie itself
        print("-", new_df.iloc[i]['title'])

if __name__ == "__main__":
    movies = load_data()
    new_df = engineer_features(movies)
    vectors, cv = vectorize_tags(new_df)
    knn = build_model(vectors)

    while True:
        title = input("\nEnter a movie title (or 'quit' to exit): ")
        if title.lower() == 'quit':
            break
        recommend(title, new_df, vectors, knn)