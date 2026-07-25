# Movie Recommender (Content-Based, KNN)

A content-based movie recommendation system built with K-Nearest Neighbors.
Given a movie title, it recommends 5 similar movies based on genre, plot
overview, cast, and director — not user ratings.

## How it works

1. Movie metadata (genres, keywords, cast, director, overview) is merged
   and cleaned from the TMDB 5000 dataset.
2. All metadata is combined into a single "tags" string per movie.
3. Tags are vectorized using `CountVectorizer` (bag-of-words, top 5000
   words, English stopwords removed).
4. A `NearestNeighbors` model (cosine distance) finds the 5 movies whose
   tag vectors are closest to the queried movie.

## Example output
Enter a movie title (or 'quit' to exit): avatar

Because you liked 'avatar', you might like:

Titan A.E.
Small Soldiers
Independence Day
Ender's Game
Aliens vs Predator: Requiem
## Dataset

[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
(Kaggle). Download `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` and
place them in the `data/` folder before running.

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

## Usage

```bash
python recommender.py
```

Enter any movie title from the dataset (e.g. `Avatar`, `The Dark Knight
Rises`, `Spectre`) and get 5 similar recommendations. Type `quit` to exit.

## What I learned

- How KNN works as a distance-based algorithm — no real "training" step,
  just similarity search at prediction time
- Feature engineering on messy nested/JSON-like text data using
  `ast.literal_eval`
- Why cosine distance is preferred over Euclidean for text-based
  similarity (direction/pattern of the vector matters more than magnitude)
- Turning text into numeric vectors with `CountVectorizer` (bag-of-words)
- Structuring an ML project into clean, reusable functions instead of one
  big script

## Possible improvements

- Swap `CountVectorizer` for `TfidfVectorizer` to weigh rare/distinctive
  words higher than common ones
- Add a collaborative filtering version using the MovieLens ratings
  dataset, to compare against this content-based approach
- Wrap this in a simple Streamlit UI instead of a terminal loop
- Add error handling for close-but-not-exact title matches (e.g. suggest
  "did you mean...?")
  