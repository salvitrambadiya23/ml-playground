import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def load_data():
    """Load the Harry Potter dialogue script into a dataframe."""
    df = pd.read_csv('data/hp_script.csv', encoding='latin-1')
    return df


def filter_characters(df):
    """Keep only rows from characters with enough dialogue to learn from."""
    main_characters = [
        'Harry Potter', 'Ron Weasley', 'Hermione Granger',
        'Rubeus Hagrid', 'Minerva McGonagall', 'Albus Dumbledore',
        'Vernon Dursley'
    ]
    filtered = df[df['character_name'].isin(main_characters)].copy()
    return filtered


def clean_text(text):
    """Lowercase text and strip out punctuation, keeping only words and spaces."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


def prepare_dataset(df):
    """Filter to main characters and clean their dialogue text."""
    filtered = filter_characters(df)
    filtered['clean_dialogue'] = filtered['dialogue'].apply(clean_text)
    return filtered
def split_data(dataset):
     """Split cleaned dialogue into train and test sets."""
     X = dataset['clean_dialogue']
     y = dataset['character_name']

     X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
     return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test):
    """Vectorize dialogue and train a Naive Bayes classifier."""
    cv = CountVectorizer()
    X_train_vec = cv.fit_transform(X_train)
    X_test_vec = cv.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nModel accuracy on test set: {accuracy:.2%}")

    return model, cv

if __name__ == "__main__":
    df = load_data()
    dataset = prepare_dataset(df)

    X_train, X_test, y_train, y_test = split_data(dataset)
    model, cv = train_model(X_train, X_test, y_train, y_test)