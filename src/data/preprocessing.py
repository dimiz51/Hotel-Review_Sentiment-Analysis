# Modules Import
import pandas as pd
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Match rating scale 1-5 with corresponding sentiment


def set_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df['Sentiment'] = df['Rating'].apply(lambda rating: 'Neutral' if rating == 3 else (
        'Negative' if rating < 3 else 'Positive'))
    df['Class'] = df['Sentiment'].apply(
        lambda sentiment: 1 if sentiment == 'Positive' else (0 if sentiment == 'Neutral' else -1))
    return df

# Dataset Cleaning routines


def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['Review'])
    df['Review'] = df['Review'].apply(lambda text: text.translate(str.maketrans('', '', string.punctuation)))
    df['Review'] = df['Review'].apply(lambda text: remove_stopwords(text))
    df['Review'] = df['Review'].apply(lambda text: remove_nums_spaces(text))
    return df

# Remove numbers/multiple spaces from reviews text as they produce a great number of useless tokens when transforming the data with tfidf 
def remove_nums_spaces(review_text: str):
    filtered_sequence = re.sub(pattern = r"\d+",repl=" ", string = review_text)
    filtered_sequence = re.sub(pattern = r"\s\s+",repl= " ",string=filtered_sequence)
    return filtered_sequence

# Remove words without further processing value for our models


def remove_stopwords(review_text: str):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(review_text)
    filtered_tokens = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = ' '.join([w.lower() for w in filtered_tokens])
    return filtered_sentence

# Export pre-processed dataset to a csv file ready for feature generation


def export_clean_data(df: pd.DataFrame) -> None:
    df.to_csv('./data/processed/clean.csv', index=False)


if __name__ == '__main__':
    df = set_sentiment(pd.read_csv('./data/raw/trip_reviews.csv'))
    df = data_cleaning(df)
    export_clean_data(df)
