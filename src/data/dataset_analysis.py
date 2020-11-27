# Modules Import
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from preprocessing import set_sentiment
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# Draw and save histogram on Reviews Ratings data
def ratings_hist(reviews_df: pd.DataFrame) -> None:
    df = set_sentiment(reviews_df)
    sns.set_style('white')
    plt.figure(1)
    sns.histplot(data=df, x='Rating', bins=5, discrete=True, palette=[
                 'Red', 'Gray', 'Blue'], hue='Sentiment', hue_order=['Negative', 'Neutral', 'Positive'])
    plt.title('Reviews Ratings', fontsize=14)
    plt.xlabel('Rating', fontsize=12)
    plt.ylabel('Number of Ratings', fontsize=12)
    fig = plt.gcf()
    fig.savefig('./data/plots/ratings_hist.png')

    plt.figure(2)
    sns.histplot(data=df, x='Class', bins=3, palette=[
        'Red', 'Gray', 'Blue'], hue='Sentiment', hue_order=['Negative', 'Neutral', 'Positive'], discrete=True)
    plt.xticks([-1,0,1])
    plt.title('Sentiments Distribution', fontsize=14)
    plt.xlabel('Sentiment Polarity Score', fontsize=12)
    plt.ylabel('Reviews', fontsize=12)
    fig = plt.gcf()
    fig.savefig('./data/plots/sentiments_hist.png')


# # Create wordcloud for the different sentiment classes
# def sentiment_wordclouds(reviews_df: pd.DataFrame) -> None:
#     try:
#         sentiments = [df[df['Sentiment'] == 'Positive'],
#                       df[df['Sentiment'] == 'Neutral'], df[df['Sentiment'] == 'Negative']]
#     except:
#         df = set_sentiment(reviews_df)
#         sentiments = [df[df['Sentiment'] == 'Positive'],
#                       df[df['Sentiment'] == 'Neutral'], df[df['Sentiment'] == 'Negative']]

#     for sentiment in sentiments:
#         sent_type = sentiment['Sentiment'].value_counts().idxmax()
#         text = ' '.join(review for review in sentiment['Review'])
#         stopwords = set(STOPWORDS)
#         wordcloud = WordCloud(
#             background_color='white',
#             stopwords=stopwords,
#             max_words=100,
#             max_font_size=40,
#             scale=3
#         ).generate(text)
#         plt.figure(3, figsize=(12, 12))
#         plt.imshow(wordcloud, interpolation='bilinear')
#         plt.axis("off")
#         plt.savefig(f'./data/plots/wordcloud_{sent_type}.png')


if __name__ == '__main__':
    df = pd.read_csv('./data/raw/trip_reviews.csv')
    ratings_hist(df)
    # sentiment_wordclouds(pd.read_csv('./data/processed/clean.csv'))
    print(df.head(5))
