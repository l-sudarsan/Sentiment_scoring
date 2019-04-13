# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 19:05:10 2019

@author: sudarsan
"""

#https://textblob.readthedocs.io/en/dev/
#https://github.com/cjhutto/vaderSentiment
#https://github.com/explosion/spaCy/blob/master/examples/deep_learning_keras.py

#Vader sentiment
import pandas as pd

df = pd.read_csv('C:\\Users\\IBM_ADMIN\\Desktop\\IBM\\Projects\\2019\\scraping\\Michel_Sarran-Toulouse_Haute_Garonne_Occitanie__en.csv')
df.columns

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

sentences = df.review_body
for sentence in sentences:
    vs = analyzer.polarity_scores(sentence)
    print("{:-<65} {}".format(sentence, str(vs)))


review = []
vs_compound = []
vs_pos = []
vs_neu = []
vs_neg = []

for i in range(0, len(df.review_body)):
    review.append(df['review_body'][i])
    vs_compound.append(analyzer.polarity_scores(df['review_body'][i])['compound'])
    vs_pos.append(analyzer.polarity_scores(df['review_body'][i])['pos'])
    vs_neu.append(analyzer.polarity_scores(df['review_body'][i])['neu'])
    vs_neg.append(analyzer.polarity_scores(df['review_body'][i])['neg'])
    
from pandas import Series, DataFrame

review_senti_df = DataFrame({'Review': review,
                        'Compound': vs_compound,
                        'Positive': vs_pos,
                        'Neutral': vs_neu,
                        'Negative': vs_neg})
review_senti_df = review_senti_df[['Review', 'Compound',
                         'Positive', 'Neutral', 'Negative']]


from textblob import TextBlob

df.review_body = df.review_body.astype(str)

df['sentiment'] = df['review_body'].apply(lambda rev: TextBlob(rev).sentiment)

    
    