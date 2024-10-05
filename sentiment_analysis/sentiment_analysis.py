# sentiment_analysis.py

import json
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def sentiment_analysis (text):

    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)

    if sentiment_scores['compound'] > 0:
        response = "Positive sentiment"
    elif sentiment_scores['compound'] < 0:
        response = "Negative sentiment"
    else:
        response = "Neutral sentiment"
    return response
  
# Define the input text
text = "I really enjoyed the movie. The acting was great and the plot was interesting."
result= sentiment_analysis(text)
print(result)