import tweepy
from textblob import TextBlob
import time
import pandas as pd

# Initialize Twitter API
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# Function to fetch tweets and analyze sentiment
def fetch_and_analyze_tweets(query):
    tweets = api.search(q=query, count=100)
    data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])

    # Add sentiment analysis
    data['Polarity'] = data['Tweets'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
    average_sentiment = data['Polarity'].mean()
    return average_sentiment

# Function to classify sentiment
def classify_sentiment(score):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

# Bot loop
while True:
    average_sentiment = fetch_and_analyze_tweets("Apple stock")
    sentiment_classification = classify_sentiment(average_sentiment)
    print(f"Sentiment for Apple stock right now: {sentiment_classification}")
    
    time.sleep(60)  # Update every 60 seconds
