from GoogleNews import GoogleNews
from textblob import TextBlob

def fetch_stock_news(ticker):
    googlenews = GoogleNews(lang='en')
    googlenews.search(f"{ticker} stock news")
    news_list = googlenews.get_texts()
    return news_list

def analyze_sentiment(news_list):
    sentiment_scores = []
    
    for news in news_list:
        analysis = TextBlob(news)
        
        if analysis.sentiment.polarity > 0:
            sentiment_scores.append('Positive')
        elif analysis.sentiment.polarity == 0:
            sentiment_scores.append('Neutral')
        else:
            sentiment_scores.append('Negative')
    
    return sentiment_scores

if __name__ == "__main__":
    ticker = "AAPL"  # Apple Inc.
    print(f"Fetching news for {ticker}")
    
    news_list = fetch_stock_news(ticker)
    sentiment_scores = analyze_sentiment(news_list)
    
    for news, sentiment in zip(news_list, sentiment_scores):
        print(f"News: {news} \nSentiment: {sentiment}\n")
