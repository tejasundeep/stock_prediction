import gradio as gr
import yfinance as yf

# Fetch historical stock data using yfinance
def fetch_stock_data(ticker, period="24d", interval="1d"):
    try:
        stock_data = yf.Ticker(ticker)
        hist_data = stock_data.history(period=period, interval=interval)
        if hist_data.empty:
            return "Failed to fetch historical data."
        return hist_data
    except Exception as e:
        return f"An error occurred: {e}"

# Analyze buy/sell sentiment based on closing prices and volume
def sentiment__analysis(hist_data):
    buy_sentiment = 0
    sell_sentiment = 0

    for i in range(len(hist_data)):
        if hist_data['Close'].iloc[i] > hist_data['Open'].iloc[i]:
            buy_sentiment += hist_data['Volume'].iloc[i]
        elif hist_data['Close'].iloc[i] < hist_data['Open'].iloc[i]:
            sell_sentiment += hist_data['Volume'].iloc[i]

    return buy_sentiment, sell_sentiment

def analyze_stock(ticker, period="24d", interval="1d"):
    hist_data = fetch_stock_data(ticker, period, interval)
    
    if isinstance(hist_data, str):
        return hist_data
    
    buy_sentiment, sell_sentiment = sentiment__analysis(hist_data)
    sentiment = f"Buy Sentiment: {buy_sentiment}, Sell Sentiment: {sell_sentiment}"

    if buy_sentiment > sell_sentiment:
        sentiment += "\nThe market sentiment is bullish."
    elif sell_sentiment > buy_sentiment:
        sentiment += "\nThe market sentiment is bearish."
    else:
        sentiment += "\nThe market sentiment is neutral."

    return sentiment

iface = gr.Interface(
    fn=analyze_stock,
    title="Stock Sentiment - Analyzer",
    inputs=[
        gr.Textbox(label="Stock Ticker"),
        gr.Dropdown(choices=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"], label="Time Period"),
        gr.Dropdown(choices=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"], label="Interval")
    ],
    outputs=gr.Textbox(label="Analysis Result"),
    css="footer {visibility: hidden}"
)

iface.launch()
