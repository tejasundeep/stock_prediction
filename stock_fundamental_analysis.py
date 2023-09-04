import yfinance as yf
import numpy as np
from typing import List, Optional, Dict, Union, Tuple

METRICS = [
    ('P/E Ratio', 'trailingPE'),
    ('P/B Ratio', 'priceToBook'),
    ('Dividend Yield', 'dividendYield'),
    ('Debt to Equity', 'debtToEquity'),
    ('Return on Assets', 'returnOnAssets'),
    ('Risk', 'beta'),
    ('EBITDA', 'ebitda'),
    ('Profitability', 'profitMargins'),
    ('Debt', 'totalDebt'),
    ('Free Cash Flow', 'freeCashflow'),
    ('Earnings Growth', 'earningsGrowth'),
    ('Book Value Per Share', 'bookValue'),
    ('Return on Equity', 'returnOnEquity'),
    ('Operating Cash Flow', 'operatingCashflow')
]

WEIGHTS = {
    'bullish': {
        'Earnings Growth': 0.20,
        'P/E Ratio': 0.15,
        'Return on Equity': 0.15,
        'Free Cash Flow': 0.10,
        'Profitability': 0.10,
        'Return on Assets': 0.08,
        'Risk': 0.05,
        'Book Value Per Share': 0.05,
        'Operating Cash Flow': 0.05,
        'Debt to Equity': 0.02
    },
    'bearish': {
        'Dividend Yield': 0.20,
        'Debt to Equity': 0.20,
        'P/B Ratio': 0.15,
        'Risk': 0.10,
        'Return on Assets': 0.08,
        'Free Cash Flow': 0.07,
        'Operating Cash Flow': 0.07,
        'Earnings Growth': 0.03
    }
}

def fetch_stock_data(ticker: str, metrics: List[Tuple[str, str]]) -> Optional[Dict[str, Union[float, None]]]:
    try:
        stock = yf.Ticker(ticker)
        stock_info = stock.info
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

    return {name: stock_info.get(metric, None) for name, metric in metrics}

def normalize_ratios(ratios_list: List[List[Union[float, None]]]) -> np.ndarray:
    ratios_array = np.array(ratios_list, dtype=object)
    n_features = ratios_array.shape[1]

    for i in range(n_features):
        column = ratios_array[:, i].astype(float)
        valid_values = column[~np.isnan(column)]

        if valid_values.size == 0:
            continue

        min_val, max_val = np.min(valid_values), np.max(valid_values)
        column[~np.isnan(column)] = 0.5 if min_val == max_val else (column[~np.isnan(column)] - min_val) / (max_val - min_val)
        ratios_array[:, i] = column

    return ratios_array.astype(float)

def calculate_score(ratios: Dict[str, Union[float, None]], weights: Dict[str, float]) -> float:
    available_metrics = [metric for metric, value in ratios.items() if value is not None and not np.isnan(value)]
    if not available_metrics:
        return 0.0

    total_weight = sum(weights.get(metric, 0) for metric in available_metrics)
    score = sum(ratios[metric] * weights.get(metric, 0) / total_weight for metric in available_metrics)

    return score

def pick_best_stocks(tickers: List[str], metrics: List[Tuple[str, str]], weights: Dict[str, float], top_n: int = 5) -> Union[List[Tuple[str, float]], str]:
    stock_scores = []
    ratios_list = []
    valid_tickers = []

    for ticker in tickers:
        ratios = fetch_stock_data(ticker, metrics)
        if ratios:
            ratios_list.append(list(ratios.values()))
            valid_tickers.append(ticker)

    if not ratios_list:
        return "No valid stocks found."

    normalized_ratios_list = normalize_ratios(ratios_list)
    for i, ticker in enumerate(valid_tickers):
        normalized_ratios = dict(zip([x[0] for x in metrics], normalized_ratios_list[i]))

        if all(np.isnan(list(normalized_ratios.values()))):
            continue

        score = calculate_score(normalized_ratios, weights)
        stock_scores.append((ticker, score))

    return sorted(stock_scores, key=lambda x: x[1], reverse=True)[:top_n]

# Main execution
if __name__ == "__main__":
    tickers = ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'TECHM.NS', 'HCLTECH.NS', 'ITC.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AAVAS.NS']
    top_n = 5
    best_stocks = pick_best_stocks(tickers, metrics=METRICS, weights=WEIGHTS['bullish'], top_n=top_n)

    if isinstance(best_stocks, str):
        print(best_stocks)
    else:
        print(f"Top {top_n} best-performing stocks:")
        for i, (stock, score) in enumerate(best_stocks):
            print(f"{i+1}. {stock} with a score of {score:.2f}")
