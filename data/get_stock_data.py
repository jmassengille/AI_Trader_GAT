# get_stock_data.py

import os
import pandas as pd
from alpha_vantage.timeseries import TimeSeries

# Replace 'YOUR_API_KEY' with your Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = 'YOUR_API_KEY'

def fetch_stock_data(symbol, output_csv):
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
    
    # The data contains 'date', '1. open', '2. high', '3. low', '4. close', '5. volume' columns
    data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    # Sort the data by date in ascending order
    data = data.sort_values('date')

    # Save the data to a CSV file
    data.to_csv(output_csv, index=False)

if __name__ == '__main__':
    # Replace 'AAPL' with the desired stock symbol (e.g., 'AAPL' for Apple Inc.)
    stock_symbol = 'AAPL'
    
    # Replace 'data/raw/stock_data.csv' with the path where you want to save the CSV file
    output_file = 'data/raw/stock_data.csv'

    if not os.path.exists('data/raw'):
        os.makedirs('data/raw')

    fetch_stock_data(stock_symbol, output_file)
