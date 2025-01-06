import yfinance as yf
import pandas as pd
import numpy as np
import talib

# def prepare_data(symbol, period='5y'):
#     try:
#         # Download data
#         df = pd.DataFrame(yf.download(symbol, period=period))
        
#         if df.empty:
#             raise ValueError(f"No data found for symbol {symbol}")

#         # Calculate technical indicators
#         # RSI
#         df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
        
#         # MACD
#         macd, macd_signal, _ = talib.MACD(df['Close'].values)
#         df['MACD'] = macd
#         df['MACD_Signal'] = macd_signal
        
#         # Bollinger Bands
#         upper, middle, lower = talib.BBANDS(df['Close'].values)
#         df['BB_Upper'] = upper
#         df['BB_Middle'] = middle
#         df['BB_Lower'] = lower
        
#         # Moving averages
#         df['SMA_20'] = talib.SMA(df['Close'].values, timeperiod=20)
#         df['SMA_50'] = talib.SMA(df['Close'].values, timeperiod=50)
        
#         # Drop any NaN values
#         df = df.dropna()

#         print(df.head())
        
#         return df
        
#     except Exception as e:
#         print(f"Error downloading data for {symbol}: {str(e)}")
#         print("Please check your internet connection and verify the stock symbol.")
#         return None
    

def prepare_data(symbol, period='5y'):
    # Download data
    ticker = yf.Ticker(symbol)

    ticker_data = ticker.history(period='5y')

    df = pd.DataFrame(ticker_data)
    
    # Calculate technical indicators
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA_ratio'] = df['MA20'] / df['MA50']
    
    # Volatility
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
    
    # Drop NaN values
    df = df.dropna()
    
    
    return df

def prepare_multiple_data(symbols, period='5y'):
    all_data = {}
    for symbol in symbols:
        # Download data for each symbol
        ticker = yf.Ticker(symbol)
        ticker_data = ticker.history(period=period)
        df = pd.DataFrame(ticker_data)
        
        # Calculate technical indicators
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA_ratio'] = df['MA20'] / df['MA50']
        
        # Volatility
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
        
        # Drop NaN values
        df = df.dropna()
        all_data[symbol] = df
    
    return all_data