import yfinance as yf
import pandas as pd

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
    print(df.head())
    
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