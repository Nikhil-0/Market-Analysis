import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
from functools import wraps
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('stock_data')

# Import settings
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import settings

def retry_decorator(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    wait_time = delay * (2 ** retries)
                    logger.warning(f"Attempt {retries}/{max_retries} failed: {str(e)}. Retrying in {wait_time}s...")
                    if retries < max_retries:
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed.")
                        raise
            return None
        return wrapper
    return decorator

def get_stock_data(ticker, start_date, end_date, include_dividends=True):
    try:
        logger.info(f"Fetching stock data for {ticker} from {start_date} to {end_date}")
        stock_data = yf.download(ticker, start=start_date, end=end_date, actions=include_dividends)
        
        if not stock_data.empty:
            # Ensure the directory exists
            output_dir = settings.DATA_DIR
            os.makedirs(output_dir, exist_ok=True)
            
            # Flatten the multi-level columns if they exist
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in stock_data.columns]
            
            # Standardize all column names to lowercase
            stock_data.columns = [x.lower() for x in stock_data.columns]
            
            # Save the data to a CSV file
            file_path = os.path.join(output_dir, f'{ticker}_stock_data.csv')
            stock_data.to_csv(file_path)
            logger.info(f"Successfully fetched and saved stock data for {ticker}")
            return file_path
        else:
            logger.warning(f"No data found for ticker {ticker}")
            return None
    except Exception as e:
        logger.error(f"An error occurred while fetching stock data for {ticker}: {e}")
        raise

@retry_decorator(max_retries=settings.MAX_RETRIES, delay=settings.RETRY_DELAY)
def get_stock_data_with_retry(ticker, start_date=None, end_date=None, lookback_days=None):
    # Set default end date to today if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Calculate start date if lookback_days is provided
    if start_date is None:
        if lookback_days is None:
            lookback_days = settings.DEFAULT_LOOKBACK_DAYS
        start_date_obj = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=lookback_days)
        start_date = start_date_obj.strftime('%Y-%m-%d')
    
    return get_stock_data(ticker, start_date, end_date)

def add_technical_indicators(stock_data_path):
    if not os.path.exists(stock_data_path):
        logger.error(f"Stock data file {stock_data_path} not found")
        return None
    
    try:
        # Load the stock data
        df = pd.read_csv(stock_data_path, index_col=0, parse_dates=True)
        
        # Only proceed if technical indicators are enabled in settings
        if settings.TECHNICAL_INDICATORS['enable']:
            logger.info(f"Adding technical indicators to {stock_data_path}")
            
            # Moving Averages
            if 'SMA20' in settings.TECHNICAL_INDICATORS['indicators']:
                df['sma20'] = df['close'].rolling(window=20).mean()
            
            if 'SMA50' in settings.TECHNICAL_INDICATORS['indicators']:
                df['sma50'] = df['close'].rolling(window=50).mean()
            
            # Relative Strength Index (RSI)
            if 'RSI14' in settings.TECHNICAL_INDICATORS['indicators']:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                df['rsi14'] = 100 - (100 / (1 + rs))
            
            # Moving Average Convergence Divergence (MACD)
            if 'MACD' in settings.TECHNICAL_INDICATORS['indicators']:
                ema12 = df['close'].ewm(span=12, adjust=False).mean()
                ema26 = df['close'].ewm(span=26, adjust=False).mean()
                df['macd'] = ema12 - ema26
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            if 'BB' in settings.TECHNICAL_INDICATORS['indicators']:
                df['bb_middle'] = df['close'].rolling(window=20).mean()
                df['bb_std'] = df['close'].rolling(window=20).std()
                df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
                df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
            
            # Save the updated data
            df.to_csv(stock_data_path)
            logger.info(f"Technical indicators added and saved to {stock_data_path}")
            
        return stock_data_path
    
    except Exception as e:
        logger.error(f"Error adding technical indicators: {e}")
        return None

if __name__ == '__main__':
    # Example usage
    ticker_symbol = 'AAPL'
    
    # Use retry function with default lookback days
    print(f"Fetching stock data for {ticker_symbol} with automatic retries...")
    stock_file = get_stock_data_with_retry(ticker_symbol)
    
    if stock_file:
        print(f"Adding technical indicators...")
        add_technical_indicators(stock_file)
        print(f"Data processing complete! Results saved to: {stock_file}")
    else:
        print("Data collection failed after multiple attempts.")
