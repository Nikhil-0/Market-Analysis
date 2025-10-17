import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("market_analysis.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def setup_data_dir():
    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def get_stock_data(ticker, start_date=None, end_date=None, period="1y"):
    try:
        logger.info(f"Downloading stock data for {ticker}")
        
        if start_date and end_date:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
        else:
            stock_data = yf.download(ticker, period=period)
        
        # Reset index to make Date a column
        stock_data = stock_data.reset_index()
        
        # Rename Date column to lowercase for consistency
        stock_data = stock_data.rename(columns={"Date": "date"})
        
        # Save to CSV
        data_dir = setup_data_dir()
        file_path = os.path.join(data_dir, f"{ticker}_stock_data.csv")
        stock_data.to_csv(file_path, index=False)
        
        logger.info(f"Stock data saved to {file_path}")
        return stock_data
        
    except Exception as e:
        logger.error(f"Error downloading stock data: {str(e)}")
        return None

def get_news_data(ticker):
    try:
        logger.info(f"Getting news data for {ticker}")
        
        # Import here to avoid circular imports
        from market_analysis.data_collection.news_scraper import scrape_finviz_news
        
        news_data = scrape_finviz_news(ticker)
        
        if news_data.empty:
            logger.warning(f"No news data found for {ticker}")
        else:
            logger.info(f"Retrieved {len(news_data)} news articles for {ticker}")
            
        return news_data
        
    except Exception as e:
        logger.error(f"Error getting news data: {str(e)}")
        return pd.DataFrame()

def analyze_sentiment(ticker):
    try:
        logger.info(f"Analyzing sentiment for {ticker}")
        
        # Import here to avoid circular imports
        from market_analysis.data_analysis.sentiment_analyzer import process_news_data
        
        sentiment_df, daily_sentiment_df = process_news_data(ticker)
        
        if sentiment_df.empty:
            logger.warning(f"No sentiment data generated for {ticker}")
        else:
            logger.info(f"Sentiment analysis completed for {len(sentiment_df)} news items")
            
        return sentiment_df, daily_sentiment_df
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def run_dashboard():
    try:
        logger.info("Starting dashboard")
        
        # Import and run the dashboard using create_dashboard API
        from market_analysis.visualization.dashboard import create_dashboard
        # Prefer project data directory
        project_root = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
        data_dir = os.path.join(project_root, 'data')
        create_dashboard(data_dir=data_dir, debug=True)
        
    except Exception as e:
        logger.error(f"Error launching dashboard: {str(e)}")
        print(f"Error: {str(e)}")
        print("Make sure all required packages are installed:")
        print("  pip install dash plotly pandas yfinance nltk")
        print("  python -c \"import nltk; nltk.download('vader_lexicon')\"")

def main():
    parser = argparse.ArgumentParser(description="Market Analysis Tool")
    
    parser.add_argument("--ticker", type=str, help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--get-stock", action="store_true", help="Download stock data")
    parser.add_argument("--get-news", action="store_true", help="Download news data")
    parser.add_argument("--analyze", action="store_true", help="Analyze sentiment")
    parser.add_argument("--dashboard", action="store_true", help="Launch dashboard")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    
    args = parser.parse_args()
    
    # Set default ticker
    ticker = args.ticker if args.ticker else "AAPL"
    
    # Create data directory
    setup_data_dir()
    
    # Process based on args
    if args.get_stock or args.all:
        get_stock_data(ticker, args.start_date, args.end_date)
        
    if args.get_news or args.all:
        get_news_data(ticker)
        
    if args.analyze or args.all:
        analyze_sentiment(ticker)
        
    if args.dashboard or args.all:
        run_dashboard()
    
    # If no specific action is specified, show help
    if not (args.get_stock or args.get_news or args.analyze or args.dashboard or args.all):
        parser.print_help()

if __name__ == "__main__":
    main()