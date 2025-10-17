import os
import sys
import logging
from datetime import datetime, timedelta
import argparse
import pandas as pd
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

def collect_stock_data(ticker, days=365):
    try:
        logger.info(f"Collecting stock data for {ticker}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Download from Yahoo Finance
        data = yf.download(ticker, start=start_date, end=end_date)
        
        if data.empty:
            logger.error(f"No stock data found for {ticker}")
            return False
            
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Rename columns for consistency
        data = data.rename(columns={"Date": "date"})
        
        # Save to CSV
        file_path = os.path.join('data', f"{ticker}_stock_data.csv")
        data.to_csv(file_path, index=False)
        
        logger.info(f"Saved stock data to {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error collecting stock data: {str(e)}")
        return False

def collect_news_data(ticker):
    try:
        logger.info(f"Collecting news data for {ticker}")
        
        # Import scraper
        try:
            from market_analysis.data_collection.news_scraper import scrape_finviz_news
        except ImportError:
            logger.error("Could not import news scraper. Make sure the module exists.")
            return False
        
        # Scrape news
        news_data = scrape_finviz_news(ticker)
        
        if news_data.empty:
            logger.warning(f"No news data found for {ticker}")
            return False
        
        logger.info(f"Collected {len(news_data)} news items for {ticker}")
        return True
    
    except Exception as e:
        logger.error(f"Error collecting news data: {str(e)}")
        return False

def analyze_sentiment(ticker):
    try:
        logger.info(f"Analyzing sentiment for {ticker}")
        
        # Import sentiment analyzer
        try:
            from market_analysis.data_analysis.sentiment_analyzer import process_news_data
        except ImportError:
            logger.error("Could not import sentiment analyzer. Make sure the module exists.")
            return False
        
        # Process news data
        sentiment_df, daily_sentiment_df = process_news_data(ticker)
        
        if sentiment_df.empty:
            logger.warning(f"No sentiment data generated for {ticker}")
            return False
        
        logger.info(f"Sentiment analysis complete for {len(sentiment_df)} news items")
        return True
    
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        return False

def extract_entities(ticker):
    try:
        logger.info(f"Extracting entities for {ticker}")
        
        # Import entity extractor
        try:
            from market_analysis.sentiment.advanced_sentiment import extract_entities_from_file
        except ImportError:
            logger.error("Could not import entity extractor. Make sure the module exists.")
            return False
        
        # Get the path to the sentiment file
        sentiment_file = os.path.join('data', f'{ticker}_sentiment_analysis.csv')
        
        if not os.path.exists(sentiment_file):
            # Try alternative filename pattern
            news_file = os.path.join('data', f'{ticker}_finviz_news.csv')
            if os.path.exists(news_file):
                sentiment_file = news_file
            else:
                logger.error(f"No suitable data file found for entity extraction: {sentiment_file}")
                return False
        
        logger.info(f"Extracting entities from {sentiment_file}...")
        entities_df, entities_file = extract_entities_from_file(sentiment_file)
        
        if entities_df is None or entities_df.empty:
            logger.warning(f"No entities extracted from {sentiment_file}")
            return False
            
        # Save a copy with the standard name expected by the dashboard
        standard_entities_file = os.path.join('data', f'{ticker}_entities.csv')
        entities_df.to_csv(standard_entities_file, index=False)
        logger.info(f"Entities saved to {standard_entities_file} for dashboard compatibility")
        
        return True
    
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}")
        return False

def run_dashboard():
    try:
        logger.info("Starting dashboard")
        
        # Import dashboard using a direct module execution
        import sys
        import importlib.util
        
        # Execute the dashboard module directly
        import subprocess
        result = subprocess.call([sys.executable, "-m", "market_analysis.visualization.dashboard"])
        
        return result == 0
    
    except Exception as e:
        logger.error(f"Error running dashboard: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Market Analysis Tool")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--days", type=int, default=365, help="Number of days of historical data to collect (default: 365)")
    parser.add_argument("--skip-stock", action="store_true", help="Skip collecting stock data")
    parser.add_argument("--skip-news", action="store_true", help="Skip collecting news data")
    parser.add_argument("--skip-sentiment", action="store_true", help="Skip sentiment analysis")
    parser.add_argument("--skip-entities", action="store_true", help="Skip entity extraction")
    parser.add_argument("--dashboard-only", action="store_true", help="Only run the dashboard")
    parser.add_argument("--entities-only", action="store_true", help="Only run entity extraction")
    
    args = parser.parse_args()
    
    ticker = args.ticker
    
    if args.dashboard_only:
        run_dashboard()
        return
        
    if args.entities_only:
        extract_entities(ticker)
        run_dashboard()
        return
    
    # Collect all data
    success = True
    
    if not args.skip_stock:
        stock_success = collect_stock_data(ticker, args.days)
        success = success and stock_success
    
    if not args.skip_news:
        news_success = collect_news_data(ticker)
        success = success and news_success
    
    if not args.skip_sentiment:
        sentiment_success = analyze_sentiment(ticker)
        success = success and sentiment_success
        
    if not args.skip_entities:
        entities_success = extract_entities(ticker)
        # Don't fail whole pipeline if entity extraction fails
        if not entities_success:
            logger.warning("Entity extraction failed or found no entities")
    
    # Run dashboard
    if success:
        logger.info("Data collection complete. Starting dashboard...")
        run_dashboard()
    else:
        logger.warning("Some data collection steps failed. Dashboard may not display correctly.")
        choice = input("Do you want to run the dashboard anyway? (y/n): ")
        if choice.lower() == 'y':
            run_dashboard()

if __name__ == "__main__":
    main()