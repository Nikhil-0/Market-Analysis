import os
import sys
import argparse
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("market_analysis.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import modules
try:
    from market_analysis.data_collection import collect_stock_data, collect_news_data
    from market_analysis.sentiment_analysis import analyze_news_sentiment, extract_named_entities
    from market_analysis.visualization.interactive_charts import (
        create_stock_sentiment_chart,
        create_correlation_chart,
        create_entity_network_chart
    )
    from market_analysis.visualization.dashboard import create_dashboard
    from config import settings
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    logger.error("Make sure you have installed all required packages and the market_analysis package is in your PYTHONPATH.")
    sys.exit(1)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Market Analysis Tool - Analyze stock prices and news sentiment"
    )
    
    # Ticker argument
    parser.add_argument(
        "--ticker", "-t",
        type=str,
        help="Stock ticker symbol(s) to analyze (comma-separated for multiple)",
        required=False
    )
    
    # Date range arguments
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for analysis (YYYY-MM-DD), default is 1 year ago",
        required=False
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for analysis (YYYY-MM-DD), default is today",
        required=False
    )
    
    # Action arguments
    parser.add_argument(
        "--collect-data",
        action="store_true",
        help="Collect stock price and news data"
    )
    
    parser.add_argument(
        "--analyze-sentiment",
        action="store_true",
        help="Analyze news sentiment"
    )
    
    parser.add_argument(
        "--extract-entities",
        action="store_true",
        help="Extract named entities from news"
    )
    
    parser.add_argument(
        "--create-charts",
        action="store_true",
        help="Create interactive charts"
    )
    
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Run the interactive dashboard"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the complete analysis pipeline"
    )
    
    # Dashboard settings
    parser.add_argument(
        "--port",
        type=int,
        default=settings.DASHBOARD_PORT,
        help=f"Port for the dashboard server (default: {settings.DASHBOARD_PORT})"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default=settings.DASHBOARD_HOST,
        help=f"Host for the dashboard server (default: {settings.DASHBOARD_HOST})"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run the dashboard in debug mode"
    )
    
    return parser.parse_args()

def validate_args(args):
    if not any([args.collect_data, args.analyze_sentiment, args.extract_entities, 
                args.create_charts, args.dashboard, args.all]):
        logger.error("No action specified. Use --help to see available actions.")
        return False
    
    if args.all:
        # If --all is specified, set all action flags to True
        args.collect_data = True
        args.analyze_sentiment = True
        args.extract_entities = True
        args.create_charts = True
        
    # If no ticker is specified and not just running dashboard
    if not args.ticker and not args.dashboard:
        logger.error("No ticker specified. Use --ticker/-t to specify a ticker symbol.")
        return False
    
    # Parse dates if provided
    today = datetime.now().date()
    
    if args.start_date:
        try:
            args.start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        except ValueError:
            logger.error("Invalid start date format. Use YYYY-MM-DD.")
            return False
    else:
        # Default to 1 year ago
        args.start_date = today - timedelta(days=365)
    
    if args.end_date:
        try:
            args.end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        except ValueError:
            logger.error("Invalid end date format. Use YYYY-MM-DD.")
            return False
    else:
        # Default to today
        args.end_date = today
    
    # Validate date range
    if args.start_date > args.end_date:
        logger.error("Start date cannot be after end date.")
        return False
    
    return True

def create_directories():
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, settings.DATA_DIR)
    charts_dir = os.path.join(project_root, settings.CHARTS_DIR)
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)
    
    return data_dir, charts_dir

def run_data_collection(args, data_dir):
    logger.info("Starting data collection...")
    
    tickers = [ticker.strip().upper() for ticker in args.ticker.split(',')]
    
    for ticker in tickers:
        logger.info(f"Collecting data for {ticker}...")
        logger.info(f"Date range: {args.start_date} to {args.end_date}")
        
        # Collect stock data
        stock_file = os.path.join(data_dir, f"{ticker}_stock_data.csv")
        collect_stock_data(ticker, args.start_date, args.end_date, stock_file)
        
        # Collect news data
        news_file = os.path.join(data_dir, f"{ticker}_news_data.csv")
        collect_news_data(ticker, args.start_date, args.end_date, news_file)
    
    logger.info("Data collection completed.")

def run_sentiment_analysis(args, data_dir):
    logger.info("Starting sentiment analysis...")
    
    tickers = [ticker.strip().upper() for ticker in args.ticker.split(',')]
    
    for ticker in tickers:
        # News sentiment analysis
        news_file = os.path.join(data_dir, f"{ticker}_news_data.csv")
        sentiment_file = os.path.join(data_dir, f"{ticker}_sentiment_analysis.csv")
        daily_sentiment_file = os.path.join(data_dir, f"{ticker}_daily_sentiment.csv")
        
        if not os.path.exists(news_file):
            logger.warning(f"News data file not found for {ticker}. Skipping sentiment analysis.")
            continue
        
        logger.info(f"Analyzing sentiment for {ticker} news...")
        analyze_news_sentiment(news_file, sentiment_file, daily_sentiment_file)
    
    logger.info("Sentiment analysis completed.")

def run_entity_extraction(args, data_dir):
    logger.info("Starting named entity extraction...")
    
    tickers = [ticker.strip().upper() for ticker in args.ticker.split(',')]
    
    for ticker in tickers:
        # Named entity extraction
        news_file = os.path.join(data_dir, f"{ticker}_news_data.csv")
        entities_file = os.path.join(data_dir, f"{ticker}_entities.csv")
        
        if not os.path.exists(news_file):
            logger.warning(f"News data file not found for {ticker}. Skipping entity extraction.")
            continue
        
        logger.info(f"Extracting entities from {ticker} news...")
        extract_named_entities(news_file, entities_file)
    
    logger.info("Entity extraction completed.")

def run_chart_creation(args, data_dir, charts_dir):
    logger.info("Creating interactive charts...")
    
    tickers = [ticker.strip().upper() for ticker in args.ticker.split(',')]
    
    for ticker in tickers:
        logger.info(f"Creating charts for {ticker}...")
        
        # Check for required files
        stock_file = os.path.join(data_dir, f"{ticker}_stock_data.csv")
        sentiment_file = os.path.join(data_dir, f"{ticker}_daily_sentiment.csv")
        
        if not os.path.exists(sentiment_file):
            sentiment_file = os.path.join(data_dir, f"{ticker}_sentiment_analysis.csv")
        
        if not os.path.exists(stock_file):
            logger.warning(f"Stock data file not found for {ticker}. Skipping chart creation.")
            continue
        
        if not os.path.exists(sentiment_file):
            logger.warning(f"Sentiment data file not found for {ticker}. Skipping chart creation.")
            continue
        
        # Create price and sentiment chart
        logger.info(f"Creating price and sentiment chart for {ticker}...")
        price_sentiment_file = os.path.join(charts_dir, f"{ticker}_price_sentiment.html")
        create_stock_sentiment_chart(stock_file, sentiment_file, price_sentiment_file)
        
        # Create correlation chart
        logger.info(f"Creating correlation chart for {ticker}...")
        correlation_file = os.path.join(charts_dir, f"{ticker}_sentiment_correlation.html")
        create_correlation_chart(stock_file, sentiment_file, correlation_file)
        
        # Create entity network chart if entities file exists
        entities_file = os.path.join(data_dir, f"{ticker}_entities.csv")
        if os.path.exists(entities_file):
            logger.info(f"Creating entity network chart for {ticker}...")
            network_file = os.path.join(charts_dir, f"{ticker}_entity_network.html")
            create_entity_network_chart(entities_file, min_occurrences=2, output_file=network_file)
    
    logger.info("Chart creation completed.")

def run_dashboard(args, data_dir):
    logger.info(f"Starting dashboard on http://{args.host}:{args.port}/...")
    create_dashboard(data_dir=data_dir, host=args.host, port=args.port, debug=args.debug)

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Validate arguments
    if not validate_args(args):
        sys.exit(1)
    
    # Create directories
    data_dir, charts_dir = create_directories()
    
    try:
        # Run requested actions
        if args.collect_data:
            run_data_collection(args, data_dir)
        
        if args.analyze_sentiment:
            run_sentiment_analysis(args, data_dir)
        
        if args.extract_entities:
            run_entity_extraction(args, data_dir)
        
        if args.create_charts:
            run_chart_creation(args, data_dir, charts_dir)
        
        if args.dashboard:
            run_dashboard(args, data_dir)
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()