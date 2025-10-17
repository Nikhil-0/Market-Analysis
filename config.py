import os

class Settings:
    # Data directories
    DATA_DIR = "data"  # Directory for storing data files relative to project root
    CHARTS_DIR = "charts"  # Directory for storing charts relative to project root
    
    # API settings
    FINVIZ_BASE_URL = "https://finviz.com"
    FINVIZ_NEWS_URL = "https://finviz.com/quote.ashx?t="
    YAHOO_FINANCE_URL = "https://finance.yahoo.com/quote/"
    
    # News settings
    MAX_NEWS_ARTICLES = 100  # Maximum number of news articles to retrieve

    # Web scraping settings
    REQUEST_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    REQUEST_TIMEOUT = 10  # seconds
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    RETRY_BACKOFF = 2  # exponential backoff multiplier

    # Analysis settings
    NEWS_SENTIMENT_WINDOW = 7  # days to aggregate for daily sentiment calculation
    TECHNICAL_INDICATOR_PERIODS = {
        'SMA': [20, 50, 200],  # Simple Moving Average periods
        'EMA': [12, 26],  # Exponential Moving Average periods
        'RSI': 14,  # Relative Strength Index period
        'MACD': {'fast': 12, 'slow': 26, 'signal': 9},  # MACD parameters
    }

    # Visualization settings
    DEFAULT_PLOT_SIZE = (16, 8)  # Default figure size in inches
    DEFAULT_PLOT_STYLE = "seaborn-v0_8-darkgrid"  # Default matplotlib style
    CHART_COLORS = {
        'price_up': '#26a69a',  # Green for price increases
        'price_down': '#ef5350',  # Red for price decreases
        'volume': '#90a4ae',  # Blue-grey for volume
        'sentiment_positive': '#1e88e5',  # Blue for positive sentiment
        'sentiment_negative': '#f4511e',  # Orange for negative sentiment
        'sentiment_neutral': '#78909c',  # Grey for neutral sentiment
        'ma20': '#7e57c2',  # Purple for 20-day moving average
        'ma50': '#26c6da',  # Cyan for 50-day moving average
        'ma200': '#ffa726',  # Orange for 200-day moving average
    }

    # Dashboard settings
    DASHBOARD_HOST = "127.0.0.1"
    DASHBOARD_PORT = 8050
    DASHBOARD_DEBUG = False

# Create a settings instance to be imported by other modules
settings = Settings()