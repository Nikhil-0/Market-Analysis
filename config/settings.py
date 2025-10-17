# Tickers to analyze
TICKERS = [
    'AAPL',  # Apple
    'MSFT',  # Microsoft
    'GOOGL', # Google/Alphabet
    'AMZN',  # Amazon
    'TSLA',  # Tesla
]

# Date ranges for historical data
DEFAULT_LOOKBACK_DAYS = 365  # 1 year of data by default

# API keys and credentials (use environment variables in production)
# NEWS_API_KEY = "your_api_key"  # Uncomment and add your key if using News API

# Web scraping settings
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries
MAX_NEWS_ARTICLES = 100  # maximum number of news articles per ticker

# Data paths
DATA_DIR = 'data'
MODELS_DIR = 'models'
CHARTS_DIR = 'charts'

# Feature settings
TECHNICAL_INDICATORS = {
    'enable': True,
    'indicators': [
        'SMA20', # Simple Moving Average 20-day
        'SMA50', # Simple Moving Average 50-day
        'RSI14', # 14-day Relative Strength Index
        'MACD',  # Moving Average Convergence Divergence
        'BB',    # Bollinger Bands
    ]
}

# Model settings
MODEL_SETTINGS = {
    'test_size': 0.2,
    'random_state': 42,
    'models': [
        'logistic_regression',
        'random_forest',
        'xgboost'
    ],
    'hyperparameters': {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1
        }
    }
}