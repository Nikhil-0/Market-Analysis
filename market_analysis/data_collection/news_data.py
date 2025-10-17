import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, date, timedelta
import logging
from functools import wraps
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('news_data')

# Import settings
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import settings

def retry_decorator(max_retries=3, delay=1):
    """
    Decorator for retrying functions with exponential backoff
    """
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

def parse_finviz_date(date_text):
    """
    Parse dates from Finviz that might include relative dates like 'Today'
    """
    today = date.today()
    
    if 'Today' in date_text:
        return today
    elif 'Yesterday' in date_text:
        return today - timedelta(days=1)
    else:
        # Try to parse as a regular date
        try:
            # Handle format like 'May-01-20'
            if re.match(r'\w{3}-\d{2}-\d{2}', date_text):
                return datetime.strptime(date_text, '%b-%d-%y').date()
            # Handle format like 'May-01'
            elif re.match(r'\w{3}-\d{2}', date_text):
                month_day = datetime.strptime(date_text, '%b-%d').date()
                # Set the correct year (assume current year)
                return date(today.year, month_day.month, month_day.day)
        except ValueError:
            logger.warning(f"Could not parse date: {date_text}")
            return None

def get_finviz_news(ticker, max_articles=None):
    """
    Scrapes financial news headlines from Finviz for a given ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        max_articles (int, optional): Maximum number of articles to retrieve
        
    Returns:
        str: Path to the saved CSV file or None if the operation fails
    """
    if max_articles is None:
        max_articles = settings.MAX_NEWS_ARTICLES
        
    try:
        logger.info(f"Scraping news for {ticker} (max: {max_articles} articles)")
        url = f'https://finviz.com/quote.ashx?t={ticker}'
        headers = settings.REQUEST_HEADERS
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        news_table = soup.find(id='news-table')
        
        if not news_table:
            logger.warning(f"No news table found for ticker {ticker}")
            return None
        
        news_list = []
        current_date = None
        
        for i, row in enumerate(news_table.findAll('tr')):
            if i >= max_articles:
                break
                
            if row.a:
                title = row.a.text.strip()
                date_cell = row.td.text.strip()
                
                # Parse date and time with better handling of relative dates
                date_parts = date_cell.split(' ')
                if len(date_parts) >= 2:
                    # Handle full date and time like 'May-31-23 12:45PM'
                    date_str = date_parts[0]
                    time_str = date_parts[1]
                    parsed_date = parse_finviz_date(date_str)
                    if parsed_date:
                        current_date = parsed_date
                else:
                    # Handle time-only like '12:45PM' (uses last known date)
                    time_str = date_parts[0]
                
                if current_date:
                    news_list.append([current_date, time_str, title])
        
        if news_list:
            df = pd.DataFrame(news_list, columns=['date', 'time', 'headline'])
            
            # Convert date strings to datetime objects
            df['date'] = pd.to_datetime(df['date'])
            
            # Add a source column for tracking
            df['source'] = 'Finviz'
            
            # No need to save to CSV here, that's handled by the caller
            logger.info(f"Successfully scraped {len(df)} news articles for {ticker}")
            
            return df
        else:
            logger.warning(f"No news articles found for {ticker}")
            return pd.DataFrame(columns=['date', 'time', 'headline', 'source'])
            
    except Exception as e:
        logger.error(f"An error occurred while scraping news for {ticker}: {e}")
        raise

@retry_decorator(max_retries=settings.MAX_RETRIES, delay=settings.RETRY_DELAY)
def get_finviz_news_with_retry(ticker, max_articles=None):
    """
    Scrapes financial news with automatic retries on failure
    
    Args:
        ticker (str): Stock ticker symbol
        max_articles (int, optional): Maximum number of articles to retrieve
        
    Returns:
        str: Path to the saved CSV file or None if all retries fail
    """
    return get_finviz_news(ticker, max_articles)

def get_marketwatch_news(ticker, max_articles=None):
    """
    Scrapes financial news headlines from MarketWatch for a given ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        max_articles (int, optional): Maximum number of articles to retrieve
        
    Returns:
        pandas.DataFrame: DataFrame with news data or None if the operation fails
    """
    if max_articles is None:
        max_articles = settings.MAX_NEWS_ARTICLES
        
    try:
        logger.info(f"Scraping MarketWatch news for {ticker} (max: {max_articles} articles)")
        url = f'https://www.marketwatch.com/investing/stock/{ticker.lower()}'
        headers = settings.REQUEST_HEADERS
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        news_list = []
        
        # Find the news section - this may need adjustment as websites change
        news_section = soup.find('div', {'class': 'collection__elements'})
        if not news_section:
            logger.warning(f"No news section found for {ticker} on MarketWatch")
            return None
            
        # Extract articles
        articles = news_section.find_all('div', {'class': 'element--article'})
        
        for i, article in enumerate(articles):
            if i >= max_articles:
                break
                
            headline_element = article.find('h3', {'class': 'article__headline'})
            if headline_element and headline_element.a:
                title = headline_element.a.text.strip()
                
                # Extract timestamp if available
                timestamp_element = article.find('span', {'class': 'article__timestamp'})
                if timestamp_element:
                    timestamp = timestamp_element.text.strip()
                    # Parse the timestamp (format varies by source)
                    try:
                        # Often in format like "Aug. 5, 2023 at 10:45 a.m. ET"
                        article_date = datetime.now().date()  # Default to today
                        article_time = "00:00"  # Default time
                    except:
                        article_date = datetime.now().date()
                        article_time = "00:00"
                else:
                    article_date = datetime.now().date()
                    article_time = "00:00"
                
                news_list.append([article_date, article_time, title])
        
        if news_list:
            df = pd.DataFrame(news_list, columns=['date', 'time', 'headline'])
            df['source'] = 'MarketWatch'
            return df
        else:
            logger.warning(f"No news articles found for {ticker} on MarketWatch")
            return None
            
    except Exception as e:
        logger.error(f"An error occurred while scraping MarketWatch news for {ticker}: {e}")
        return None

def get_multiple_news_sources(ticker, sources=None, max_articles=None):
    """
    Collects news from multiple sources and combines them
    
    Args:
        ticker (str): Stock ticker symbol
        sources (list, optional): List of news sources to use
        max_articles (int, optional): Maximum number of articles per source
        
    Returns:
        str: Path to the saved CSV file with combined news
    """
    if sources is None:
        sources = ['finviz', 'marketwatch']
    
    if max_articles is None:
        max_articles = settings.MAX_NEWS_ARTICLES
    
    all_news = []
    
    try:
        logger.info(f"Collecting news from multiple sources for {ticker}")
        
        # Collect news from each source
        for source in sources:
            if source.lower() == 'finviz':
                try:
                    finviz_file = get_finviz_news_with_retry(ticker, max_articles)
                    if finviz_file:
                        finviz_news = pd.read_csv(finviz_file)
                        all_news.append(finviz_news)
                except Exception as e:
                    logger.error(f"Error collecting Finviz news: {e}")
            
            elif source.lower() == 'marketwatch':
                try:
                    marketwatch_news = get_marketwatch_news(ticker, max_articles)
                    if marketwatch_news is not None:
                        all_news.append(marketwatch_news)
                except Exception as e:
                    logger.error(f"Error collecting MarketWatch news: {e}")
        
        # Combine all news sources if we have any
        if all_news:
            combined_news = pd.concat(all_news, ignore_index=True)
            combined_news = combined_news.drop_duplicates(subset=['headline'])
            
            # Sort by date, most recent first
            combined_news['date'] = pd.to_datetime(combined_news['date'])
            combined_news = combined_news.sort_values('date', ascending=False)
            
            # Save combined news
            output_dir = settings.DATA_DIR
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, f'{ticker}_combined_news.csv')
            combined_news.to_csv(file_path, index=False)
            
            logger.info(f"Saved {len(combined_news)} news articles from {len(all_news)} sources")
            return file_path
        else:
            logger.warning(f"No news articles found from any source for {ticker}")
            return None
            
    except Exception as e:
        logger.error(f"An error occurred while collecting news from multiple sources: {e}")
        return None

if __name__ == '__main__':
    # Example usage
    ticker_symbol = 'AAPL'
    
    # Collect news from multiple sources
    print(f"Collecting news for {ticker_symbol} from multiple sources...")
    news_file = get_multiple_news_sources(ticker_symbol)
    
    if news_file:
        print(f"News collection complete! Results saved to: {news_file}")
    else:
        print("News collection failed.")