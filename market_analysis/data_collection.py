from .data_collection.stock_data import get_stock_data, get_stock_data_with_retry
from .data_collection.news_data import get_finviz_news, get_finviz_news_with_retry, get_multiple_news_sources

def collect_stock_data(ticker, start_date, end_date, output_file=None):

    return get_stock_data_with_retry(ticker, start_date, end_date, output_file)

def collect_news_data(ticker, output_file=None):
   
    return get_finviz_news_with_retry(ticker, output_file)