import os
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time

def get_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if not stock_data.empty:
            # Ensure the directory exists
            output_dir = 'data'
            os.makedirs(output_dir, exist_ok=True)
            # Save the data to a CSV file
            file_path = os.path.join(output_dir, f'{ticker}_stock_data.csv')
            stock_data.to_csv(file_path)
            print(f"Successfully fetched and saved stock data for {ticker}")
            return file_path
        else:
            print(f"No data found for ticker {ticker}")
            return None
    except Exception as e:
        print(f"An error occurred while fetching stock data for {ticker}: {e}")
        return None

def get_finviz_news(ticker, max_articles=50):
    try:
        url = f'https://finviz.com/quote.ashx?t={ticker}'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        news_table = soup.find(id='news-table')
        
        if not news_table:
            print(f"No news found for ticker {ticker}")
            return None
        
        news_list = []
        current_date = None
        
        for i, row in enumerate(news_table.findAll('tr')):
            if i >= max_articles:
                break
                
            if row.a:
                title = row.a.text.strip()
                date_cell = row.td.text.strip()
                
                # Parse date and time
                date_parts = date_cell.split(' ')
                if len(date_parts) == 2:
                    current_date = date_parts[0]
                    time_str = date_parts[1]
                else:
                    time_str = date_parts[0]
                
                news_list.append([current_date, time_str, title])
        
        if news_list:
            df = pd.DataFrame(news_list, columns=['date', 'time', 'headline'])
            df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
            
            # Save to CSV
            output_dir = 'data'
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, f'{ticker}_news.csv')
            df.to_csv(file_path, index=False)
            print(f"Successfully scraped {len(df)} news articles for {ticker}")
            return file_path
        else:
            print(f"No news articles found for {ticker}")
            return None
            
    except Exception as e:
        print(f"An error occurred while scraping news for {ticker}: {e}")
        return None

if __name__ == '__main__':
    # Example usage
    ticker_symbol = 'AAPL'
    start = '2023-01-01'
    end = '2024-01-01'
    
    print("Fetching stock data...")
    stock_file = get_stock_data(ticker_symbol, start, end)
    
    print("\nScraping financial news...")
    news_file = get_finviz_news(ticker_symbol)
    
    if stock_file and news_file:
        print(f"\nData collection complete!")
        print(f"Stock data saved to: {stock_file}")
        print(f"News data saved to: {news_file}")
    else:
        print("\nSome data collection failed. Please check the error messages above.")
