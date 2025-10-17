import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime

def scrape_finviz_news(ticker):
    print(f'Scraping Finviz news for {ticker}...')
    
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    
    # Add headers to mimic a browser visit
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5'
    }
    
    try:
        # Add a delay to avoid getting blocked
        time.sleep(random.uniform(1, 2))
        
        # Get the page content
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f'Error: Received status code {response.status_code}')
            return pd.DataFrame(columns=['date', 'time', 'headline', 'ticker'])
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for the news table
        news_table = soup.find('table', {'class': 'fullview-news-outer'})
        
        if news_table is None:
            print('News table not found')
            return pd.DataFrame(columns=['date', 'time', 'headline', 'ticker'])
        
        # Process the news table
        news_data = []
        rows = news_table.find_all('tr')
        
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        for row in rows:
            cells = row.find_all('td')
            
            if len(cells) >= 2:
                date_time_cell = cells[0].text.strip()
                date_time_parts = date_time_cell.split()
                
                if len(date_time_parts) >= 2:
                    news_date = date_time_parts[0]
                    news_time = date_time_parts[1]
                else:
                    news_date = date_time_cell
                    news_time = 'N/A'
                
                # Handle special date formats
                if news_date.lower() == 'today':
                    news_date = today
                elif news_date.lower() == 'yesterday':
                    news_date = yesterday
                
                headline = cells[1].text.strip()
                
                news_data.append({
                    'date': news_date,
                    'time': news_time,
                    'headline': headline,
                    'ticker': ticker
                })
        
        # Convert to DataFrame
        news_df = pd.DataFrame(news_data)
        
        print(f'Successfully scraped {len(news_df)} news articles')
        
        # Make sure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Save to CSV
        news_df.to_csv(os.path.join('data', f'{ticker}_finviz_news.csv'), index=False)
        
        return news_df
    
    except Exception as e:
        print(f'Error: {str(e)}')
        return pd.DataFrame(columns=['date', 'time', 'headline', 'ticker'])
