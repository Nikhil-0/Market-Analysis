"""
Sentiment analysis module for analyzing financial news sentiment.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def clean_text(text):
    """
    Clean and prepare text for sentiment analysis
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ''
    
    # Remove special characters and extra whitespace
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = ' '.join(text.split())
    return text

def analyze_sentiment(df, text_column='headline'):
    """
    Analyze sentiment of text data using VADER
    
    Args:
        df (pd.DataFrame): DataFrame containing text data
        text_column (str): Column name containing text to analyze
        
    Returns:
        pd.DataFrame: DataFrame with sentiment scores added
    """
    print(f'Analyzing sentiment for {len(df)} items...')
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Make sure the text column exists
    if text_column not in result_df.columns:
        print(f'Error: Column {text_column} not found in DataFrame')
        print(f'Available columns: {result_df.columns.tolist()}')
        return result_df
    
    # Initialize VADER sentiment analyzer
    try:
        sia = SentimentIntensityAnalyzer()
    except Exception as e:
        print(f'Error initializing VADER: {str(e)}')
        print('Make sure NLTK and VADER are installed:')
        print('  pip install nltk')
        print('  python -c "import nltk; nltk.download(\'vader_lexicon\')"')
        return result_df
    
    # Clean text and add a new column
    result_df['cleaned_text'] = result_df[text_column].apply(clean_text)
    
    # Initialize sentiment columns
    result_df['sentiment_compound'] = 0.0
    result_df['sentiment_positive'] = 0.0
    result_df['sentiment_neutral'] = 0.0
    result_df['sentiment_negative'] = 0.0
    result_df['sentiment_category'] = 'Neutral'
    
    # Process each row
    for i, row in result_df.iterrows():
        text = row['cleaned_text']
        
        if not text:
            continue
            
        try:
            # Get sentiment scores
            sentiment = sia.polarity_scores(text)
            
            # Store scores
            result_df.at[i, 'sentiment_compound'] = sentiment['compound']
            result_df.at[i, 'sentiment_positive'] = sentiment['pos']
            result_df.at[i, 'sentiment_neutral'] = sentiment['neu']
            result_df.at[i, 'sentiment_negative'] = sentiment['neg']
            
            # Determine sentiment category
            if sentiment['compound'] >= 0.05:
                result_df.at[i, 'sentiment_category'] = 'Positive'
            elif sentiment['compound'] <= -0.05:
                result_df.at[i, 'sentiment_category'] = 'Negative'
            else:
                result_df.at[i, 'sentiment_category'] = 'Neutral'
        
        except Exception as e:
            print(f'Error analyzing sentiment for text: {text[:50]}...: {str(e)}')
    
    print('Sentiment analysis completed')
    return result_df

def fix_dates(df):
    """
    Fix special date formats like 'Today' and convert to proper dates
    
    Args:
        df (pd.DataFrame): DataFrame with date column
        
    Returns:
        pd.DataFrame: DataFrame with fixed dates
    """
    if 'date' not in df.columns:
        print('Error: date column not found')
        return df
    
    today = datetime.now().date()
    yesterday = (datetime.now() - timedelta(days=1)).date()
    
    # Create a new column for processed dates
    df['processed_date'] = None
    
    for i, row in df.iterrows():
        date_str = str(row['date']).strip()
        
        # Handle special cases
        if date_str.lower() == 'today':
            df.at[i, 'processed_date'] = today
        elif date_str.lower() == 'yesterday':
            df.at[i, 'processed_date'] = yesterday
        else:
            # Try to parse the date
            try:
                parsed_date = pd.to_datetime(date_str).date()
                df.at[i, 'processed_date'] = parsed_date
            except:
                print(f'Could not parse date: {date_str}')
                df.at[i, 'processed_date'] = today  # Default to today
    
    # Replace original date column with processed dates
    df['date'] = df['processed_date']
    df = df.drop(columns=['processed_date'])
    
    return df

def create_daily_sentiment(ticker, sentiment_df=None):
    """
    Create daily aggregated sentiment data
    
    Args:
        ticker (str): Stock ticker symbol
        sentiment_df (pd.DataFrame, optional): DataFrame with sentiment scores
        
    Returns:
        pd.DataFrame: Daily aggregated sentiment data
    """
    print(f'Creating daily sentiment aggregates for {ticker}...')
    
    # If sentiment_df is not provided, try to load from file
    if sentiment_df is None:
        sentiment_file = os.path.join('data', f'{ticker}_sentiment_analysis.csv')
        
        if not os.path.exists(sentiment_file):
            print(f'Error: Sentiment file not found: {sentiment_file}')
            return pd.DataFrame()
        
        sentiment_df = pd.read_csv(sentiment_file)
    
    # Fix dates
    sentiment_df = fix_dates(sentiment_df)
    
    # Ensure date is datetime
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    
    # Get numeric columns only - filter out non-numeric columns to avoid aggregation issues
    skip_columns = ['sentiment_category', 'time', 'headline', 'cleaned_text', 'ticker']
    
    # Create aggregation dictionary
    agg_dict = {}
    for col in sentiment_df.columns:
        if col != 'date' and col not in skip_columns:
            try:
                # Test if column is numeric
                pd.to_numeric(sentiment_df[col])
                agg_dict[col] = 'mean'
                print(f'Using column for aggregation: {col}')
            except (ValueError, TypeError):
                print(f'Skipping non-numeric column: {col}')
    
    # Count headlines per day
    if 'headline' in sentiment_df.columns:
        agg_dict['headline'] = 'count'
    
    # Group by date
    daily_sentiment = sentiment_df.groupby('date').agg(agg_dict).reset_index()
    
    # Rename columns to match dashboard expectations
    rename_dict = {}
    if 'sentiment_compound' in daily_sentiment.columns:
        rename_dict['sentiment_compound'] = 'sentiment_score'
    elif 'compound' in daily_sentiment.columns:
        rename_dict['compound'] = 'sentiment_score'
        
    if 'headline' in daily_sentiment.columns:
        rename_dict['headline'] = 'headline_count'
    
    daily_sentiment = daily_sentiment.rename(columns=rename_dict)
    
    # Ensure required columns exist
    required_columns = ['date', 'sentiment_score', 'headline_count']
    
    for col in required_columns:
        if col not in daily_sentiment.columns:
            if col == 'sentiment_score':
                daily_sentiment[col] = 0.0
            elif col == 'headline_count':
                daily_sentiment[col] = 5
    
    # Save daily sentiment data
    output_file = os.path.join('data', f'{ticker}_daily_sentiment.csv')
    daily_sentiment.to_csv(output_file, index=False)
    print(f'Saved daily sentiment to {output_file}')
    
    return daily_sentiment

def process_news_data(ticker):
    """
    Process news data: load, analyze sentiment, and create daily aggregates
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        tuple: (sentiment_df, daily_sentiment_df)
    """
    print(f'Processing news data for {ticker}...')
    
    # Check for news file
    news_file = os.path.join('data', f'{ticker}_finviz_news.csv')
    
    if not os.path.exists(news_file):
        print(f'News file not found: {news_file}')
        return pd.DataFrame(), pd.DataFrame()
    
    # Load news data
    news_df = pd.read_csv(news_file)
    print(f'Loaded {len(news_df)} news articles')
    
    # Check if we have data
    if news_df.empty:
        print('News data is empty')
        return news_df, pd.DataFrame()
    
    # Analyze sentiment
    sentiment_df = analyze_sentiment(news_df)
    
    # Save processed data
    output_file = os.path.join('data', f'{ticker}_sentiment_analysis.csv')
    sentiment_df.to_csv(output_file, index=False)
    print(f'Saved sentiment analysis to {output_file}')
    
    # Create aggregated daily sentiment data
    daily_sentiment = create_daily_sentiment(ticker, sentiment_df)
    
    return sentiment_df, daily_sentiment