import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('vader_sentiment')

# Import settings
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import settings

def clean_text(text):
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters but keep spaces and basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def analyze_sentiment(text):
    if not text or pd.isna(text):
        return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
    
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(str(text))
    return sentiment

def get_sentiment_category(compound_score):
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def enhance_vader_lexicon_for_finance():
    analyzer = SentimentIntensityAnalyzer()
    
    # Financial specific lexicon enhancements
    financial_lexicon = {
        # Positive financial terms
        'bullish': 3.0,
        'outperform': 2.0,
        'beat': 2.0,
        'beats': 2.0,
        'exceeded': 1.5,
        'exceeds': 1.5,
        'upgrade': 1.5,
        'upgraded': 1.5,
        'buy': 1.0,
        'overweight': 1.0,
        'strong buy': 2.0,
        'growth': 1.0,
        'profit': 1.5,
        'profitable': 1.5,
        'gain': 1.0,
        'gains': 1.0,
        'record high': 2.0,
        
        # Negative financial terms
        'bearish': -3.0,
        'underperform': -2.0,
        'miss': -2.0,
        'missed': -2.0,
        'misses': -2.0,
        'downgrade': -1.5,
        'downgraded': -1.5,
        'sell': -1.0,
        'underweight': -1.0,
        'strong sell': -2.0,
        'decline': -1.0,
        'declines': -1.0,
        'declined': -1.0,
        'loss': -1.5,
        'losses': -1.5,
        'bankruptcy': -3.0,
        'bankrupt': -3.0,
        'recession': -2.0,
        'recession fears': -2.5,
        'layoff': -1.5,
        'layoffs': -1.5,
        'litigation': -1.5,
        'lawsuit': -1.5,
        'regulatory': -1.0,
        'probe': -1.5,
        'investigation': -1.5,
    }
    
    # Add the financial lexicon to VADER's lexicon
    analyzer.lexicon.update(financial_lexicon)
    return analyzer

def process_sentiment_data(input_file_path, output_file_path=None, text_column='headline', use_finance_lexicon=True):
    try:
        logger.info(f"Loading data from {input_file_path}...")
        data = pd.read_csv(input_file_path)
        
        # Check for text column (try multiple possible column names)
        possible_columns = [text_column, 'text', 'headline', 'title', 'content']
        text_col = None
        
        for col in possible_columns:
            if col in data.columns:
                text_col = col
                break
        
        if text_col is None:
            logger.error(f"No suitable text column found. Available columns: {list(data.columns)}")
            return None
        
        logger.info(f"Using '{text_col}' column for sentiment analysis...")
        
        # Clean the text data
        data['cleaned_text'] = data[text_col].apply(clean_text)
        
        # Use finance-enhanced lexicon if requested
        if use_finance_lexicon:
            logger.info("Using finance-enhanced sentiment lexicon...")
            analyzer = enhance_vader_lexicon_for_finance()
            
            # Analyze sentiment with enhanced lexicon
            data['sentiment_compound'] = data['cleaned_text'].apply(
                lambda x: analyzer.polarity_scores(str(x))['compound']
            )
            data['sentiment_positive'] = data['cleaned_text'].apply(
                lambda x: analyzer.polarity_scores(str(x))['pos']
            )
            data['sentiment_neutral'] = data['cleaned_text'].apply(
                lambda x: analyzer.polarity_scores(str(x))['neu']
            )
            data['sentiment_negative'] = data['cleaned_text'].apply(
                lambda x: analyzer.polarity_scores(str(x))['neg']
            )
        else:
            # Analyze sentiment with standard lexicon
            logger.info("Using standard sentiment lexicon...")
            sentiment_results = data['cleaned_text'].apply(analyze_sentiment)
            
            # Extract sentiment components
            data['sentiment_compound'] = sentiment_results.apply(lambda x: x['compound'])
            data['sentiment_positive'] = sentiment_results.apply(lambda x: x['pos'])
            data['sentiment_neutral'] = sentiment_results.apply(lambda x: x['neu'])
            data['sentiment_negative'] = sentiment_results.apply(lambda x: x['neg'])
        
        # Add sentiment category
        data['sentiment_category'] = data['sentiment_compound'].apply(get_sentiment_category)
        
        # If output path is not provided, generate one
        if output_file_path is None:
            base_name = os.path.basename(input_file_path)
            name_parts = os.path.splitext(base_name)
            output_file_path = os.path.join(settings.DATA_DIR, f"{name_parts[0]}_sentiment{name_parts[1]}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        # Save results
        data.to_csv(output_file_path, index=False)
        
        # Print summary statistics
        logger.info(f"Sentiment Analysis Complete!")
        logger.info(f"Results saved to: {output_file_path}")
        logger.info(f"Sentiment Summary:")
        logger.info(data['sentiment_category'].value_counts().to_string())
        logger.info(f"Average sentiment score: {data['sentiment_compound'].mean():.3f}")
        
        return output_file_path
        
    except FileNotFoundError:
        logger.error(f"Error: The file {input_file_path} was not found.")
        return None
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None

def aggregate_sentiment_by_date(sentiment_file):
    try:
        logger.info(f"Aggregating sentiment data from {sentiment_file} by date...")
        sentiment_data = pd.read_csv(sentiment_file)
        
        # Ensure date column is datetime
        sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
        
        # Group by date and calculate mean sentiment
        daily_sentiment = sentiment_data.groupby(sentiment_data['date'].dt.date).agg({
            'sentiment_compound': 'mean',
            'sentiment_positive': 'mean',
            'sentiment_negative': 'mean',
            'sentiment_neutral': 'mean',
            'headline': 'count'  # Count of headlines per day
        }).reset_index()
        
        # Rename count column
        daily_sentiment = daily_sentiment.rename(columns={'headline': 'headline_count'})
        
        # Add sentiment category based on aggregated compound score
        daily_sentiment['sentiment_category'] = daily_sentiment['sentiment_compound'].apply(get_sentiment_category)
        
        logger.info(f"Aggregated {len(sentiment_data)} headlines into {len(daily_sentiment)} daily entries")
        
        return daily_sentiment
    
    except Exception as e:
        logger.error(f"Error aggregating sentiment by date: {e}")
        return None

if __name__ == '__main__':
    # Example usage
    # Get the absolute path to the project's root directory
    PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Example usage - analyze news data collected from data_collection.py
    ticker = 'AAPL'
    input_file = os.path.join(PROJ_ROOT, 'data', f'{ticker}_combined_news.csv')
    
    if not os.path.exists(input_file):
        input_file = os.path.join(PROJ_ROOT, 'data', f'{ticker}_news.csv')
    
    print(f"Looking for news data file: {input_file}")
    
    if os.path.exists(input_file):
        print(f"Found news data! Starting sentiment analysis...")
        output_file = os.path.join(PROJ_ROOT, 'data', f'{ticker}_sentiment_analysis.csv')
        
        result = process_sentiment_data(
            input_file_path=input_file,
            output_file_path=output_file,
            text_column='headline',
            use_finance_lexicon=True
        )
        
        if result:
            print(f"\n Sentiment analysis completed successfully!")
            print(f" Results saved to: {result}")
            
            # Aggregate by date
            daily_sentiment = aggregate_sentiment_by_date(result)
            if daily_sentiment is not None:
                daily_file = os.path.join(PROJ_ROOT, 'data', f'{ticker}_daily_sentiment.csv')
                daily_sentiment.to_csv(daily_file, index=False)
                print(f" Daily aggregated sentiment saved to: {daily_file}")
    else:
        print(f" News data file not found: {input_file}")
        print(f" Run data_collection.py first to collect news data.")
