import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import os
from datetime import datetime

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

def process_sentiment_data(input_file_path, output_file_path, text_column='headline'):
    try:
        print(f"Loading data from {input_file_path}...")
        data = pd.read_csv(input_file_path)
        
        # Check for text column (try multiple possible column names)
        possible_columns = [text_column, 'text', 'headline', 'title', 'content']
        text_col = None
        
        for col in possible_columns:
            if col in data.columns:
                text_col = col
                break
        
        if text_col is None:
            print(f"No suitable text column found. Available columns: {list(data.columns)}")
            return None
        
        print(f"Using '{text_col}' column for sentiment analysis...")
        
        # Clean the text data
        data['cleaned_text'] = data[text_col].apply(clean_text)
        
        # Analyze sentiment
        print("Analyzing sentiment...")
        sentiment_results = data['cleaned_text'].apply(analyze_sentiment)
        
        # Extract sentiment components
        data['sentiment_compound'] = sentiment_results.apply(lambda x: x['compound'])
        data['sentiment_positive'] = sentiment_results.apply(lambda x: x['pos'])
        data['sentiment_neutral'] = sentiment_results.apply(lambda x: x['neu'])
        data['sentiment_negative'] = sentiment_results.apply(lambda x: x['neg'])
        data['sentiment_category'] = data['sentiment_compound'].apply(get_sentiment_category)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        # Save results
        data.to_csv(output_file_path, index=False)
        
        # Print summary statistics
        print(f"\nSentiment Analysis Complete!")
        print(f"Results saved to: {output_file_path}")
        print(f"\nSentiment Summary:")
        print(data['sentiment_category'].value_counts())
        print(f"\nAverage sentiment score: {data['sentiment_compound'].mean():.3f}")
        
        return output_file_path
        
    except FileNotFoundError:
        print(f"Error: The file {input_file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == '__main__':
    # Get the absolute path to the project's root directory
    PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Example usage - analyze news data collected from data_collection.py
    ticker = 'AAPL'
    input_file = os.path.join(PROJ_ROOT, 'data', f'{ticker}_news.csv')
    output_file = os.path.join(PROJ_ROOT, 'data', f'{ticker}_sentiment_analysis.csv')
    
    print(f"Looking for news data file: {input_file}")
    
    if os.path.exists(input_file):
        print(f"Found news data! Starting sentiment analysis...")
        result = process_sentiment_data(input_file, output_file, text_column='headline')
        if result:
            print(f"\nSentiment analysis completed successfully!")
            print(f"Results saved to: {result}")
    else:
        print(f" News data file not found: {input_file}")
        print(f" Run data_collection.py first to collect news data.")
        print(f"\n Example usage:")
        print(f"   process_sentiment_data('data/your_news.csv', 'data/sentiment_results.csv')")
