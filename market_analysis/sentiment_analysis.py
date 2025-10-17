from .sentiment.vader_sentiment import analyze_sentiment, process_sentiment_data
from .sentiment.advanced_sentiment import extract_entities, extract_topics

def analyze_news_sentiment(news_file, output_file=None, daily_output_file=None):
    import os
    import pandas as pd
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Process the sentiment data
        sentiment_df = process_sentiment_data(news_file, output_file)
        
        # Aggregate sentiment data by date if needed
        if daily_output_file:
            # Make sure the date column is datetime
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
            
            # Group by date
            daily_sentiment = sentiment_df.groupby('date').agg({
                'compound': 'mean',
                'pos': 'mean', 
                'neu': 'mean',
                'neg': 'mean',
                'headline': 'count'
            }).reset_index()
            
            # Rename columns
            daily_sentiment = daily_sentiment.rename(columns={
                'compound': 'sentiment_score',
                'pos': 'positive_score',
                'neu': 'neutral_score', 
                'neg': 'negative_score',
                'headline': 'headline_count'
            })
            
            # Save daily sentiment if output file is specified
            os.makedirs(os.path.dirname(os.path.abspath(daily_output_file)), exist_ok=True)
            daily_sentiment.to_csv(daily_output_file, index=False)
            logger.info(f"Daily sentiment saved to {daily_output_file}")
            
            return sentiment_df, daily_sentiment
        
        return sentiment_df, None
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        return pd.DataFrame(), pd.DataFrame()

def extract_named_entities(news_file, output_file=None):
    from .sentiment.advanced_sentiment import extract_entities_from_file
    return extract_entities_from_file(news_file)