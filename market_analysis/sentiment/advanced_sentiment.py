import pandas as pd
import numpy as np
import os
import logging
import re
from collections import Counter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('advanced_sentiment')

# Import settings
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import settings

try:
    import spacy
    SPACY_AVAILABLE = True
    # Try to load the model - this will fail if it's not installed
    try:
        nlp = spacy.load("en_core_web_sm")
    except IOError:
        logger.warning("spaCy model 'en_core_web_sm' not found. Entity extraction will be limited.")
        SPACY_AVAILABLE = False
except ImportError:
    logger.warning("spaCy not installed. Entity extraction will be limited.")
    SPACY_AVAILABLE = False

def extract_entities(text):
    if not SPACY_AVAILABLE:
        logger.warning("spaCy not available. Entity extraction skipped.")
        return {}
        
    if not text or pd.isna(text):
        return {}
    
    try:
        # Make sure text is a string and not too long to avoid processing issues
        text_str = str(text)
        if len(text_str) > 1000:  # Limit very long texts
            text_str = text_str[:1000]
            
        # Process with spaCy
        doc = nlp(text_str)
        
        # Extract entities and filter out very short entities
        entities = {}
        for ent in doc.ents:
            # Only include entities with reasonable length and valid entity types
            if len(ent.text) >= 2 and ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'DATE', 'MONEY', 'PERCENT']:
                entities[ent.text] = ent.label_
                
        return entities
    except Exception as e:
        logger.warning(f"Error extracting entities: {e}")
        return {}

def extract_entities_from_file(input_file, text_column='headline'):
    try:
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available. Install with: pip install spacy")
            logger.warning("Then download the model with: python -m spacy download en_core_web_sm")
            return None, None
            
        logger.info(f"Loading data from {input_file}...")
        data = pd.read_csv(input_file)
        
        # Check if the text column exists
        if text_column not in data.columns:
            possible_columns = ['headline', 'text', 'title', 'content']
            for col in possible_columns:
                if col in data.columns:
                    text_column = col
                    break
        
        if text_column not in data.columns:
            logger.error(f"No suitable text column found. Available columns: {list(data.columns)}")
            return None, None
            
        logger.info(f"Extracting entities from '{text_column}' column...")
        
        # Extract entities from each text
        entity_data = []
        
        for i, row in data.iterrows():
            text = row[text_column]
            if pd.isna(text) or not text:
                continue
                
            entities = extract_entities(text)
            
            for entity, entity_type in entities.items():
                entity_data.append({
                    'date': row.get('date', None),
                    'entity': entity,
                    'entity_type': entity_type,
                    'source_text': text
                })
        
        if entity_data:
            entities_df = pd.DataFrame(entity_data)
            
            # Save to CSV
            base_name = os.path.basename(input_file)
            name_parts = os.path.splitext(base_name)
            output_file = os.path.join(settings.DATA_DIR, f"{name_parts[0]}_entities{name_parts[1]}")
            
            entities_df.to_csv(output_file, index=False)
            
            # Also save a standardized version for the dashboard
            ticker = name_parts[0].split('_')[0]  # Extract ticker from filename
            std_output_file = os.path.join(settings.DATA_DIR, f"{ticker}_entities.csv")
            entities_df.to_csv(std_output_file, index=False)
            
            logger.info(f"Extracted {len(entities_df)} entities from {len(data)} texts")
            logger.info(f"Results saved to: {output_file}")
            logger.info(f"Also saved to dashboard-compatible format: {std_output_file}")
            
            # Provide a summary of entity types found
            type_counts = entities_df['entity_type'].value_counts()
            logger.info("Entity types found:")
            logger.info(type_counts.to_string())
            
            return entities_df, output_file
        else:
            logger.warning("No entities found in the texts")
            return None, None
            
    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        return None, None

def extract_topics(texts, num_topics=5):
    # Convert to list if it's a series
    if hasattr(texts, 'tolist'):
        texts = texts.tolist()
        
    # Remove None values and convert to strings
    texts = [str(t) for t in texts if t is not None]
    
    # Combine all texts
    combined_text = ' '.join(texts).lower()
    
    # Remove punctuation and split into words
    words = re.findall(r'\b[a-z]{3,}\b', combined_text)  # Only words with 3+ letters
    
    # Remove common stop words
    stop_words = {
        'the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'for', 'on', 'with',
        'said', 'its', 'was', 'by', 'as', 'at', 'from', 'be', 'an', 'are', 'has',
        'have', 'had', 'will', 'would', 'could', 'should', 'may', 'can'
    }
    
    filtered_words = [word for word in words if word not in stop_words]
    
    # Count words and return the most common
    word_counts = Counter(filtered_words)
    
    return word_counts.most_common(num_topics)

def extract_topics_from_file(input_file, text_column='headline', num_topics=10):
    try:
        logger.info(f"Loading data from {input_file}...")
        data = pd.read_csv(input_file)
        
        # Check if the text column exists
        if text_column not in data.columns:
            possible_columns = ['headline', 'text', 'title', 'content']
            for col in possible_columns:
                if col in data.columns:
                    text_column = col
                    break
        
        if text_column not in data.columns:
            logger.error(f"No suitable text column found. Available columns: {list(data.columns)}")
            return None
            
        logger.info(f"Extracting topics from '{text_column}' column...")
        
        topics = extract_topics(data[text_column], num_topics)
        
        logger.info("Main topics found:")
        for topic, count in topics:
            logger.info(f"  {topic}: {count} occurrences")
            
        return topics
            
    except Exception as e:
        logger.error(f"Error extracting topics: {e}")
        return None

if __name__ == '__main__':
    # Example usage
    # Get the absolute path to the project's root directory
    PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Example usage - analyze sentiment data
    ticker = 'AAPL'
    input_file = os.path.join(PROJ_ROOT, 'data', f'{ticker}_sentiment_analysis.csv')
    
    if os.path.exists(input_file):
        print(f"Found sentiment data! Starting advanced analysis...")
        
        # Extract entities if spaCy is available
        if SPACY_AVAILABLE:
            print("Extracting named entities...")
            entities_df, entities_file = extract_entities_from_file(input_file)
            
            if entities_df is not None:
                print(f"Entity extraction completed successfully!")
                print(f"Results saved to: {entities_file}")
                
                # Show top entities by type
                org_entities = entities_df[entities_df['entity_type'] == 'ORG']['entity'].value_counts().head(5)
                print("\nTop mentioned organizations:")
                for entity, count in org_entities.items():
                    print(f"  {entity}: {count} mentions")
        else:
            print("spaCy not available. Entity extraction skipped.")
            print("Install with: pip install spacy")
            print("Then download the model with: python -m spacy download en_core_web_sm")
        
        # Extract topics
        print("\nExtracting main topics...")
        topics = extract_topics_from_file(input_file)
        
        if topics:
            print("\nMain topics in the news:")
            for topic, count in topics:
                print(f"  {topic}: {count} occurrences")
    else:
        print(f"Sentiment data file not found: {input_file}")
        print(f"Run sentiment_analysis.py first to perform sentiment analysis.")
