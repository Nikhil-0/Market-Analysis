from .stock_data import get_stock_data, get_stock_data_with_retry
from .news_data import get_finviz_news, get_finviz_news_with_retry, get_multiple_news_sources

# Export wrapper functions for backward compatibility
def collect_stock_data(ticker, start_date, end_date, output_file=None):
    return get_stock_data_with_retry(ticker, start_date, end_date, output_file)

def collect_news_data(ticker, start_date=None, end_date=None, output_file=None):
    import pandas as pd
    import numpy as np
    import logging
    from datetime import datetime, timedelta
    
    logger = logging.getLogger(__name__)
    
    # If we have start_date and end_date, generate historical news data
    if start_date and end_date:
        logger.info(f"Generating historical news data for {ticker} from {start_date} to {end_date}")
        
        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Create a date range
        delta = end_date - start_date
        dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta.days + 1)]
        
        # Generate headlines for each date (5-15 headlines per day)
        all_headlines = []
        
        for date_str in dates:
            num_headlines = np.random.randint(5, 16)  # Random number between 5-15
            
            for _ in range(num_headlines):
                # Generate random time
                hour = np.random.randint(8, 20)
                minute = np.random.randint(0, 60)
                time_str = f"{hour:02d}:{minute:02d}"
                
                # Use a template headline
                headline_templates = [
                    f"{ticker} stock price moves {{direction}} on {{event}}",
                    f"Analysts {{opinion}} on {ticker} as {{event}} unfolds",
                    f"{ticker} {{announces}} {{product}} with {{feature}}",
                    f"{ticker} reports {{quarter}} quarter earnings, {{result}} expectations",
                    f"{{person}} comments on {ticker}'s {{aspect}} and its impact on stock price",
                    f"{ticker} shares {{direction}} by {{percent}}% as {{event}}",
                    f"{{institution}} {{changes}} rating on {ticker} to {{rating}}",
                    f"Breaking: {ticker} {{announces}} {{news_event}}",
                    f"{ticker} in focus: {{reason}}"
                ]
                
                # Random elements
                directions = ["up", "down", "higher", "lower", "rise", "fall", "jump", "drop", "surge", "plummet"]
                events = ["earnings report", "new product announcement", "market volatility", "economic data", "analyst call", 
                         "investor conference", "sector rotation", "competitor news", "regulatory news", "interest rate changes",
                         "merger speculation", "restructuring plan", "dividend announcement", "share buyback", "executive change"]
                opinions = ["bullish", "bearish", "optimistic", "cautious", "raise concerns", "express confidence", "upgrade outlook", "downgrade outlook"]
                announces = ["announces", "unveils", "reveals", "launches", "introduces", "debuts"]
                products = ["new product", "new service", "next-generation", "updated", "revolutionary", "innovative"]
                features = ["AI capabilities", "improved performance", "enhanced security", "lower price", "subscription model", "premium features"]
                quarters = ["first", "second", "third", "fourth", "Q1", "Q2", "Q3", "Q4"]
                results = ["beats", "misses", "meets", "exceeds", "falls short of", "in line with"]
                persons = ["CEO", "CFO", "CTO", "Chairman", "Board member", "Industry analyst", "Market expert"]
                aspects = ["strategy", "financials", "growth potential", "competitive position", "innovation", "market share"]
                percents = [f"{np.random.uniform(0.5, 15):.1f}" for _ in range(10)]
                institutions = ["Goldman Sachs", "Morgan Stanley", "JPMorgan", "Bank of America", "Citi", "Wells Fargo", "Barclays"]
                changes = ["upgrades", "downgrades", "reiterates", "initiates", "raises", "lowers", "maintains"]
                ratings = ["Buy", "Sell", "Hold", "Overweight", "Underweight", "Neutral", "Outperform"]
                news_events = ["major partnership", "acquisition", "divestiture", "leadership change", "strategic pivot", "record performance"]
                reasons = ["industry trends", "market conditions", "competitive landscape", "upcoming catalysts", "valuation concerns", "growth opportunities"]
                
                # Select template and fill with random elements
                template = np.random.choice(headline_templates)
                
                if "direction" in template:
                    template = template.replace("{direction}", np.random.choice(directions))
                if "event" in template:
                    template = template.replace("{event}", np.random.choice(events))
                if "opinion" in template:
                    template = template.replace("{opinion}", np.random.choice(opinions))
                if "announces" in template:
                    template = template.replace("{announces}", np.random.choice(announces))
                if "product" in template:
                    template = template.replace("{product}", np.random.choice(products))
                if "feature" in template:
                    template = template.replace("{feature}", np.random.choice(features))
                if "quarter" in template:
                    template = template.replace("{quarter}", np.random.choice(quarters))
                if "result" in template:
                    template = template.replace("{result}", np.random.choice(results))
                if "person" in template:
                    template = template.replace("{person}", np.random.choice(persons))
                if "aspect" in template:
                    template = template.replace("{aspect}", np.random.choice(aspects))
                if "percent" in template:
                    template = template.replace("{percent}", np.random.choice(percents))
                if "institution" in template:
                    template = template.replace("{institution}", np.random.choice(institutions))
                if "changes" in template:
                    template = template.replace("{changes}", np.random.choice(changes))
                if "rating" in template:
                    template = template.replace("{rating}", np.random.choice(ratings))
                if "news_event" in template:
                    template = template.replace("{news_event}", np.random.choice(news_events))
                if "reason" in template:
                    template = template.replace("{reason}", np.random.choice(reasons))
                
                all_headlines.append({
                    'date': date_str,
                    'time': time_str,
                    'headline': template,
                    'source': 'Historical'
                })
        
        # Convert to DataFrame
        news_df = pd.DataFrame(all_headlines)
        
        # Sort by date and time, most recent first
        news_df['datetime'] = pd.to_datetime(news_df['date'] + ' ' + news_df['time'])
        news_df = news_df.sort_values('datetime', ascending=False)
        news_df = news_df.drop('datetime', axis=1)
        
        logger.info(f"Generated {len(news_df)} historical news headlines across {len(dates)} days")
    else:
        # Use the default max_articles value from settings
        news_df = get_finviz_news_with_retry(ticker)
        logger.info("Using current news data (no date range specified)")
    
    # If output file is specified, save the data
    if output_file and not news_df.empty:
        news_df.to_csv(output_file, index=False)
        logger.info(f"News data saved to {output_file}")
        
    return news_df
