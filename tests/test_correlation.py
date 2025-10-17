import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the correlation chart function
from market_analysis.visualization.interactive_charts import create_correlation_chart

def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    ticker = 'AAPL'
    
    # Paths to data files
    stock_file = os.path.join(data_dir, f"{ticker}_stock_data_clean.csv")
    if not os.path.exists(stock_file):
        stock_file = os.path.join(data_dir, f"{ticker}_stock_data.csv")
    
    sentiment_file = os.path.join(data_dir, f"{ticker}_daily_sentiment.csv")
    
    print(f"Stock file found: {os.path.exists(stock_file)}")
    print(f"Sentiment file found: {os.path.exists(sentiment_file)}")
    
    if os.path.exists(stock_file) and os.path.exists(sentiment_file):
        print("Creating correlation chart...")
        
        # Load the data to inspect
        stock_data = pd.read_csv(stock_file)
        sentiment_data = pd.read_csv(sentiment_file)
        
        print(f"Stock data shape: {stock_data.shape}")
        print(f"Sentiment data shape: {sentiment_data.shape}")
        
        # Print some sample data
        print("\nSentiment data sample:")
        print(sentiment_data.head())
        
        # Create the chart
        chart_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'charts', f"{ticker}_correlation_test.html")
        fig = create_correlation_chart(stock_file, sentiment_file, output_file=chart_file)
        
        print(f"\nChart saved to: {chart_file}")
        print("Open this file in your web browser to view the chart.")

if __name__ == "__main__":
    main()