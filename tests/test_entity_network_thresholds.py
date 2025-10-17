import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the entity network chart function
from market_analysis.visualization.interactive_charts import create_entity_network_chart

def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    ticker = 'AAPL'
    
    # Path to entities file
    entities_file = os.path.join(data_dir, f"{ticker}_entities.csv")
    
    if os.path.exists(entities_file):
        print(f"Entity file found: {entities_file}")
        
        # Create charts with different minimum occurrence thresholds
        for min_occurrences in [1, 3, 5]:
            print(f"\nCreating entity network chart with min_occurrences = {min_occurrences}...")
            chart_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                    'charts', 
                                    f"{ticker}_entity_network_min{min_occurrences}.html")
            
            # Create the chart
            fig = create_entity_network_chart(entities_file, 
                                            min_occurrences=min_occurrences, 
                                            output_file=chart_file)
            
            if fig is not None:
                print(f"Entity network chart created successfully: {chart_file}")
            else:
                print(f"Failed to create entity network chart with min_occurrences = {min_occurrences}")
    else:
        print(f"Entity file not found: {entities_file}")

if __name__ == "__main__":
    main()