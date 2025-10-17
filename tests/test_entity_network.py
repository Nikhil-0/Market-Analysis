import os
import sys
from market_analysis.visualization.interactive_charts import create_entity_network_chart

# Get the absolute path to the project's root directory
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))

# Example ticker
ticker = 'AAPL'

# Try to create entity network chart if entities file exists
entities_file = os.path.join(PROJ_ROOT, 'data', f'{ticker}_entities.csv')
if os.path.exists(entities_file):
    print(f"Entity file found: {entities_file}")
    print("Creating entity network chart...")
    try:
        network_file = os.path.join(PROJ_ROOT, 'charts', f'{ticker}_entity_network.html')
        os.makedirs(os.path.dirname(network_file), exist_ok=True)
        fig3 = create_entity_network_chart(entities_file, min_occurrences=1, output_file=network_file)
        if fig3:
            print(f"Entity network chart created successfully: {network_file}")
    except Exception as e:
        print(f"Error creating entity network chart: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"Entity file not found: {entities_file}")