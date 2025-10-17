import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('interactive_charts')

# Import settings
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import settings

# Try to import Plotly (it's optional)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("Plotly not installed. Interactive charts will not be available.")
    logger.warning("Install with: pip install plotly")
    PLOTLY_AVAILABLE = False

def create_stock_sentiment_chart(stock_file, sentiment_file, output_file=None):
    if not PLOTLY_AVAILABLE:
        logger.error("Plotly not installed. Cannot create interactive chart.")
        return None
    
    try:
        # Load data: handle both clean and original stock data format
        try:
            stock_data = pd.read_csv(stock_file)
            
            # Check if it's the original format with the extra header row
            if len(stock_data) > 0 and 'AAPL' in str(stock_data.iloc[0].values):
                stock_data = pd.read_csv(stock_file, skiprows=1)
                # Rename columns to lowercase for consistency
                stock_data.columns = ['date', 'close', 'high', 'low', 'open', 'volume']
                logger.info("Detected and skipped extra header row in stock data")
        except Exception as e:
            logger.error(f"Error loading stock data: {e}")
            return None
            
        sentiment_data = pd.read_csv(sentiment_file)
        
        # Ensure date columns are datetime
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
        
        # Merge data
        merged_data = pd.merge(stock_data, sentiment_data, on='date', how='left')
        
        # Extract ticker symbol from filename
        ticker = os.path.basename(stock_file).split('_')[0].upper()
        
        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add candlestick chart for stock price
        fig.add_trace(
            go.Candlestick(
                x=merged_data['date'],
                open=merged_data['open'],
                high=merged_data['high'],
                low=merged_data['low'],
                close=merged_data['close'],
                name="Price"
            ),
            secondary_y=False
        )
        
        # Add sentiment score line
        fig.add_trace(
            go.Scatter(
                x=merged_data['date'],
                y=merged_data['sentiment_score'],
                mode='lines',
                name='Sentiment',
                line=dict(color='purple', width=2)
            ),
            secondary_y=True
        )
        
        # Add volume as bar chart
        fig.add_trace(
            go.Bar(
                x=merged_data['date'],
                y=merged_data['volume'],
                name='Volume',
                marker=dict(color='rgba(128, 128, 128, 0.5)')
            ),
            secondary_y=False
        )
        
        # Update axes titles
        fig.update_yaxes(title_text="Price ($)", secondary_y=False)
        fig.update_yaxes(title_text="Sentiment Score", secondary_y=True)
        
        # Update layout
        fig.update_layout(
            title=f"{ticker} Stock Price and Sentiment Analysis",
            xaxis_title="Date",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=800,
            template='plotly_white'
        )
        
        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        # Save to HTML if output file provided
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            fig.write_html(output_file)
            logger.info(f"Interactive chart saved to {output_file}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating interactive chart: {e}")
        return None

def create_correlation_chart(stock_file, sentiment_file, output_file=None):
    if not PLOTLY_AVAILABLE:
        logger.error("Plotly not installed. Cannot create correlation chart.")
        return None
    
    try:
        logger.info(f"Creating correlation chart with stock file: {stock_file}")
        logger.info(f"Sentiment file: {sentiment_file}")
        
        # Load stock data
        stock_data = pd.read_csv(stock_file)
        logger.info(f"Loaded stock data with columns: {stock_data.columns.tolist()}")
        
        # Check if it's the original format with the extra header row
        if len(stock_data) > 0 and 'AAPL' in str(stock_data.iloc[0].values):
            stock_data = pd.read_csv(stock_file, skiprows=1)
            # Rename columns to lowercase for consistency
            stock_data.columns = ['date', 'close', 'high', 'low', 'open', 'volume']
            logger.info("Detected and skipped extra header row in stock data")
        
        # Load sentiment data
        sentiment_data = pd.read_csv(sentiment_file)
        logger.info(f"Loaded sentiment data with columns: {sentiment_data.columns.tolist()}")
        
        # Ensure date columns are datetime
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
        
        # Merge data
        merged_data = pd.merge(stock_data, sentiment_data, on='date', how='left')
        logger.info(f"Merged data shape: {merged_data.shape}")
        
        # Calculate daily returns
        merged_data['price_change'] = merged_data['close'].pct_change()
        
        # Drop NaN values but keep zero sentiment scores
        merged_data = merged_data.dropna(subset=['price_change'])
        # Make sure sentiment_score exists, set to 0 if NaN
        if 'sentiment_score' in merged_data.columns:
            merged_data['sentiment_score'] = merged_data['sentiment_score'].fillna(0)
        logger.info(f"Valid data points for correlation: {len(merged_data)}")
        
        # Calculate correlation coefficient
        correlation = merged_data['sentiment_score'].corr(merged_data['price_change'])
        logger.info(f"Calculated correlation: {correlation:.4f}")
        
        # Create a basic scatter plot - with trendline handling
        try:
            # Use all data points to ensure we have enough points for visualization
            filtered_data = merged_data
            
            # Log the number of data points we're using
            logger.info(f"Using {len(filtered_data)} data points for correlation visualization")
                
            # Try with OLS trendline first
            fig = px.scatter(
                filtered_data,
                x='sentiment_score',
                y='price_change',
                trendline='ols',  # Add trend line with ordinary least squares
                hover_data=['date'],  # Show date on hover
                labels={
                    'sentiment_score': 'Sentiment Score',
                    'price_change': 'Price Change (%)'
                },
                title='Correlation between Sentiment and Price Change'
            )
        except ImportError:
            # Fall back to basic plot without trendline if statsmodels is not installed
            logger.warning("statsmodels not installed. Creating correlation chart without trendline.")
            # Use all data points for better visualization
            filtered_data = merged_data
            
            # Log the number of data points we're using
            logger.info(f"Using {len(filtered_data)} data points for correlation visualization without trendline")
                
            fig = px.scatter(
                filtered_data,
                x='sentiment_score',
                y='price_change',
                hover_data=['date'],  # Show date on hover
                labels={
                    'sentiment_score': 'Sentiment Score',
                    'price_change': 'Price Change (%)'
                },
                title='Correlation between Sentiment and Price Change'
            )
        
        # Format the y-axis as percentage
        fig.update_layout(
            yaxis=dict(
                tickformat='.1%',
                title='Price Change (%)'
            ),
            xaxis=dict(
                title='Sentiment Score',
                range=[-1, 1]  # Set x-axis range to the full sentiment range
            )
        )
        
        # Add annotation with correlation coefficient
        fig.add_annotation(
            xref='paper',
            yref='paper',
            x=0.02,
            y=0.98,
            text=f'Correlation: {correlation:.3f}',
            showarrow=False,
            font=dict(size=14),
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
        
        # Save to HTML if output file provided
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            fig.write_html(output_file)
            logger.info(f"Correlation chart saved to {output_file}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating correlation chart: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Create an error figure
        fig = go.Figure()
        fig.update_layout(
            title=f"Error creating correlation chart: {str(e)}",
            xaxis_title="",
            yaxis_title=""
        )
        fig.add_annotation(
            text="An error occurred while generating this chart. Please check the logs.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
        
        # Format the y-axis as percentage
        fig.update_layout(
            yaxis=dict(
                tickformat='.1%',
                title='Price Change (%)'
            ),
            xaxis=dict(
                title='Sentiment Score',
                range=[-1, 1]  # Set x-axis range to the full sentiment range
            )
        )
        
        # Calculate correlation coefficient
        # We are using valid_data which has NaN values removed
        try:
            correlation = merged_data['sentiment_score'].corr(merged_data['price_change'])
            logger.info(f"Calculated correlation: {correlation:.4f}")
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            correlation = 0  # Default value if correlation can't be calculated
        
        # Add annotation with correlation coefficient
        fig.add_annotation(
            xref='paper',
            yref='paper',
            x=0.02,
            y=0.98,
            text=f'Correlation: {correlation:.3f}',
            showarrow=False,
            font=dict(size=14),
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
        
        # Save to HTML if output file provided
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            fig.write_html(output_file)
            logger.info(f"Correlation chart saved to {output_file}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating correlation chart: {e}")
        return None

def create_entity_network_chart(entities_file, min_occurrences=2, output_file=None):
    if not PLOTLY_AVAILABLE:
        logger.error("Plotly not installed. Cannot create network chart.")
        return None
    
    # Check if the entities file exists first
    if not os.path.exists(entities_file):
        logger.error(f"Entities file not found: {entities_file}")
        return None
        
    # Try to import networkx
    try:
        import networkx as nx
    except ImportError:
        logger.error("NetworkX not installed. Cannot create network chart.")
        logger.error("Install with: pip install networkx")
        return None
    
    try:
        # Load entity data
        entities_df = pd.read_csv(entities_file)
        
        # Debug information
        logger.info(f"Loaded entities file with {len(entities_df)} records and columns: {entities_df.columns.tolist()}")
        logger.info(f"Entity types in data: {entities_df['entity_type'].unique().tolist() if 'entity_type' in entities_df.columns else 'No entity_type column'}")
        
        # Count occurrences and print top entities for debugging
        entity_counts = entities_df['entity'].value_counts()
        logger.info(f"Top 5 entities by count: {entity_counts.head(5).to_dict()}")
        
        # Always use minimum occurrence of 1 to ensure we get enough entities
        # Use the provided threshold only if it results in at least 5 entities
        min_occ = min_occurrences
        frequent_entities = entity_counts[entity_counts >= min_occ].index.tolist()
        
        # If we don't have at least 5 entities, reduce the threshold until we do
        while len(frequent_entities) < 5 and min_occ > 1:
            min_occ -= 1
            frequent_entities = entity_counts[entity_counts >= min_occ].index.tolist()
        
        logger.info(f"Found {len(frequent_entities)} entities with {min_occ}+ occurrences (requested: {min_occurrences})")
        
        if len(frequent_entities) < 2:
            logger.warning(f"Not enough entities in data file. Need at least 2 entities.")
            return go.Figure().update_layout(
                title="Not enough entities in data",
                xaxis_title="",
                yaxis_title="",
                annotations=[dict(
                    text="Entity data does not contain enough entities for visualization.\nTry extracting more entities or lowering the threshold.",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16)
                )]
            )
        
        # Filter dataframe to frequent entities only
        filtered_df = entities_df[entities_df['entity'].isin(frequent_entities)]
        
        # Create a graph
        G = nx.Graph()
        
        # Add nodes (entities)
        entity_count = 0
        for entity, count in entity_counts[entity_counts >= min_occ].items():
            # Get entity type safely with a default in case of issues
            try:
                entity_rows = filtered_df[filtered_df['entity'] == entity]
                if len(entity_rows) > 0:
                    entity_type = entity_rows['entity_type'].iloc[0]
                else:
                    # Fallback if entity doesn't have a type record
                    entity_type = "MISC"
            except Exception as e:
                logger.warning(f"Error getting entity type for {entity}: {e}")
                entity_type = "MISC"
                
            # Only add the top 50 entities to avoid visualization clutter
            if entity_count < 50:
                G.add_node(entity, size=count, group=entity_type)
                entity_count += 1
        
        # Add edges based on co-occurrence in same text
        text_groups = entities_df.groupby('source_text')
        
        for _, group in text_groups:
            entities_in_text = group['entity'].tolist()
            # Create edges between all pairs of entities in the same text
            for i, entity1 in enumerate(entities_in_text):
                if entity1 not in frequent_entities:
                    continue
                for entity2 in entities_in_text[i+1:]:
                    if entity2 not in frequent_entities:
                        continue
                    if G.has_edge(entity1, entity2):
                        G[entity1][entity2]['weight'] += 1
                    else:
                        G.add_edge(entity1, entity2, weight=1)
        
        # Get node positions using a layout algorithm
        pos = nx.spring_layout(G, seed=42)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2]['weight'])
        
        # Scale edge weights for visualization but use a single width
        # Plotly doesn't support array of line widths in this context
        max_weight = max(edge_weights) if edge_weights else 1
        avg_width = 2  # Use a fixed width for all edges
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=avg_width, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        # Define colors for different entity types
        color_map = {
            'PERSON': '#e41a1c',
            'ORG': '#377eb8',
            'GPE': '#4daf4a',
            'LOC': '#984ea3',
            'PRODUCT': '#ff7f00',
            'DATE': '#ffff33',
            'MONEY': '#a65628',
            'PERCENT': '#f781bf',
            'TIME': '#999999',
        }
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Get entity attributes with defaults
            size = G.nodes[node].get('size', 1)
            entity_type = G.nodes[node].get('group', 'MISC')
            
            node_text.append(f"{node} ({entity_type})<br>Occurrences: {size}")
            node_size.append(size * 5)  # Scale size for visualization
            
            # Set node color based on entity type
            node_color.append(color_map.get(entity_type, '#1f77b4'))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=False,
                color=node_color,
                size=node_size,
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        title=dict(text='Entity Network Graph', font=dict(size=16)),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=700,
                        template='plotly_white'
                    )
                )
        
        # Add legend for entity types
        for entity_type, color in color_map.items():
            # Safely get group attributes to check for entity type presence
            entity_types_in_graph = [G.nodes[node].get('group', 'MISC') for node in G.nodes()]
            if entity_type in entity_types_in_graph:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    name=entity_type
                ))
        
        # Save to HTML if output file provided
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            fig.write_html(output_file)
            logger.info(f"Entity network chart saved to {output_file}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating entity network chart: {e}")
        return None

if __name__ == '__main__':
    # Example usage
    # Get the absolute path to the project's root directory
    PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Example ticker
    ticker = 'AAPL'
    stock_file = os.path.join(PROJ_ROOT, settings.DATA_DIR, f'{ticker}_stock_data.csv')
    sentiment_file = os.path.join(PROJ_ROOT, settings.DATA_DIR, f'{ticker}_daily_sentiment.csv')
    
    if not os.path.exists(sentiment_file):
        sentiment_file = os.path.join(PROJ_ROOT, settings.DATA_DIR, f'{ticker}_sentiment_analysis.csv')
    
    # Check if files exist
    if not os.path.exists(stock_file) or not os.path.exists(sentiment_file):
        print(f"Required data files not found.")
        print(f"Run data_collection.py and sentiment_analysis.py first.")
        sys.exit(1)
    
    print(f"Creating interactive charts for {ticker}...")
    
    # Create output directory for charts
    charts_dir = os.path.join(PROJ_ROOT, 'charts')
    os.makedirs(charts_dir, exist_ok=True)
    
    # Create and save stock/sentiment chart
    if PLOTLY_AVAILABLE:
        print("Creating price and sentiment chart...")
        stock_sentiment_file = os.path.join(charts_dir, f'{ticker}_price_sentiment.html')
        fig1 = create_stock_sentiment_chart(stock_file, sentiment_file, stock_sentiment_file)
        
        print("Creating correlation chart...")
        correlation_file = os.path.join(charts_dir, f'{ticker}_sentiment_correlation.html')
        fig2 = create_correlation_chart(stock_file, sentiment_file, correlation_file)
        
        # Try to create entity network chart if entities file exists
        entities_file = os.path.join(PROJ_ROOT, settings.DATA_DIR, f'{ticker}_entities.csv')
        if os.path.exists(entities_file):
            print("Creating entity network chart...")
            try:
                network_file = os.path.join(charts_dir, f'{ticker}_entity_network.html')
                fig3 = create_entity_network_chart(entities_file, min_occurrences=2, output_file=network_file)
                if fig3:
                    print(f"Entity network chart created: {network_file}")
            except Exception as e:
                print(f"Error creating entity network chart: {e}")
        
        print("\nInteractive charts created successfully!")
        print(f"Stock and sentiment chart: {stock_sentiment_file}")
        print(f"Correlation chart: {correlation_file}")
    else:
        print("Plotly not installed. Interactive charts cannot be created.")
        print("Install with: pip install plotly")