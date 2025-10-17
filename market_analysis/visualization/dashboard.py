import os
import pandas as pd
import logging
from datetime import datetime, timedelta
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dashboard')

# Import settings
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import settings

# Try to import Plotly and Dash (optional)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("Plotly not installed. Interactive charts will not be available.")
    logger.warning("Install with: pip install plotly")
    PLOTLY_AVAILABLE = False

try:
    import dash
    from dash import dcc, html, dash_table
    from dash.dependencies import Input, Output
    DASH_AVAILABLE = True
except ImportError:
    logger.warning("Dash not installed. Interactive dashboards will not be available.")
    logger.warning("Install with: pip install dash")
    DASH_AVAILABLE = False

# Import our interactive charts module
from market_analysis.visualization.interactive_charts import (
    create_stock_sentiment_chart,
    create_correlation_chart,
    create_entity_network_chart
)


# Simple helper for formatted spans
def create_formatted_span(value, color=None, bold=False):
    style = {}
    if color:
        style['color'] = color
    if bold:
        style['fontWeight'] = 'bold'
    
    return html.Span(value, style=style)

# Helper to stringify elements
def element_to_string(elem):
    if isinstance(elem, (html.Span, html.Div)):
        return str(elem)
    return str(elem)

# Helper function to parse dates safely
def parse_date(date_input):
    if not date_input:
        return None
    
    # If it's already a datetime.date object, return it directly
    from datetime import date
    if isinstance(date_input, date):
        return date_input
    
    # Handle string dates (most common from date picker)
    try:
        if isinstance(date_input, str):
            # Handle ISO format dates with optional 'T' separator
            if 'T' in date_input:
                date_part = date_input.split('T')[0]
            else:
                date_part = date_input
                
            # Try to parse the date string
            return datetime.strptime(date_part, '%Y-%m-%d').date()
        else:
            # For other types (timestamps, etc.), convert using pandas
            return pd.Timestamp(date_input).date()
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not parse date: {date_input}, error: {e}")
        return None

class MarketAnalysisDashboard:
    # Helper functions for formatting display values with colors
    @staticmethod
    def format_sentiment(value):
        try:
            if value == "N/A" or pd.isna(value):
                return create_formatted_span("N/A", color="gray")
                
            # Convert to float and format
            sentiment_val = float(value)
            # Clip tiny values to zero to avoid displaying -0.0000
            if abs(sentiment_val) < 0.00005:
                sentiment_val = 0.0
                
            # Always ensure 4 decimal places are shown
            formatted_val = f"{sentiment_val:.4f}"
            
            # Simple direct styling
            if sentiment_val > 0.1:
                return create_formatted_span(formatted_val, color="green", bold=True)
            elif sentiment_val < -0.1:
                return create_formatted_span(formatted_val, color="red", bold=True)
            else:
                # Ensure neutral values are always gray, not default black
                return create_formatted_span(formatted_val, color="#888888")
        except Exception as e:
            logger.error(f"Error formatting sentiment: {e}")
            return create_formatted_span(str(value), color="gray")
            
    @staticmethod
    def format_sentiment_component(value):
        try:
            if value == "N/A" or pd.isna(value):
                return create_formatted_span("N/A", color="gray")
                
            # Convert to float and format with 4 decimal places
            sentiment_val = float(value)
            if abs(sentiment_val) < 0.00005:
                sentiment_val = 0.0
                
            formatted_val = f"{sentiment_val:.4f}"
            
            # Always use gray color for sentiment component columns
            return create_formatted_span(formatted_val, color="gray")
        except Exception as e:
            logger.error(f"Error formatting sentiment component: {e}")
            return create_formatted_span(str(value), color="gray")
    
    @staticmethod
    @staticmethod
    def format_count(value):
        try:
            # Critical fix: Don't treat 0 specially
            if value == "N/A" or pd.isna(value):
                return create_formatted_span("0", bold=True)
                
            if isinstance(value, (int, float)):
                # Always format as number with thousands separator, even if zero
                return create_formatted_span(f"{int(value):,}", bold=True)
            elif isinstance(value, str):
                if value.isdigit():
                    return create_formatted_span(f"{int(value):,}", bold=True)
                return create_formatted_span(value, bold=True)
            else:
                return create_formatted_span("0", bold=True)
        except Exception as e:
            logger.error(f"Error formatting count: {e}")
            return create_formatted_span("0", bold=True)

    def __init__(self, data_dir=None, host='127.0.0.1', port=8050, debug=False):
        if not DASH_AVAILABLE or not PLOTLY_AVAILABLE:
            logger.error("Dash or Plotly not installed. Cannot create dashboard.")
            return None
        
        # Set up data directory
        if data_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.data_dir = os.path.join(project_root, settings.DATA_DIR)
        else:
            self.data_dir = data_dir
        
        self.host = host
        self.port = port
        self.debug = debug
        
        # Discover available tickers
        self.available_tickers = self._discover_tickers()
        if not self.available_tickers:
            logger.error("No ticker data found. Cannot create dashboard.")
            return None
            
        # Determine date range for all data
        self.min_date, self.max_date = self._get_data_date_range()
            
        # Initialize Dash app with responsive design
        assets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
        self.app = dash.Dash(__name__, 
                            title='Market Analysis Dashboard',
                            assets_folder=assets_path,
                            suppress_callback_exceptions=True,
                            meta_tags=[
                                {'name': 'viewport', 'content': 'width=device-width, initial-scale=1'},
                                {'name': 'theme-color', 'content': '#1e88e5'},
                                {'name': 'description', 'content': 'Interactive financial analysis with sentiment data'}
                            ],
                            external_stylesheets=[
                                'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css'
                            ])

        # Set custom index to provide a favicon and avoid /favicon.ico errors
        self.app.index_string = (
            "<!DOCTYPE html>\n"
            "<html>\n"
            "  <head>\n"
            "    {%metas%}\n"
            "    <title>Market Analysis Dashboard</title>\n"
            "    <link rel=\"icon\" type=\"image/x-icon\" href=\"/assets/favicon.ico\" />\n"
            "    {%css%}\n"
            "  </head>\n"
            "  <body>\n"
            "    {%app_entry%}\n"
            "    <footer>\n"
            "      {%config%}\n"
            "      {%scripts%}\n"
            "      {%renderer%}\n"
            "    </footer>\n"
            "  </body>\n"
            "</html>\n"
        )

        # Explicit favicon route: prefer assets/favicon.ico, then assets/logo.svg, else 204
        @self.app.server.route('/favicon.ico')
        def _favicon_ok():  # noqa: D401
            try:
                ico_path = os.path.join(assets_path, 'favicon.ico')
                if os.path.exists(ico_path):
                    with open(ico_path, 'rb') as f:
                        data = f.read()
                    return (data, 200, {'Content-Type': 'image/x-icon'})
                svg_path = os.path.join(assets_path, 'logo.svg')
                if os.path.exists(svg_path):
                    with open(svg_path, 'rb') as f:
                        data = f.read()
                    # Some browsers accept SVG favicon via image/svg+xml
                    return (data, 200, {'Content-Type': 'image/svg+xml'})
                return ('', 204)
            except Exception as _e:
                logger.warning(f"/favicon.ico handler error: {_e}")
                return ('', 204)

        # Global error handler to log full tracebacks for 500s
        # Note: rely on Dash/Flask default error handling; avoid overriding to keep clear diagnostics
        
        # Set up the layout
        self._setup_layout()
        
        # Set up callbacks
        self._setup_callbacks()
    
    def _discover_tickers(self):
        tickers = set()
        
        # Look for stock data files
        if os.path.exists(self.data_dir):
            for file in os.listdir(self.data_dir):
                if file.endswith('_stock_data.csv'):
                    ticker = file.split('_')[0].upper()
                    # Check if we also have sentiment data
                    sentiment_file = os.path.join(self.data_dir, f'{ticker}_daily_sentiment.csv')
                    alt_sentiment_file = os.path.join(self.data_dir, f'{ticker}_sentiment_analysis.csv')
                    
                    if os.path.exists(sentiment_file) or os.path.exists(alt_sentiment_file):
                        tickers.add(ticker)
        
        logger.info(f"Discovered {len(tickers)} tickers with data: {', '.join(tickers)}")
        return sorted(list(tickers))
        
    def _get_data_date_range(self):
        min_date = None
        max_date = None
        
        for ticker in self.available_tickers:
            # Try to read stock data file
            stock_file = os.path.join(self.data_dir, f"{ticker}_stock_data_clean.csv")
            if not os.path.exists(stock_file):
                stock_file = os.path.join(self.data_dir, f"{ticker}_stock_data.csv")
                
            if os.path.exists(stock_file):
                try:
                    df = pd.read_csv(stock_file)
                    # Support multiple possible date column names
                    date_col = None
                    for col in ['Date', 'date', 'Timestamp', 'timestamp']:
                        if col in df.columns:
                            date_col = col
                            break
                    if date_col:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        # Drop NaT to avoid min/max issues
                        valid_dates = df[date_col].dropna()
                        if not valid_dates.empty:
                            ticker_min = valid_dates.min()
                            ticker_max = valid_dates.max()
                            if min_date is None or ticker_min < min_date:
                                min_date = ticker_min
                            if max_date is None or ticker_max > max_date:
                                max_date = ticker_max
                        else:
                            logger.warning(f"No valid dates found in {stock_file} using column '{date_col}'")
                    else:
                        logger.warning(f"No date column found in {stock_file}; columns: {list(df.columns)}")
                except Exception as e:
                    logger.warning(f"Error reading {stock_file}: {e}")
        
        # Convert to datetime.date objects for the DatePickerRange
        if min_date and max_date:
            min_date = min_date.date()
            max_date = max_date.date()
            logger.info(f"Data date range: {min_date} to {max_date}")
            return min_date, max_date
        else:
            # Default to a recent date range if no data available
            today = datetime.now().date()
            # Create a more reliable start date (1 year ago) with proper error handling
            try:
                start = today.replace(year=today.year - 1)
            except ValueError:
                # Handle leap year edge case (Feb 29)
                if today.month == 2 and today.day == 29:
                    start = today.replace(year=today.year - 1, day=28)
                else:
                    start = today - pd.Timedelta(days=365)
                    
            logger.warning(f"Could not determine data date range, using default: {start} to {today}")
            return start, today
    
    def _setup_layout(self):
        self.app.layout = html.Div(
            [
                html.Div(
                    [
                        html.Div([
                            html.Img(src='assets/logo.svg', className="dashboard-logo"),
                            html.Div([
                                html.H1("Market Analysis Dashboard", className="dashboard-title"),
                                html.P("Interactive financial analysis with sentiment data", className="dashboard-description"),
                            ]),
                        ], className="header-content"),
                        html.Div([
                            html.Span("Powered by VADER Sentiment Analysis", className="header-badge"),
                            html.Span("AI-Enhanced", className="header-badge accent"),
                        ], className="header-badges"),
                    ],
                    className="header",
                ),
                
                # Ticker selection and info card
                html.Div(
                    [
                        html.Div([
                            html.I(className="fas fa-chart-line ticker-icon"),
                            html.Div([
                                html.Label("Select Stock:", id="ticker-label", className="ticker-label"),
                                dcc.Dropdown(
                                    id="ticker-dropdown",
                                    options=[{"label": ticker, "value": ticker} for ticker in self.available_tickers],
                                    value=self.available_tickers[0] if self.available_tickers else None,
                                    clearable=False,
                                    className="dropdown"
                                ),
                            ], className="ticker-control", **{"role": "group", "aria-labelledby": "ticker-label"}),
                        ], className="ticker-header"),
                        html.Div([
                            html.Div([
                                html.Label("Date Range", id="date-range-display-label", className="info-label"),
                                html.Div("Loading...", id="date-range-display", className="info-value", **{"role": "status", "aria-live": "polite"}),
                            ], className="ticker-info-item", **{"role": "group", "aria-labelledby": "date-range-display-label"}),
                            # Removed headline count display
                            html.Div([
                                html.Label("Avg. Sentiment", id="avg-sentiment-label", className="info-label"),
                                html.Div("Loading...", id="avg-sentiment-display", className="info-value neutral-sentiment", **{"role": "status", "aria-live": "polite"}),
                            ], className="ticker-info-item", **{"role": "group", "aria-labelledby": "avg-sentiment-label"}),
                        ], className="ticker-info"),
                    ],
                    className="ticker-selector",
                ),
                
                # Date range picker
                html.Div(
                    [
                        html.Div([
                            html.I(className="fas fa-calendar-alt date-icon"),
                            html.Label("Select Date Range:", id="date-picker-range-label", className="date-label"),
                        ], className="date-header"),
                        html.Div([
                            dcc.DatePickerRange(
                                id="date-picker-range",
                                min_date_allowed=self.min_date if hasattr(self, 'min_date') else None,
                                max_date_allowed=self.max_date if hasattr(self, 'max_date') else None,
                                start_date=self.min_date if hasattr(self, 'min_date') else None,
                                end_date=self.max_date if hasattr(self, 'max_date') else None,
                                display_format="DD-MM-YYYY",
                                className="date-picker"
                            ),
                        ], className="date-picker-container", **{"role": "group", "aria-labelledby": "date-picker-range-label"}),
                    ],
                    className="date-range-selector card-shadow",
                ),
                
                # Tabs for different analyses
                dcc.Tabs(
                    [
                        # Tab 1: Stock Price and Sentiment
                        dcc.Tab(
                            label="Price & Sentiment",
                            children=[
                                html.Div(
                                    [
                                        html.H3("Stock Price and Sentiment Analysis", id="price-sentiment-title"),
                                        dcc.Loading(
                                            id="loading-price-sentiment",
                                            type="circle",
                                            children=[
                                                dcc.Graph(id="price-sentiment-chart"),
                                            ],
                                        ),
                                    ],
                                    className="chart-container",
                                )
                            ],
                        ),
                        
                        # Tab 2: Correlation Analysis
                        dcc.Tab(
                            label="Correlation Analysis",
                            children=[
                                html.Div(
                                    [
                                        html.H3("Price Change vs. Sentiment Correlation", id="correlation-title"),
                                        dcc.Loading(
                                            id="loading-correlation",
                                            type="circle",
                                            children=[
                                                dcc.Graph(id="correlation-chart"),
                                            ],
                                        ),
                                    ],
                                    className="chart-container",
                                )
                            ],
                        ),
                        
                        # Tab 3: Entity Network
                        dcc.Tab(
                            label="Entity Network",
                            children=[
                                html.Div(
                                    [
                                        html.H3("Named Entity Network Analysis", id="entity-network-title"),
                                        html.Div(
                                            [
                                                html.Label("Minimum Entity Occurrences:"),
                                                dcc.Slider(
                                                    id="entity-occurrence-slider",
                                                    min=1,
                                                    max=10,
                                                    step=1,
                                                    value=2,
                                                    marks={i: str(i) for i in range(1, 11)},
                                                ),
                                            ],
                                            className="slider-container",
                                        ),
                                        dcc.Loading(
                                            id="loading-entity-network",
                                            type="circle",
                                            children=[
                                                html.Div(id="entity-network-container", children=[
                                                    dcc.Graph(id="entity-network-chart"),
                                                ]),
                                            ],
                                        ),
                                    ],
                                    className="chart-container",
                                )
                            ],
                        ),
                        
                        # Tab 4: Data Table
                        dcc.Tab(
                            label="Data Tables",
                            children=[
                                html.Div(
                                    [
                                        html.H3("Stock and Sentiment Data", id="data-title"),
                                        dcc.Tabs(
                                            [
                                                dcc.Tab(
                                                    label="Stock Data",
                                                    children=[
                                                        dcc.Loading(
                                                            id="loading-stock-table",
                                                            type="circle",
                                                            children=[
                                                                html.Div(id="stock-table-container"),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                dcc.Tab(
                                                    label="Sentiment Data",
                                                    children=[
                                                        dcc.Loading(
                                                            id="loading-sentiment-table",
                                                            type="circle",
                                                            children=[
                                                                html.Div(id="sentiment-table-container"),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                    className="table-container",
                                )
                            ],
                        ),
                    ],
                    className="tabs-container",
                ),
                
                # Footer
                html.Div(
                    [
                        html.P(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"),
                        html.P([
                            html.A("GitHub", href="https://github.com/Nikhil-0/Market-Analysis", target="_blank"),
                        ]),
                    ],
                    className="footer",
                ),
            ],
            className="dashboard-container",
        )
    
    def _setup_callbacks(self):       
        # Callback for price and sentiment chart
        @self.app.callback(
            Output("price-sentiment-chart", "figure"),
            [
                Input("ticker-dropdown", "value"),
                Input("date-picker-range", "start_date"),
                Input("date-picker-range", "end_date")
            ]
        )
        def update_price_sentiment_chart(ticker, start_date, end_date):
            try:
                if not ticker:
                    return go.Figure().update_layout(
                        title="No ticker selected",
                        xaxis_title="",
                        yaxis_title="",
                    )
                    
                # Convert string dates to datetime objects if needed
                if start_date and isinstance(start_date, str):
                    start_date = datetime.strptime(start_date.split('T')[0], '%Y-%m-%d').date()
                if end_date and isinstance(end_date, str):
                    end_date = datetime.strptime(end_date.split('T')[0], '%Y-%m-%d').date()
                
                stock_file = os.path.join(self.data_dir, f"{ticker}_stock_data_clean.csv")
                if not os.path.exists(stock_file):
                    stock_file = os.path.join(self.data_dir, f"{ticker}_stock_data.csv")
                    
                sentiment_file = os.path.join(self.data_dir, f"{ticker}_daily_sentiment.csv")
                
                # Try alternative sentiment file name if the first doesn't exist
                if not os.path.exists(sentiment_file):
                    sentiment_file = os.path.join(self.data_dir, f"{ticker}_sentiment_analysis.csv")
                
                if not os.path.exists(stock_file) or not os.path.exists(sentiment_file):
                    return go.Figure().update_layout(
                        title=f"Data files for {ticker} not found",
                        xaxis_title="",
                        yaxis_title="",
                    )
                
                # Apply date filter to the data before creating the chart
                if start_date and end_date:
                    # First, create a full chart for loading the data
                    fig = create_stock_sentiment_chart(stock_file, sentiment_file)
                    
                    # Apply date filter using Plotly's range feature
                    fig.update_layout(
                        xaxis=dict(
                            range=[start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')],
                            rangeslider=dict(
                                range=[start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')]
                            )
                        )
                    )
                    return fig
                else:
                    # If no date range selected, show all data
                    fig = create_stock_sentiment_chart(stock_file, sentiment_file)
                    return fig
                
            except Exception as e:
                logger.error(f"Error updating price sentiment chart: {e}")
                return go.Figure().update_layout(
                    title=f"Error loading chart: {str(e)}",
                    xaxis_title="",
                    yaxis_title="",
                )
        
        # Callback for correlation chart
        @self.app.callback(
            Output("correlation-chart", "figure"),
            [
                Input("ticker-dropdown", "value"),
                Input("date-picker-range", "start_date"),
                Input("date-picker-range", "end_date")
            ]
        )
        def update_correlation_chart(ticker, start_date, end_date):
            # Convert string dates to datetime objects if needed
            if start_date and isinstance(start_date, str):
                start_date = datetime.strptime(start_date.split('T')[0], '%Y-%m-%d').date()
            if end_date and isinstance(end_date, str):
                end_date = datetime.strptime(end_date.split('T')[0], '%Y-%m-%d').date()
            try:
                if not ticker:
                    return go.Figure().update_layout(
                        title="No ticker selected",
                        xaxis_title="",
                        yaxis_title="",
                    )
                
                # Log detailed information for debugging
                logger.info(f"Generating correlation chart for ticker: {ticker}")
                
                # First try the clean stock data file
                stock_file = os.path.join(self.data_dir, f"{ticker}_stock_data_clean.csv")
                if not os.path.exists(stock_file):
                    stock_file = os.path.join(self.data_dir, f"{ticker}_stock_data.csv")
                    logger.info(f"Using original stock data file: {stock_file}")
                else:
                    logger.info(f"Using clean stock data file: {stock_file}")
                    
                sentiment_file = os.path.join(self.data_dir, f"{ticker}_daily_sentiment.csv")
                
                # Try alternative sentiment file name if the first doesn't exist
                if not os.path.exists(sentiment_file):
                    sentiment_file = os.path.join(self.data_dir, f"{ticker}_sentiment_analysis.csv")
                    logger.info(f"Using alternative sentiment file: {sentiment_file}")
                else:
                    logger.info(f"Using sentiment file: {sentiment_file}")
                
                if not os.path.exists(stock_file):
                    logger.error(f"Stock data file not found: {stock_file}")
                    return go.Figure().update_layout(
                        title=f"Stock data file for {ticker} not found",
                        xaxis_title="",
                        yaxis_title="",
                    )
                    
                if not os.path.exists(sentiment_file):
                    logger.error(f"Sentiment data file not found: {sentiment_file}")
                    return go.Figure().update_layout(
                        title=f"Sentiment data file for {ticker} not found",
                        xaxis_title="",
                        yaxis_title="",
                    )
                
                # Create the correlation chart with date filtering
                if start_date and end_date:
                    # Convert dates using our helper
                    start_date_obj = parse_date(start_date)
                    end_date_obj = parse_date(end_date)
                    
                    # Load the data first to filter by date
                    try:
                        stock_data = pd.read_csv(stock_file)
                        sentiment_data = pd.read_csv(sentiment_file)
                        
                        # Ensure date columns are datetime
                        stock_data['date'] = pd.to_datetime(stock_data['date'])
                        sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
                        
                        # Filter by date range using timestamps for proper comparison
                        start_ts = pd.Timestamp(start_date_obj)
                        end_ts = pd.Timestamp(end_date_obj)
                        
                        logger.info(f"Filtering correlation data by date range: {start_ts} to {end_ts}")
                        
                        stock_data_filtered = stock_data[(stock_data['date'] >= start_ts) & 
                                                        (stock_data['date'] <= end_ts)]
                        sentiment_data_filtered = sentiment_data[(sentiment_data['date'] >= start_ts) & 
                                                                (sentiment_data['date'] <= end_ts)]
                        
                        logger.info(f"After filtering: {len(stock_data_filtered)} stock records, {len(sentiment_data_filtered)} sentiment records")
                        
                        # Save filtered data to temporary files
                        temp_stock_file = os.path.join(self.data_dir, f"{ticker}_stock_data_temp.csv")
                        temp_sentiment_file = os.path.join(self.data_dir, f"{ticker}_sentiment_temp.csv")
                        
                        stock_data_filtered.to_csv(temp_stock_file, index=False)
                        sentiment_data_filtered.to_csv(temp_sentiment_file, index=False)
                        
                        # Create chart with filtered data
                        fig = create_correlation_chart(temp_stock_file, temp_sentiment_file)
                        
                        # Clean up temporary files
                        try:
                            os.remove(temp_stock_file)
                            os.remove(temp_sentiment_file)
                        except:
                            pass  # Ignore cleanup errors
                            
                    except Exception as e:
                        logger.error(f"Error filtering data by date range: {e}")
                        # Fallback to unfiltered data
                        fig = create_correlation_chart(stock_file, sentiment_file)
                else:
                    # No date filtering
                    fig = create_correlation_chart(stock_file, sentiment_file)
                
                if fig is None:
                    logger.error("Correlation chart function returned None")
                    return go.Figure().update_layout(
                        title=f"Error: Could not generate correlation chart",
                        xaxis_title="",
                        yaxis_title="",
                    )
                    
                logger.info("Correlation chart generated successfully")
                return fig
                
            except Exception as e:
                logger.error(f"Error updating correlation chart: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return go.Figure().update_layout(
                    title=f"Error loading chart: {str(e)}",
                    xaxis_title="",
                    yaxis_title="",
                )
        
        # Callback for entity network chart
        @self.app.callback(
            Output("entity-network-chart", "figure"),
            [
                Input("ticker-dropdown", "value"),
                Input("entity-occurrence-slider", "value"),
                Input("date-picker-range", "start_date"),
                Input("date-picker-range", "end_date")
            ]
        )
        def update_entity_network_chart(ticker, min_occurrences, start_date, end_date):
            # Convert string dates to datetime objects if needed
            if start_date and isinstance(start_date, str):
                start_date = datetime.strptime(start_date.split('T')[0], '%Y-%m-%d').date()
            if end_date and isinstance(end_date, str):
                end_date = datetime.strptime(end_date.split('T')[0], '%Y-%m-%d').date()
            try:
                if not ticker:
                    return go.Figure().update_layout(
                        title="No ticker selected",
                        xaxis_title="",
                        yaxis_title="",
                    )
                
                # Actually use the min_occurrences from the slider
                actual_min_occurrences = min_occurrences
                logger.info(f"Looking for entity data for ticker {ticker} with minimum {actual_min_occurrences} occurrences")
                
                entities_file = os.path.join(self.data_dir, f"{ticker}_entities.csv")
                
                if not os.path.exists(entities_file):
                    return go.Figure().update_layout(
                        title=f"Entity data for {ticker} not found",
                        xaxis_title="",
                        yaxis_title="",
                        annotations=[dict(
                            text="Entity extraction not performed for this ticker",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5,
                            showarrow=False,
                            font=dict(size=16)
                        )]
                    )
                
                logger.info(f"Found entity file: {entities_file}")
                
                # Filter entities by date if needed
                if start_date and end_date:
                    try:
                        # Load entity data and join with news data to get dates
                        entities_df = pd.read_csv(entities_file)
                        
                        # Try to find sentiment data file to get dates
                        sentiment_file = os.path.join(self.data_dir, f"{ticker}_daily_sentiment.csv")
                        if not os.path.exists(sentiment_file):
                            sentiment_file = os.path.join(self.data_dir, f"{ticker}_sentiment_analysis.csv")
                        
                        if os.path.exists(sentiment_file):
                            sentiment_df = pd.read_csv(sentiment_file)
                            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
                            
                            # Filter sentiment data by date range - properly convert dates
                            start_date_obj = parse_date(start_date)
                            end_date_obj = parse_date(end_date)
                            
                            start_ts = pd.Timestamp(start_date_obj)
                            end_ts = pd.Timestamp(end_date_obj)
                            
                            filtered_sentiment = sentiment_df[
                                (sentiment_df['date'] >= start_ts) & 
                                (sentiment_df['date'] <= end_ts)
                            ]
                            logger.info(f"Filtered entity data by date range: {start_ts} to {end_ts}, {len(filtered_sentiment)} rows")
                            
                            # If we have source_text in both dataframes, filter entities based on it
                            if 'source_text' in entities_df.columns and 'headline' in filtered_sentiment.columns:
                                # Only keep entities from texts in the filtered date range
                                entities_df = entities_df[entities_df['source_text'].isin(filtered_sentiment['headline'])]
                                
                                # Save to temporary file
                                temp_entities_file = os.path.join(self.data_dir, f"{ticker}_entities_temp.csv")
                                entities_df.to_csv(temp_entities_file, index=False)
                                
                                # Use the temporary file for the chart
                                fig = create_entity_network_chart(temp_entities_file, min_occurrences=actual_min_occurrences)
                                
                                # Clean up
                                try:
                                    os.remove(temp_entities_file)
                                except:
                                    pass  # Ignore cleanup errors
                            else:
                                # If we don't have the linking fields, use the original file
                                fig = create_entity_network_chart(entities_file, min_occurrences=actual_min_occurrences)
                        else:
                            # If no sentiment data with dates, use the original file
                            fig = create_entity_network_chart(entities_file, min_occurrences=actual_min_occurrences)
                    except Exception as e:
                        logger.error(f"Error filtering entities by date: {e}")
                        # Fall back to unfiltered entities
                        fig = create_entity_network_chart(entities_file, min_occurrences=actual_min_occurrences)
                else:
                    # No date filtering
                    fig = create_entity_network_chart(entities_file, min_occurrences=actual_min_occurrences)
                    
                if fig is None:
                    return go.Figure().update_layout(
                        title=f"Not enough entities with {actual_min_occurrences}+ occurrences",
                        xaxis_title="",
                        yaxis_title="",
                        annotations=[dict(
                            text="Try lowering the minimum occurrences threshold",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5,
                            showarrow=False,
                            font=dict(size=16)
                        )]
                    )
                return fig
                
            except Exception as e:
                logger.error(f"Error updating entity network chart: {e}")
                return go.Figure().update_layout(
                    title=f"Error loading chart: {str(e)}",
                    xaxis_title="",
                    yaxis_title="",
                )
        
        # Callback for stock data table
        @self.app.callback(
            Output("stock-table-container", "children"),
            [
                Input("ticker-dropdown", "value"),
                Input("date-picker-range", "start_date"),
                Input("date-picker-range", "end_date")
            ]
        )
        def update_stock_table(ticker, start_date, end_date):
            # Convert string dates to datetime objects if needed
            if start_date and isinstance(start_date, str):
                start_date = datetime.strptime(start_date.split('T')[0], '%Y-%m-%d').date()
            if end_date and isinstance(end_date, str):
                end_date = datetime.strptime(end_date.split('T')[0], '%Y-%m-%d').date()
            try:
                if not ticker:
                    return html.Div("No ticker selected")
                
                stock_file = os.path.join(self.data_dir, f"{ticker}_stock_data_clean.csv")
                if not os.path.exists(stock_file):
                    stock_file = os.path.join(self.data_dir, f"{ticker}_stock_data.csv")
                
                if not os.path.exists(stock_file):
                    return html.Div(f"Stock data for {ticker} not found")
                
                # Load and format data
                df = pd.read_csv(stock_file)
                # Check if second row contains ticker symbols (indicating an extra header row)
                if len(df) > 0 and 'AAPL' in str(df.iloc[0].values):
                    df = pd.read_csv(stock_file, skiprows=1)
                    logger.info("Detected and skipped extra header row in stock data")
                
                # Convert date to datetime for filtering
                df['date'] = pd.to_datetime(df['date'])
                
                # Apply date filtering if provided
                if start_date and end_date:
                    # Convert start_date and end_date to pandas Timestamps for proper comparison
                    start_ts = pd.Timestamp(start_date)
                    end_ts = pd.Timestamp(end_date)
                    
                    # Filter using the timestamp comparison
                    df = df[(df['date'] >= start_ts) & (df['date'] <= end_ts)]
                
                # Sort by date descending to show most recent first
                df = df.sort_values('date', ascending=False)
                
                # Format date for display
                df['date'] = df['date'].dt.strftime('%Y-%m-%d')
                
                # Format numeric columns
                numeric_cols = ['open', 'high', 'low', 'close', 'Open', 'High', 'Low', 'Close']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = df[col].round(2).apply(lambda x: f"${x:.2f}")
                
                volume_cols = ['volume', 'Volume']
                for col in volume_cols:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: f"{x:,}")
                
                # Take only the most recent rows
                df = df.head(10)
                
                # Create a table
                table = html.Table([
                    # Header
                    html.Thead(html.Tr([html.Th(col) for col in df.columns])),
                    # Body
                    html.Tbody([html.Tr([html.Td(df.iloc[i][col]) for col in df.columns]) for i in range(len(df))])
                ])
                
                return html.Div([
                    html.Div([
                        html.I(className="fas fa-chart-line mr-2", style={"color": "#1e88e5", "margin-right": "8px"}),
                        html.H4(f"Recent Stock Data for {ticker}")
                    ], style={"display": "flex", "align-items": "center"}),
                    html.Div(table, className="table-responsive")
                ], className="data-section")
                
            except Exception as e:
                logger.error(f"Error updating stock table: {e}")
                return html.Div(f"Error loading stock data: {str(e)}")
        
        # Callbacks for info displays
        @self.app.callback(
            [
                Output("date-range-display", "children"),
                Output("avg-sentiment-display", "children"),
                Output("avg-sentiment-display", "className"),
            ],
            [
                Input("ticker-dropdown", "value"),
                Input("date-picker-range", "start_date"),
                Input("date-picker-range", "end_date")
            ]
        )
        def update_info_displays(ticker, start_date, end_date):
            logger.debug(f"Updating info displays for ticker: {ticker}, dates: {start_date} to {end_date}")
            
            if not ticker:
                # Return simple formatted spans when no ticker is selected
                return (
                    create_formatted_span("No date range selected"), 
                    create_formatted_span("N/A", color="gray"),
                    "info-value neutral-sentiment",
                )
                
            # Convert string dates to datetime objects using our helper
            start_date_obj = parse_date(start_date)
            end_date_obj = parse_date(end_date)

            # Use the class min/max dates if no date range is selected
            if not start_date_obj:
                start_date_obj = self.min_date
            if not end_date_obj:
                end_date_obj = self.max_date
                
            # Format the date range text
            date_range_text = f"{start_date_obj.strftime('%b %d, %Y')} - {end_date_obj.strftime('%b %d, %Y')}"
            
            # Return formatted spans for each output
            try:
                sentiment_file = os.path.join(self.data_dir, f"{ticker}_daily_sentiment.csv")
                # Try alternative sentiment file name if the first doesn't exist
                if not os.path.exists(sentiment_file):
                    sentiment_file = os.path.join(self.data_dir, f"{ticker}_sentiment_analysis.csv")
                    
                if os.path.exists(sentiment_file):
                    try:
                        df = pd.read_csv(sentiment_file)
                        logger.info(f"Loaded sentiment file with {len(df)} rows and columns: {df.columns.tolist()}")
                        
                        # Ensure date column is datetime
                        if 'date' in df.columns:
                            df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=False)
                            before_drop = len(df)
                            df = df.dropna(subset=['date'])
                            if len(df) != before_drop:
                                logger.info(f"Dropped {before_drop - len(df)} rows with invalid dates from sentiment data")
                            
                            # Apply date filtering - ensure we compare date objects to date objects or timestamp to timestamp
                            start_timestamp = pd.Timestamp(start_date_obj)
                            end_timestamp = pd.Timestamp(end_date_obj)
                            # Convert dataframe dates to date objects for proper comparison
                            filtered_df = df[(df['date'] >= start_timestamp) & (df['date'] <= end_timestamp)]
                            logger.info(
                                f"After timestamp filtering: {len(filtered_df)} rows (from {len(df)}), "
                                f"requested: {start_timestamp.date()} to {end_timestamp.date()}, "
                                f"data range: {df['date'].min().date() if not df.empty else 'n/a'} to {df['date'].max().date() if not df.empty else 'n/a'}"
                            )
                            # Fallback: compare on date-only if timestamp filter yields 0 but data exists
                            if len(filtered_df) == 0 and not df.empty:
                                logger.info(f"Initial timestamp filtering yielded 0 rows, trying date-only fallback")
                                logger.info(f"Date range: {start_date_obj} to {end_date_obj}")
                                logger.info(f"DataFrame date range: {df['date'].min()} to {df['date'].max()}")
                                
                                df_dates = df.copy()
                                try:
                                    # Extract just the date component
                                    df_dates['__date_only'] = df_dates['date'].dt.date
                                    logger.info(f"Created __date_only column, values: {df_dates['__date_only'].head().tolist()}")
                                    
                                    # Debug dates before filtering
                                    logger.info(f"start_date_obj type: {type(start_date_obj)}, value: {start_date_obj}")
                                    logger.info(f"end_date_obj type: {type(end_date_obj)}, value: {end_date_obj}")
                                    
                                    # Filter with date-only comparison
                                    filtered_df = df_dates[(df_dates['__date_only'] >= start_date_obj) & (df_dates['__date_only'] <= end_date_obj)]
                                    logger.info(f"Fallback date-only filtering produced {len(filtered_df)} rows")
                                    
                                    # Drop helper column for downstream processing
                                    if '__date_only' in filtered_df.columns:
                                        filtered_df = filtered_df.drop(columns=['__date_only'])
                                except Exception as _fe:
                                    logger.warning(f"Date-only fallback filtering failed: {_fe}")
                            
                            # Calculate average sentiment
                            avg_sentiment_val = None
                            if len(filtered_df) > 0:
                                sentiment_col = None
                                if 'sentiment_score' in filtered_df.columns:
                                    sentiment_col = 'sentiment_score'
                                elif 'sentiment_compound' in filtered_df.columns:
                                    sentiment_col = 'sentiment_compound'
                                elif {'sentiment_positive','sentiment_negative'}.issubset(filtered_df.columns):
                                    # Fallback approximation: positive - negative
                                    filtered_df = filtered_df.copy()
                                    filtered_df['__sentiment_tmp'] = pd.to_numeric(filtered_df['sentiment_positive'], errors='coerce').fillna(0) - \
                                                                     pd.to_numeric(filtered_df['sentiment_negative'], errors='coerce').fillna(0)
                                    sentiment_col = '__sentiment_tmp'
                                if sentiment_col:
                                    avg_sentiment_val = pd.to_numeric(filtered_df[sentiment_col], errors='coerce').mean()
                            
                            # Format the sentiment for display
                            if avg_sentiment_val is not None:
                                # Clip tiny values to zero to avoid -0.0000
                                if abs(avg_sentiment_val) < 0.00005:
                                    avg_sentiment_val = 0.0
                                avg_sentiment_text = f"{avg_sentiment_val:.4f}"
                                logger.info(f"Calculated average sentiment: {avg_sentiment_text}")
                                avg_sentiment_span = MarketAnalysisDashboard.format_sentiment(avg_sentiment_text)
                            else:
                                logger.warning("Could not calculate average sentiment")
                                avg_sentiment_span = create_formatted_span("N/A", color="gray")
                            # Determine sentiment class for container
                            sentiment_class = "info-value neutral-sentiment"
                            try:
                                if avg_sentiment_val is not None:
                                    if avg_sentiment_val > 0.1:
                                        sentiment_class = "info-value sentiment-positive"
                                    elif avg_sentiment_val < -0.1:
                                        sentiment_class = "info-value sentiment-negative"
                                    else:
                                        sentiment_class = "info-value neutral-sentiment"
                            except Exception:
                                sentiment_class = "info-value neutral-sentiment"
                            
                            # Get the total number of data points in the selected date range
                            row_count = len(filtered_df)
                            logger.info(f"Total data points in selected range: {row_count} rows")
                            
                            # Create a simple, safe display value
                            headline_count_span = create_formatted_span(f"{row_count:,}", bold=True)
                            # This displays the number of data points in the selected date range
                            
                            # Return the formatted spans
                            return (
                                create_formatted_span(date_range_text),
                                avg_sentiment_span,
                                sentiment_class,
                            )
                        else:
                            logger.error("No date column found in sentiment file")
                            return (
                                create_formatted_span(date_range_text),
                                create_formatted_span("N/A", color="gray"),
                                "info-value neutral-sentiment",
                            )
                            
                    except Exception as e:
                        logger.error(f"Error processing sentiment data: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                # Return default values if we couldn't process the file
                return (
                    create_formatted_span(date_range_text),
                    create_formatted_span("N/A", color="gray"),
                    "info-value neutral-sentiment",
                )
                
            except Exception as e:
                logger.error(f"Error in update_info_displays: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
                # Return default values on error
                return (
                    create_formatted_span(date_range_text if date_range_text else "Error"),
                    create_formatted_span("Error", color="red"),
                    create_formatted_span("Error", color="red"),
                    "info-value neutral-sentiment",
                )
        
        # Callback for sentiment data table
        @self.app.callback(
            Output("sentiment-table-container", "children"),
            [
                Input("ticker-dropdown", "value"),
                Input("date-picker-range", "start_date"),
                Input("date-picker-range", "end_date")
            ]
        )
        def update_sentiment_table(ticker, start_date, end_date):
            # Convert string dates to datetime objects if needed
            if start_date and isinstance(start_date, str):
                start_date = datetime.strptime(start_date.split('T')[0], '%Y-%m-%d').date()
            if end_date and isinstance(end_date, str):
                end_date = datetime.strptime(end_date.split('T')[0], '%Y-%m-%d').date()
            try:
                if not ticker:
                    return html.Div("No ticker selected")
                
                sentiment_file = os.path.join(self.data_dir, f"{ticker}_daily_sentiment.csv")
                
                # Try alternative sentiment file name if the first doesn't exist
                if not os.path.exists(sentiment_file):
                    sentiment_file = os.path.join(self.data_dir, f"{ticker}_sentiment_analysis.csv")
                
                if not os.path.exists(sentiment_file):
                    return html.Div(f"Sentiment data for {ticker} not found")
                
                # Load and format data
                df = pd.read_csv(sentiment_file)
                
                if 'date' in df.columns:
                    # Convert to datetime for filtering
                    df['date'] = pd.to_datetime(df['date'])
                    
                    # Apply date filtering if provided
                    if start_date and end_date:
                        # Convert string dates to datetime objects using our helper
                        start_date_obj = parse_date(start_date)
                        end_date_obj = parse_date(end_date)
                        
                        # Convert start_date and end_date to pandas Timestamps for proper comparison
                        start_ts = pd.Timestamp(start_date_obj)
                        end_ts = pd.Timestamp(end_date_obj)
                        
                        logger.info(f"Filtering sentiment table by date range: {start_ts} to {end_ts}")
                        
                        # Filter using the timestamp comparison
                        df = df[(df['date'] >= start_ts) & (df['date'] <= end_ts)]
                        logger.info(f"After date filtering: {len(df)} rows")
                        
                    # Sort by date descending to show most recent first
                    df = df.sort_values('date', ascending=False)
                    
                    # Format for display
                    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
                
                # Take only the most recent rows first
                df = df.head(10)
                
                # Create table rows with simple text for all cells
                
                # Create a native HTML table instead of DataTable to better handle HTML content
                try:
                    logger.info(f"Creating HTML table with columns: {df.columns.tolist()}")
                    
                    # Make a copy of the DataFrame for table display
                    table_df = df.copy()
                    
                    # Filter out any columns we don't want to display
                    display_columns = [col for col in table_df.columns if col != 'sentiment_color']
                    
                    # Create table headers
                    table_header = html.Thead(
                        html.Tr([
                            html.Th(col.replace('_', ' ').title(), scope="col") 
                            for col in display_columns
                        ])
                    )
                    
                    # Create table rows with proper formatting for each cell type
                    rows = []
                    for _, row in table_df.iterrows():
                        cells = []
                        for col in display_columns:
                            cell_content = row[col]
                            
                            # Use our helper functions to format cell content
                            if col == 'sentiment_score' or col == 'sentiment_compound':
                                try:
                                    # Use our format_sentiment helper for consistent styling
                                    cell_content = MarketAnalysisDashboard.format_sentiment(cell_content)
                                except Exception as e:
                                    logger.error(f"Error formatting sentiment in table: {e}")
                                    cell_content = create_formatted_span(str(cell_content), color="gray")
                            
                            # Apply sentiment formatting to all sentiment columns for consistency
                            elif 'sentiment_' in col and col not in ['sentiment_color']:
                                try:
                                    # Use the dedicated format_sentiment_component function for sentiment components
                                    # This ensures they are always displayed in gray
                                    cell_content = MarketAnalysisDashboard.format_sentiment_component(cell_content)
                                except Exception as e:
                                    logger.error(f"Error formatting sentiment component in table: {e}")
                                    cell_content = create_formatted_span(str(cell_content))
                                    
                            # Apply special formatting for count values
                            elif 'count' in col.lower():
                                try:
                                    # Use our format_count helper for consistent styling
                                    cell_content = MarketAnalysisDashboard.format_count(cell_content)
                                except Exception as e:
                                    logger.error(f"Error formatting count in table: {e}")
                                    cell_content = create_formatted_span("0", bold=True)
                                    
                            # Format dates
                            elif col == 'date' and pd.notnull(cell_content):
                                try:
                                    if isinstance(cell_content, str):
                                        cell_content = pd.to_datetime(cell_content)
                                    formatted_date = cell_content.strftime('%Y-%m-%d')
                                    cell_content = create_formatted_span(formatted_date)
                                except Exception as e:
                                    logger.error(f"Error formatting date in table: {e}")
                                    cell_content = create_formatted_span(str(cell_content))
                            # For all other types, convert to string and use formatted span
                            elif not isinstance(cell_content, (html.Span, html.Div)):
                                cell_content = create_formatted_span(str(cell_content))
                                
                            # Add the cell to our row
                            cells.append(html.Td(cell_content))
                        
                        rows.append(html.Tr(cells))
                    
                    table_body = html.Tbody(rows)
                    
                    # Create the final HTML table
                    table = html.Table(
                        [table_header, table_body],
                        id='sentiment-data-table',
                        className='sentiment-table',
                        style={
                            'width': '100%',
                            'borderCollapse': 'collapse',
                            'marginTop': '15px',
                            'marginBottom': '15px'
                        }
                    )
                    
                except Exception as e:
                    logger.error(f"Error creating table: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # Create a simple error message as fallback
                    table = html.Div([
                        html.P("Error rendering data table. See logs for details.", style={"color": "red"}),
                        html.Pre(str(df.head(5)))
                    ])

                
                return html.Div([
                    html.Div([
                        html.I(className="fas fa-comments mr-2", style={"color": "#ff9800", "margin-right": "8px"}),
                        html.H4(f"Recent Sentiment Data for {ticker}", id="sentiment-table-heading")
                    ], style={"display": "flex", "align-items": "center"}),
                    html.Div(table, className="table-responsive", **{'aria-labelledby': 'sentiment-table-heading'})
                ], className="data-section", id="sentiment-table-section")
                
            except Exception as e:
                logger.error(f"Error updating sentiment table: {e}")
                return html.Div(f"Error loading sentiment data: {str(e)}")
    
    def run(self):
        if not hasattr(self, 'app'):
            logger.error("Dashboard not properly initialized. Cannot run.")
            return
        
        logger.info(f"Starting dashboard on http://{self.host}:{self.port}/")
        self.app.run(host=self.host, port=self.port, debug=self.debug)


def create_dashboard(data_dir=None, host='127.0.0.1', port=8050, debug=False):
    if not DASH_AVAILABLE or not PLOTLY_AVAILABLE:
        logger.error("Dash or Plotly not installed. Cannot create dashboard.")
        logger.error("Install with: pip install dash plotly")
        return None

    dashboard = MarketAnalysisDashboard(data_dir=data_dir, host=host, port=port, debug=debug)
    
    dashboard.run()
    
    return dashboard


if __name__ == '__main__':
    # Example usage
    # Get the absolute path to the project's root directory
    PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Set up data directory
    data_dir = os.path.join(PROJ_ROOT, settings.DATA_DIR)
    
    print("Starting Market Analysis Dashboard")
    print(f"Looking for data in: {data_dir}")
    
    # Create and run dashboard
    dashboard = create_dashboard(data_dir=data_dir, debug=True)
    
    if dashboard is None:
        print("Failed to create dashboard. Check logs for details.")
        print("Make sure to install required packages:")
        print("pip install dash plotly pandas")
