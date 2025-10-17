# Try importing required packages
try:
    import plotly
    import dash
    
    print("✅ Successfully imported plotly version:", plotly.__version__)
    print("✅ Successfully imported dash version:", dash.__version__)
    
except ImportError as e:
    print("❌ Import error:", e)

# Try creating a simple Dash app
try:
    app = dash.Dash(__name__)
    app.layout = dash.html.Div([
        dash.html.H1("Dash Test")
    ])
    print("✅ Successfully created a Dash app")
    
except Exception as e:
    print("❌ Error creating Dash app:", e)