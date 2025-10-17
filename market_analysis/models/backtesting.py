import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('backtesting')

# Import settings
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import settings

def calculate_returns(prices, positions, initial_capital=10000):
    # Make a copy of the data
    returns = pd.DataFrame(index=prices.index)
    returns['price'] = prices
    returns['position'] = positions
    
    # Calculate daily returns of the asset
    returns['asset_returns'] = returns['price'].pct_change()
    
    # Calculate strategy returns (shifted because we act on signals from the previous day)
    returns['strategy_returns'] = returns['position'].shift(1) * returns['asset_returns']
    
    # Replace NaN values with 0
    returns['strategy_returns'] = returns['strategy_returns'].fillna(0)
    
    # Calculate cumulative returns
    returns['asset_cumulative'] = (1 + returns['asset_returns']).cumprod() * initial_capital
    returns['strategy_cumulative'] = (1 + returns['strategy_returns']).cumprod() * initial_capital
    
    # Calculate drawdowns
    returns['asset_peak'] = returns['asset_cumulative'].cummax()
    returns['strategy_peak'] = returns['strategy_cumulative'].cummax()
    returns['asset_drawdown'] = (returns['asset_cumulative'] - returns['asset_peak']) / returns['asset_peak']
    returns['strategy_drawdown'] = (returns['strategy_cumulative'] - returns['strategy_peak']) / returns['strategy_peak']
    
    # Calculate performance metrics
    total_days = len(returns)
    trading_days_per_year = 252
    
    # Total return
    total_return = returns['strategy_cumulative'].iloc[-1] / initial_capital - 1
    
    # Annualized return
    years = total_days / trading_days_per_year
    annualized_return = (1 + total_return) ** (1 / years) - 1
    
    # Maximum drawdown
    max_drawdown = returns['strategy_drawdown'].min()
    
    # Sharpe ratio (annualized)
    risk_free_rate = 0.02  # Assuming 2% annual risk-free rate
    daily_risk_free = (1 + risk_free_rate) ** (1 / trading_days_per_year) - 1
    excess_returns = returns['strategy_returns'] - daily_risk_free
    sharpe_ratio = np.sqrt(trading_days_per_year) * excess_returns.mean() / excess_returns.std()
    
    # Win rate
    wins = (returns['strategy_returns'] > 0).sum()
    losses = (returns['strategy_returns'] < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    
    # Create metrics dictionary
    metrics = {
        'initial_capital': initial_capital,
        'final_value': returns['strategy_cumulative'].iloc[-1],
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'trading_days': total_days
    }
    
    return returns, metrics

def backtest_model(model, X, y, prices, initial_capital=10000):
    try:
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]  # Probability of class 1 (price going up)
        
        # Create position signals (1 for long, 0 for flat)
        # Here we're using a simple long-only strategy
        positions = pd.Series(predictions, index=X.index)
        
        # Calculate returns and metrics
        returns, metrics = calculate_returns(prices, positions, initial_capital)
        
        # Add model predictions to the results
        returns['prediction'] = predictions
        returns['probability'] = probabilities
        returns['actual'] = y
        
        # Calculate model accuracy metrics
        metrics['accuracy'] = (predictions == y).mean()
        
        # Calculate confusion matrix elements
        true_positives = ((predictions == 1) & (y == 1)).sum()
        false_positives = ((predictions == 1) & (y == 0)).sum()
        true_negatives = ((predictions == 0) & (y == 0)).sum()
        false_negatives = ((predictions == 0) & (y == 1)).sum()
        
        metrics['true_positives'] = true_positives
        metrics['false_positives'] = false_positives
        metrics['true_negatives'] = true_negatives
        metrics['false_negatives'] = false_negatives
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        
        return returns, metrics
    
    except Exception as e:
        logger.error(f"Error during backtesting: {e}")
        return None, None

def plot_backtest_results(returns, metrics, title="Model Backtest Results"):
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 14), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot cumulative returns
        ax1 = axes[0]
        returns['asset_cumulative'].plot(ax=ax1, color='blue', label='Buy & Hold')
        returns['strategy_cumulative'].plot(ax=ax1, color='green', label='Strategy')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title(f"{title} - Initial Capital: ${metrics['initial_capital']:,.0f}")
        ax1.legend()
        ax1.grid(True)
        
        # Plot drawdowns
        ax2 = axes[1]
        returns['asset_drawdown'].plot(ax=ax2, color='blue', label='Buy & Hold')
        returns['strategy_drawdown'].plot(ax=ax2, color='green', label='Strategy')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_ylim(-1, 0)  # Drawdowns are negative
        ax2.legend()
        ax2.grid(True)
        
        # Plot positions and predictions
        ax3 = axes[2]
        ax3.scatter(returns.index, returns['position'] * 0.5, color='green', s=5, label='Position (1=Long, 0=Flat)')
        ax3.scatter(returns.index, returns['actual'] * 0.75, color='blue', s=5, label='Actual Direction')
        ax3.scatter(returns.index, returns['prediction'], color='red', s=5, label='Predicted Direction')
        ax3.set_ylabel('Signals')
        ax3.set_yticks([0, 0.5, 0.75, 1])
        ax3.set_yticklabels(['Flat/Down', 'Long Position', 'Actual Up', 'Predicted Up'])
        ax3.legend()
        ax3.grid(True)
        
        # Add metrics as text
        metrics_text = (
            f"Total Return: {metrics['total_return']:.2%}\n"
            f"Annualized Return: {metrics['annualized_return']:.2%}\n"
            f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Win Rate: {metrics['win_rate']:.2%}\n"
            f"Model Accuracy: {metrics['accuracy']:.2%}\n"
            f"Precision: {metrics['precision']:.2f}\n"
            f"Recall: {metrics['recall']:.2f}\n"
            f"F1 Score: {metrics['f1_score']:.2f}"
        )
        
        plt.figtext(0.15, 0.01, metrics_text, fontsize=12, ha='left')
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting backtest results: {e}")
        return None

def backtest_sentiment_strategy(stock_file, sentiment_file, threshold=0.05):
    try:
        # Load stock data
        stock_data = pd.read_csv(stock_file)
        
        # Load sentiment data
        sentiment_data = pd.read_csv(sentiment_file)
        
        # Convert dates to datetime
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
        
        # Set dates as index
        stock_data.set_index('date', inplace=True)
        sentiment_data.set_index('date', inplace=True)
        
        # Merge data
        data = stock_data.merge(sentiment_data[['sentiment_compound']], 
                                left_index=True, right_index=True, how='left')
        
        # Fill missing sentiment values with 0 (neutral)
        data['sentiment_compound'] = data['sentiment_compound'].fillna(0)
        
        # Create position signals based on sentiment
        # 1 for bullish (sentiment > threshold)
        # 0 for neutral or bearish (sentiment <= threshold)
        data['position'] = (data['sentiment_compound'] > threshold).astype(int)
        
        # Get the target variable (price going up next day)
        data['next_close'] = data['close'].shift(-1)
        data['price_change'] = data['next_close'] - data['close']
        data['actual'] = (data['price_change'] > 0).astype(int)
        
        # Drop rows with NaN values
        data = data.dropna()
        
        # Calculate returns
        returns, metrics = calculate_returns(data['close'], data['position'])
        
        # Add sentiment and actual direction to the results
        returns['sentiment'] = data['sentiment_compound']
        returns['actual'] = data['actual']
        
        # Calculate strategy accuracy
        correct_predictions = ((data['position'] == 1) & (data['actual'] == 1)).sum() + \
                            ((data['position'] == 0) & (data['actual'] == 0)).sum()
        metrics['accuracy'] = correct_predictions / len(data)
        
        # Calculate confusion matrix elements
        true_positives = ((data['position'] == 1) & (data['actual'] == 1)).sum()
        false_positives = ((data['position'] == 1) & (data['actual'] == 0)).sum()
        true_negatives = ((data['position'] == 0) & (data['actual'] == 0)).sum()
        false_negatives = ((data['position'] == 0) & (data['actual'] == 1)).sum()
        
        metrics['true_positives'] = true_positives
        metrics['false_positives'] = false_positives
        metrics['true_negatives'] = true_negatives
        metrics['false_negatives'] = false_negatives
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        
        return returns, metrics
    
    except Exception as e:
        logger.error(f"Error during sentiment strategy backtesting: {e}")
        return None, None

if __name__ == '__main__':
    # Example usage
    # Get the absolute path to the project's root directory
    PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Make sure the output directories exist
    os.makedirs(os.path.join(PROJ_ROOT, settings.DATA_DIR), exist_ok=True)
    os.makedirs(os.path.join(PROJ_ROOT, settings.MODELS_DIR), exist_ok=True)
    
    # Example ticker
    ticker = 'AAPL'
    stock_file = os.path.join(PROJ_ROOT, settings.DATA_DIR, f'{ticker}_stock_data.csv')
    sentiment_file = os.path.join(PROJ_ROOT, settings.DATA_DIR, f'{ticker}_daily_sentiment.csv')
    
    if not os.path.exists(sentiment_file):
        sentiment_file = os.path.join(PROJ_ROOT, settings.DATA_DIR, f'{ticker}_sentiment_analysis.csv')
    
    # Check if files exist
    if not os.path.exists(stock_file):
        print(f"Stock data file not found: {stock_file}")
        print(f"Run data_collection.py first to collect stock data.")
        sys.exit(1)
    
    if not os.path.exists(sentiment_file):
        print(f"Sentiment data file not found: {sentiment_file}")
        print(f"Run sentiment_analysis.py first to analyze sentiment.")
        sys.exit(1)
    
    print(f"Backtesting sentiment-based strategy for {ticker}...")
    returns, metrics = backtest_sentiment_strategy(stock_file, sentiment_file)
    
    if returns is not None and metrics is not None:
        print("\nBacktesting completed!")
        
        # Display performance metrics
        print("\nStrategy Performance:")
        print(f"Initial Capital: ${metrics['initial_capital']:,.2f}")
        print(f"Final Value: ${metrics['final_value']:,.2f}")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        
        # Plot results
        fig = plot_backtest_results(returns, metrics, f"{ticker} Sentiment Strategy Backtest")
        
        # Save the plot
        plot_path = os.path.join(PROJ_ROOT, settings.DATA_DIR, f'{ticker}_sentiment_backtest.png')
        fig.savefig(plot_path)
        
        print(f"\nBacktest plot saved to: {plot_path}")
        plt.show()
    else:
        print("Backtesting failed. Check the logs for details.")
