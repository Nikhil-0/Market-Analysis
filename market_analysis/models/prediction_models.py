import pandas as pd
import numpy as np
import os
import logging
import pickle
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('prediction_models')

# Import settings
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import settings

# Try to import XGBoost (it's optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost not installed. XGBoost models will not be available.")
    XGBOOST_AVAILABLE = False

def prepare_data(stock_file, sentiment_file=None, target_shift=1, test_size=0.2):
    try:
        logger.info(f"Loading stock data from {stock_file}")
        stock_data = pd.read_csv(stock_file)
        
        # Set date as index if it's not already
        if 'date' in stock_data.columns:
            stock_data['date'] = pd.to_datetime(stock_data['date'])
            stock_data = stock_data.set_index('date')
        
        # Add target variable: whether price goes up in the next `target_shift` days
        stock_data['price_change'] = stock_data['close'].diff(periods=target_shift).shift(-target_shift)
        stock_data['target'] = (stock_data['price_change'] > 0).astype(int)
        
        # Merge with sentiment data if provided
        if sentiment_file and os.path.exists(sentiment_file):
            logger.info(f"Loading sentiment data from {sentiment_file}")
            sentiment_data = pd.read_csv(sentiment_file)
            
            # Ensure date is in the right format
            sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
            
            # If it's the raw sentiment file with multiple entries per day, aggregate by day
            if 'sentiment_compound' in sentiment_data.columns and len(sentiment_data) > len(stock_data):
                sentiment_data = sentiment_data.groupby(sentiment_data['date'].dt.date).agg({
                    'sentiment_compound': 'mean',
                    'sentiment_positive': 'mean',
                    'sentiment_negative': 'mean',
                    'sentiment_neutral': 'mean'
                }).reset_index()
                sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
            
            # Merge with stock data
            sentiment_data = sentiment_data.set_index('date')
            data = stock_data.merge(sentiment_data, left_index=True, right_index=True, how='left')
            
            # Fill missing sentiment values with 0 (neutral)
            sentiment_columns = [col for col in data.columns if 'sentiment' in col]
            data[sentiment_columns] = data[sentiment_columns].fillna(0)
        else:
            data = stock_data
        
        # Drop rows with NaN values (due to diff and shift operations)
        data = data.dropna()
        
        # Define features based on what's available in the data
        feature_candidates = [
            # Price and volume features
            'open', 'high', 'low', 'close', 'volume',
            # Technical indicators (if they exist)
            'sma20', 'sma50', 'rsi14', 'macd', 'bb_upper', 'bb_lower',
            # Sentiment features (if they exist)
            'sentiment_compound', 'sentiment_positive', 'sentiment_negative', 'sentiment_neutral'
        ]
        
        # Only use features that actually exist in the data
        features = [f for f in feature_candidates if f in data.columns]
        logger.info(f"Using features: {features}")
        
        # Select features and target
        X = data[features]
        y = data['target']
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        
        # Split data with time series consideration
        train_idx = int(len(X_scaled) * (1 - test_size))
        X_train = X_scaled.iloc[:train_idx]
        X_test = X_scaled.iloc[train_idx:]
        y_train = y.iloc[:train_idx]
        y_test = y.iloc[train_idx:]
        
        logger.info(f"Data prepared: {len(X_train)} training samples, {len(X_test)} test samples")
        logger.info(f"Target distribution in training data: {y_train.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test, features
        
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return None, None, None, None, None

def train_logistic_regression(X_train, y_train, hyperparams=None):
    logger.info("Training Logistic Regression model")
    
    # Default hyperparameters
    if hyperparams is None:
        hyperparams = {
            'C': 1.0,
            'penalty': 'l2',
            'max_iter': 1000,
            'random_state': 42
        }
    
    model = LogisticRegression(**hyperparams)
    model.fit(X_train, y_train)
    
    return model

def train_random_forest(X_train, y_train, hyperparams=None):
    logger.info("Training Random Forest model")
    
    # Default hyperparameters from settings or use these
    if hyperparams is None:
        hyperparams = settings.MODEL_SETTINGS.get('hyperparameters', {}).get('random_forest', {})
        if not hyperparams:
            hyperparams = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'random_state': 42
            }
    
    model = RandomForestClassifier(**hyperparams)
    model.fit(X_train, y_train)
    
    return model

def train_xgboost(X_train, y_train, hyperparams=None):
    if not XGBOOST_AVAILABLE:
        logger.error("XGBoost not installed. Install with: pip install xgboost")
        return None
        
    logger.info("Training XGBoost model")
    
    # Default hyperparameters from settings or use these
    if hyperparams is None:
        hyperparams = settings.MODEL_SETTINGS.get('hyperparameters', {}).get('xgboost', {})
        if not hyperparams:
            hyperparams = {
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1,
                'random_state': 42
            }
    
    model = xgb.XGBClassifier(**hyperparams)
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    logger.info(f"Evaluating {model_name}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Log results
    logger.info(f"{model_name} Accuracy: {accuracy:.4f}")
    logger.info(f"{model_name} Precision: {precision:.4f}")
    logger.info(f"{model_name} Recall: {recall:.4f}")
    logger.info(f"{model_name} F1 Score: {f1:.4f}")
    logger.info(f"{model_name} Confusion Matrix:\n{conf_matrix}")
    
    # Return metrics as a dictionary
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix.tolist()
    }
    
    return metrics

def save_model(model, file_path, metadata=None):
    try:
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Create a model package with metadata
        model_package = {
            'model': model,
            'metadata': metadata or {},
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save the model package
        with open(file_path, 'wb') as f:
            pickle.dump(model_package, f)
            
        logger.info(f"Model saved to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False

def load_model(file_path):
    try:
        with open(file_path, 'rb') as f:
            model_package = pickle.load(f)
            
        model = model_package['model']
        metadata = model_package.get('metadata', {})
        timestamp = model_package.get('timestamp', 'unknown')
        
        logger.info(f"Loaded model from {file_path} (created on {timestamp})")
        return model, metadata
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

def compare_models(stock_file, sentiment_file=None):
    # Prepare data
    X_train, X_test, y_train, y_test, features = prepare_data(
        stock_file, sentiment_file, 
        test_size=settings.MODEL_SETTINGS.get('test_size', 0.2)
    )
    
    if X_train is None:
        return None
    
    results = {}
    models = {}
    
    # Check if we have enough samples from both classes
    class_counts = np.bincount(y_train)
    if len(class_counts) < 2 or min(class_counts) < 2:
        logger.warning("Not enough samples in the minority class for reliable training")
        logger.warning(f"Class distribution: {np.bincount(y_train)}")
        logger.warning("Consider using more data or a different target definition")
    
    # Train and evaluate Logistic Regression
    log_reg = train_logistic_regression(X_train, y_train)
    results['logistic_regression'] = evaluate_model(log_reg, X_test, y_test, "Logistic Regression")
    models['logistic_regression'] = log_reg
    
    # Train and evaluate Random Forest
    rf = train_random_forest(X_train, y_train)
    results['random_forest'] = evaluate_model(rf, X_test, y_test, "Random Forest")
    models['random_forest'] = rf
    
    # Train and evaluate XGBoost if available
    if XGBOOST_AVAILABLE:
        xgb_model = train_xgboost(X_train, y_train)
        if xgb_model:
            results['xgboost'] = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
            models['xgboost'] = xgb_model
    
    # Find the best model based on F1 score
    best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
    best_model = models[best_model_name]
    
    logger.info(f"Best model: {best_model_name} with F1 score: {results[best_model_name]['f1']:.4f}")
    
    # Save the best model
    model_metadata = {
        'features': features,
        'metrics': results[best_model_name],
        'training_data': {
            'stock_file': stock_file,
            'sentiment_file': sentiment_file,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
    }
    
    # Generate a filename based on the ticker symbol
    ticker = os.path.basename(stock_file).split('_')[0]
    model_filename = f"{ticker}_prediction_model.pkl"
    model_path = os.path.join(settings.MODELS_DIR, model_filename)
    
    save_model(best_model, model_path, model_metadata)
    
    return {
        'results': results,
        'best_model': best_model_name,
        'best_model_path': model_path,
        'features': features
    }

if __name__ == '__main__':
    # Example usage
    # Get the absolute path to the project's root directory
    PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Make sure the models directory exists
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
    
    print(f"Comparing prediction models for {ticker}...")
    print(f"Using stock data: {stock_file}")
    
    if os.path.exists(sentiment_file):
        print(f"Using sentiment data: {sentiment_file}")
        comparison_results = compare_models(stock_file, sentiment_file)
    else:
        print(f"No sentiment data found. Training with stock data only.")
        comparison_results = compare_models(stock_file)
    
    if comparison_results:
        print("\nModel comparison completed!")
        
        # Show results summary
        print("\nModel Performance Comparison:")
        results = comparison_results['results']
        
        # Format as a table
        headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
        rows = []
        
        for model_name, metrics in results.items():
            row = [
                model_name,
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1']:.4f}"
            ]
            rows.append(row)
        
        # Print the table
        print("-" * 80)
        print(f"{headers[0]:<20} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10} {headers[4]:<10}")
        print("-" * 80)
        for row in rows:
            print(f"{row[0]:<20} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10}")
        print("-" * 80)
        
        # Highlight the best model
        print(f"\nBest model: {comparison_results['best_model']}")
        print(f"Model saved to: {comparison_results['best_model_path']}")
    else:
        print("Model comparison failed. Check the logs for details.")
