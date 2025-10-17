# Market Analysis Project 📊

A comprehensive Python-based market analysis tool that leverages machine learning and natural language processing to analyze market trends, predict stock movements, and extract insights from financial news. This project combines advanced AI techniques with traditional market analysis to provide a holistic view of market dynamics and relationships.

## 🌟 Features

- **AI-Powered Stock Analysis**
  - Machine learning models for price prediction (XGBoost, Random Forest)
  - Automated feature engineering and selection
  - Technical indicator calculations
  - Historical pattern recognition
  - Price movement analysis with ML insights

- **Advanced NLP & Sentiment Analysis**
  - Deep learning-based sentiment classification
  - VADER sentiment analysis implementation
  - Named Entity Recognition (NER) using spaCy
  - Advanced entity relationship extraction
  - Temporal sentiment pattern analysis

- **Entity Network Analysis**
  - Company and entity relationship mapping
  - Interactive network visualizations
  - Configurable correlation thresholds
  - Dynamic network filtering

- **Interactive Dashboard**
  - Near real-time data updates from Yahoo Finance
  - Live news feed integration from Finviz
  - Dynamic visualization components
  - Real-time sentiment scoring
  - Interactive entity network visualization

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/market-analysis-project.git
cd market-analysis-project
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Install additional NLP dependencies:
```bash
py -m spacy download en_core_web_sm
```

### Configuration

1. Review and modify settings in `config/settings.py` as needed
2. Ensure data directories exist (will be created automatically if missing):
   - `data/` - For storing collected data
   - `charts/` - For storing generated visualizations

### Data Collection Modes

The project supports multiple data collection modes:

1. **Real-time Monitoring**
   - Live stock price updates via Yahoo Finance API
   - Real-time news scraping from Finviz
   - Immediate sentiment analysis of incoming news

2. **Historical Analysis**
   - Historical stock data retrieval
   - Archived news data collection
   - Batch sentiment analysis

3. **Update Frequency**
   - Stock prices: Near real-time (up to 1-minute intervals)
   - News feeds: Real-time with configurable polling interval
   - Sentiment analysis: Immediate processing of new data

## 📊 Usage

### Basic Analysis

Run the main analysis script:
```bash
py run_analysis.py
```

### Interactive Dashboard

Launch the interactive dashboard:
```bash
py -m market_analysis.visualization.dashboard
```

### Jupyter Notebook

For exploratory analysis, use the provided notebook:
```bash
jupyter notebook notebooks/market_analysis.ipynb
```

## 📁 Project Structure

```
market_analysis/
├── data_collection/       # Data collection modules
├── data_analysis/        # Analysis components
├── models/              # Prediction models
├── sentiment/           # Sentiment analysis
├── visualization/       # Visualization tools
└── utils/              # Utility functions
```

## 🧪 Testing

Run the test suite:
```bash
py -m pytest tests/
```

Key test modules:
- `test_correlation.py` - Tests for correlation analysis
- `test_entity_network.py` - Tests for entity network generation
- `test_imports.py` - Package import validation

## 📋 Dependencies

Core dependencies:
- pandas (≥1.3.0) - Data manipulation
- numpy (≥1.20.0) - Numerical computations
- scikit-learn (≥1.0.0) - ML algorithms and model evaluation
- XGBoost (optional) - Advanced gradient boosting
- VADER Sentiment (≥3.3.2) - Rule-based sentiment analysis
- spaCy (≥3.3.0) - Industrial-strength NLP
- Plotly (≥5.3.0) - Interactive visualizations

For a complete list, see `requirements.txt`.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📬 Contact

Nikhil Madeti
Project Link: [https://github.com/yourusername/market-analysis-project](https://github.com/yourusername/market-analysis-project)

## 🙏 Acknowledgments

- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) for sentiment analysis
- [Plotly](https://plotly.com/) for interactive visualizations
- [spaCy](https://spacy.io/) for NLP capabilities