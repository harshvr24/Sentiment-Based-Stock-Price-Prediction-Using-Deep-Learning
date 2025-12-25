# 🎯 Model Training Guide

## 📋 Overview

This guide explains how to provide datasets to train your Sentiment-Based Stock Price Prediction model. The training process involves preparing financial news data, training a Bayesian CNN model, and evaluating its performance.

## 🗂️ Dataset Requirements

### Required Format

Your dataset must be in one of these formats:
- **CSV** (`.csv`)
- **JSON** (`.json`) 
- **Excel** (`.xlsx`)

### Required Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `headline` | string | News headline or article title | "Apple reports record-breaking earnings" |
| `label` | integer | Binary classification (0=DOWN, 1=UP) | 1 |
| `date` | string | Publication date (optional) | "2024-01-15" |

### Example Dataset Structure

#### CSV Format
```csv
headline,label,date
"Apple announces record-breaking iPhone sales",1,2024-01-15
"Tech stocks plummet on market uncertainty",0,2024-01-15
"Microsoft reports strong cloud revenue growth",1,2024-01-14
"Market volatility causes major sell-off",0,2024-01-14
```

#### JSON Format
```json
[
  {
    "headline": "Apple announces record-breaking iPhone sales",
    "label": 1,
    "date": "2024-01-15"
  },
  {
    "headline": "Tech stocks plummet on market uncertainty",
    "label": 0,
    "date": "2024-01-15"
  }
]
```

## 📊 Dataset Sources

### 1. Financial News APIs
- **Finnhub News API**: Comprehensive financial news
- **Alpha Vantage News API**: Market news and sentiment
- **NewsAPI.org**: General news with financial focus

### 2. Financial Websites
- **Reuters Business News**: Professional financial reporting
- **Bloomberg News**: Market analysis and company news
- **CNBC News**: Business and market coverage
- **Yahoo Finance News**: Stock-specific news

### 3. Stock Market Data Providers
- **yfinance news data**: Stock-specific news
- **Quandl news datasets**: Historical financial news
- **IEX Cloud news data**: Real-time market news

## 🛠️ Quick Start: Create Sample Datasets

### Step 1: Generate Sample Data
```bash
python src/create_sample_dataset.py
```

This creates three sample datasets:
- `data/us_stocks_dataset.csv` (2000 samples)
- `data/indian_stocks_dataset.csv` (1500 samples)  
- `data/tech_stocks_dataset.csv` (1000 samples)

### Step 2: Train with Sample Data
```bash
python src/train_model.py --data_paths data/us_stocks_dataset.csv data/indian_stocks_dataset.csv data/tech_stocks_dataset.csv
```

## 🚀 Training Your Own Model

### Step 1: Prepare Your Dataset

1. **Collect Data**: Gather financial news headlines with price movement labels
2. **Clean Data**: Remove duplicates, handle missing values
3. **Label Data**: Assign 0 (DOWN) or 1 (UP) based on price movement
4. **Format Data**: Save in CSV, JSON, or Excel format

### Step 2: Run Training

#### Basic Training
```bash
python src/train_model.py --data_paths your_dataset.csv
```

#### Advanced Training with Multiple Datasets
```bash
python src/train_model.py \
  --data_paths dataset1.csv dataset2.json dataset3.xlsx \
  --epochs 15 \
  --batch_size 64 \
  --max_words 15000 \
  --max_length 150
```

### Step 3: Monitor Training

The training script provides:
- **Real-time progress**: Rich console output with progress bars
- **Dataset statistics**: Sample counts, balance ratios
- **Training metrics**: Accuracy, loss, precision, recall
- **Visualizations**: Confusion matrix, training history plots

## ⚙️ Training Parameters

### Command Line Arguments

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_paths` | Required | Paths to dataset files |
| `--epochs` | 10 | Number of training epochs |
| `--batch_size` | 32 | Batch size for training |
| `--max_words` | 10000 | Maximum vocabulary size |
| `--max_length` | 100 | Maximum sequence length |
| `--embedding_dim` | 128 | Embedding dimension |

### Model Architecture

The Bayesian CNN model includes:
- **Embedding Layer**: Converts words to vectors
- **Convolutional Layer**: Extracts features from text
- **Global Max Pooling**: Reduces dimensionality
- **Dense Layers**: Final classification
- **Dropout**: Prevents overfitting

## 📈 Training Process

### 1. Data Loading
- Loads multiple datasets from different sources
- Combines and standardizes data
- Validates required columns and data types

### 2. Text Preprocessing
- Tokenizes headlines into words
- Creates vocabulary mapping
- Pads sequences to consistent length
- Splits into training/test sets (70%/30%)

### 3. Model Training
- Creates Bayesian CNN architecture
- Trains with early stopping
- Saves best model during training
- Monitors validation metrics

### 4. Evaluation
- Calculates accuracy, precision, recall
- Generates confusion matrix
- Creates training history plots
- Saves performance metrics

## 📁 Output Files

After training, you'll get:

| File | Description |
|------|-------------|
| `models/bayesian_cnn_model.keras` | Final trained model |
| `models/best_model.keras` | Best model during training |
| `models/metrics.json` | Training performance metrics |
| `models/confusion_matrix.png` | Confusion matrix visualization |
| `models/training_history.png` | Training/validation curves |
| `data/processed/vocabulary.json` | Word-to-index mapping |

## 🎯 Data Collection Strategies

### 1. Historical Data Collection
```python
# Example: Collect historical news with price data
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def collect_historical_data(symbol, start_date, end_date):
    # Get stock data
    stock = yf.Ticker(symbol)
    hist = stock.history(start=start_date, end=end_date)
    
    # Get news data
    news = stock.news
    
    # Create dataset
    dataset = []
    for article in news:
        # Determine label based on price movement
        # This is a simplified example
        label = 1 if price_increased else 0
        
        dataset.append({
            'headline': article['title'],
            'label': label,
            'date': article['published']
        })
    
    return pd.DataFrame(dataset)
```

### 2. Real-time Data Collection
```python
# Example: Collect real-time news
import finnhub

def collect_realtime_news(symbol):
    finnhub_client = finnhub.Client(api_key='your_api_key')
    
    news = finnhub_client.company_news(
        symbol,
        _from='2024-01-01',
        to='2024-01-31'
    )
    
    # Process and label news
    # (Labeling requires historical price analysis)
    
    return news
```

## 🔍 Data Labeling Strategies

### Method 1: Price Movement Based
- **UP (1)**: Stock price increased 1-3 days after news
- **DOWN (0)**: Stock price decreased 1-3 days after news
- **Time Window**: 1-3 days after news publication

### Method 2: Sentiment Analysis Based
- **UP (1)**: Positive sentiment keywords (growth, profit, success)
- **DOWN (0)**: Negative sentiment keywords (loss, decline, risk)

### Method 3: Expert Labeling
- **Manual Review**: Human experts label news based on market knowledge
- **Consistency**: Multiple experts for validation

## 📊 Dataset Quality Guidelines

### Size Requirements
- **Minimum**: 5,000 labeled headlines
- **Recommended**: 50,000+ labeled headlines
- **Optimal**: 100,000+ labeled headlines

### Balance Requirements
- **UP/DOWN Ratio**: Aim for 40-60% balance
- **Avoid Imbalance**: Don't have 90% UP or 90% DOWN

### Quality Requirements
- **Relevance**: Focus on financial/business news
- **Accuracy**: Ensure correct labeling
- **Diversity**: Include various companies and sectors
- **Timeliness**: Use recent data (last 2-3 years)

## 🚨 Common Issues & Solutions

### Issue 1: "No valid datasets could be loaded"
**Solution**: Check file paths and format
```bash
# Verify file exists and format
ls -la your_dataset.csv
head -5 your_dataset.csv
```

### Issue 2: "Missing required columns"
**Solution**: Ensure column names match exactly
```python
# Check column names
import pandas as pd
df = pd.read_csv('your_dataset.csv')
print(df.columns.tolist())
```

### Issue 3: "Poor model performance"
**Solutions**:
- Increase dataset size
- Improve data quality
- Adjust model parameters
- Check for data leakage

### Issue 4: "Memory errors during training"
**Solutions**:
- Reduce batch size: `--batch_size 16`
- Reduce vocabulary size: `--max_words 5000`
- Use smaller sequences: `--max_length 50`

## 🔄 Retraining Workflow

### 1. Collect New Data
```bash
# Add new datasets to your collection
python src/create_sample_dataset.py
```

### 2. Retrain Model
```bash
# Train with all datasets
python src/train_model.py --data_paths data/*.csv
```

### 3. Compare Performance
```bash
# Compare old vs new metrics
diff models/metrics_old.json models/metrics.json
```

### 4. Deploy New Model
```bash
# Backup old model
cp models/bayesian_cnn_model.keras models/bayesian_cnn_model_backup.keras

# Use new model
cp models/best_model.keras models/bayesian_cnn_model.keras
```

## 📚 Example Workflows

### Workflow 1: Quick Start with Sample Data
```bash
# 1. Create sample datasets
python src/create_sample_dataset.py

# 2. Train model
python src/train_model.py --data_paths data/*.csv

# 3. Test predictions
python src/real_time_predictor.py AAPL MSFT
```

### Workflow 2: Custom Dataset Training
```bash
# 1. Prepare your dataset (CSV format)
# your_dataset.csv with columns: headline,label,date

# 2. Train with custom parameters
python src/train_model.py \
  --data_paths your_dataset.csv \
  --epochs 20 \
  --batch_size 64 \
  --max_words 20000

# 3. Evaluate results
cat models/metrics.json
```

### Workflow 3: Multi-Dataset Training
```bash
# 1. Prepare multiple datasets
# us_stocks.csv, indian_stocks.csv, tech_stocks.csv

# 2. Train on all datasets
python src/train_model.py \
  --data_paths us_stocks.csv indian_stocks.csv tech_stocks.csv \
  --epochs 15

# 3. Test on different markets
python src/real_time_predictor.py AAPL RELIANCE.NS
```

## 🎯 Best Practices

### Data Collection
1. **Diverse Sources**: Use multiple news sources
2. **Time Coverage**: Collect data over 2-3 years
3. **Market Coverage**: Include US and Indian markets
4. **Quality Control**: Validate data accuracy

### Training
1. **Validation Split**: Use 70/15/15 split (train/val/test)
2. **Early Stopping**: Prevent overfitting
3. **Hyperparameter Tuning**: Experiment with parameters
4. **Model Checkpointing**: Save best model during training

### Evaluation
1. **Multiple Metrics**: Use accuracy, precision, recall, F1
2. **Confusion Matrix**: Understand model behavior
3. **Cross-Validation**: Ensure robust performance
4. **Real-world Testing**: Test on live data

---

**Next Steps**: After training, use your model with `python src/real_time_predictor.py` to make predictions on live stock data! 