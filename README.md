# Sentiment-Based Stock Price Prediction Using Deep Learning

## 📊 Project Overview

This project implements a real-time stock price prediction system that analyzes news sentiment to predict whether a stock's price will go UP or DOWN. The system uses a Bayesian Convolutional Neural Network (CNN) trained on financial news data to make predictions based on recent news headlines.

## 🎯 Key Features

- **Multi-Market Support**: Analyzes both US and Indian stocks
- **Real-Time News Analysis**: Fetches latest news from multiple sources
- **Sentiment-Based Predictions**: Uses deep learning to predict price movements
- **Rich Terminal Interface**: Beautiful console output with color-coded results
- **Multiple Data Sources**: yfinance (primary), Finnhub (secondary), Google News (fallback)
- **Currency Support**: Displays prices in INR for Indian stocks, USD for US stocks

## 🏗️ Project Architecture

```
Sentiment-Based-Stock-Price-Prediction-Using-Deep-Learning/
├── src/
│   └── real_time_predictor.py      # Main prediction script
├── models/
│   └── bayesian_cnn_model.keras    # Trained deep learning model
├── data/
│   └── processed/
│       └── vocabulary.json         # Word-to-index mapping
└── README.md                       # This file
```

## 🧠 Model Architecture

### Bayesian Convolutional Neural Network (CNN)

The project uses a Bayesian CNN model (`bayesian_cnn_model.keras`) that:

- **Input**: Preprocessed news headlines (text converted to numerical sequences)
- **Architecture**: Convolutional layers with Bayesian inference capabilities
- **Output**: Binary classification (UP/DOWN) with confidence scores
- **Training Data**: Financial news headlines with labeled price movements

### Model Artifacts & Experiment Results

Due to GitHub file size limitations, the following files are not included
in the repository:

- `models/bayesian_cnn_model.keras`
- `models/cross_validation_results.json`

The repository contains:
- Complete model architecture code
- Training and evaluation scripts
- Fold-wise result summaries
- Performance metrics and plots

### Text Preprocessing Pipeline

1. **Tokenization**: Splits text into individual words
2. **Lowercase Conversion**: Standardizes text format
3. **Stopword Removal**: Removes common words (the, and, is, etc.)
4. **Punctuation Removal**: Cleans special characters
5. **Vocabulary Mapping**: Converts words to numerical indices
6. **Sequence Padding**: Ensures consistent input length (100 tokens)

## 📈 Supported Markets

### US Stocks
- **Format**: Standard symbols (e.g., AAPL, MSFT, TSLA)
- **Currency**: USD
- **Data Sources**: yfinance, Finnhub, Google News

### Indian Stocks
- **Format**: Symbol with exchange suffix (e.g., RELIANCE.NS, HDFC.NS)
- **Suffixes**: 
  - `.NS` - National Stock Exchange (NSE)
  - `.BO` - Bombay Stock Exchange (BSE)
- **Currency**: INR
- **Data Sources**: yfinance, Finnhub, Google News

## 🔧 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Required Dependencies

Create a `requirements.txt` file with the following packages:

```txt
tensorflow>=2.10.0
pandas>=1.5.0
numpy>=1.21.0
yfinance>=0.2.0
finnhub-python>=2.4.0
python-dotenv>=0.19.0
nltk>=3.7
feedparser>=6.0.0
rich>=12.0.0
```

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Sentiment-Based-Stock-Price-Prediction-Using-Deep-Learning
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   FINNHUB_API_KEY=your_finnhub_api_key_here
   ```

4. **Get Finnhub API Key**:
   - Visit [https://finnhub.io/](https://finnhub.io/)
   - Sign up for a free account
   - Copy your API key to the `.env` file

5. **Download NLTK data** (automatic on first run):
   The script will automatically download required NLTK data (punkt, stopwords)

## 🚀 Usage

### Basic Usage

Run the prediction script:

```bash
python src/real_time_predictor.py
```

### Command Line Arguments

Specify stocks directly via command line:

```bash
python src/real_time_predictor.py AAPL MSFT RELIANCE.NS
```

### Interactive Mode

If no arguments provided, the script will prompt for stock symbols:

```
Enter stock symbols to analyze (comma or space separated, e.g. AAPL MSFT RELIANCE.NS):
```

### Example Output

```
╭─────────────────────────────────────────────────────────────────────────────╮
│                    Real-time Stock Price Prediction System                   │
╰─────────────────────────────────────────────────────────────────────────────╯
───────────────────────────────────────────────────────────────────────────────
Initialization successful!
───────────────────────────────────────────────────────────────────────────────

Enter Indian stocks as SYMBOL.NS (e.g. RELIANCE.NS), US stocks as SYMBOL (e.g. AAPL)
Enter stock symbols to analyze (comma or space separated, e.g. AAPL MSFT RELIANCE.NS): AAPL RELIANCE.NS

Companies specified interactively:
- AAPL
- RELIANCE.NS

───────────────────────────────────────────────────────────────────────────────
Starting analysis...
───────────────────────────────────────────────────────────────────────────────

───────────────────────────────────────────────────────────────────────────────
Analyzing AAPL
───────────────────────────────────────────────────────────────────────────────

Current Price: 175.43 USD

Recent News Analysis:
┌──────────────────┬──────────────────────────────────────────────────────────────┬──────────────┬────────────┐
│ Date             │ Headline                                                     │ Prediction   │ Confidence │
├──────────────────┼──────────────────────────────────────────────────────────────┼──────────────┼────────────┤
│ 2024-01-15 14:30 │ Apple announces new iPhone features                         │ UP           │ 85.2%      │
│ 2024-01-15 12:15 │ Apple reports strong quarterly earnings                     │ UP           │ 92.1%      │
│ 2024-01-15 10:45 │ Apple stock faces market pressure                          │ DOWN         │ 67.3%      │
└──────────────────┴──────────────────────────────────────────────────────────────┴──────────────┴────────────┘

╭─────────────────────────────────────────────────────────────────────────────╮
│ Summary for AAPL                                                             │
│                                                                               │
│ UP predictions: 2                                                            │
│ DOWN predictions: 1                                                          │
│ Overall sentiment: BULLISH                                                   │
╰─────────────────────────────────────────────────────────────────────────────╯
```

## 📊 Dataset Requirements

### Training Dataset Format

To train the model, you need a dataset with the following structure:

#### Required Columns:
1. **`headline`** (str): News headline or article title
2. **`label`** (int): Binary label (0 = DOWN, 1 = UP)
3. **`date`** (datetime): Publication date (optional but recommended)

#### Example Dataset Structure:
```csv
headline,label,date
"Apple announces record-breaking iPhone sales",1,2024-01-15
"Tech stocks plummet on market uncertainty",0,2024-01-15
"Microsoft reports strong cloud revenue growth",1,2024-01-14
"Market volatility causes major sell-off",0,2024-01-14
```

### Dataset Sources

#### Recommended Sources:
1. **Financial News APIs**:
   - Finnhub News API
   - Alpha Vantage News API
   - NewsAPI.org

2. **Financial Websites**:
   - Reuters Business News
   - Bloomberg News
   - CNBC News
   - Yahoo Finance News

3. **Stock Market Data Providers**:
   - yfinance news data
   - Quandl news datasets
   - IEX Cloud news data

#### Data Collection Strategy:
1. **Time Period**: Collect news from at least 2-3 years
2. **Volume**: Aim for 50,000+ labeled headlines
3. **Balance**: Ensure balanced UP/DOWN labels
4. **Quality**: Focus on high-quality financial news sources
5. **Relevance**: Include company-specific and market-wide news

### Data Preprocessing Requirements

#### Text Cleaning:
- Remove HTML tags and special characters
- Convert to lowercase
- Remove extra whitespace
- Handle contractions (don't → do not)

#### Labeling Strategy:
- **UP (1)**: News that typically leads to positive price movement
- **DOWN (0)**: News that typically leads to negative price movement
- **Time Window**: Label based on price movement within 1-3 days after news

#### Validation Split:
- Training: 70%
- Validation: 15%
- Test: 15%

## 🔍 How It Works

### 1. Stock Symbol Detection
The system automatically detects market type:
- **US Stocks**: Standard symbols (AAPL, MSFT)
- **Indian Stocks**: Symbols with `.NS` or `.BO` suffix

### 2. Price Fetching
- **US Stocks**: Fetches USD prices via yfinance
- **Indian Stocks**: Fetches INR prices via yfinance with appropriate exchange suffix

### 3. News Collection (Multi-Source Fallback)
1. **Primary**: yfinance news API
2. **Secondary**: Finnhub news API
3. **Fallback**: Google News RSS feed

### 4. Text Preprocessing
1. Tokenizes news headlines
2. Removes stopwords and punctuation
3. Converts to numerical sequences using vocabulary
4. Pads sequences to consistent length

### 5. Prediction Generation
1. Feeds preprocessed text to Bayesian CNN
2. Gets confidence score (0-1)
3. Determines direction (UP if > 0.5, DOWN if ≤ 0.5)
4. Calculates final confidence

### 6. Results Display
- Rich terminal interface with color coding
- Summary statistics (UP vs DOWN predictions)
- Overall sentiment analysis

## 🎨 Features & Capabilities

### Rich Terminal Interface
- **Color-coded output**: Green for UP, Red for DOWN predictions
- **Formatted tables**: Clean display of news and predictions
- **Progress indicators**: Real-time status updates
- **Error handling**: Graceful error messages and fallbacks

### Multi-Source News Integration
- **yfinance**: Primary source for both US and Indian stocks
- **Finnhub**: Secondary source with comprehensive financial news
- **Google News**: Fallback for broader news coverage

### Robust Error Handling
- **API failures**: Automatic fallback to alternative sources
- **Invalid symbols**: Graceful skipping with informative messages
- **Network issues**: Retry mechanisms and timeout handling
- **Data parsing**: Robust parsing of various news formats

### Market-Specific Features
- **Currency display**: INR for Indian stocks, USD for US stocks
- **Exchange handling**: Automatic NSE/BSE suffix management
- **Regional news**: Optimized news fetching for different markets

## 🔧 Configuration

### Environment Variables
```env
FINNHUB_API_KEY=your_api_key_here
```

### Model Parameters
- **Max sequence length**: 100 tokens
- **Vocabulary size**: Based on training data
- **Prediction threshold**: 0.5 (configurable)

### News Fetching Parameters
- **News window**: 7 days (configurable)
- **Max articles**: 10 per stock (configurable)
- **Duplicate removal**: Enabled

## 🚨 Troubleshooting

### Common Issues

1. **"Finnhub API key not found"**
   - Ensure `.env` file exists with correct API key
   - Verify API key is valid and active

2. **"Model file not found"**
   - Ensure `models/bayesian_cnn_model.keras` exists
   - Run training script if model is missing

3. **"Vocabulary file not found"**
   - Ensure `data/processed/vocabulary.json` exists
   - Run preprocessing script if vocabulary is missing

4. **"No recent news found"**
   - Check internet connection
   - Verify stock symbol is valid
   - Try different stock symbols

5. **"Error fetching price"**
   - Verify stock symbol format
   - Check if stock is actively traded
   - Ensure market is open (for real-time prices)

### Performance Optimization

1. **Reduce API calls**: Use command line arguments instead of interactive mode
2. **Batch processing**: Analyze multiple stocks in one run
3. **Caching**: Consider implementing news caching for repeated analysis

## 📈 Model Performance

### Training Metrics
- **Accuracy**: Model accuracy on test set
- **Precision/Recall**: Per-class performance metrics
- **F1-Score**: Balanced performance measure
- **ROC-AUC**: Overall model discrimination ability

### Prediction Confidence
- **High Confidence (>80%)**: Strong model prediction
- **Medium Confidence (60-80%)**: Moderate prediction strength
- **Low Confidence (<60%)**: Weak prediction, consider additional analysis

## 🔮 Future Enhancements

### Planned Features
1. **Technical Analysis Integration**: Combine with price patterns
2. **Sentiment Intensity**: Fine-grained sentiment scoring
3. **Portfolio Analysis**: Multi-stock portfolio predictions
4. **Historical Backtesting**: Performance validation on historical data
5. **Real-time Alerts**: Automated notification system
6. **Web Interface**: Browser-based analysis tool

### Model Improvements
1. **Transformer Models**: BERT/GPT-based sentiment analysis
2. **Multi-modal Analysis**: Text + image + audio news
3. **Time Series Integration**: LSTM/GRU for temporal patterns
4. **Ensemble Methods**: Combine multiple model predictions

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team
- Check the troubleshooting section above

---


**Disclaimer**: This tool is for educational and research purposes only. Stock predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always conduct thorough research and consult with financial advisors before making investment decisions. 

