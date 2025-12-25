import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
import finnhub
from dotenv import load_dotenv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import yfinance as yf
import sys
import feedparser
# Add rich imports
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

# Setup rich console with custom theme
custom_theme = Theme({
    "info": "bold cyan",
    "warning": "bold yellow",
    "error": "bold red",
    "success": "bold green",
    "headline": "bold white",
    "prediction_up": "bold green",
    "prediction_down": "bold red",
    "summary": "bold magenta"
})
console = Console(theme=custom_theme)

# Load environment variables
load_dotenv()

# Check for API key
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
if not FINNHUB_API_KEY or FINNHUB_API_KEY == 'your_api_key_here':
    print("Error: Finnhub API key not found!")
    print("Please follow these steps:")
    print("1. Go to https://finnhub.io/")
    print("2. Sign up for a free account")
    print("3. Get your API key")
    print("4. Add your API key to the .env file:")
    print("   FINNHUB_API_KEY=your_actual_api_key")
    sys.exit(1)

try:
    # Initialize Finnhub client
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    
    # Test API connection
    _ = finnhub_client.company_news('AAPL', 
                                  _from=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                                  to=datetime.now().strftime('%Y-%m-%d'))
except Exception as e:
    print(f"Error connecting to Finnhub API: {str(e)}")
    print("Please check your API key and internet connection.")
    sys.exit(1)

# Download NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except Exception as e:
    print(f"Error downloading NLTK data: {str(e)}")
    sys.exit(1)

# Load vocabulary
try:
    with open('data/processed/vocabulary.json', 'r') as f:
        vocabulary = json.load(f)
except FileNotFoundError:
    print("Error: vocabulary.json not found!")
    print("Please run the preprocessing script first:")
    print("python src/preprocess_news.py")
    sys.exit(1)
except Exception as e:
    print(f"Error loading vocabulary: {str(e)}")
    sys.exit(1)

# Load model
try:
    # Try to load the simple model first
    model = tf.keras.models.load_model('models/simple_model.h5')
except FileNotFoundError:
    try:
        # Fallback to the final model
        model = tf.keras.models.load_model('models/final_model.h5')
    except FileNotFoundError:
        try:
            # Fallback to the old model
            model = tf.keras.models.load_model('models/bayesian_cnn_model.keras')
        except FileNotFoundError:
            print("Error: No model file found!")
            print("Please run the training script first:")
            print("python src/train_model.py --save_final_model")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading old model: {str(e)}")
            print("Please run the training script to create a new model:")
            print("python src/train_model.py --save_final_model")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading final model: {str(e)}")
        sys.exit(1)
except Exception as e:
    print(f"Error loading simple model: {str(e)}")
    sys.exit(1)

def preprocess_text(text):
    """Preprocess text for prediction"""
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and punctuation
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
        
        # Convert to indices
        indices = [vocabulary.get(token, 0) for token in tokens]
        
        # Pad sequence
        max_length = 150  # Same as training
        padded = np.zeros(max_length, dtype=np.int32)
        padded[:len(indices)] = indices[:max_length]
        
        return padded
    except Exception as e:
        print(f"Error preprocessing text: {str(e)}")
        return None

def is_indian_stock(symbol):
    # Only treat as Indian if it ends with .NS or .BO
    return symbol.upper().endswith('.NS') or symbol.upper().endswith('.BO')

# Mapping for Indian stock symbols to company names (add more as needed)
INDIAN_SYMBOL_TO_NAME = {
    'RELIANCE.NS': 'Reliance Industries',
    'HDFC.NS': 'HDFC Bank',
    # Add more mappings as needed
}

def fetch_google_news_headlines(company_name_or_symbol, max_results=10):
    # Use company name if available, else symbol
    query = INDIAN_SYMBOL_TO_NAME.get(company_name_or_symbol.upper(), company_name_or_symbol)
    url = f"https://news.google.com/rss/search?q={query}+stock"
    feed = feedparser.parse(url)
    news_list = []
    for entry in feed.entries[:max_results]:
        news_list.append({
            'headline': entry.title,
            'datetime': entry.published if hasattr(entry, 'published') else '',
            'url': entry.link
        })
    return news_list

def fetch_recent_news(symbol):
    """Fetch recent news for a symbol (yfinance primary, Finnhub fallback, Google News as last resort)"""
    # Try yfinance first
    try:
        if is_indian_stock(symbol):
            yf_symbol = symbol if symbol.upper().endswith(('.NS', '.BO')) else f"{symbol.upper()}.NS"
        else:
            yf_symbol = symbol.upper()
        stock = yf.Ticker(yf_symbol)
        news = getattr(stock, 'news', None)
        if news and isinstance(news, list) and len(news) > 0:
            news_list = []
            for article in news:
                news_list.append(article)
            return news_list[:10]
    except Exception as e:
        print(f"Error fetching news from yfinance for {symbol}: {str(e)}")
    # Fallback to Finnhub
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        if is_indian_stock(symbol):
            base_symbol = symbol.replace('.NS', '').replace('.BO', '')
            finnhub_symbol = f"NSE:{base_symbol.upper()}"
        else:
            finnhub_symbol = symbol.upper()
        news = finnhub_client.company_news(
            finnhub_symbol,
            _from=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d')
        )
        if news and len(news) > 0:
            news.sort(key=lambda x: x.get('datetime', 0), reverse=True)
            seen_headlines = set()
            unique_news = []
            for article in news:
                headline = article.get('headline', '').lower()
                if headline not in seen_headlines:
                    seen_headlines.add(headline)
                    unique_news.append(article)
            return unique_news[:10]
    except Exception as e:
        print(f"Error fetching news from Finnhub for {symbol}: {str(e)}")
    # Final fallback: Google News RSS
    news = fetch_google_news_headlines(symbol)
    return news

def parse_news_item(article):
    # If 'content' key exists, use it
    content = article.get('content', article)
    # Try all possible headline/title fields
    headline = (
        content.get('headline') or
        content.get('title') or
        content.get('Title') or
        content.get('name') or
        'N/A'
    )
    # Try all possible date fields
    dt_val = (
        content.get('datetime') or
        content.get('providerPublishTime') or
        content.get('pubDate') or
        content.get('published') or
        content.get('date') or
        ''
    )
    # Try to parse UNIX timestamp
    if isinstance(dt_val, int) and dt_val > 0:
        try:
            dt = datetime.fromtimestamp(dt_val)
            date_str = dt.strftime('%Y-%m-%d %H:%M')
        except Exception:
            date_str = str(dt_val)
    elif isinstance(dt_val, str) and dt_val:
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(dt_val)
            date_str = dt.strftime('%Y-%m-%d %H:%M')
        except Exception:
            date_str = dt_val
    else:
        date_str = 'N/A'
    return headline, date_str

def fetch_current_price(symbol):
    """Fetch current stock price (supports US and Indian stocks)"""
    try:
        if is_indian_stock(symbol):
            yf_symbol = symbol if symbol.upper().endswith(('.NS', '.BO')) else f"{symbol.upper()}.NS"
            currency = 'INR'
        else:
            yf_symbol = symbol.upper()
            currency = 'USD'
        stock = yf.Ticker(yf_symbol)
        current_price = stock.history(period='1d')['Close'].iloc[-1]
        return current_price, currency
    except Exception as e:
        print(f"Error fetching price for {symbol}: {str(e)}")
        return None, None

def make_prediction(news_text):
    """Make prediction using the model"""
    try:
        # Preprocess text
        processed_text = preprocess_text(news_text)
        if processed_text is None:
            return None, None
        
        # Reshape for model input
        processed_text = np.expand_dims(processed_text, axis=0)
        
        # Make prediction
        prediction = model.predict(processed_text, verbose=0)  # Disable progress bar
        
        # Get confidence
        confidence = float(prediction[0][0])
        
        # Determine direction
        direction = "UP" if confidence > 0.5 else "DOWN"
        confidence = confidence if direction == "UP" else 1 - confidence
        
        return direction, confidence
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None, None

def analyze_company(symbol):
    """Analyze a company's news and make predictions"""
    console.rule(f"[headline]Analyzing {symbol}")
    
    # Fetch current price
    current_price, currency = fetch_current_price(symbol)
    if current_price is None:
        console.print(f"[warning]Warning: Skipping invalid or unsupported symbol: {symbol} (could not fetch price)")
        return
    
    console.print(f"[info]Current Price: [bold]{current_price:.2f} {currency}")
    
    # Fetch recent news
    news = fetch_recent_news(symbol)
    if not news:
        console.print(f"[warning]Warning: Skipping {symbol} (no recent news found or symbol not supported by news API)")
        return
    
    console.print("\n[info]Recent News Analysis:")
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Date", style="dim", width=18)
    table.add_column("Headline", style="headline", width=60, overflow="fold")
    table.add_column("Prediction", style="bold", width=18)
    table.add_column("Confidence", style="bold", width=12)
    
    predictions_made = False
    up_predictions = 0
    down_predictions = 0
    
    for article in news:
        headline, date_str = parse_news_item(article)
        direction, confidence = make_prediction(headline)
        if direction and confidence:
            predictions_made = True
            if direction == "UP":
                up_predictions += 1
                pred_style = "prediction_up"
            else:
                down_predictions += 1
                pred_style = "prediction_down"
            table.add_row(
                str(date_str),
                str(headline),
                f"[{pred_style}]{direction}[/{pred_style}]",
                f"{confidence:.2%}"
            )
    
    if not predictions_made:
        console.print("[warning]No predictions could be made from the recent news")
    else:
        console.print(table)
        summary_panel = Panel(
            f"[success]UP predictions:[/] {up_predictions}\n[error]DOWN predictions:[/] {down_predictions}\n[summary]Overall sentiment:[/] {'BULLISH' if up_predictions > down_predictions else 'BEARISH'}",
            title=f"Summary for {symbol}",
            style="summary"
        )
        console.print(summary_panel)

def main():
    """Main function to run real-time predictions"""
    console.print("\n[bold underline]Real-time Stock Price Prediction System", style="info")
    console.rule("Initialization successful!", style="success")

    # --- Custom Company Input ---
    console.print("[info]Enter Indian stocks as SYMBOL.NS (e.g. RELIANCE.NS), US stocks as SYMBOL (e.g. AAPL)")
    if len(sys.argv) > 1:
        companies = sys.argv[1:]
        console.print("Companies specified via command-line:", style="info")
    else:
        user_input = input("Enter stock symbols to analyze (comma or space separated, e.g. AAPL MSFT RELIANCE.NS): ")
        if "," in user_input:
            companies = [s.strip().upper() for s in user_input.split(",") if s.strip()]
        else:
            companies = [s.strip().upper() for s in user_input.split() if s.strip()]
        console.print("Companies specified interactively:", style="info")
    for symbol in companies:
        console.print(f"- {symbol}", style="headline")

    console.rule("Starting analysis...", style="info")
    try:
        for symbol in companies:
            analyze_company(symbol)
            console.rule()
    except KeyboardInterrupt:
        console.print("\nGracefully shutting down...", style="warning")
    except Exception as e:
        console.print(f"\n[error]An unexpected error occurred: {str(e)}")
    finally:
        console.print("\nAnalysis complete!", style="success")

if __name__ == "__main__":
    main() 