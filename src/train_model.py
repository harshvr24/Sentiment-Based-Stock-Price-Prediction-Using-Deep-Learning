import os
import argparse
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data only if not already present
def download_nltk_data():
    """Download required NLTK data if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('tokenizers/stopwords')
    except LookupError:
        logger.info("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train an LSTM model for stock sentiment prediction.')
    parser.add_argument('--data_paths', nargs='+', required=False, 
                       default=['data/my_train_data.csv'], 
                       help='Paths to training datasets (CSV).')
    parser.add_argument('--epochs', type=int, default=30, 
                       help='Number of epochs (default: 30).')
    parser.add_argument('--max_words', type=int, default=20000, 
                       help='Max vocabulary size (default: 20000).')
    parser.add_argument('--max_length', type=int, default=150, 
                       help='Max sequence length (default: 150).')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size (try 16, 32, or 64).')
    parser.add_argument('--embedding_dim', type=int, default=128, 
                       help='Embedding dimension.')
    parser.add_argument('--glove_path', type=str, 
                       default='data/processed/glove.6B.100d.txt',
                       help='Path to GloVe embeddings file.')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Directory to save models and results.')
    parser.add_argument('--save_final_model', action='store_true',
                       help='Save the final trained model.')
    return parser.parse_args()

def validate_data_paths(paths):
    """Validate that all data paths exist."""
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        if not path.endswith('.csv'):
            raise ValueError(f"Unsupported file format: {path}")

def load_datasets(paths):
    """Load and combine datasets from multiple CSV files."""
    logger.info(f"Loading datasets from: {paths}")
    dfs = []
    for path in paths:
        try:
            df = pd.read_csv(path)
            logger.info(f"Loaded {len(df)} rows from {path}")
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            raise
    
    if not dfs:
        raise ValueError("No valid datasets loaded")
    
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined dataset has {len(df)} rows")
    return df

def validate_dataset(df):
    """Validate dataset structure and content."""
    required_columns = {'headline', 'label'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f'Dataset must contain columns: {required_columns}')
    
    if df.empty:
        raise ValueError("Dataset is empty")
    
    # Check for valid labels
    unique_labels = df['label'].unique()
    valid_labels = {0, 1}
    if not all(label in valid_labels for label in unique_labels):
        raise ValueError(f"Labels must be 0 or 1, found: {unique_labels}")
    
    # Check if we have both classes
    if len(unique_labels) < 2:
        raise ValueError(f"Dataset must contain both classes (0 and 1), found only: {unique_labels}")
    
    logger.info(f"Dataset validation passed. Labels: {unique_labels}")

def clean_dataset(df):
    """Clean the dataset by removing duplicates and null values."""
    logger.info(f"Initial dataset size: {len(df)}")
    
    # Remove duplicates
    initial_size = len(df)
    df = df.drop_duplicates(subset=['headline'])
    logger.info(f"Removed {initial_size - len(df)} duplicate headlines")
    
    # Remove null values
    initial_size = len(df)
    df = df.dropna(subset=['headline', 'label'])
    logger.info(f"Removed {initial_size - len(df)} rows with null values")
    
    if df.empty:
        raise ValueError("Dataset is empty after cleaning")
    
    logger.info(f"Final cleaned dataset size: {len(df)}")
    return df

def find_near_duplicates(headlines, similarity_threshold=0.8, max_pairs=10, large_dataset_threshold=1000):
    """Find near-duplicate headlines efficiently."""
    near_duplicates = []
    n = len(headlines)
    
    # Use a more efficient approach for large datasets
    if n > large_dataset_threshold:
        logger.info(f"Large dataset detected, limiting near-duplicate search to first {large_dataset_threshold} headlines")
        headlines = headlines[:large_dataset_threshold]
        n = large_dataset_threshold
    
    for i in range(n):
        for j in range(i + 1, min(i + 20, n)):  # Check next 20 headlines
            ratio = SequenceMatcher(None, headlines[i], headlines[j]).ratio()
            if similarity_threshold < ratio < 1.0:
                near_duplicates.append((headlines[i], headlines[j], ratio))
                if len(near_duplicates) >= max_pairs:
                    break
        if len(near_duplicates) >= max_pairs:
            break
    
    return near_duplicates

def balance_classes(df):
    """Balance the dataset by undersampling the majority class."""
    label_counts = df['label'].value_counts()
    logger.info(f"Class distribution before balancing: {label_counts.to_dict()}")
    
    min_class_count = label_counts.min()
    balanced_dfs = []
    
    for label in [0, 1]:
        class_df = df[df['label'] == label]
        if len(class_df) > min_class_count:
            class_df = class_df.sample(min_class_count, random_state=42)
        balanced_dfs.append(class_df)
    
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Class distribution after balancing: {df_balanced['label'].value_counts().to_dict()}")
    return df_balanced

def preprocess_texts(texts):
    """Preprocess text data by tokenizing and cleaning."""
    processed = []
    for text in texts:
        if pd.isna(text):
            processed.append('')
            continue
        
        # Tokenize and clean
        tokens = word_tokenize(str(text).lower())
        tokens = [t for t in tokens if t.isalpha() and len(t) > 1]
        processed.append(' '.join(tokens))
    
    return processed

def load_glove_embeddings(glove_path, max_words, embedding_dim):
    """Load GloVe embeddings if available."""
    if not os.path.exists(glove_path):
        logger.warning(f"GloVe embeddings not found at {glove_path}")
        return None
    
    logger.info("Loading GloVe embeddings...")
    embeddings_index = {}
    
    try:
        with open(glove_path, encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        
        # Check if embedding dimension matches expected dimension
        if embeddings_index:
            actual_dim = len(next(iter(embeddings_index.values())))
            if actual_dim != embedding_dim:
                logger.warning(f"GloVe embedding dimension ({actual_dim}) doesn't match expected dimension ({embedding_dim})")
                logger.warning("Using actual GloVe dimension for embedding matrix")
                embedding_dim = actual_dim
        
        logger.info(f"Loaded {len(embeddings_index)} word vectors with dimension {embedding_dim}")
        return embeddings_index, embedding_dim
    except Exception as e:
        logger.error(f"Error loading GloVe embeddings: {e}")
        return None, embedding_dim

def create_embedding_matrix(tokenizer, embeddings_index, max_words, embedding_dim):
    """Create embedding matrix from GloVe embeddings."""
    if embeddings_index is None:
        return None
    
    embedding_matrix = np.zeros((max_words, embedding_dim))
    found_words = 0
    
    for word, i in tokenizer.word_index.items():
        if i >= max_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            found_words += 1
    
    logger.info(f"Found {found_words} words in GloVe embeddings")
    return embedding_matrix

def create_model(max_words, max_length, embedding_dim, embedding_matrix=None):
    """Create the LSTM model."""
    if embedding_matrix is not None:
        embedding_layer = Embedding(
            max_words, embedding_dim, 
            weights=[embedding_matrix], 
            input_length=max_length, 
            trainable=False
        )
    else:
        embedding_layer = Embedding(
            max_words, embedding_dim, 
            input_length=max_length
        )
    
    model = Sequential([
        embedding_layer,
        LSTM(128, dropout=0.6, recurrent_dropout=0.3, return_sequences=False),
        Dropout(0.6),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def sanity_check(X, y, model_config, max_samples=10):
    """Perform sanity check by overfitting a small batch."""
    logger.info("Performing sanity check...")
    
    if len(X) < max_samples:
        max_samples = len(X)
    
    tiny_X, tiny_y = X[:max_samples], y[:max_samples]
    
    # Create a simple model for sanity check
    simple_model = Sequential([
        Embedding(model_config['max_words'], model_config['embedding_dim'], 
                 input_length=model_config['max_length']),
        GlobalMaxPooling1D(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    simple_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    history = simple_model.fit(
        tiny_X, tiny_y, 
        epochs=20, 
        batch_size=2, 
        verbose=0
    )
    
    final_acc = history.history['accuracy'][-1]
    logger.info(f"Sanity check accuracy: {final_acc:.3f}")
    
    if final_acc < 0.8:
        logger.warning("Sanity check failed - model may have issues")
        return False
    else:
        logger.info("Sanity check passed")
        return True

def train_with_cross_validation(X, y, model_config, args, output_dir):
    """Train model using k-fold cross-validation."""
    logger.info("Starting k-fold cross-validation...")
    
    k = 5
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    metrics_list = []
    confusion_matrices = []
    
    for fold in range(1, k + 1):
        logger.info(f"Training fold {fold}/{k}")
        
        # Split data
        train_indices, val_indices = list(skf.split(X, y))[fold - 1]
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        # Create model for this fold
        model = create_model(**model_config)
        
        # Callbacks
        checkpoint_path = os.path.join(output_dir, f'best_model_fold{fold}.h5')
        checkpoint = ModelCheckpoint(
            checkpoint_path, 
            save_best_only=True, 
            monitor='val_accuracy', 
            mode='max'
        )
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[checkpoint, early_stop],
            verbose=0
        )
        
        # Evaluate
        y_pred = (model.predict(X_val) > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0)
        }
        
        cm = confusion_matrix(y_val, y_pred)
        
        logger.info(f"Fold {fold} - Accuracy: {metrics['accuracy']:.3f}, "
                   f"Precision: {metrics['precision']:.3f}, "
                   f"Recall: {metrics['recall']:.3f}, "
                   f"F1: {metrics['f1']:.3f}")
        
        metrics_list.append(metrics)
        confusion_matrices.append(cm)
        
        # Save fold results
        fold_results = {
            'fold': fold,
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(y_val, y_pred, output_dict=True)
        }
        
        with open(os.path.join(output_dir, f'fold_{fold}_results.json'), 'w') as f:
            json.dump(fold_results, f, indent=2)
    
    return metrics_list, confusion_matrices

def save_final_model(X, y, model_config, args, output_dir):
    """Train and save a final model on the full dataset."""
    if not args.save_final_model:
        return
    
    logger.info("Training final model on full dataset...")
    
    model = create_model(**model_config)
    
    # Train on full dataset
    history = model.fit(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model.h5')
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2)
    
    # Plot and save training history
    plot_training_history(history, output_dir)

def plot_training_history(history, output_dir):
    """Plot and save training history."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main training function."""
    logger.info("Starting sentiment analysis model training...")
    
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Download NLTK data
    download_nltk_data()
    
    # Validate data paths
    validate_data_paths(args.data_paths)
    
    # Load and validate data
    df = load_datasets(args.data_paths)
    validate_dataset(df)
    
    # Clean data
    df = clean_dataset(df)
    
    # Find near-duplicates
    headlines = df['headline'].tolist()
    near_duplicates = find_near_duplicates(headlines)
    if near_duplicates:
        logger.info(f"Found {len(near_duplicates)} near-duplicate pairs")
        for h1, h2, ratio in near_duplicates[:3]:  # Show first 3
            logger.info(f"Similarity {ratio:.2f}: {h1[:50]}... | {h2[:50]}...")
    
    # Balance classes
    df = balance_classes(df)
    
    # Preprocess text
    texts = preprocess_texts(df['headline'])
    labels = df['label'].astype(int).values
    
    # Tokenize
    tokenizer = Tokenizer(num_words=args.max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=args.max_length, padding='post', truncating='post')
    y = labels
    
    # Save vocabulary
    vocab_path = os.path.join('data/processed', 'vocabulary.json')
    with open(vocab_path, 'w') as f:
        json.dump(tokenizer.word_index, f, indent=2)
    logger.info(f"Vocabulary saved to {vocab_path}")
    
    # Load GloVe embeddings
    embeddings_index, actual_embedding_dim = load_glove_embeddings(args.glove_path, args.max_words, args.embedding_dim)
    embedding_matrix = create_embedding_matrix(tokenizer, embeddings_index, args.max_words, actual_embedding_dim)
    
    # Model configuration
    model_config = {
        'max_words': args.max_words,
        'max_length': args.max_length,
        'embedding_dim': actual_embedding_dim,
        'embedding_matrix': embedding_matrix
    }
    
    # Sanity check
    if not sanity_check(X, y, model_config):
        logger.warning("Sanity check failed, but continuing...")
    
    # Train with cross-validation
    metrics_list, confusion_matrices = train_with_cross_validation(
        X, y, model_config, args, args.output_dir
    )
    
    # Calculate average metrics
    avg_metrics = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        avg_metrics[metric] = float(np.mean([m[metric] for m in metrics_list]))
    
    avg_cm = np.mean(confusion_matrices, axis=0).tolist()
    
    # Save cross-validation results
    cv_results = {
        'average_metrics': avg_metrics,
        'average_confusion_matrix': avg_cm,
        'fold_metrics': metrics_list,
        'model_config': {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in model_config.items()},
        'training_params': vars(args)
    }
    
    cv_results_path = os.path.join(args.output_dir, 'cross_validation_results.json')
    with open(cv_results_path, 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("CROSS-VALIDATION RESULTS")
    logger.info("="*50)
    logger.info(f"Average Accuracy:  {avg_metrics['accuracy']:.3f}")
    logger.info(f"Average Precision: {avg_metrics['precision']:.3f}")
    logger.info(f"Average Recall:    {avg_metrics['recall']:.3f}")
    logger.info(f"Average F1-Score:  {avg_metrics['f1']:.3f}")
    logger.info("="*50)
    
    # Save final model if requested
    save_final_model(X, y, model_config, args, args.output_dir)
    
    logger.info("Training completed successfully!")

if __name__ == '__main__':
    main() 