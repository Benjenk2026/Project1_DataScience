import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import sys
import json
from pathlib import Path
import re
import pickle

# Scikit-learn imports for ML
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.preprocessing import LabelEncoder

# Text preprocessing
from sklearn.pipeline import Pipeline


def select_file(filetypes=None, title="Select file"):
    """Open file dialog to select a file.
    
    Parameters:
    -----------
    filetypes : list, optional
        List of tuples specifying file type filters
    title : str
        Dialog window title
        
    Returns:
    --------
    str
        Path to selected file, or empty string if cancelled
    """
    if filetypes is None:
        filetypes = [
            ("CSV files", "*.csv"),
            ("Excel files", ("*.xlsx", "*.xls")),
            ("All files", "*.*"),
        ]
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return file_path


def load_complaint_data(file_path, text_column=None, label_column=None, max_rows=None):
    """
    Load complaint/review data from a file.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the data file (complaints, reviews, etc.)
    text_column : str, optional
        Name of the column containing text/description
    label_column : str, optional
        Name of the column containing category labels
    max_rows : int, optional
        Maximum number of rows to load
        
    Returns:
    --------
    DataFrame or None
        Loaded data with text and labels
    """
    p = Path(file_path) if not isinstance(file_path, Path) else file_path
    if not p.exists():
        print(f"File not found: {file_path}")
        return None
    
    extension = p.suffix.lower()
    
    try:
        if extension == ".csv":
            print(f"Loading CSV file...")
            df = pd.read_csv(p, nrows=max_rows)
        elif extension in [".xlsx", ".xls"]:
            print(f"Loading Excel file...")
            df = pd.read_excel(p, nrows=max_rows)
        else:
            print(f"Unsupported file format: {extension}")
            return None
            
        print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
        
        # Display available columns
        print(f"\nAvailable columns: {', '.join(df.columns.tolist())}")
        
        # Detect dataset type and suggest defaults
        suggested_text, suggested_label = detect_dataset_type(df)
        if suggested_text or suggested_label:
            print(f"\nDetected dataset type - Suggestions:")
            if suggested_text:
                print(f"  Text column: '{suggested_text}'")
            if suggested_label:
                print(f"  Label column: '{suggested_label}'")
        
        return df
        
    except Exception as e:
        print(f"Failed to load file: {e}")
        return None


def detect_dataset_type(df):
    """
    Detect the type of dataset and suggest appropriate columns.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
        
    Returns:
    --------
    tuple
        (suggested_text_column, suggested_label_column)
    """
    cols = df.columns.tolist()
    text_col = None
    label_col = None
    
    # Check for Yelp review dataset
    if 'text' in cols and 'stars' in cols:
        text_col = 'text'
        label_col = 'stars'
    # Check for complaint dataset
    elif 'subject' in cols and 'service_name' in cols:
        text_col = 'subject'
        label_col = 'service_name'
    # Generic detection
    else:
        # Look for common text column names
        text_candidates = ['text', 'description', 'subject', 'content', 'review', 'message', 'complaint']
        for candidate in text_candidates:
            if candidate in cols:
                text_col = candidate
                break
        
        # Look for common label column names
        label_candidates = ['category', 'label', 'class', 'type', 'service_name', 'stars', 'rating', 'sentiment']
        for candidate in label_candidates:
            if candidate in cols:
                label_col = candidate
                break
    
    return text_col, label_col


def stars_to_sentiment(stars):
    """
    Convert star ratings to sentiment categories.
    
    Parameters:
    -----------
    stars : int or float
        Star rating (typically 1-5)
        
    Returns:
    --------
    str
        Sentiment category: 'Negative', 'Neutral', or 'Positive'
    """
    try:
        rating = float(stars)
        if rating <= 2:
            return 'Negative'
        elif rating == 3:
            return 'Neutral'
        else:  # 4-5
            return 'Positive'
    except (ValueError, TypeError):
        return 'Unknown'


def preprocess_text(text):
    """
    Basic text preprocessing for text descriptions.
    
    Parameters:
    -----------
    text : str
        Raw text to preprocess
        
    Returns:
    --------
    str
        Preprocessed text
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def prepare_data(df, text_column, label_column, preprocess=True, min_samples_per_class=5, 
                convert_stars_to_sentiment=False):
    """
    Prepare text data for classification.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe with text data
    text_column : str
        Column name containing text
    label_column : str
        Column name containing category labels
    preprocess : bool
        Whether to apply text preprocessing
    min_samples_per_class : int
        Minimum number of samples required per class (default: 5)
    convert_stars_to_sentiment : bool
        If True and label_column is 'stars', convert to sentiment categories
        
    Returns:
    --------
    tuple
        (X, y, label_mapping) where:
        - X: preprocessed text data (Series)
        - y: encoded labels (array)
        - label_mapping: dict mapping encoded labels to original names
    """
    # Validate columns
    if text_column not in df.columns:
        print(f"Error: Text column '{text_column}' not found in data")
        return None, None, None
    
    if label_column not in df.columns:
        print(f"Error: Label column '{label_column}' not found in data")
        return None, None, None
    
    # Create working copy
    data = df[[text_column, label_column]].copy()
    
    # Convert stars to sentiment if requested
    if convert_stars_to_sentiment and label_column == 'stars':
        print("Converting star ratings to sentiment categories...")
        data['sentiment'] = data[label_column].apply(stars_to_sentiment)
        label_column = 'sentiment'
        print("  1-2 stars → Negative")
        print("  3 stars → Neutral")
        print("  4-5 stars → Positive")
    
    # Remove rows with missing values
    original_count = len(data)
    data = data.dropna()
    if len(data) < original_count:
        print(f"Removed {original_count - len(data)} rows with missing values")
    
    # Filter out classes with too few samples
    class_counts = data[label_column].value_counts()
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    data = data[data[label_column].isin(valid_classes)]
    
    removed_classes = len(class_counts) - len(valid_classes)
    if removed_classes > 0:
        print(f"Removed {removed_classes} classes with fewer than {min_samples_per_class} samples")
    
    # Preprocess text if requested
    if preprocess:
        print("Preprocessing text...")
        data[text_column] = data[text_column].apply(preprocess_text)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data[label_column])
    
    # Create label mapping
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    
    print(f"\nData preparation complete:")
    print(f"  Total samples: {len(data):,}")
    print(f"  Number of classes: {len(label_mapping)}")
    print(f"  Class distribution:")
    
    class_dist = pd.Series(y).value_counts().sort_index()
    for idx, count in class_dist.items():
        print(f"    {label_mapping[idx]}: {count} ({count/len(data)*100:.1f}%)")
    
    return data[text_column], y, label_mapping


def build_classifier(model_type='logistic_regression', vectorizer_type='tfidf', max_features=5000):
    """
    Build a classification pipeline with text vectorization and ML model.
    
    Parameters:
    -----------
    model_type : str
        Type of classifier: 'logistic_regression', 'svm', 'random_forest', 'naive_bayes'
    vectorizer_type : str
        Type of text vectorizer: 'tfidf' or 'count'
    max_features : int
        Maximum number of features for vectorizer
        
    Returns:
    --------
    Pipeline
        Sklearn pipeline with vectorizer and classifier
    """
    # Select vectorizer
    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,  # Ignore terms that appear in fewer than 2 documents
            max_df=0.8,  # Ignore terms that appear in more than 80% of documents
            stop_words='english'
        )
    else:
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )
    
    # Select classifier
    if model_type == 'logistic_regression':
        classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )
    elif model_type == 'svm':
        classifier = LinearSVC(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
    elif model_type == 'random_forest':
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1  # Use all CPU cores
        )
    elif model_type == 'naive_bayes':
        classifier = MultinomialNB(alpha=0.1)
    else:
        print(f"Unknown model type: {model_type}. Using logistic regression.")
        classifier = LogisticRegression(max_iter=1000, random_state=42)
    
    # Create pipeline
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])
    
    return pipeline


def train_and_evaluate(X, y, label_mapping, model_type='logistic_regression', 
                      vectorizer_type='tfidf', test_size=0.2, cv_folds=5):
    """
    Train a classification model and evaluate its performance.
    
    Parameters:
    -----------
    X : array-like
        Text data
    y : array-like
        Labels
    label_mapping : dict
        Mapping from encoded labels to original names
    model_type : str
        Type of classifier
    vectorizer_type : str
        Type of text vectorizer
    test_size : float
        Proportion of data to use for testing
    cv_folds : int
        Number of cross-validation folds
        
    Returns:
    --------
    tuple
        (trained_model, test_results) where test_results is a dict with metrics
    """
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} classifier with {vectorizer_type.upper()} vectorization")
    print(f"{'='*60}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\nDataset split:")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")
    
    # Build model
    model = build_classifier(model_type, vectorizer_type)
    
    # Train model
    print(f"\nTraining model...")
    model.fit(X_train, y_train)
    print("Training complete!")
    
    # Cross-validation on training set
    print(f"\nPerforming {cv_folds}-fold cross-validation on training set...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n{'='*60}")
    print("TEST SET PERFORMANCE")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Detailed classification report
    print(f"\n{'='*60}")
    print("DETAILED CLASSIFICATION REPORT")
    print(f"{'='*60}")
    target_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    
    # Store results
    test_results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_test': y_test,
        'y_pred': y_pred,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return model, test_results


def save_model(model, label_mapping, output_path):
    """
    Save trained model and label mapping to disk.
    
    Parameters:
    -----------
    model : Pipeline
        Trained model pipeline
    label_mapping : dict
        Label mapping dictionary
    output_path : str or Path
        Path to save the model
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and mapping together
        model_data = {
            'model': model,
            'label_mapping': label_mapping
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Failed to save model: {e}")
        return False


def load_model(model_path):
    """
    Load a saved model and label mapping.
    
    Parameters:
    -----------
    model_path : str or Path
        Path to the saved model file
        
    Returns:
    --------
    tuple
        (model, label_mapping) or (None, None) on failure
    """
    try:
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"Model file not found: {model_path}")
            return None, None
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"Model loaded from: {model_path}")
        return model_data['model'], model_data['label_mapping']
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None, None


def predict_complaints(model, label_mapping, complaints):
    """
    Predict categories for new complaint descriptions.
    
    Parameters:
    -----------
    model : Pipeline
        Trained model
    label_mapping : dict
        Label mapping dictionary
    complaints : list or str
        Single complaint text or list of complaint texts
        
    Returns:
    --------
    DataFrame
        Predictions with complaint text and predicted category
    """
    # Handle single complaint
    if isinstance(complaints, str):
        complaints = [complaints]
    
    # Preprocess
    preprocessed = [preprocess_text(c) for c in complaints]
    
    # Predict
    predictions = model.predict(preprocessed)
    
    # Map back to original labels
    predicted_labels = [label_mapping[pred] for pred in predictions]
    
    # Create results dataframe
    results = pd.DataFrame({
        'complaint_text': complaints,
        'predicted_category': predicted_labels
    })
    
    return results


def compare_models(X, y, label_mapping, test_size=0.2):
    """
    Compare performance of multiple models on the same dataset.
    
    Parameters:
    -----------
    X : array-like
        Text data
    y : array-like
        Labels
    label_mapping : dict
        Label mapping dictionary
    test_size : float
        Proportion of data for testing
        
    Returns:
    --------
    DataFrame
        Comparison results with model names and metrics
    """
    models_to_test = [
        ('Logistic Regression', 'logistic_regression'),
        ('Support Vector Machine', 'svm'),
        ('Random Forest', 'random_forest'),
        ('Naive Bayes', 'naive_bayes')
    ]
    
    results = []
    
    print(f"\n{'='*60}")
    print("COMPARING MULTIPLE MODELS")
    print(f"{'='*60}")
    
    for name, model_type in models_to_test:
        print(f"\nTesting {name}...")
        model, test_results = train_and_evaluate(
            X, y, label_mapping, 
            model_type=model_type,
            test_size=test_size,
            cv_folds=3  # Reduce folds for faster comparison
        )
        
        results.append({
            'Model': name,
            'Accuracy': test_results['accuracy'],
            'Precision': test_results['precision'],
            'Recall': test_results['recall'],
            'F1 Score': test_results['f1_score'],
            'CV Mean': test_results['cv_mean'],
            'CV Std': test_results['cv_std']
        })
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('F1 Score', ascending=False)
    
    print(f"\n{'='*60}")
    print("MODEL COMPARISON RESULTS")
    print(f"{'='*60}")
    print(comparison_df.to_string(index=False))
    
    return comparison_df


if __name__ == "__main__":
    print("=" * 60)
    print("TEXT CLASSIFIER - Traditional ML Models")
    print("Supports: Complaints, Reviews, and Other Text Data")
    print("=" * 60)
    
    # Main menu
    print("\nSelect an action:")
    print("  1 - Train and evaluate a single model")
    print("  2 - Compare multiple models")
    print("  3 - Make predictions with a saved model")
    print("  4 - Exit")
    
    action = input("\nEnter choice (1-4): ").strip()
    
    if action == "1":
        # Train and evaluate single model
        file_path = select_file(title="Select data file (complaints, reviews, etc.)")
        if not file_path:
            print("No file selected. Exiting.")
            sys.exit(0)
        
        print(f"\nSelected file: {file_path}")
        
        # Load data
        max_rows_input = input("Maximum rows to load (press Enter for all): ").strip()
        max_rows = int(max_rows_input) if max_rows_input else None
        
        df = load_complaint_data(file_path, max_rows=max_rows)
        if df is None:
            sys.exit(1)
        
        # Get column names with defaults
        suggested_text, suggested_label = detect_dataset_type(df)
        
        text_prompt = f"\nEnter the text column name"
        if suggested_text:
            text_prompt += f" (press Enter for '{suggested_text}')"
        text_prompt += ": "
        text_col = input(text_prompt).strip() or suggested_text
        
        label_prompt = f"Enter the label column name"
        if suggested_label:
            label_prompt += f" (press Enter for '{suggested_label}')"
        label_prompt += ": "
        label_col = input(label_prompt).strip() or suggested_label
        
        # Check if we should convert stars to sentiment
        convert_to_sentiment = False
        if label_col == 'stars':
            sentiment_choice = input("\nConvert star ratings to sentiment? (y/n, default: n): ").strip().lower()
            convert_to_sentiment = (sentiment_choice == 'y')
        
        # Prepare data
        X, y, label_mapping = prepare_data(df, text_col, label_col, 
                                           convert_stars_to_sentiment=convert_to_sentiment)
        if X is None:
            sys.exit(1)
        
        # Select model
        print("\nAvailable models:")
        print("  1 - Logistic Regression")
        print("  2 - Support Vector Machine (SVM)")
        print("  3 - Random Forest")
        print("  4 - Naive Bayes")
        
        model_choice = input("\nSelect model (1-4, default: 1): ").strip() or "1"
        model_map = {
            '1': 'logistic_regression',
            '2': 'svm',
            '3': 'random_forest',
            '4': 'naive_bayes'
        }
        model_type = model_map.get(model_choice, 'logistic_regression')
        
        # Select vectorizer
        vectorizer_choice = input("Vectorizer type (1=TF-IDF, 2=Count, default: 1): ").strip() or "1"
        vectorizer_type = 'tfidf' if vectorizer_choice == '1' else 'count'
        
        # Train and evaluate
        model, results = train_and_evaluate(X, y, label_mapping, model_type, vectorizer_type)
        
        # Save model
        save_choice = input("\nSave trained model? (y/n): ").strip().lower()
        if save_choice == 'y':
            output_dir = Path("models")
            output_dir.mkdir(exist_ok=True)
            model_name = input("Enter model name (e.g., 'complaint_classifier'): ").strip() or "complaint_classifier"
            output_path = output_dir / f"{model_name}.pkl"
            save_model(model, label_mapping, output_path)
    
    elif action == "2":
        # Compare multiple models
        file_path = select_file(title="Select data file (complaints, reviews, etc.)")
        if not file_path:
            print("No file selected. Exiting.")
            sys.exit(0)
        
        print(f"\nSelected file: {file_path}")
        
        # Load data
        max_rows_input = input("Maximum rows to load (press Enter for all): ").strip()
        max_rows = int(max_rows_input) if max_rows_input else None
        
        df = load_complaint_data(file_path, max_rows=max_rows)
        if df is None:
            sys.exit(1)
        
        # Get column names with defaults
        suggested_text, suggested_label = detect_dataset_type(df)
        
        text_prompt = f"\nEnter the text column name"
        if suggested_text:
            text_prompt += f" (press Enter for '{suggested_text}')"
        text_prompt += ": "
        text_col = input(text_prompt).strip() or suggested_text
        
        label_prompt = f"Enter the label column name"
        if suggested_label:
            label_prompt += f" (press Enter for '{suggested_label}')"
        label_prompt += ": "
        label_col = input(label_prompt).strip() or suggested_label
        
        # Check if we should convert stars to sentiment
        convert_to_sentiment = False
        if label_col == 'stars':
            sentiment_choice = input("\nConvert star ratings to sentiment? (y/n, default: n): ").strip().lower()
            convert_to_sentiment = (sentiment_choice == 'y')
        
        # Prepare data
        X, y, label_mapping = prepare_data(df, text_col, label_col,
                                           convert_stars_to_sentiment=convert_to_sentiment)
        if X is None:
            sys.exit(1)
        
        # Compare models
        comparison_results = compare_models(X, y, label_mapping)
        
        # Save comparison results
        save_choice = input("\nSave comparison results? (y/n): ").strip().lower()
        if save_choice == 'y':
            output_dir = Path("analysis_findings")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / "model_comparison_results.csv"
            comparison_results.to_csv(output_path, index=False)
            print(f"Comparison results saved to: {output_path}")
    
    elif action == "3":
        # Make predictions with saved model
        model_path = select_file(
            filetypes=[("Model files", "*.pkl"), ("All files", "*.*")],
            title="Select saved model file"
        )
        if not model_path:
            print("No model selected. Exiting.")
            sys.exit(0)
        
        model, label_mapping = load_model(model_path)
        if model is None:
            sys.exit(1)
        
        print(f"\nModel loaded with {len(label_mapping)} categories:")
        for idx, label in sorted(label_mapping.items()):
            print(f"  {idx}: {label}")
        
        # Get complaints to classify
        print("\nEnter complaint descriptions to classify (one per line)")
        print("Enter an empty line when done:")
        
        complaints = []
        while True:
            line = input("> ").strip()
            if not line:
                break
            complaints.append(line)
        
        if complaints:
            predictions = predict_complaints(model, label_mapping, complaints)
            print(f"\n{'='*60}")
            print("PREDICTIONS")
            print(f"{'='*60}")
            print(predictions.to_string(index=False))
            
            # Save predictions
            save_choice = input("\nSave predictions? (y/n): ").strip().lower()
            if save_choice == 'y':
                output_dir = Path("data/processed")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / "complaint_predictions.csv"
                predictions.to_csv(output_path, index=False)
                print(f"Predictions saved to: {output_path}")
        else:
            print("No complaints entered.")
    
    elif action == "4":
        print("Exiting.")
        sys.exit(0)
    
    else:
        print(f"Invalid choice: {action}")
        sys.exit(1)
    
    print("\nDone!")
