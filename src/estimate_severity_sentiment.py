import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import re
import pickle

# Lexicon-based sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("Warning: VADER not installed. Run: pip install vaderSentiment")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("Warning: TextBlob not installed. Run: pip install textblob")

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.preprocessing import LabelEncoder


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


def load_data(file_path, max_rows=None):
    """
    Load data from a file.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the data file
    max_rows : int, optional
        Maximum number of rows to load
        
    Returns:
    --------
    DataFrame or None
        Loaded data
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
        print(f"\nAvailable columns: {', '.join(df.columns.tolist())}")
        
        return df
        
    except Exception as e:
        print(f"Failed to load file: {e}")
        return None


def preprocess_text(text):
    """
    Basic text preprocessing.
    
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
    return str(text).strip()


# ==================== LEXICON-BASED METHODS ====================

def analyze_vader_sentiment(text):
    """
    Analyze sentiment using VADER (Valence Aware Dictionary and Sentiment Reasoner).
    
    Parameters:
    -----------
    text : str
        Text to analyze
        
    Returns:
    --------
    dict
        Dictionary with 'neg', 'neu', 'pos', 'compound' scores
    """
    if not VADER_AVAILABLE:
        return {'neg': np.nan, 'neu': np.nan, 'pos': np.nan, 'compound': np.nan}
    
    analyzer = SentimentIntensityAnalyzer()
    try:
        scores = analyzer.polarity_scores(str(text))
        return scores
    except Exception:
        return {'neg': np.nan, 'neu': np.nan, 'pos': np.nan, 'compound': np.nan}


def analyze_textblob_sentiment(text):
    """
    Analyze sentiment using TextBlob.
    
    Parameters:
    -----------
    text : str
        Text to analyze
        
    Returns:
    --------
    dict
        Dictionary with 'polarity' (-1 to 1) and 'subjectivity' (0 to 1)
    """
    if not TEXTBLOB_AVAILABLE:
        return {'polarity': np.nan, 'subjectivity': np.nan}
    
    try:
        blob = TextBlob(str(text))
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    except Exception:
        return {'polarity': np.nan, 'subjectivity': np.nan}


def estimate_severity_lexicon(text, severity_keywords=None):
    """
    Estimate severity using a keyword-based lexicon approach.
    
    Parameters:
    -----------
    text : str
        Text to analyze
    severity_keywords : dict, optional
        Dictionary mapping severity levels to lists of keywords
        
    Returns:
    --------
    dict
        Dictionary with severity scores and estimated level
    """
    if severity_keywords is None:
        # Default severity keywords (can be customized)
        severity_keywords = {
            'critical': ['urgent', 'critical', 'emergency', 'severe', 'dangerous', 
                        'immediately', 'asap', 'hazard', 'threat', 'crisis'],
            'high': ['important', 'serious', 'major', 'significant', 'broken', 
                    'not working', 'failed', 'problem', 'issue', 'concern'],
            'medium': ['moderate', 'minor', 'small', 'inconvenience', 'annoying',
                      'somewhat', 'occasionally', 'sometimes'],
            'low': ['trivial', 'cosmetic', 'suggestion', 'enhancement', 'nice to have',
                   'minor', 'slight', 'barely']
        }
    
    text_lower = str(text).lower()
    
    scores = {
        'critical': 0,
        'high': 0,
        'medium': 0,
        'low': 0
    }
    
    # Count keyword matches for each severity level
    for level, keywords in severity_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                scores[level] += 1
    
    # Determine overall severity level
    total_matches = sum(scores.values())
    if total_matches == 0:
        estimated_level = 'unknown'
        confidence = 0.0
    else:
        max_level = max(scores.items(), key=lambda x: x[1])
        estimated_level = max_level[0]
        confidence = max_level[1] / total_matches
    
    return {
        'critical_score': scores['critical'],
        'high_score': scores['high'],
        'medium_score': scores['medium'],
        'low_score': scores['low'],
        'total_matches': total_matches,
        'estimated_severity': estimated_level,
        'confidence': confidence
    }


def apply_lexicon_analysis(df, text_column, include_vader=True, include_textblob=True, 
                          include_severity=False):
    """
    Apply lexicon-based analysis to a dataframe.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    text_column : str
        Name of the text column to analyze
    include_vader : bool
        Whether to include VADER analysis
    include_textblob : bool
        Whether to include TextBlob analysis
    include_severity : bool
        Whether to include severity estimation
        
    Returns:
    --------
    DataFrame
        DataFrame with added analysis columns
    """
    if text_column not in df.columns:
        print(f"Error: Column '{text_column}' not found")
        return None
    
    result_df = df.copy()
    
    print("\nApplying lexicon-based analysis...")
    
    # VADER analysis
    if include_vader and VADER_AVAILABLE:
        print("  Computing VADER sentiment scores...")
        vader_results = result_df[text_column].apply(analyze_vader_sentiment)
        result_df['vader_negative'] = vader_results.apply(lambda x: x['neg'])
        result_df['vader_neutral'] = vader_results.apply(lambda x: x['neu'])
        result_df['vader_positive'] = vader_results.apply(lambda x: x['pos'])
        result_df['vader_compound'] = vader_results.apply(lambda x: x['compound'])
        
        # Categorize based on compound score
        def vader_category(compound):
            if pd.isna(compound):
                return 'unknown'
            if compound >= 0.05:
                return 'positive'
            elif compound <= -0.05:
                return 'negative'
            else:
                return 'neutral'
        
        result_df['vader_sentiment'] = result_df['vader_compound'].apply(vader_category)
        print(f"    VADER sentiment distribution:")
        print(result_df['vader_sentiment'].value_counts().to_string())
    
    # TextBlob analysis
    if include_textblob and TEXTBLOB_AVAILABLE:
        print("\n  Computing TextBlob sentiment scores...")
        textblob_results = result_df[text_column].apply(analyze_textblob_sentiment)
        result_df['textblob_polarity'] = textblob_results.apply(lambda x: x['polarity'])
        result_df['textblob_subjectivity'] = textblob_results.apply(lambda x: x['subjectivity'])
        
        # Categorize based on polarity
        def textblob_category(polarity):
            if pd.isna(polarity):
                return 'unknown'
            if polarity > 0.1:
                return 'positive'
            elif polarity < -0.1:
                return 'negative'
            else:
                return 'neutral'
        
        result_df['textblob_sentiment'] = result_df['textblob_polarity'].apply(textblob_category)
        print(f"    TextBlob sentiment distribution:")
        print(result_df['textblob_sentiment'].value_counts().to_string())
    
    # Severity estimation
    if include_severity:
        print("\n  Estimating severity levels...")
        severity_results = result_df[text_column].apply(estimate_severity_lexicon)
        result_df['severity_critical'] = severity_results.apply(lambda x: x['critical_score'])
        result_df['severity_high'] = severity_results.apply(lambda x: x['high_score'])
        result_df['severity_medium'] = severity_results.apply(lambda x: x['medium_score'])
        result_df['severity_low'] = severity_results.apply(lambda x: x['low_score'])
        result_df['severity_level'] = severity_results.apply(lambda x: x['estimated_severity'])
        result_df['severity_confidence'] = severity_results.apply(lambda x: x['confidence'])
        print(f"    Severity level distribution:")
        print(result_df['severity_level'].value_counts().to_string())
    
    print("\nLexicon-based analysis complete!")
    return result_df


# ==================== ML-BASED METHODS ====================

def create_labeled_subset(df, text_column, label_column, label_mapping=None):
    """
    Create a labeled subset for training ML models.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    text_column : str
        Name of the text column
    label_column : str
        Name of the label column (can be from lexicon analysis or existing labels)
    label_mapping : dict, optional
        Mapping to consolidate labels (e.g., {'positive': 'positive', 'negative': 'negative', 'neutral': 'neutral'})
        
    Returns:
    --------
    tuple
        (X, y) where X is text data and y is labels
    """
    if text_column not in df.columns:
        print(f"Error: Text column '{text_column}' not found")
        return None, None
    
    if label_column not in df.columns:
        print(f"Error: Label column '{label_column}' not found")
        return None, None
    
    # Filter out rows with missing values
    data = df[[text_column, label_column]].dropna()
    
    if label_mapping:
        data[label_column] = data[label_column].map(label_mapping)
        data = data.dropna()
    
    # Remove 'unknown' labels if present
    if 'unknown' in data[label_column].values:
        data = data[data[label_column] != 'unknown']
    
    print(f"\nLabeled subset created:")
    print(f"  Total samples: {len(data):,}")
    print(f"  Label distribution:")
    print(data[label_column].value_counts().to_string())
    
    return data[text_column], data[label_column]


def train_ml_classifier(X, y, model_type='logistic_regression', test_size=0.2, 
                       max_features=5000, ngram_range=(1, 2)):
    """
    Train a simple ML classifier on labeled data.
    
    Parameters:
    -----------
    X : Series or array
        Text data
    y : Series or array
        Labels
    model_type : str
        Type of model: 'logistic_regression', 'naive_bayes', 'random_forest'
    test_size : float
        Proportion of data for testing
    max_features : int
        Maximum number of TF-IDF features
    ngram_range : tuple
        N-gram range for TF-IDF
        
    Returns:
    --------
    tuple
        (model, vectorizer, label_encoder, test_results)
    """
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} classifier")
    print(f"{'='*60}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )
    
    print(f"\nDataset split:")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.8,
        stop_words='english'
    )
    
    # Transform text to features
    print("\nVectorizing text...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Select and train classifier
    if model_type == 'logistic_regression':
        classifier = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    elif model_type == 'naive_bayes':
        classifier = MultinomialNB(alpha=0.1)
    elif model_type == 'random_forest':
        classifier = RandomForestClassifier(n_estimators=100, random_state=42, 
                                           class_weight='balanced', n_jobs=-1)
    else:
        print(f"Unknown model type: {model_type}. Using logistic regression.")
        classifier = LogisticRegression(max_iter=1000, random_state=42)
    
    print(f"Training {model_type} model...")
    classifier.fit(X_train_vec, y_train)
    print("Training complete!")
    
    # Evaluate
    print("\nEvaluating on test set...")
    y_pred = classifier.predict(X_test_vec)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print("TEST SET PERFORMANCE")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f}")
    
    print(f"\n{'='*60}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*60}")
    target_names = label_encoder.classes_
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    
    # Store results
    test_results = {
        'accuracy': accuracy,
        'y_test': y_test,
        'y_pred': y_pred,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'label_encoder': label_encoder
    }
    
    return classifier, vectorizer, label_encoder, test_results


def apply_ml_predictions(df, text_column, classifier, vectorizer, label_encoder, 
                        output_column_name='ml_prediction'):
    """
    Apply trained ML model to make predictions on new data.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    text_column : str
        Name of the text column
    classifier : trained model
        Trained classifier
    vectorizer : fitted vectorizer
        Fitted TF-IDF vectorizer
    label_encoder : fitted LabelEncoder
        Fitted label encoder
    output_column_name : str
        Name for the output prediction column
        
    Returns:
    --------
    DataFrame
        DataFrame with added prediction column
    """
    result_df = df.copy()
    
    print(f"\nApplying ML predictions to {len(df):,} rows...")
    
    # Vectorize text
    X_vec = vectorizer.transform(result_df[text_column].fillna(''))
    
    # Predict
    predictions = classifier.predict(X_vec)
    
    # Decode labels
    result_df[output_column_name] = label_encoder.inverse_transform(predictions)
    
    print(f"Prediction distribution:")
    print(result_df[output_column_name].value_counts().to_string())
    
    return result_df


def compare_methods(df, text_column, lexicon_column, ml_column):
    """
    Compare results from lexicon-based and ML-based methods.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with both lexicon and ML predictions
    text_column : str
        Name of the text column
    lexicon_column : str
        Name of the lexicon-based prediction column
    ml_column : str
        Name of the ML-based prediction column
        
    Returns:
    --------
    DataFrame
        Comparison statistics
    """
    print(f"\n{'='*60}")
    print("COMPARING LEXICON-BASED VS ML-BASED METHODS")
    print(f"{'='*60}")
    
    # Filter valid comparisons
    comparison_df = df[[lexicon_column, ml_column]].dropna()
    
    if len(comparison_df) == 0:
        print("No valid data for comparison")
        return None
    
    # Calculate agreement
    agreement = (comparison_df[lexicon_column] == comparison_df[ml_column]).mean()
    
    print(f"\nTotal comparisons: {len(comparison_df):,}")
    print(f"Agreement rate: {agreement:.2%}")
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    print(f"(Rows: {lexicon_column}, Columns: {ml_column})")
    confusion = pd.crosstab(
        comparison_df[lexicon_column], 
        comparison_df[ml_column], 
        margins=True
    )
    print(confusion)
    
    # Disagreement analysis
    disagreements = comparison_df[comparison_df[lexicon_column] != comparison_df[ml_column]]
    if len(disagreements) > 0:
        print(f"\nDisagreement breakdown:")
        disagree_pairs = disagreements.groupby([lexicon_column, ml_column]).size()
        for (lex, ml), count in disagree_pairs.items():
            print(f"  {lex} (lexicon) vs {ml} (ML): {count} ({count/len(disagreements)*100:.1f}%)")
    
    return confusion


def save_results(df, output_path):
    """
    Save results to a CSV file.
    
    Parameters:
    -----------
    df : DataFrame
        Results dataframe
    output_path : str or Path
        Path to save the file
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\nResults saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Failed to save results: {e}")
        return False


def save_model(classifier, vectorizer, label_encoder, output_path):
    """
    Save trained model components to disk.
    
    Parameters:
    -----------
    classifier : trained model
        Trained classifier
    vectorizer : fitted vectorizer
        Fitted TF-IDF vectorizer
    label_encoder : fitted LabelEncoder
        Fitted label encoder
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
        
        model_data = {
            'classifier': classifier,
            'vectorizer': vectorizer,
            'label_encoder': label_encoder
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Failed to save model: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("SEVERITY & SENTIMENT ESTIMATION")
    print("Lexicon-based (VADER, TextBlob) + ML Classifiers")
    print("=" * 60)
    
    # Check for required libraries
    if not VADER_AVAILABLE:
        print("\n⚠ VADER not available. Install with: pip install vaderSentiment")
    if not TEXTBLOB_AVAILABLE:
        print("\n⚠ TextBlob not available. Install with: pip install textblob")
    
    # Main menu
    print("\nSelect an action:")
    print("  1 - Apply lexicon-based analysis (VADER, TextBlob, Severity)")
    print("  2 - Train ML classifier on labeled data")
    print("  3 - Apply ML classifier to new data")
    print("  4 - Full pipeline: Lexicon + ML + Comparison")
    print("  5 - Exit")
    
    action = input("\nEnter choice (1-5): ").strip()
    
    if action == "1":
        # Lexicon-based analysis only
        file_path = select_file(title="Select data file")
        if not file_path:
            print("No file selected. Exiting.")
            sys.exit(0)
        
        print(f"\nSelected file: {file_path}")
        
        # Load data
        max_rows_input = input("Maximum rows to load (press Enter for all): ").strip()
        max_rows = int(max_rows_input) if max_rows_input else None
        
        df = load_data(file_path, max_rows=max_rows)
        if df is None:
            sys.exit(1)
        
        # Get text column
        text_col = input("\nEnter the name of the text column: ").strip()
        
        # Choose analyses
        include_vader = input("Include VADER sentiment? (y/n, default: y): ").strip().lower() != 'n'
        include_textblob = input("Include TextBlob sentiment? (y/n, default: y): ").strip().lower() != 'n'
        include_severity = input("Include severity estimation? (y/n, default: n): ").strip().lower() == 'y'
        
        # Apply analysis
        result_df = apply_lexicon_analysis(df, text_col, include_vader, include_textblob, include_severity)
        
        if result_df is not None:
            # Save results
            save_choice = input("\nSave results? (y/n): ").strip().lower()
            if save_choice == 'y':
                output_dir = Path("data/processed")
                output_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(file_path).stem
                output_path = output_dir / f"{stem}_with_sentiment.csv"
                save_results(result_df, output_path)
    
    elif action == "2":
        # Train ML classifier
        file_path = select_file(title="Select data file with labels")
        if not file_path:
            print("No file selected. Exiting.")
            sys.exit(0)
        
        print(f"\nSelected file: {file_path}")
        
        # Load data
        max_rows_input = input("Maximum rows to load (press Enter for all): ").strip()
        max_rows = int(max_rows_input) if max_rows_input else None
        
        df = load_data(file_path, max_rows=max_rows)
        if df is None:
            sys.exit(1)
        
        # Get columns
        text_col = input("\nEnter the name of the text column: ").strip()
        label_col = input("Enter the name of the label column: ").strip()
        
        # Create labeled subset
        X, y = create_labeled_subset(df, text_col, label_col)
        if X is None:
            sys.exit(1)
        
        # Select model
        print("\nAvailable models:")
        print("  1 - Logistic Regression")
        print("  2 - Naive Bayes")
        print("  3 - Random Forest")
        
        model_choice = input("\nSelect model (1-3, default: 1): ").strip() or "1"
        model_map = {'1': 'logistic_regression', '2': 'naive_bayes', '3': 'random_forest'}
        model_type = model_map.get(model_choice, 'logistic_regression')
        
        # Train model
        classifier, vectorizer, label_encoder, results = train_ml_classifier(X, y, model_type)
        
        # Save model
        save_choice = input("\nSave trained model? (y/n): ").strip().lower()
        if save_choice == 'y':
            output_dir = Path("models")
            output_dir.mkdir(exist_ok=True)
            model_name = input("Enter model name (e.g., 'sentiment_classifier'): ").strip() or "sentiment_classifier"
            output_path = output_dir / f"{model_name}.pkl"
            save_model(classifier, vectorizer, label_encoder, output_path)
    
    elif action == "3":
        # Apply ML classifier to new data
        print("\nFirst, select the trained model file:")
        model_path = select_file(
            filetypes=[("Model files", "*.pkl"), ("All files", "*.*")],
            title="Select trained model file"
        )
        if not model_path:
            print("No model selected. Exiting.")
            sys.exit(0)
        
        # Load model
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            classifier = model_data['classifier']
            vectorizer = model_data['vectorizer']
            label_encoder = model_data['label_encoder']
            print(f"Model loaded from: {model_path}")
            print(f"Classes: {', '.join(label_encoder.classes_)}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            sys.exit(1)
        
        # Load data
        print("\nNow, select the data file:")
        file_path = select_file(title="Select data file")
        if not file_path:
            print("No file selected. Exiting.")
            sys.exit(0)
        
        max_rows_input = input("Maximum rows to load (press Enter for all): ").strip()
        max_rows = int(max_rows_input) if max_rows_input else None
        
        df = load_data(file_path, max_rows=max_rows)
        if df is None:
            sys.exit(1)
        
        # Get text column
        text_col = input("\nEnter the name of the text column: ").strip()
        
        # Apply predictions
        result_df = apply_ml_predictions(df, text_col, classifier, vectorizer, label_encoder)
        
        # Save results
        save_choice = input("\nSave results? (y/n): ").strip().lower()
        if save_choice == 'y':
            output_dir = Path("data/processed")
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(file_path).stem
            output_path = output_dir / f"{stem}_ml_predictions.csv"
            save_results(result_df, output_path)
    
    elif action == "4":
        # Full pipeline: Lexicon + ML + Comparison
        file_path = select_file(title="Select data file")
        if not file_path:
            print("No file selected. Exiting.")
            sys.exit(0)
        
        print(f"\nSelected file: {file_path}")
        
        # Load data
        max_rows_input = input("Maximum rows to load (press Enter for all): ").strip()
        max_rows = int(max_rows_input) if max_rows_input else None
        
        df = load_data(file_path, max_rows=max_rows)
        if df is None:
            sys.exit(1)
        
        # Get text column
        text_col = input("\nEnter the name of the text column: ").strip()
        
        # Step 1: Apply lexicon-based analysis
        print("\n" + "="*60)
        print("STEP 1: LEXICON-BASED ANALYSIS")
        print("="*60)
        
        result_df = apply_lexicon_analysis(df, text_col, 
                                          include_vader=VADER_AVAILABLE, 
                                          include_textblob=TEXTBLOB_AVAILABLE, 
                                          include_severity=False)
        
        if result_df is None:
            sys.exit(1)
        
        # Step 2: Train ML classifier using lexicon results as labels
        print("\n" + "="*60)
        print("STEP 2: TRAIN ML CLASSIFIER")
        print("="*60)
        
        # Choose which lexicon method to use as labels
        label_choices = []
        if VADER_AVAILABLE:
            label_choices.append(('vader_sentiment', 'VADER'))
        if TEXTBLOB_AVAILABLE:
            label_choices.append(('textblob_sentiment', 'TextBlob'))
        
        if not label_choices:
            print("No lexicon methods available to use as labels.")
            sys.exit(1)
        
        if len(label_choices) == 1:
            label_col = label_choices[0][0]
            print(f"Using {label_choices[0][1]} as labels for ML training")
        else:
            print("\nSelect which lexicon method to use as labels:")
            for i, (col, name) in enumerate(label_choices, 1):
                print(f"  {i} - {name}")
            choice = input(f"\nSelect (1-{len(label_choices)}, default: 1): ").strip() or "1"
            idx = int(choice) - 1 if choice.isdigit() else 0
            label_col = label_choices[idx][0]
        
        # Create labeled subset
        X, y = create_labeled_subset(result_df, text_col, label_col)
        if X is None:
            print("Insufficient labeled data for ML training.")
        else:
            # Train classifier
            classifier, vectorizer, label_encoder, ml_results = train_ml_classifier(
                X, y, model_type='logistic_regression'
            )
            
            # Step 3: Apply ML predictions to full dataset
            print("\n" + "="*60)
            print("STEP 3: APPLY ML PREDICTIONS")
            print("="*60)
            
            result_df = apply_ml_predictions(result_df, text_col, classifier, vectorizer, 
                                           label_encoder, output_column_name='ml_sentiment')
            
            # Step 4: Compare methods
            print("\n" + "="*60)
            print("STEP 4: COMPARE METHODS")
            print("="*60)
            
            comparison = compare_methods(result_df, text_col, label_col, 'ml_sentiment')
        
        # Save results
        save_choice = input("\nSave final results? (y/n): ").strip().lower()
        if save_choice == 'y':
            output_dir = Path("data/processed")
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(file_path).stem
            output_path = output_dir / f"{stem}_full_analysis.csv"
            save_results(result_df, output_path)
            
            # Optionally save the ML model
            save_model_choice = input("Save ML model? (y/n): ").strip().lower()
            if save_model_choice == 'y':
                model_dir = Path("models")
                model_dir.mkdir(exist_ok=True)
                model_path = model_dir / "sentiment_ml_model.pkl"
                save_model(classifier, vectorizer, label_encoder, model_path)
    
    elif action == "5":
        print("Exiting.")
        sys.exit(0)
    
    else:
        print(f"Invalid choice: {action}")
        sys.exit(1)
    
    print("\nDone!")
