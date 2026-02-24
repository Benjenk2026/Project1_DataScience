import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional

# NLP utilities
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not installed. Run: pip install nltk")

# TF-IDF for token importance
from sklearn.feature_extraction.text import TfidfVectorizer


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


def preprocess_text(text, lowercase=True, remove_special=True, remove_extra_spaces=True):
    """
    Basic text preprocessing.
    
    Parameters:
    -----------
    text : str
        Raw text to preprocess
    lowercase : bool
        Convert to lowercase
    remove_special : bool
        Remove special characters
    remove_extra_spaces : bool
        Remove extra whitespace
        
    Returns:
    --------
    str
        Preprocessed text
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    if lowercase:
        text = text.lower()
    
    if remove_special:
        # Keep alphanumeric, spaces, and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\:\-]', ' ', text)
    
    if remove_extra_spaces:
        text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def get_stopwords(custom_stopwords=None):
    """
    Get a set of stopwords for filtering.
    
    Parameters:
    -----------
    custom_stopwords : list, optional
        Additional stopwords to include
        
    Returns:
    --------
    set
        Set of stopwords
    """
    # Default stopwords (common English words)
    default_stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
        'us', 'them', 'what', 'which', 'who', 'when', 'where', 'why', 'how'
    }
    
    if custom_stopwords:
        default_stopwords.update(custom_stopwords)
    
    return default_stopwords


def extract_ngrams(text, n=2, join=True, remove_stopwords=False):
    """
    Extract n-grams from text.
    
    Parameters:
    -----------
    text : str
        Input text
    n : int
        N-gram size (default: 2 for bigrams)
    join : bool
        If True, join n-grams as strings; if False, return as tuples
    remove_stopwords : bool
        Whether to filter out stopwords
        
    Returns:
    --------
    list
        List of n-grams
    """
    words = text.lower().split()
    
    if remove_stopwords:
        stopwords_set = get_stopwords()
        words = [w for w in words if w not in stopwords_set]
    
    # Create n-grams
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = words[i:i+n]
        if join:
            ngrams.append(' '.join(ngram))
        else:
            ngrams.append(tuple(ngram))
    
    return ngrams


def extract_key_phrases_tfidf(texts, n_phrases=5, ngram_range=(1, 3)):
    """
    Extract key phrases using TF-IDF scoring.
    
    Parameters:
    -----------
    texts : list or Series
        List of text documents
    n_phrases : int
        Number of top phrases to extract
    ngram_range : tuple
        Range of n-gram sizes to consider
        
    Returns:
    --------
    list
        List of top key phrases
    """
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=100,
        stop_words='english',
        min_df=2  # Appear in at least 2 documents
    )
    
    try:
        # Fit to texts
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Get feature names and scores
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate mean TF-IDF score per feature
        mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        
        # Get top phrases
        top_indices = np.argsort(mean_scores)[::-1][:n_phrases]
        top_phrases = [feature_names[i] for i in top_indices]
        
        return top_phrases
    
    except Exception as e:
        print(f"Error extracting key phrases: {e}")
        return []


def extract_key_phrases_frequency(text, n_phrases=5, ngram_range=(1, 2), 
                                  remove_stopwords=True, min_freq=1):
    """
    Extract key phrases using frequency-based approach.
    
    Parameters:
    -----------
    text : str
        Input text
    n_phrases : int
        Number of top phrases to extract
    ngram_range : tuple
        Range of n-gram sizes to consider
    remove_stopwords : bool
        Whether to filter stopwords
    min_freq : int
        Minimum frequency threshold
        
    Returns:
    --------
    list
        List of top key phrases with frequencies
    """
    phrases = Counter()
    
    # Extract n-grams for each n in range
    for n in range(ngram_range[0], ngram_range[1] + 1):
        text_lower = text.lower()
        ngrams = extract_ngrams(text_lower, n=n, remove_stopwords=remove_stopwords)
        phrases.update(ngrams)
    
    # Filter by minimum frequency and get top phrases
    valid_phrases = [(phrase, count) for phrase, count in phrases.items() if count >= min_freq]
    valid_phrases.sort(key=lambda x: x[1], reverse=True)
    
    return valid_phrases[:n_phrases]


def extract_important_tokens(text, n_tokens=5, remove_stopwords=True, 
                            min_length=3, pos_filter=None):
    """
    Extract important tokens from text.
    
    Parameters:
    -----------
    text : str
        Input text
    n_tokens : int
        Number of top tokens to extract
    remove_stopwords : bool
        Whether to filter stopwords
    min_length : int
        Minimum token length
    pos_filter : list, optional
        Part-of-speech tags to include (e.g., ['NN', 'VB', 'JJ'])
        Requires NLTK
        
    Returns:
    --------
    list
        List of important tokens
    """
    # Tokenize
    text_lower = text.lower()
    words = text_lower.split()
    
    # Filter by length
    words = [w for w in words if len(w) >= min_length]
    
    # Filter stopwords
    if remove_stopwords:
        stopwords_set = get_stopwords()
        words = [w for w in words if w not in stopwords_set]
    
    # Filter by POS tags if requested and NLTK available
    if pos_filter and NLTK_AVAILABLE:
        try:
            pos_tags = pos_tag(words)
            words = [word for word, pos in pos_tags if pos in pos_filter]
        except Exception:
            pass  # Fall back to no POS filtering
    
    # Get frequency of remaining tokens
    token_counts = Counter(words)
    top_tokens = token_counts.most_common(n_tokens)
    
    return top_tokens


def extract_first_sentence(text, sentence_limit=1):
    """
    Extract first N sentences from text.
    
    Parameters:
    -----------
    text : str
        Input text
    sentence_limit : int
        Number of sentences to extract
        
    Returns:
    --------
    str
        Extracted sentences
    """
    text = str(text).strip()
    
    if NLTK_AVAILABLE:
        try:
            sentences = sent_tokenize(text)
        except Exception:
            # Fallback to simple sentence splitting
            sentences = [s.strip() for s in text.split('.') if s.strip()]
    else:
        # Simple sentence splitting by period
        sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    selected = sentences[:sentence_limit]
    return '. '.join(selected) + ('.' if selected else '')


def extract_leading_phrase(text, max_length=100, word_limit=None):
    """
    Extract leading phrase from text.
    
    Parameters:
    -----------
    text : str
        Input text
    max_length : int
        Maximum character length
    word_limit : int, optional
        Maximum number of words
        
    Returns:
    --------
    str
        Leading phrase
    """
    text = str(text).strip()
    
    # Split into words
    words = text.split()
    
    # Apply word limit if specified
    if word_limit:
        words = words[:word_limit]
    
    result = ' '.join(words)
    
    # Apply character limit
    if len(result) > max_length:
        # Truncate and find last full word
        truncated = result[:max_length]
        last_space = truncated.rfind(' ')
        if last_space > 0:
            result = truncated[:last_space]
        else:
            result = truncated
        result += '...'
    
    return result


def generate_summary_by_method(text, method='key_phrases_freq', n_items=5, **kwargs):
    """
    Generate summary using specified method.
    
    Parameters:
    -----------
    text : str
        Input text
    method : str
        Summarization method:
        - 'first_sentence': First sentence
        - 'leading_phrase': Leading portion
        - 'key_phrases_freq': Frequency-based key phrases
        - 'important_tokens': Frequency-based tokens
    n_items : int
        Number of items to extract
    **kwargs : dict
        Additional arguments for specific methods
        
    Returns:
    --------
    str
        Generated summary
    """
    if pd.isna(text) or not text:
        return ""
    
    text = str(text).strip()
    
    if method == 'first_sentence':
        return extract_first_sentence(text, sentence_limit=n_items)
    
    elif method == 'leading_phrase':
        return extract_leading_phrase(text, word_limit=n_items)
    
    elif method == 'key_phrases_freq':
        phrases = extract_key_phrases_frequency(
            text, 
            n_phrases=n_items,
            **kwargs
        )
        if phrases:
            return ', '.join([phrase for phrase, _ in phrases])
        return ""
    
    elif method == 'important_tokens':
        tokens = extract_important_tokens(text, n_tokens=n_items, **kwargs)
        if tokens:
            return ', '.join([token for token, _ in tokens])
        return ""
    
    else:
        print(f"Unknown method: {method}")
        return ""


def apply_summarization(df, text_column, methods=None, n_items=5, 
                       save_original=False, save_all_methods=True):
    """
    Apply summarization to a dataframe.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    text_column : str
        Name of the text column
    methods : list, optional
        List of methods to apply. If None, defaults to all methods
    n_items : int
        Number of items to extract per method
    save_original : bool
        Whether to preserve original text column
    save_all_methods : bool
        Whether to save all methods in separate columns
        
    Returns:
    --------
    DataFrame
        DataFrame with summaries
    """
    if text_column not in df.columns:
        print(f"Error: Column '{text_column}' not found")
        return None
    
    if methods is None:
        methods = ['first_sentence', 'leading_phrase', 'key_phrases_freq', 'important_tokens']
    
    result_df = df.copy()
    
    print(f"\nApplying summarization methods to {len(df):,} rows...")
    print(f"Methods: {', '.join(methods)}")
    
    for method in methods:
        print(f"\n  Applying {method}...")
        col_name = f"summary_{method}"
        
        try:
            if method == 'first_sentence':
                result_df[col_name] = result_df[text_column].apply(
                    lambda x: extract_first_sentence(x, sentence_limit=n_items)
                )
            
            elif method == 'leading_phrase':
                result_df[col_name] = result_df[text_column].apply(
                    lambda x: extract_leading_phrase(x, word_limit=n_items)
                )
            
            elif method == 'key_phrases_freq':
                result_df[col_name] = result_df[text_column].apply(
                    lambda x: generate_summary_by_method(x, method='key_phrases_freq', n_items=n_items)
                )
            
            elif method == 'important_tokens':
                result_df[col_name] = result_df[text_column].apply(
                    lambda x: generate_summary_by_method(x, method='important_tokens', n_items=n_items)
                )
            
            # Print sample results
            sample = result_df[col_name].dropna().iloc[0] if not result_df[col_name].dropna().empty else "[No output]"
            print(f"    Sample output: {sample[:80]}...")
        
        except Exception as e:
            print(f"    Error: {e}")
    
    # If not saving all methods, keep only the first one
    if not save_all_methods and len(methods) > 0:
        cols_to_keep = [col for col in result_df.columns 
                       if col == f"summary_{methods[0]}" or col in df.columns]
        result_df = result_df[cols_to_keep].copy()
        # Rename to generic 'summary'
        result_df.rename(columns={f"summary_{methods[0]}": 'summary'}, inplace=True)
    
    # Optionally remove original text column
    if not save_original and text_column in result_df.columns and text_column not in df.columns:
        result_df = result_df.drop(columns=[text_column])
    
    print(f"\nSummarization complete!")
    return result_df


def analyze_text_statistics(df, text_column):
    """
    Analyze statistics about texts in a column.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    text_column : str
        Name of the text column
        
    Returns:
    --------
    dict
        Dictionary with statistics
    """
    if text_column not in df.columns:
        print(f"Error: Column '{text_column}' not found")
        return None
    
    texts = df[text_column].dropna()
    
    # Calculate statistics
    lengths = texts.astype(str).apply(len)
    word_counts = texts.astype(str).apply(lambda x: len(x.split()))
    sentence_counts = texts.astype(str).apply(lambda x: len(x.split('.')))
    
    stats = {
        'total_texts': len(texts),
        'null_texts': df[text_column].isna().sum(),
        'avg_char_length': lengths.mean(),
        'min_char_length': lengths.min(),
        'max_char_length': lengths.max(),
        'avg_word_count': word_counts.mean(),
        'min_word_count': word_counts.min(),
        'max_word_count': word_counts.max(),
        'avg_sentence_count': sentence_counts.mean(),
    }
    
    return stats


def print_text_statistics(stats):
    """
    Print text statistics in a formatted way.
    
    Parameters:
    -----------
    stats : dict
        Statistics dictionary from analyze_text_statistics
    """
    if stats is None:
        return
    
    print(f"\n{'='*60}")
    print("TEXT STATISTICS")
    print(f"{'='*60}")
    print(f"Total texts: {stats['total_texts']:,}")
    print(f"Null values: {stats['null_texts']}")
    print(f"\nCharacter Length:")
    print(f"  Average: {stats['avg_char_length']:.0f}")
    print(f"  Min: {stats['min_char_length']:.0f}")
    print(f"  Max: {stats['max_char_length']:.0f}")
    print(f"\nWord Count:")
    print(f"  Average: {stats['avg_word_count']:.1f}")
    print(f"  Min: {stats['min_word_count']:.0f}")
    print(f"  Max: {stats['max_word_count']:.0f}")
    print(f"\nSentence Count:")
    print(f"  Average: {stats['avg_sentence_count']:.1f}")


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


if __name__ == "__main__":
    print("=" * 60)
    print("TEXT SUMMARIZATION TOOL")
    print("Heuristic-based complaint/review summarization")
    print("=" * 60)
    
    # Check for NLTK
    if not NLTK_AVAILABLE:
        print("\n⚠ NLTK not available. Install with: pip install nltk")
        print("  Some features (sentence tokenization) will use fallback methods.")
    
    # Main menu
    print("\nSelect an action:")
    print("  1 - Analyze text statistics")
    print("  2 - Generate summaries using selected methods")
    print("  3 - Compare multiple summarization methods")
    print("  4 - Extract key phrases from entire dataset")
    print("  5 - Exit")
    
    action = input("\nEnter choice (1-5): ").strip()
    
    if action == "1":
        # Analyze text statistics
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
        
        # Analyze statistics
        stats = analyze_text_statistics(df, text_col)
        print_text_statistics(stats)
    
    elif action == "2":
        # Generate summaries
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
        
        # Show available methods
        print("\nAvailable summarization methods:")
        print("  1 - First sentence")
        print("  2 - Leading phrase")
        print("  3 - Key phrases (frequency-based)")
        print("  4 - Important tokens (frequency-based)")
        print("  5 - All methods")
        
        method_choice = input("\nSelect method (1-5, default: 5): ").strip() or "5"
        
        method_map = {
            '1': ['first_sentence'],
            '2': ['leading_phrase'],
            '3': ['key_phrases_freq'],
            '4': ['important_tokens'],
            '5': ['first_sentence', 'leading_phrase', 'key_phrases_freq', 'important_tokens']
        }
        
        methods = method_map.get(method_choice, method_map['5'])
        
        # Get parameters
        n_items = input("Number of items to extract per method (default: 5): ").strip()
        n_items = int(n_items) if n_items else 5
        
        save_all = input("Save all method results? (y/n, default: n): ").strip().lower() != 'y'
        
        # Apply summarization
        result_df = apply_summarization(df, text_col, methods=methods, 
                                        n_items=n_items, save_all_methods=save_all)
        
        if result_df is not None:
            # Display samples
            print(f"\n{'='*60}")
            print("SAMPLE RESULTS")
            print(f"{'='*60}")
            
            display_cols = [col for col in result_df.columns if 'summary' in col or col == text_col]
            display_df = result_df[display_cols].head(3)
            for idx, row in display_df.iterrows():
                print(f"\nRow {idx}:")
                for col in display_cols:
                    print(f"  {col}: {row[col][:100]}..." if len(str(row[col])) > 100 else f"  {col}: {row[col]}")
            
            # Save results
            save_choice = input("\nSave results? (y/n): ").strip().lower()
            if save_choice == 'y':
                output_dir = Path("data/processed")
                output_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(file_path).stem
                output_path = output_dir / f"{stem}_with_summaries.csv"
                save_results(result_df, output_path)
    
    elif action == "3":
        # Compare multiple methods
        file_path = select_file(title="Select data file")
        if not file_path:
            print("No file selected. Exiting.")
            sys.exit(0)
        
        print(f"\nSelected file: {file_path}")
        
        # Load data (limited for comparison)
        df = load_data(file_path, max_rows=100)
        if df is None:
            sys.exit(1)
        
        # Get text column
        text_col = input("\nEnter the name of the text column: ").strip()
        
        # Select texts to compare
        n_samples = input("Number of samples to compare (default: 5): ").strip()
        n_samples = int(n_samples) if n_samples else 5
        
        # Display comparison
        print(f"\n{'='*60}")
        print("METHOD COMPARISON")
        print(f"{'='*60}")
        
        samples = df[text_col].dropna().head(n_samples)
        
        for idx, text in enumerate(samples, 1):
            print(f"\n{'='*60}")
            print(f"SAMPLE {idx}")
            print(f"{'='*60}")
            print(f"\nOriginal text:\n{text}\n")
            
            print(f"Summaries ({10} items each):\n")
            
            # First sentence
            first_sent = extract_first_sentence(text, sentence_limit=1)
            print(f"  • First sentence:\n    {first_sent}\n")
            
            # Leading phrase
            leading = extract_leading_phrase(text, word_limit=10)
            print(f"  • Leading phrase:\n    {leading}\n")
            
            # Key phrases
            phrases = extract_key_phrases_frequency(text, n_phrases=5)
            if phrases:
                phrase_str = ', '.join([p for p, _ in phrases])
                print(f"  • Key phrases:\n    {phrase_str}\n")
            
            # Important tokens
            tokens = extract_important_tokens(text, n_tokens=5)
            if tokens:
                token_str = ', '.join([t for t, _ in tokens])
                print(f"  • Important tokens:\n    {token_str}\n")
    
    elif action == "4":
        # Extract dataset-level key phrases
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
        
        # Get parameters
        n_phrases = input("Number of key phrases to extract (default: 20): ").strip()
        n_phrases = int(n_phrases) if n_phrases else 20
        
        print(f"\nExtracting {n_phrases} key phrases from {len(df):,} documents...")
        
        # Preprocess texts
        texts = df[text_col].dropna().astype(str).apply(preprocess_text).tolist()
        
        if not texts:
            print("No valid texts to process.")
            sys.exit(1)
        
        # Extract key phrases using TF-IDF
        key_phrases = extract_key_phrases_tfidf(texts, n_phrases=n_phrases)
        
        print(f"\n{'='*60}")
        print("TOP KEY PHRASES (TF-IDF)")
        print(f"{'='*60}")
        
        for idx, phrase in enumerate(key_phrases, 1):
            print(f"  {idx:2d}. {phrase}")
        
        # Save to file
        save_choice = input("\nSave key phrases to file? (y/n): ").strip().lower()
        if save_choice == 'y':
            output_dir = Path("analysis_findings")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "extracted_key_phrases.txt"
            
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("TOP KEY PHRASES (TF-IDF)\n")
                    f.write("=" * 60 + "\n\n")
                    for idx, phrase in enumerate(key_phrases, 1):
                        f.write(f"{idx:2d}. {phrase}\n")
                
                print(f"Key phrases saved to: {output_path}")
            except Exception as e:
                print(f"Failed to save key phrases: {e}")
    
    elif action == "5":
        print("Exiting.")
        sys.exit(0)
    
    else:
        print(f"Invalid choice: {action}")
        sys.exit(1)
    
    print("\nDone!")
