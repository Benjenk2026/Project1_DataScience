"""
cleaning.py
Phase 2 - Data Cleaning Pipeline
CS 4/5630 Project 1

Cleans the Philadelphia 311 CSV and all Yelp JSON files.
Tools: pandas (required), re, pathlib, argparse

Usage:
    python src/cleaning.py --all
    python src/cleaning.py --file 311
    python src/cleaning.py --file yelp_review --chunked
    python src/cleaning.py --file yelp_user --chunked

Use --chunked for large files (yelp_review, yelp_user) to avoid
loading the entire file into memory at once. Processes in batches
of 100,000 rows, writing incrementally to the output CSV.
"""

import argparse
import re
import json
from pathlib import Path
import pandas as pd


def openfile(file_path, max_rows=None):
    """
    Load a data file with optional row limit.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the file to open
    max_rows : int, optional
        Maximum number of rows to load (default: None = load all)
        Set to a number to limit rows
    
    Returns:
    --------
    DataFrame or None
    """
    # Ensure we have a Path object so we can safely access `.suffix`
    p = Path(file_path) if not isinstance(file_path, Path) else file_path
    if isinstance(p, Path) and not p.exists():
        print(f"File not found: {file_path}")
        return None

    print(f"Loading {p.name}...")
    try:
        if extension == ".json":
            print(f"Reading JSON file (attempting JSON lines, max {max_rows:,} rows)..." if max_rows else "Reading JSON file (attempting JSON lines)...")
            try:
                df = pd.read_json(p, lines=True, nrows=max_rows)
            except ValueError:
                try:
                    # Standard JSON doesn't support nrows, load all and truncate
                    df = pd.read_json(p)
                    if max_rows and len(df) > max_rows:
                        print(f"Warning: Loaded {len(df):,} rows, limiting to {max_rows:,} rows")
                        df = df.head(max_rows)
                except ValueError:
                    rows = []
                    with open(p, "r", encoding="utf-8-sig") as f:
                        for i, line in enumerate(f, 1):
                            if max_rows and len(rows) >= max_rows:
                                print(f"Reached maximum row limit of {max_rows:,} rows")
                                break
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rows.append(json.loads(line))
                            except json.JSONDecodeError as je:
                                print(f"Warning: JSON decode error on line {i}: {je}. Skipping line.")
                                continue
                    df = pd.DataFrame(rows)
        elif extension in [".xlsx", ".xls"]:
            print(f"Reading Excel file (max {max_rows:,} rows)..." if max_rows else "Reading Excel file...")
            df = pd.read_excel(p, nrows=max_rows)
        elif extension == ".csv":
            print(f"Reading CSV file (max {max_rows:,} rows)..." if max_rows else "Reading CSV file...")
            df = pd.read_csv(p, nrows=max_rows)
        else:
            # Fallback: try common pandas readers
            try:
                df = pd.read_json(p)
                if max_rows and len(df) > max_rows:
                    print(f"Warning: Loaded {len(df):,} rows, limiting to {max_rows:,} rows")
                    df = df.head(max_rows)
            except Exception:
                df = pd.read_csv(p, nrows=max_rows)
    except Exception as e:
        print(f"ERROR loading {p}: {e}")
        return None

    # return the dataframe to the caller
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    return df


def process_file_in_chunks(file_path, chunk_size=100000, process_func=None, **kwargs):
    """
    Process a large file in chunks to avoid memory issues.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the file to process
    chunk_size : int
        Number of rows per chunk (default: 100,000)
    process_func : callable, optional
        Function to apply to each chunk. Should accept (chunk_df, chunk_num, **kwargs)
        and return a processed DataFrame. If None, just yields chunks.
    **kwargs : dict
        Additional arguments to pass to process_func
        
    Yields:
    -------
    DataFrame
        Processed chunks if process_func is provided, otherwise raw chunks
    """
    p = Path(file_path) if not isinstance(file_path, Path) else file_path
    if not p.exists():
        print(f"File not found: {file_path}")
        return
    
    extension = p.suffix.lower()
    chunk_num = 0
    total_rows = 0
    
    print(f"\nProcessing file in chunks of {chunk_size:,} rows...")
    
    try:
        if extension == ".csv":
            # CSV supports efficient chunking
            for chunk in pd.read_csv(p, chunksize=chunk_size):
                chunk_num += 1
                total_rows += len(chunk)
                print(f"\nChunk {chunk_num}: {len(chunk):,} rows (total so far: {total_rows:,})")
                
                if process_func:
                    processed = process_func(chunk, chunk_num, **kwargs)
                    yield processed
                else:
                    yield chunk
                    
        elif extension in [".xlsx", ".xls"]:
            # Excel doesn't support native chunking, so we read in chunks
            skip_rows = 0
            while True:
                chunk = pd.read_excel(p, skiprows=range(1, skip_rows+1) if skip_rows > 0 else None, 
                                     nrows=chunk_size)
                if chunk.empty:
                    break
                    
                chunk_num += 1
                total_rows += len(chunk)
                print(f"\nChunk {chunk_num}: {len(chunk):,} rows (total so far: {total_rows:,})")
                
                if process_func:
                    processed = process_func(chunk, chunk_num, **kwargs)
                    yield processed
                else:
                    yield chunk
                
                if len(chunk) < chunk_size:
                    break
                skip_rows += chunk_size
                
        elif extension == ".json":
            # JSON lines supports chunking
            try:
                for chunk in pd.read_json(p, lines=True, chunksize=chunk_size):
                    chunk_num += 1
                    total_rows += len(chunk)
                    print(f"\nChunk {chunk_num}: {len(chunk):,} rows (total so far: {total_rows:,})")
                    
                    if process_func:
                        processed = process_func(chunk, chunk_num, **kwargs)
                        yield processed
                    else:
                        yield chunk
            except ValueError:
                # Standard JSON - load all and yield chunks
                df = pd.read_json(p)
                for i in range(0, len(df), chunk_size):
                    chunk = df.iloc[i:i+chunk_size]
                    chunk_num += 1
                    total_rows += len(chunk)
                    print(f"\nChunk {chunk_num}: {len(chunk):,} rows (total so far: {total_rows:,})")
                    
                    if process_func:
                        processed = process_func(chunk, chunk_num, **kwargs)
                        yield processed
                    else:
                        yield chunk
        else:
            print(f"Unsupported file type for chunking: {extension}")
            return
            
    except Exception as e:
        print(f"Error processing chunks: {e}")
        return
    
    print(f"\nFinished processing {chunk_num} chunks ({total_rows:,} total rows)")


def standardize_data(file_path_or_df, save=True, processed_dir="data/processed", casefold_values=False, trim_whitespace=True):
    """Read and standardize the file at `file_path` then optionally save to `processed_dir`.

    Strategy:
    - Drop rows missing critical ID or timestamp columns (unusable without them)
    - Drop columns with >= 80% missing values (too sparse to be useful)
    - Impute numeric fields with median where specified
    """
    df = df.copy()
    original_rows = len(df)

    if drop_rows:
        valid = [c for c in drop_rows if c in df.columns]
        if valid:
            df = df.dropna(subset=valid)
            print(f"  Dropped {original_rows - len(df):,} rows missing {valid}")

    sparse_cols = df.columns[df.isna().mean() >= 0.8].tolist()
    if sparse_cols:
        df = df.drop(columns=sparse_cols)
        print(f"  Dropped {len(sparse_cols)} sparse columns (>=80% missing): {sparse_cols}")

    if impute:
        for col, strategy in impute.items():
            if col in df.columns and df[col].isna().any():
                fill = df[col].median() if strategy == "median" else df[col].mean()
                df[col] = df[col].fillna(fill)
                print(f"  Imputed '{col}' with {strategy} ({fill:.2f})")

    print(f"  Rows: {original_rows:,} -> {len(df):,}")
    return df


# ==============================================================================
# DEDUPLICATION
# Required: identify and remove rows representing the same real-world event
# ==============================================================================

def deduplicate_records(df: pd.DataFrame, subset: list) -> pd.DataFrame:
    """Remove duplicates keeping the most complete record per group.

    Strategy: within each group sharing the same key, keep the row with
    the fewest missing values.
    Justification: preserves the richest version of each real-world event.
    """
    original = len(df)
    if not subset or not all(c in df.columns for c in subset):
        print(f"  Skipping dedup — key columns not found: {subset}")
        return df

    df = df.copy()
    df["_missing"] = df.isna().sum(axis=1)
    df = (df.sort_values(subset + ["_missing"])
            .drop_duplicates(subset=subset, keep="first")
            .drop(columns=["_missing"]))

    print(f"  Dedup on {subset}: {original:,} -> {len(df):,} "
          f"({original - len(df):,} removed)")
    return df


# ==============================================================================
# CHUNK CLEANING  (for large files)
# ==============================================================================

def _clean_chunk(df: pd.DataFrame, name: str, cfg: dict) -> pd.DataFrame:
    """Apply all cleaning steps to a single chunk. Used by clean_file_chunked."""
    df = standardize_columns(df)

    if name == "311":
        source = next((c for c in ["subject", "service_name", "service_notice"]
                       if c in df.columns), None)
        if source:
            df["complaint_type"] = df[source].map(normalize_complaint_type)
        df = clean_location(df)
    elif name == "yelp_business":
        if "categories" in df.columns:
            results = df["categories"].map(normalize_yelp_categories)
            df["categories_clean"] = results.map(lambda x: x[0])
            df["broad_category"]   = results.map(lambda x: x[1])
        df = clean_location(df)

    # Drop rows missing critical columns
    if cfg["drop_rows"]:
        valid = [c for c in cfg["drop_rows"] if c in df.columns]
        if valid:
            df = df.dropna(subset=valid)

    # Fill text columns
    for col in cfg["text_fill"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype("string")

    # Within-chunk dedup only (cross-chunk dedup not feasible without full load)
    id_col = cfg["id_col"]
    if id_col in df.columns:
        df["_missing"] = df.isna().sum(axis=1)
        df = (df.sort_values([id_col, "_missing"])
                .drop_duplicates(subset=[id_col], keep="first")
                .drop(columns=["_missing"]))

    return df


def deduplicate_records(df_or_path, subset=None, keep='first', strategy='subset', verbose=True):
    """Deduplicate records from a DataFrame based on specified columns.
    
    Identifies and removes redundant rows that represent the same real-world entity
    or event.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to deduplicate.
    subset : str, list, or None, default None
        Column(s) to consider for identifying duplicate rows.
        - If None: uses all columns
        - If str: a single column name (e.g., 'id')
        - If list: multiple column names (e.g., ['id', 'name'])
    keep : {'first', 'last', 'most_complete'}, default 'first'
        Which duplicate row to keep:
        - 'first': keeps the first occurrence
        - 'last': keeps the last occurrence
        - 'most_complete': keeps the row with the fewest NaN/null values
    strategy : {'subset', 'approximate'}, default 'subset'
        Deduplication strategy:
        - 'subset': exact duplicate matching on specified columns
        - 'approximate': can be extended for fuzzy matching (placeholder)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with duplicate rows removed.
    dict
        Summary statistics with keys:
        - 'original_rows': number of rows before deduplication
        - 'final_rows': number of rows after deduplication
        - 'duplicates_removed': number of duplicate rows removed
        - 'duplicate_groups': number of groups with duplicates
    
    Examples
    --------
    # Remove rows with duplicate 'id'
    >>> dedup_df, stats = deduplicate_records(df, subset='id')
    
    # Remove rows with duplicate 'id' and 'name' combinations
    >>> dedup_df, stats = deduplicate_records(df, subset=['id', 'name'])
    
    # Keep the row with the most non-null values
    >>> dedup_df, stats = deduplicate_records(df, subset='id', keep='most_complete')
    """
    # Accept either a DataFrame or a file path
    if isinstance(df_or_path, (str, Path)):
        df = openfile(df_or_path)
        if df is None:
            return None, {
                'original_rows': 0,
                'final_rows': 0,
                'duplicates_removed': 0,
                'duplicate_groups': 0
            }
    else:
        df = df_or_path

    if df is None or df.empty:
        return df, {
            'original_rows': 0,
            'final_rows': 0,
            'duplicates_removed': 0,
            'duplicate_groups': 0
        }
    
    original_count = len(df)

    # Create a working copy to avoid modifying original
    df_work = df.copy()
    
    # Convert unhashable types (dicts, lists) to strings for deduplication
    # This is necessary for JSON data with nested structures
    for col in df_work.columns:
        try:
            # Check if column contains unhashable types (dicts or lists)
            sample = df_work[col].dropna().head(1)
            if not sample.empty:
                first_val = sample.iloc[0]
                if isinstance(first_val, (dict, list)):
                    # Convert to string representation
                    df_work[col] = df_work[col].astype(str)
        except Exception:
            pass
    
    # Validate inputs
    if subset is not None:
        if isinstance(subset, str):
            subset = [subset]
        # Check if all columns exist
        missing_cols = set(subset) - set(df_work.columns)
        if missing_cols:
            print(f"Warning: Columns {missing_cols} not found in DataFrame. Using all columns.")
            subset = None
    
    # Validate keep parameter
    if keep not in ['first', 'last', 'most_complete']:
        print(f"Warning: keep='{keep}' not recognized. Using 'first'.")
        keep = 'first'
    
    # Handle 'most_complete' strategy
    if keep == 'most_complete':
        # Reset index to track original order
        df_work = df_work.reset_index(drop=True).copy()
        
        # Count non-null values per row
        df_work['_null_count'] = df_work.isna().sum(axis=1)
        
        if subset:
            # For each duplicate group, keep the row with fewest nulls
            dedup_df = df_work.loc[
                df_work.groupby(subset, dropna=False)['_null_count'].idxmin()
            ]
        else:
            # When no subset is provided, keep the single row with fewest nulls.
            # Use double-brackets to ensure a DataFrame is returned (not a Series).
            dedup_df = df_work.loc[[df_work['_null_count'].idxmin()]]
        
        dedup_df = dedup_df.drop(columns=['_null_count'])
    else:
        # Use standard pandas drop_duplicates
        dedup_df = df_work.drop_duplicates(subset=subset, keep=keep)
    
    final_count = len(dedup_df)
    duplicates_removed = original_count - final_count
    
    # Calculate number of duplicate groups
    if duplicates_removed > 0 and subset:
        duplicate_groups = df_work.groupby(subset, dropna=False).size()
        duplicate_groups = (duplicate_groups > 1).sum()
    else:
        duplicate_groups = 0
    
    stats = {
        'original_rows': original_count,
        'final_rows': final_count,
        'duplicates_removed': duplicates_removed,
        'duplicate_groups': duplicate_groups
    }
    
    if verbose:
        print(f"Deduplication Summary:")
        print(f"  Original rows: {original_count}")
        print(f"  Final rows: {final_count}")
        print(f"  Duplicates removed: {duplicates_removed}")
        if duplicate_groups > 0:
            print(f"  Duplicate groups found: {duplicate_groups}")
    
    return dedup_df, stats


def handle_missing_values(df, drop_rows_subset=None, drop_cols_threshold=None, 
                          impute_strategy=None, impute_values=None):
    """Handle missing values in a DataFrame through dropping or imputation.
    
    Provides flexible strategies for handling missing data: removing rows/columns
    or imputing missing values with defaults.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with potential missing values.
    drop_rows_subset : str, list, or None, default None
        Column(s) where missing values trigger row deletion.
        - If str: single column (e.g., 'id')
        - If list: multiple columns (e.g., ['id', 'required_field'])
        - If None: no rows dropped based on missing values
        Example: drop_rows_subset='id' removes all rows with missing IDs
    drop_cols_threshold : float, default None
        Drop columns with missing value percentage >= threshold (0-100).
        - If 50: drops columns with >=50% missing values
        - If None: no columns dropped
    impute_strategy : dict or None, default None
        Dictionary mapping column names to imputation methods.
        Methods: 'mean', 'median', 'mode', 'forward_fill', 'backward_fill'
        Example: {'age': 'mean', 'category': 'mode', 'value': 'forward_fill'}
    impute_values : dict or None, default None
        Dictionary mapping column names to specific default values.
        Takes precedence over impute_strategy.
        Example: {'department': 'Unknown', 'status': 'Inactive'}
    
    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled.
    dict
        Summary statistics with keys:
        - 'original_rows': rows before processing
        - 'final_rows': rows after processing
        - 'rows_dropped': number of rows removed
        - 'cols_dropped': number of columns removed
        - 'cells_imputed': number of values imputed
    
    Examples
    --------
    # Drop rows with missing ID
    >>> df_clean, stats = handle_missing_values(df, drop_rows_subset='id')
    
    # Drop columns with >80% missing values
    >>> df_clean, stats = handle_missing_values(df, drop_cols_threshold=80)
    
    # Impute age with mean, category with mode
    >>> df_clean, stats = handle_missing_values(
    ...     df, 
    ...     impute_strategy={'age': 'mean', 'category': 'mode'}
    ... )
    
    # Impute with specific values
    >>> df_clean, stats = handle_missing_values(
    ...     df, 
    ...     drop_rows_subset='id',
    ...     impute_values={'department': 'Unknown', 'salary': 0}
    ... )
    """
    if name not in FILE_CONFIG:
        print(f"ERROR: Unknown file '{name}'.")
        return

    cfg = FILE_CONFIG[name]
    p = cfg["input"]

    if not p.exists():
        print(f"ERROR: File not found: {p}")
        return

    print(f"\n{'='*50}\nCleaning (chunked): {name}\n{'='*50}")
    print(f"  Source: {p}  ({p.stat().st_size / 1e9:.2f} GB)")
    print(f"  Chunk size: {CHUNK_SIZE:,} rows")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg["output"]

    total_in = 0
    total_out = 0
    first_chunk = True

    for chunk in iter_json_chunks(p, CHUNK_SIZE):
        total_in += len(chunk)
        cleaned = _clean_chunk(chunk, name, cfg)
        total_out += len(cleaned)

        # Write header only on first chunk
        cleaned.to_csv(out, mode="w" if first_chunk else "a",
                       header=first_chunk, index=False, encoding="utf-8")
        first_chunk = False

    print(f"\n  Done. {total_in:,} rows read -> {total_out:,} rows saved.")
    print(f"  Saved -> {out}")


# ==============================================================================
# MAIN CLEANING PIPELINE  (standard, full-file)
# ==============================================================================

def clean_file(name: str) -> pd.DataFrame:
    """Run the full cleaning pipeline for a named dataset."""
    if name not in FILE_CONFIG:
        print(f"ERROR: Unknown file '{name}'. Options: {list(FILE_CONFIG.keys())}")
        return None

    cfg = FILE_CONFIG[name]
    print(f"\n{'='*50}\nCleaning: {name}\n{'='*50}")

    df = openfile(cfg["input"])
    if df is None:
        return None

    # 1. Standardize column names
    df = standardize_columns(df)

    # 2. Dataset-specific enrichment
    if name == "311":
        source = next((c for c in ["subject", "service_name", "service_notice"]
                       if c in df.columns), None)
        if source:
            df["complaint_type"] = df[source].map(normalize_complaint_type)
        df = clean_location(df)

    elif name == "yelp_business":
        if "categories" in df.columns:
            results = df["categories"].map(normalize_yelp_categories)
            df["categories_clean"] = results.map(lambda x: x[0])
            df["broad_category"]   = results.map(lambda x: x[1])
        df = clean_location(df)

    # 3. Handle missing values
    df = handle_missing_values(df, drop_rows=cfg["drop_rows"], impute=cfg["impute"])

    # 4. Fill text columns so downstream NLP steps don't break
    for col in cfg["text_fill"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype("string")

    # 5. Deduplicate
    df = deduplicate_records(df, subset=[cfg["id_col"]])

    # 6. Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg["output"], index=False, encoding="utf-8")
    print(f"  Saved -> {cfg['output']}  ({len(df):,} rows, {df.shape[1]} cols)")

    return df


def run_all(chunked_names: list = None) -> dict:
    """Clean every raw file. Optionally specify which files to run chunked."""
    chunked_names = chunked_names or []
    results = {}
    for name in FILE_CONFIG:
        if name in chunked_names:
            clean_file_chunked(name)
            results[name] = None  # chunked mode doesn't return a df
        else:
            results[name] = clean_file(name)
    print("\n========== All Cleaning Complete ==========")
    for name, df in results.items():
        status = f"{len(df):,} rows" if df is not None else "done (chunked)"
        print(f"  {name}: {status}")
    return results


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    print("==========================")
    print("Data Cleaning Utility")
    print("==========================") 
    
    # Ask if user wants to process in chunks for large files
    chunk_mode = input("Process large files in chunks? (y/n, default: n): ").strip().lower() == 'y'
    chunk_size = 100000  # Default chunk size
    
    if chunk_mode:
        chunk_input = input(f"Enter chunk size (rows per batch, default: {chunk_size:,}): ").strip()
        if chunk_input:
            try:
                chunk_size = int(chunk_input)
            except ValueError:
                print(f"Invalid input, using default chunk size of {chunk_size:,}")
    
    action = input("\nSelect an action: 1(standardize), 2(deduplicate), 3(handle missing values), 4(exit) ").strip()
    
    #call standardization function
    if action == "1":
        file_path = select_file()
        if not file_path:
            print("No file selected. Exiting.")
            sys.exit(0)
        else:
            print(f"Selected file: {file_path}")
            
            if chunk_mode:
                # Process in chunks and save each chunk
                output_dir = Path("data/processed")
                output_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(file_path).stem
                
                all_chunks = []
                for chunk_df in process_file_in_chunks(file_path, chunk_size=chunk_size):
                    # Standardize this chunk
                    standardized = standardize_data(chunk_df, save=False)
                    if standardized is not None:
                        all_chunks.append(standardized)
                
                if all_chunks:
                    # Combine all chunks
                    print(f"\nCombining {len(all_chunks)} chunks...")
                    final_df = pd.concat(all_chunks, ignore_index=True)
                    
                    # Save combined result
                    output_path = output_dir / f"{stem}_standardized.csv"
                    try:
                        final_df.to_csv(output_path, index=False, encoding='utf-8')
                        print(f"Saved standardized file ({len(final_df):,} rows) to: {output_path}")
                    except Exception as e:
                        print(f"Failed to save: {e}")
            else:
                standardize_data(file_path)
    
    elif action == "2":
        file_path = select_file()
        if not file_path:
            print("No file selected. Exiting.")
            sys.exit(0)
        else:
            print(f"Selected file: {file_path}")
            
            if chunk_mode:
                # Load first chunk to get column names
                first_chunk = None
                for chunk in process_file_in_chunks(file_path, chunk_size=chunk_size):
                    first_chunk = chunk
                    break
                
                if first_chunk is None:
                    print("Failed to load file.")
                    sys.exit(0)
                
                print(f"\nAvailable columns: {', '.join(first_chunk.columns.tolist())}")
                subset_input = input("Enter column(s) to check for duplicates (comma-separated, or press Enter for all columns): ").strip()
                
                if subset_input:
                    subset = [col.strip() for col in subset_input.split(',')]
                else:
                    subset = None
                
                keep_input = input("Keep strategy - 1(first), 2(last), 3(most_complete): ").strip()
                keep_map = {'1': 'first', '2': 'last', '3': 'most_complete'}
                keep = keep_map.get(keep_input, 'first')
                
                # Process all chunks and combine
                all_chunks = []
                for chunk in process_file_in_chunks(file_path, chunk_size=chunk_size):
                    all_chunks.append(chunk)
                
                print(f"\nCombining {len(all_chunks)} chunks for deduplication...")
                df = pd.concat(all_chunks, ignore_index=True)
                
                dedup_df, stats = deduplicate_records(df, subset=subset, keep=keep)
                
                save_choice = input("\nSave deduplicated file? (y/n): ").strip().lower()
                if save_choice == 'y':
                    out_dir = Path("data/processed")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    stem = Path(file_path).stem
                    out_path = out_dir / f"{stem}_deduplicated.csv"
                    try:
                        dedup_df.to_csv(out_path, index=False, encoding='utf-8')
                        print(f"Wrote deduplicated file to: {out_path}")
                    except Exception as e:
                        print(f"Failed to write deduplicated file: {e}")
            else:
                df = openfile(file_path)
                if df is None:
                    print("Failed to load file.")
                    sys.exit(0)
                
                print(f"\nAvailable columns: {', '.join(df.columns.tolist())}")
                subset_input = input("Enter column(s) to check for duplicates (comma-separated, or press Enter for all columns): ").strip()
                
                if subset_input:
                    subset = [col.strip() for col in subset_input.split(',')]
                else:
                    subset = None
                
                keep_input = input("Keep strategy - 1(first), 2(last), 3(most_complete): ").strip()
                keep_map = {'1': 'first', '2': 'last', '3': 'most_complete'}
                keep = keep_map.get(keep_input, 'first')
                
                dedup_df, stats = deduplicate_records(df, subset=subset, keep=keep)
                
                save_choice = input("\nSave deduplicated file? (y/n): ").strip().lower()
                if save_choice == 'y':
                    out_dir = Path("data/processed")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    stem = Path(file_path).stem
                    out_path = out_dir / f"{stem}_deduplicated.csv"
                    try:
                        dedup_df.to_csv(out_path, index=False, encoding='utf-8')
                        print(f"Wrote deduplicated file to: {out_path}")
                    except Exception as e:
                        print(f"Failed to write deduplicated file: {e}")
    
    elif action == "3":
        file_path = select_file()
        if not file_path:
            print("No file selected. Exiting.")
            sys.exit(0)
        else:
            print(f"Selected file: {file_path}")
            
            if chunk_mode:
                # Load in chunks and combine for analysis
                all_chunks = []
                for chunk in process_file_in_chunks(file_path, chunk_size=chunk_size):
                    all_chunks.append(chunk)
                
                print(f"\nCombining {len(all_chunks)} chunks...")
                df = pd.concat(all_chunks, ignore_index=True)
            else:
                df = openfile(file_path)
            
            if df is None:
                print("Failed to load file.")
                sys.exit(0)
            
            print(f"\nAvailable columns: {', '.join(df.columns.tolist())}")
            print(f"Missing values summary:")
            missing_summary = df.isna().sum()
            for col in missing_summary[missing_summary > 0].index:
                pct = (missing_summary[col] / len(df)) * 100
                print(f"  {col}: {missing_summary[col]} ({pct:.1f}%)")
            
            drop_rows_input = input("\nEnter column(s) where missing values should drop rows (comma-separated, or press Enter to skip): ").strip()
            drop_rows_subset = [col.strip() for col in drop_rows_input.split(',')] if drop_rows_input else None
            
            # Summary after drop_rows selection
            if drop_rows_subset:
                print(f"\n→ Will drop rows with missing values in: {', '.join(drop_rows_subset)}")
                # Actually drop the rows and update the dataframe
                df = df.dropna(subset=drop_rows_subset, how='any')
                print(f"→ Dropped rows. New total: {len(df)} rows")
                
                # Print updated missing values summary after dropping rows
                print(f"\nUpdated missing values summary:")
                missing_summary = df.isna().sum()
                if missing_summary.sum() == 0:
                    print("  (No missing values remaining)")
                else:
                    for col in missing_summary[missing_summary > 0].index:
                        pct = (missing_summary[col] / len(df)) * 100
                        print(f"  {col}: {missing_summary[col]} ({pct:.1f}%)")
            else:
                print("\n→ No columns selected for row dropping (will skip this step)")
            
            drop_cols_input = input("\nEnter threshold % to drop columns with missing values (0-100, or press Enter to skip): ").strip()
            drop_cols_threshold = float(drop_cols_input) if drop_cols_input else None
            
            impute_choice = input("Impute missing values? (y/n): ").strip().lower()
            impute_strategy = None
            impute_values = None
            
            if impute_choice == 'y':
                print("Imputation strategies: mean, median, mode, forward_fill, backward_fill, or specific value")
                impute_input = input("Enter column:strategy pairs (e.g., 'age:mean,dept:Unknown', leave blank to skip): ").strip()
                
                if impute_input:
                    impute_strategy = {}
                    impute_values = {}
                    pairs = impute_input.split(',')
                    for pair in pairs:
                        if ':' in pair:
                            col, strategy = pair.split(':', 1)
                            col = col.strip()
                            strategy = strategy.strip()
                            
                            # Check if it's a numeric value for direct imputation
                            try:
                                # Try to convert to number
                                num_val = float(strategy)
                                impute_values[col] = num_val
                            except ValueError:
                                # It's a strategy or string value
                                if strategy in ['mean', 'median', 'mode', 'forward_fill', 'backward_fill']:
                                    impute_strategy[col] = strategy
                                else:
                                    # Treat as a string value to impute
                                    impute_values[col] = strategy
            
            clean_df, stats = handle_missing_values(
                df, 
                drop_rows_subset=None,
                drop_cols_threshold=drop_cols_threshold,
                impute_strategy=impute_strategy,
                impute_values=impute_values
            )
            
            save_choice = input("\nSave cleaned file? (y/n): ").strip().lower()
            if save_choice == 'y':
                out_dir = Path("data/processed")
                out_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(file_path).stem
                out_path = out_dir / f"{stem}_handled_missing.csv"
                try:
                    clean_df.to_csv(out_path, index=False, encoding='utf-8')
                    print(f"Wrote cleaned file to: {out_path}")
                except Exception as e:
                    print(f"Failed to write cleaned file: {e}")
    
    elif action == "4":
        print("Exiting.")
        sys.exit(0)
 
