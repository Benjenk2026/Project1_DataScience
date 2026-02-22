import tkinter as tk
from tkinter import filedialog
import pandas as pd
import sys
import json
from pathlib import Path
import re




#open file dialog to select file
def select_file(filetypes=None, title="Select file"):
    if filetypes is None:
        filetypes = [
            ("Excel, CSV, or JSON", ("*.xlsx", "*.xls", "*.csv", "*.json")),
            ("Excel files", ("*.xlsx", "*.xls")),
            ("CSV files", "*.csv"),
            ("JSON files", "*.json"),
            ("All files", "*.*"),
        ]
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return file_path


def openfile(file_path):
    # Ensure we have a Path object so we can safely access `.suffix`
    p = Path(file_path) if not isinstance(file_path, Path) else file_path
    if isinstance(p, Path) and not p.exists():
        print(f"File not found: {file_path}")
        return None
    extension = p.suffix.lower()
    try:
        if extension == ".json":
            print("Reading JSON file (attempting JSON lines)...")
            try:
                df = pd.read_json(p, lines=True)
            except ValueError:
                try:
                    df = pd.read_json(p)
                except ValueError:
                    rows = []
                    with open(p, "r", encoding="utf-8-sig") as f:
                        for i, line in enumerate(f, 1):
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
            print("Reading Excel file...")
            df = pd.read_excel(p)
        elif extension == ".csv":
            print("Reading CSV file...")
            df = pd.read_csv(p)
        else:
            # Fallback: try common pandas readers
            try:
                df = pd.read_json(p)
            except Exception:
                df = pd.read_csv(p)
    except Exception as e:
        print(f"Failed to read file '{file_path}': {e}")
        return None

    # return the dataframe to the caller
    return df


def standardize_data(file_path_or_df, save=True, processed_dir="data/processed", casefold_values=False, trim_whitespace=True):
    """Read and standardize the file at `file_path` then optionally save to `processed_dir`.

    Returns the standardized DataFrame or None on failure.
    """
    # Allow passing either a path or an already-loaded DataFrame
    file_path = None
    if isinstance(file_path_or_df, (str, Path)):
        file_path = file_path_or_df
        df = openfile(file_path)
        if df is None:
            return None
    else:
        df = file_path_or_df
        if df is None:
            return None

    # Normalize column names to snake_case
    def to_snake_case(name: str) -> str:
        s = str(name)
        s = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', s)
        s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
        s = re.sub(r'[^0-9a-zA-Z]+', '_', s)
        s = re.sub(r'_+', '_', s)
        s = s.strip('_').lower()
        return s


    try:
        orig_cols = df.columns.astype(str).tolist()
        norm_cols = [to_snake_case(c) for c in orig_cols]

        # Optionally trim whitespace from string-like values
        if trim_whitespace:
            try:
                for col in df.columns:
                    if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                        df[col] = df[col].where(df[col].isna(), df[col].astype(str).str.strip())
            except Exception:
                pass

        # Merge columns that normalize to the same snake_case name
        from collections import OrderedDict

        groups = OrderedDict()
        for orig, norm in zip(orig_cols, norm_cols):
            groups.setdefault(norm, []).append(orig)

        merged_df = pd.DataFrame()
        for norm, cols in groups.items():
            if len(cols) == 1:
                merged_df[norm] = df[cols[0]]
            else:
                # take the first non-null value across the duplicate columns
                merged_df[norm] = df[cols].bfill(axis=1).iloc[:, 0]

        df = merged_df

        # Further token-level normalization and consistent prefixes
        replacements = {
            'lat': 'latitude',
            'long': 'longitude',
            'lon': 'longitude',
            'addr': 'address',
            'add': 'address',
            'zip': 'postal_code',
            'postcode': 'postal_code',
            'st': 'state',
            'biz': 'business',
        }

        def apply_token_replacements(name: str) -> str:
            parts = [p for p in name.split('_') if p]
            parts = [replacements.get(p, p) for p in parts]
            return '_'.join(parts)

        new_names = []
        for col in df.columns.astype(str):
            n = apply_token_replacements(col)

            # canonical latitude/longitude names
            if re.search(r'(^|_)lat($|_)', n) or 'latitude' in n:
                n = 'latitude'
            if re.search(r'(^|_)(lon|long)($|_)', n) or 'longitude' in n:
                n = 'longitude'

            # counts -> prefix with num_ and pluralize base when sensible
            if 'count' in n.split('_') or re.search(r'(^|_)count($|_)', n):
                base = re.sub(r'(_?count$)', '', n)
                base = base.strip('_')
                if base.endswith('s'):
                    plural = base
                else:
                    plural = base + 's' if base else 'count'
                n = f'num_{plural}'

            # boolean detection: values only 0/1 or True/False
            try:
                sample = df[col].dropna().head(200)
                if not sample.empty:
                    vals = set(str(x).strip().lower() for x in sample.astype(str).unique())
                    if vals.issubset({'0', '1', 'true', 'false', 't', 'f'}):
                        if not n.startswith('is_'):
                            n = 'is_' + n
            except Exception:
                pass

            # final cleanup: remove any leftover non-alphanumeric/underscore
            n = re.sub(r'[^0-9a-z_]+', '_', n)
            n = re.sub(r'_+', '_', n).strip('_').lower()

            new_names.append(n)

        # Ensure uniqueness
        final_names = []
        seen = {}
        for n in new_names:
            base = n
            i = 1
            while n in seen:
                i += 1
                n = f"{base}_{i}"
            seen[n] = True
            final_names.append(n)

        df.columns = final_names
    except Exception:
        pass
    # Optionally case-fold string-like values for consistent textual normalization
    if casefold_values:
        try:
            for col in df.columns:
                # only attempt for object/string dtypes
                if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                    try:
                        df[col] = df[col].where(df[col].isna(), df[col].astype(str).str.casefold())
                    except Exception:
                        # fallback: convert to str then casefold where possible
                        try:
                            df[col] = df[col].astype(str).where(df[col].isna(), df[col].astype(str).str.casefold())
                        except Exception:
                            pass
        except Exception:
            pass
    file_label = file_path if file_path else "DataFrame"
    print(f"Standardized data for file '{file_label}' with {len(df)} rows and {len(df.columns)} columns.")

    if save:
        out_dir = Path(processed_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(file_path).stem if file_path else "dataframe"
        out_path = out_dir / f"{stem}_standardized.csv"
        try:
            df.to_csv(out_path, index=False, encoding='utf-8')
            print(f"Wrote standardized file to: {out_path}")
        except Exception as e:
            print(f"Failed to write standardized file: {e}")

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
            # Not really useful for 'most_complete' with no subset
            dedup_df = df_work.loc[df_work['_null_count'].idxmin()]
        
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
    if df is None or df.empty:
        return df, {
            'original_rows': 0,
            'final_rows': 0,
            'rows_dropped': 0,
            'cols_dropped': 0,
            'cells_imputed': 0
        }
    
    df_work = df.copy()
    original_rows = len(df_work)
    original_cols = len(df_work.columns)
    cells_imputed = 0
    
    # Step 1: Drop rows with missing values in critical columns
    if drop_rows_subset is not None:
        if isinstance(drop_rows_subset, str):
            drop_rows_subset = [drop_rows_subset]
        
        # Validate columns exist
        valid_cols = [col for col in drop_rows_subset if col in df_work.columns]
        if not valid_cols and drop_rows_subset:
            print(f"Warning: Columns {set(drop_rows_subset) - set(df_work.columns)} not found.")
        
        if valid_cols:
            rows_before = len(df_work)
            # Drop rows where any of the specified columns have NaN
            df_work = df_work.dropna(subset=valid_cols, how='any')
            rows_dropped = rows_before - len(df_work)
            print(f"  Dropped {rows_dropped} rows with missing values in {valid_cols}")
        else:
            rows_dropped = 0
    else:
        rows_dropped = 0
    
    # Step 2: Drop columns with high missing percentage
    if drop_cols_threshold is not None:
        if not (0 <= drop_cols_threshold <= 100):
            print(f"Warning: drop_cols_threshold must be 0-100. Using None.")
            drop_cols_threshold = None
        else:
            missing_pct = (df_work.isna().sum() / len(df_work)) * 100
            cols_to_drop = missing_pct[missing_pct >= drop_cols_threshold].index.tolist()
            if cols_to_drop:
                df_work = df_work.drop(columns=cols_to_drop)
                print(f"  Dropped {len(cols_to_drop)} columns with >={drop_cols_threshold}% missing: {cols_to_drop}")
            cols_dropped = len(cols_to_drop)
        if drop_cols_threshold is None:
            cols_dropped = 0
    else:
        cols_dropped = 0
    
    # Step 3: Impute missing values with specific values
    if impute_values is not None:
        for col, fill_value in impute_values.items():
            if col in df_work.columns:
                missing_count = df_work[col].isna().sum()
                if missing_count > 0:
                    df_work[col].fillna(fill_value, inplace=True)
                    cells_imputed += missing_count
                    print(f"  Imputed {missing_count} missing values in '{col}' with '{fill_value}'")
    
    # Step 4: Impute missing values with statistical measures
    if impute_strategy is not None:
        for col, strategy in impute_strategy.items():
            if col not in df_work.columns:
                print(f"Warning: Column '{col}' not found. Skipping imputation.")
                continue
            
            missing_count = df_work[col].isna().sum()
            if missing_count == 0:
                continue
            
            try:
                if strategy == 'mean':
                    fill_value = df_work[col].mean()
                    df_work[col].fillna(fill_value, inplace=True)
                    print(f"  Imputed {missing_count} missing values in '{col}' with mean ({fill_value:.2f})")
                
                elif strategy == 'median':
                    fill_value = df_work[col].median()
                    df_work[col].fillna(fill_value, inplace=True)
                    print(f"  Imputed {missing_count} missing values in '{col}' with median ({fill_value:.2f})")
                
                elif strategy == 'mode':
                    fill_value = df_work[col].mode()[0] if not df_work[col].mode().empty else None
                    if fill_value is not None:
                        df_work[col].fillna(fill_value, inplace=True)
                        print(f"  Imputed {missing_count} missing values in '{col}' with mode ({fill_value})")
                
                elif strategy == 'forward_fill':
                    df_work[col].fillna(method='ffill', inplace=True)
                    still_missing = df_work[col].isna().sum()
                    imputed_count = missing_count - still_missing
                    df_work[col].fillna(method='bfill', inplace=True)
                    still_missing = df_work[col].isna().sum()
                    imputed_count = missing_count - still_missing
                    print(f"  Imputed {imputed_count} missing values in '{col}' with forward fill")
                
                elif strategy == 'backward_fill':
                    df_work[col].fillna(method='bfill', inplace=True)
                    still_missing = df_work[col].isna().sum()
                    imputed_count = missing_count - still_missing
                    print(f"  Imputed {imputed_count} missing values in '{col}' with backward fill")
                
                else:
                    print(f"Warning: Unknown imputation strategy '{strategy}' for column '{col}'")
                    continue
                
                cells_imputed += missing_count - df_work[col].isna().sum()
            
            except Exception as e:
                print(f"Warning: Could not impute '{col}' with strategy '{strategy}': {e}")
    
    final_rows = len(df_work)
    final_cols = len(df_work.columns)
    
    stats = {
        'original_rows': original_rows,
        'final_rows': final_rows,
        'rows_dropped': rows_dropped,
        'cols_dropped': cols_dropped,
        'cells_imputed': cells_imputed
    }
    
    print(f"\nMissing Values Handling Summary:")
    print(f"  Original rows: {original_rows}, Final rows: {final_rows} (dropped {rows_dropped})")
    print(f"  Original columns: {original_cols}, Final columns: {final_cols} (dropped {cols_dropped})")
    print(f"  Cells imputed: {cells_imputed}")
    
    return df_work, stats


if __name__ == "__main__":
    print("==========================")
    print("Data Cleaning Utility")
    print("==========================") 
    action = input("Select an action: 1(standardize), 2(deduplicate), 3(handle missing values), 4(exit) ").strip()
    
    #call standardization function
    if action == "1":
        file_path = select_file()
        if not file_path:
            print("No file selected. Exiting.")
            sys.exit(0)
        else:
            print(f"Selected file: {file_path}")
            standardize_data(file_path)
    
    elif action == "2":
        file_path = select_file()
        if not file_path:
            print("No file selected. Exiting.")
            sys.exit(0)
        else:
            print(f"Selected file: {file_path}")
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
 