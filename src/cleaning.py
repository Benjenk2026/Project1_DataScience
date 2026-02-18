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
            ("Excel or JSON", ("*.xlsx", "*.xls", "*.json")),
            ("Excel files", ("*.xlsx", "*.xls")),
            ("JSON files", "*.json"),
            ("All files", "*.*"),
        ]
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return file_path


def openfile(file_path):
    print("Standardizing data...")
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
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            rows.append(json.loads(line))
                    df = pd.DataFrame(rows)
        elif extension in [".xlsx", ".xls"]:
            print("Reading Excel file...")
            df = pd.read_excel(p)
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


def standardize_data(file_path, save=True, processed_dir="data/processed"):
    """Read and standardize the file at `file_path` then optionally save to `processed_dir`.

    Returns the standardized DataFrame or None on failure.
    """
    df = openfile(file_path)
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

    print(f"Standardized data for file '{file_path}' with {len(df)} rows and {len(df.columns)} columns.")

    if save:
        out_dir = Path(processed_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(file_path).stem
        out_path = out_dir / f"{stem}_standardized.csv"
        try:
            df.to_csv(out_path, index=False, encoding='utf-8')
            print(f"Wrote standardized file to: {out_path}")
        except Exception as e:
            print(f"Failed to write standardized file: {e}")

    return df



if __name__ == "__main__":
    print("==========================")
    print("Data Cleaning Utility")
    print("==========================") 
    action = input("Select an action 1(standarize), 2(exit) ").strip()
    
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
        print("Exiting.")
        sys.exit(0)
 