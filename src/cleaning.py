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


# CONFIGURATION
# Default file paths — edit these if your directory structure differs

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CHUNK_SIZE = 100_000  
FILE_CONFIG = {
    "311": {
        "input":     RAW_DIR / "public_cases_fc.csv",
        "output":    PROCESSED_DIR / "311_cleaned.csv",
        "id_col":    "service_request_id",
        "drop_rows": ["service_request_id", "requested_datetime"],
        "text_fill": ["subject", "status", "status_notes", "service_name",
                      "service_notice", "address", "media_url"],
        "impute":    {},
    },
    "yelp_business": {
        "input":     RAW_DIR / "yelp_academic_dataset_business.json",
        "output":    PROCESSED_DIR / "yelp_business_cleaned.csv",
        "id_col":    "business_id",
        "drop_rows": ["business_id"],
        "text_fill": [],
        "impute":    {"stars": "median", "review_count": "median"},
    },
    "yelp_review": {
        "input":     RAW_DIR / "yelp_academic_dataset_review.json",
        "output":    PROCESSED_DIR / "yelp_review_cleaned.csv",
        "id_col":    "review_id",
        "drop_rows": ["review_id", "business_id"],
        "text_fill": ["text"],
        "impute":    {},
    },
    "yelp_checkin": {
        "input":     RAW_DIR / "yelp_academic_dataset_checkin.json",
        "output":    PROCESSED_DIR / "yelp_checkin_cleaned.csv",
        "id_col":    "business_id",
        "drop_rows": ["business_id"],
        "text_fill": [],
        "impute":    {},
    },
    "yelp_tip": {
        "input":     RAW_DIR / "yelp_academic_dataset_tip.json",
        "output":    PROCESSED_DIR / "yelp_tip_cleaned.csv",
        "id_col":    "business_id",
        "drop_rows": ["business_id", "user_id"],
        "text_fill": ["text"],
        "impute":    {},
    },
    "yelp_user": {
        "input":     RAW_DIR / "yelp_academic_dataset_user.json",
        "output":    PROCESSED_DIR / "yelp_user_cleaned.csv",
        "id_col":    "user_id",
        "drop_rows": ["user_id"],
        "text_fill": [],
        "impute":    {"review_count": "median", "average_stars": "median"},
    },
}


# FILE LOADING

def openfile(file_path: Path) -> pd.DataFrame:
    """Load a CSV or JSON file into a DataFrame."""
    p = Path(file_path)
    if not p.exists():
        print(f"ERROR: File not found: {p}")
        return None

    print(f"Loading {p.name}...")
    try:
        if p.suffix == ".csv":
            return pd.read_csv(p, low_memory=False)
        elif p.suffix == ".json":
            try:
                return pd.read_json(p, lines=True)
            except ValueError:
                rows = []
                with open(p, "r", encoding="utf-8-sig") as f:
                    for i, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            print(f"  Skipping bad JSON on line {i}")
                return pd.DataFrame(rows)
    except Exception as e:
        print(f"ERROR loading {p}: {e}")
        return None


def iter_json_chunks(file_path: Path, chunk_size: int):
    """Read a JSON lines file in chunks without loading it all into memory.

    Yields DataFrames of up to chunk_size rows at a time.
    This allows processing of files larger than available RAM.
    """
    p = Path(file_path)
    rows = []
    chunk_num = 0
    with open(p, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  Skipping bad JSON on line {i}")
                continue
            if len(rows) >= chunk_size:
                chunk_num += 1
                print(f"  Processing chunk {chunk_num} "
                      f"(rows {(chunk_num-1)*chunk_size+1:,} - {chunk_num*chunk_size:,})...")
                yield pd.DataFrame(rows)
                rows = []
    if rows:
        chunk_num += 1
        print(f"  Processing final chunk {chunk_num} ({len(rows):,} rows)...")
        yield pd.DataFrame(rows)


# STANDARDIZE COLUMN NAMES
# Required: consistent snake_case names

_ALIASES = {
    "lat": "latitude", "long": "longitude", "lon": "longitude",
    "zip": "zipcode",  "postal_code": "zipcode",
}

def to_snake_case(name: str) -> str:
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", str(name))
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_").lower()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to snake_case and resolve common aliases."""
    df = df.copy()
    df.columns = [to_snake_case(c) for c in df.columns]
    df = df.rename(columns={c: _ALIASES.get(c, c) for c in df.columns})
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].astype(str).str.strip().replace("nan", pd.NA)
    return df


# COMPLAINT TYPE NORMALIZATION  (311 only)
# Required: case folding, trimming, consistent labels

_COMPLAINT_RULES = [
    (r"\bpotholes?\b",                         "potholes"),
    (r"\btrash\b|\brubbish\b|\bgarbage\b",     "trash"),
    (r"\bnoise\b",                             "noise"),
    (r"\bstreet\s*lights?\b|\blight\s*out\b", "street lights"),
    (r"\bparking\b",                           "parking"),
    (r"\bgraffiti\b",                          "graffiti"),
    (r"\bwater\b",                             "water"),
    (r"\bsidewalk\b",                          "sidewalk"),
    (r"\brat\b|\brodent\b",                    "rodents"),
    (r"\btree\b",                              "trees"),
]

def normalize_complaint_type(value) -> str:
    """Lowercase + trim + apply rule-based label mapping."""
    if pd.isna(value):
        return ""
    s = re.sub(r"\s+", " ", str(value).strip().casefold())
    for pattern, label in _COMPLAINT_RULES:
        if re.search(pattern, s):
            return label
    return s


# YELP CATEGORY NORMALIZATION  (yelp_business only)
# Required: string matching and grouping rules

_CATEGORY_RULES = [
    ("public",      ["public services & government", "government", "post office",
                     "police", "fire station", "city hall", "public library"]),
    ("restaurants", ["restaurant", "food", "pizza", "cafe", "coffee",
                     "burger", "sushi", "diner", "bakery", "seafood"]),
    ("nightlife",   ["nightlife", "pub", "cocktail", "club", "brewery",
                     "wine bar", "bars"]),
    ("retail",      ["shopping", "retail", "grocery", "market", "store", "clothing"]),
    ("services",    ["service", "repair", "salon", "beauty", "spa",
                     "cleaning", "plumbing", "contractor"]),
    ("health",      ["doctor", "dentist", "medical", "health", "pharmacy",
                     "hospital", "fitness", "gym"]),
    ("auto",        ["auto", "car", "tire", "mechanic"]),
]

def normalize_yelp_categories(value) -> tuple:
    """Return (categories_clean, broad_category) for a Yelp category string."""
    if pd.isna(value):
        return "", "other"
    cats = [c.strip().lower() for c in str(value).split(",") if c.strip()]
    broad = "other"
    for broad_name, keywords in _CATEGORY_RULES:
        if any(any(k in c for k in keywords) for c in cats):
            broad = broad_name
            break
    return ", ".join(cats), broad


# LOCATION CLEANING
# Required: clean lat/lon and ZIP codes

def clean_location(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce lat/lon to numeric, null impossible values, extract 5-digit ZIPs."""
    df = df.copy()
    if "latitude" in df.columns:
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df.loc[df["latitude"].abs() > 90, "latitude"] = pd.NA
    if "longitude" in df.columns:
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df.loc[df["longitude"].abs() > 180, "longitude"] = pd.NA
    if "zipcode" in df.columns:
        df["zipcode"] = (df["zipcode"].astype("string")
                         .str.replace(r"\.0$", "", regex=True)
                         .str.extract(r"(\d{5})", expand=False)
                         .astype("string"))
    return df


# MISSING VALUES
# Required: documented drop and impute strategies

def handle_missing_values(df: pd.DataFrame, drop_rows: list = None,
                          impute: dict = None) -> pd.DataFrame:
    """Handle missing values with documented strategies.

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


# DEDUPLICATION
# Required: identify and remove rows representing the same real-world event

def deduplicate_records(df: pd.DataFrame, subset: list) -> pd.DataFrame:
   
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


# CHUNK CLEANING  (for large files)

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


def clean_file_chunked(name: str) -> None:

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


# MAIN CLEANING PIPELINE  (standard, full-file)

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


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleaning Pipeline - CS 4/5630 Project 1")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all",  action="store_true", help="Clean all raw files")
    group.add_argument("--file", choices=list(FILE_CONFIG.keys()),
                       help="Clean a single file by name")
    parser.add_argument("--chunked", action="store_true",
                        help="Process in chunks (recommended for yelp_review and yelp_user)")
    args = parser.parse_args()

    if args.all:
        # Auto-use chunked mode for the two large files when running --all
        run_all(chunked_names=["yelp_review", "yelp_user"])
    elif args.chunked:
        clean_file_chunked(args.file)
    else:
        clean_file(args.file)
