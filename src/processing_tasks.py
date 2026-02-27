# src/processing_tasks.py
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import argparse
import joblib
from .cleaning import openfile, standardize_data, handle_missing_values, deduplicate_records

# -----------------------------
# Basic text helpers
# -----------------------------
_WORD_RE = re.compile(r"[a-z0-9]+")

def basic_clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    words = _WORD_RE.findall(s)
    return " ".join(words)

def top_keywords(text: str, k: int = 5) -> str:
    """Heuristic summarization: return top-k frequent tokens in the text."""
    tokens = basic_clean_text(text).split()
    if not tokens:
        return ""
    # tiny stopword list to keep results reasonable
    stop = set(["the","and","a","an","to","of","in","on","for","is","it","this","that","with","as","at","by","be"])
    tokens = [t for t in tokens if t not in stop and len(t) >= 3]
    if not tokens:
        return ""
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    top = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:k]
    return ", ".join([w for w, _ in top])

# -----------------------------
# Yelp category normalization
# -----------------------------
BROAD_CATEGORY_RULES = [
    ("restaurants", ["restaurant", "food", "pizza", "cafe", "coffee", "barbeque", "burger", "sushi", "diner", "bakery"]),
    ("nightlife", ["bar", "pub", "cocktail", "nightlife", "club"]),
    ("retail", ["shopping", "retail", "grocery", "market", "store", "clothing"]),
    ("services", ["service", "repair", "salon", "beauty", "spa", "cleaning", "plumbing", "electric", "contractor"]),
    ("health", ["doctor", "dentist", "medical", "health", "pharmacy", "hospital"]),
    ("auto", ["auto", "car", "tire", "mechanic", "oil change"]),
    ("public", ["government", "public", "library", "school", "police", "fire"]),
]

def normalize_yelp_categories(cat_value) -> tuple[str, str]:
    """
    Returns (categories_clean, broad_category)
    - categories_clean: comma-separated, cleaned lower-case categories
    - broad_category: one of the broad buckets above (or 'other')
    """
    if pd.isna(cat_value):
        return "", "other"

    # Yelp categories are often "A, B, C"
    cats = [c.strip().lower() for c in str(cat_value).split(",") if c.strip()]
    categories_clean = ", ".join(cats)

    # broad bucket
    broad = "other"
    for broad_name, keywords in BROAD_CATEGORY_RULES:
        if any(any(k in c for k in keywords) for c in cats):
            broad = broad_name
            break

    return categories_clean, broad

# -----------------------------
# Main processing
# -----------------------------
def process_311(df311: pd.DataFrame) -> pd.DataFrame:
    df = df311.copy()

    # choose a text field if present
    text_col = None
    for c in ["description", "status_notes", "service_notice", "subject", "service_name"]:
        if c in df.columns:
            text_col = c
            break

    if text_col is None:
        df["complaint_text_clean"] = ""
        df["complaint_keywords"] = ""
        return df

    df["complaint_text_clean"] = df[text_col].astype(str).map(basic_clean_text)
    df["complaint_keywords"] = df[text_col].astype(str).map(lambda x: top_keywords(x, k=6))
    return df

def process_yelp(dfy: pd.DataFrame) -> pd.DataFrame:
    df = dfy.copy()

    if "categories" in df.columns:
        cats_clean = []
        broad = []
        for v in df["categories"]:
            c_clean, b = normalize_yelp_categories(v)
            cats_clean.append(c_clean)
            broad.append(b)
        df["categories_clean"] = cats_clean
        df["broad_category"] = broad
    else:
        df["categories_clean"] = ""
        df["broad_category"] = "other"

    return df

def run_processing(
    path_311: str = "data/raw/test_data.xlsx",
    path_yelp: str = "data/raw/yelp_JSON_test.json",
):
    # Load raw
    raw_311 = openfile(path_311)
    raw_yelp = openfile(path_yelp)

    # Standardize cols using your cleaning.py normalize system
    df311 = standardize_data(raw_311, save=False)
    dfy = standardize_data(raw_yelp, save=False)

    # Minimal missing handling
    if "service_request_id" in df311.columns:
        df311, _ = deduplicate_records(df311, subset=["service_request_id"], keep="most_complete")
    if "business_id" in dfy.columns:
        dfy, _ = deduplicate_records(dfy, subset=["business_id"], keep="most_complete")

    # Process
    df311 = process_311(df311)
    dfy = process_yelp(dfy)

    # Save 
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    df311.to_csv(out_dir / "311_processed.csv", index=False, encoding="utf-8")
    dfy.to_csv(out_dir / "yelp_processed.csv", index=False, encoding="utf-8")

    print("Saved:")
    print(" - data/processed/311_processed.csv")
    print(" - data/processed/yelp_processed.csv")

    return df311, dfy

def classify_complaints(path: str):
    # -----------------------------
    # 1. LOAD MULTIPLE FILES
    # -----------------------------

    file_paths = [
    "complaints1.xlsx",
    "complaints2.json"
]

    dataframes = []

    for path in file_paths:
        df = openfile(path)   # using imported function
        if df is not None:
            dataframes.append(df)

    if not dataframes:
        raise ValueError("No valid files loaded.")

    df = pd.concat(dataframes, ignore_index=True)

    # -----------------------------
    # 2. VALIDATE REQUIRED COLUMNS
    # -----------------------------

    required_cols = ["text", "category"]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols].dropna()

    # -----------------------------
    # 3. SPLIT DATA
    # -----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["category"],
        test_size=0.2,
        random_state=42
    )

    # -----------------------------
    # 4. TEXT VECTORIZATION
    # -----------------------------

    vectorizer = TfidfVectorizer(stop_words="english")

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # -----------------------------
    # 5. TRAIN MODEL
    # -----------------------------

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # -----------------------------
    # 6. EVALUATE
    # -----------------------------

    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)

    print("Model Accuracy:", accuracy)

    # -----------------------------
    # 7. PREDICT NEW TEXT
    # -----------------------------

    new_complaint = ["I was charged twice for my order"]

    new_vec = vectorizer.transform(new_complaint)
    prediction = model.predict(new_vec)

    print("Predicted Category:", prediction[0])
