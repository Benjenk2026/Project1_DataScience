"""
classification.py
Phase 2 — Text & Category Processing
CS 4/5630 Project 1

Produces TWO unified output CSVs, each containing ALL enrichment columns:

  data/processed/311_enriched.csv
      Original 311 columns
      + complaint_category      (rule-based seed label)
      + predicted_category      (best ML model: LR / SVM / RF)
      + top_ngrams              (top bigrams/trigrams for that category)

  data/processed/yelp_reviews_enriched.csv
      Original Yelp review columns
      + review_aspect           (rule-based aspect: Food Quality, Service, ...)
      + vader_compound / vader_pos / vader_neg
      + sentiment_label         (Positive / Neutral / Negative from VADER)
      + textblob_polarity / textblob_subjectivity
      + ml_severity             (LR classifier trained on star labels)
      + severity                (unified: High/Medium/Low — High means bad/complaint)
      + top_ngrams              (top bigrams/trigrams for that review aspect)
      + broad_category_rule     (business group from string matching)
      + cluster_id / cluster_label  (K-Means cluster of business category)

Design: single streaming pass per source file
  Phase A - Fit (in-memory sample):
    - Fit TF-IDF + train/evaluate all three classifiers (311)
    - Fit TF-IDF + train/evaluate severity LR (Yelp reviews)
    - Compute n-gram summaries per category (stored as a lookup dict)
    - Fit TF-IDF + K-Means on Yelp businesses (stored as a lookup dict by business_id)
  Phase B - Stream & enrich (chunked):
    - Read source CSV in PREDICT_CHUNK_SIZE batches
    - Apply all fitted models + lookup dicts to each batch
    - Append enriched batch to the output CSV immediately (low RAM footprint)

Usage:
    python src/classification.py                    # run both pipelines
    python src/classification.py --source 311       # only 311
    python src/classification.py --source yelp      # only Yelp
    python src/classification.py --chunk-size 25000 # smaller batches

Inputs  (data/processed/):
    311_cleaned.csv
    yelp_review_cleaned.csv
    yelp_business_cleaned.csv

Outputs (data/processed/):
    311_enriched.csv
    yelp_reviews_enriched.csv
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PROCESSED_DIR = Path("data/processed")

TRAIN_SAMPLE       = 200_000   # rows used to fit 311 models
YELP_TRAIN_SAMPLE  = 50_000    # smaller sample for Yelp — reviews are much
                               # larger/denser than 311 rows; 50k is enough
                               # to train a solid sentiment/severity classifier
PREDICT_CHUNK_SIZE = 50_000    # rows per streaming prediction batch

COMPLAINT_RULES = [
    (r"\bpotholes?\b|\bpavement\b|\broad\s*damage\b",         "Infrastructure"),
    (r"\btrash\b|\brubbish\b|\bgarbage\b|\blitter\b",          "Sanitation"),
    (r"\bnoise\b|\bloud\b|\bmusic\b",                          "Noise"),
    (r"\bstreet\s*lights?\b|\blight\s*out\b|\billumination\b", "Lighting"),
    (r"\bparking\b|\bvehicle\b|\bcar\b",                       "Parking"),
    (r"\bgraffiti\b|\bvandal\b",                               "Graffiti"),
    (r"\bwater\b|\bflooding?\b|\bleaks?\b",                    "Water"),
    (r"\bsidewalk\b|\bcurb\b|\bpedestrian\b",                  "Sidewalk"),
    (r"\brat\b|\brodent\b|\bpest\b|\bvermin\b",                "Rodents/Pests"),
    (r"\btree\b|\bbranch\b|\bfoliage\b",                       "Trees/Vegetation"),
    (r"\bsewer\b|\bdrain\b|\bmanhole\b",                       "Sewer/Drainage"),
    (r"\bhomeless\b|\bencampment\b|\bshelter\b",               "Homeless Services"),
    (r"\bbuilding\b|\bproperty\b|\bzoning\b|\bpermit\b",       "Building/Property"),
    (r"\bcrime\b|\bsafety\b|\bpolice\b|\btheft\b",             "Public Safety"),
]

REVIEW_RULES = [
    (r"\bburger\b|\bpizza\b|\bsushi\b|\bfood\b|\bmeal\b|\bate\b|\bdelicious\b|\bmenu\b|\bdish\b",
     "Food Quality"),
    (r"\bservice\b|\bstaff\b|\bwaiter\b|\bwaitress\b|\brude\b|\bfriendly\b|\bhelpful\b",
     "Service"),
    (r"\batmosphere\b|\bambien\b|\bdecor\b|\bvibe\b|\binterior\b|\bloud\b|\bcozy\b",
     "Atmosphere"),
    (r"\bprice\b|\bexpensive\b|\bcheap\b|\bvalue\b|\bworth\b|\bcost\b|\baffordable\b",
     "Price/Value"),
    (r"\bwait\b|\bslow\b|\bfast\b|\bquick\b|\bhours\b|\blong\b|\btime\b|\bdelay\b",
     "Wait Time"),
    (r"\bclean\b|\bdirty\b|\bhygiene\b|\bsanit\b|\bmess\b",
     "Cleanliness"),
]

CATEGORY_GROUPS = {
    "Food & Dining":  ["restaurant", "food", "bar", "cafe", "coffee", "pizza",
                       "sushi", "bakery", "diner", "brewery", "winery"],
    "Retail":         ["shop", "store", "boutique", "market", "clothing", "mall",
                       "outlet", "hardware"],
    "Health":         ["medical", "doctor", "dentist", "pharmacy", "hospital",
                       "clinic", "health", "urgent care"],
    "Beauty":         ["salon", "spa", "nail", "hair", "barber", "massage",
                       "skincare", "beauty"],
    "Automotive":     ["auto", "car", "tire", "mechanic", "oil change",
                       "body shop", "dealership"],
    "Services":       ["plumber", "electrician", "contractor", "cleaning",
                       "lawn", "repair", "movers", "locksmith"],
    "Entertainment":  ["gym", "fitness", "yoga", "movie", "theatre", "bowling",
                       "arcade", "escape room", "gaming"],
    "Education":      ["school", "tutor", "university", "college", "training",
                       "daycare", "childcare"],
}


# ==============================================================================
# SHARED HELPERS
# ==============================================================================

def load(path, sample=None, seed=42):
    """Load a CSV fully, optionally down-sampling to `sample` rows."""
    path = Path(path)
    if not path.exists():
        print(f"  [SKIP] Not found: {path}")
        return None
    print(f"  Loading {path.name}...")
    df = pd.read_csv(path, low_memory=False)
    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=seed).reset_index(drop=True)
        print(f"    Sampled {sample:,} rows from {path.name}")
    return df


def iter_csv_chunks(path, chunk_size):
    """Stream a CSV in fixed-size chunks. Yields (chunk_index, DataFrame)."""
    for i, chunk in enumerate(pd.read_csv(path, chunksize=chunk_size, low_memory=False)):
        yield i, chunk


def rule_label(text, rules, default="Other"):
    """Apply ordered regex rules and return the first matching label."""
    if not isinstance(text, str):
        return default
    t = text.lower()
    for pattern, label in rules:
        if re.search(pattern, t):
            return label
    return default


def append_csv(df, path, first):
    """Append (or create) a CSV chunk. Writes header only on the first call."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="w" if first else "a", header=first,
              index=False, encoding="utf-8")


def ngrams_per_category(df, text_col, cat_col, top_n=12):
    """
    Return {category: "term1 | term2 | ..."} -- a pipe-separated string of
    the top bigrams/trigrams per category, suitable for a CSV column value.
    """
    lookup = {}
    for cat in df[cat_col].dropna().unique():
        subset = df[df[cat_col] == cat][text_col].fillna("")
        if len(subset) < 5:
            continue
        vec = TfidfVectorizer(ngram_range=(2, 3), stop_words="english",
                              max_features=2_000, sublinear_tf=True)
        try:
            X = vec.fit_transform(subset)
        except ValueError:
            continue
        scores = np.asarray(X.mean(axis=0)).flatten()
        top_terms = [vec.get_feature_names_out()[i]
                     for i in scores.argsort()[-top_n:][::-1]]
        lookup[str(cat)] = " | ".join(top_terms)
    return lookup


# ==============================================================================
# PER-ROW HEURISTIC SUMMARIZER
# Extracts a short summary from an individual text by:
#   1. Scoring every token by its TF-IDF weight from a pre-fitted vectorizer
#   2. Picking the top-N highest-scoring tokens as "key terms"
#   3. Returning the first sentence that contains any key term, or falling back
#      to the top key terms joined as a phrase
# This is purely heuristic (no LLM), satisfying the classical NLP requirement.
# ==============================================================================

def build_summarizer(texts, max_features=5_000):
    """Fit a TF-IDF vectorizer on a corpus and return a per-row summary function.

    The returned function takes a single text string and returns a short
    extractive summary (one sentence or a key-phrase string, <= 20 words).
    """
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    vec.fit(texts.fillna("").astype(str))
    vocab = vec.vocabulary_          # term -> column index
    idf   = vec.idf_                 # idf weights
    # token_score: term -> its idf weight (higher = more informative)
    token_score = {term: idf[idx] for term, idx in vocab.items()}

    def summarize(text, top_n=5, max_words=20):
        if not isinstance(text, str) or not text.strip():
            return ""
        # Score each word/bigram in the text
        words = re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
        scored = {}
        for w in words:
            if w in token_score:
                scored[w] = token_score[w]
        # Also score bigrams
        for i in range(len(words) - 1):
            bg = words[i] + " " + words[i+1]
            if bg in token_score:
                scored[bg] = token_score[bg]

        if not scored:
            # Fallback: return first max_words words
            return " ".join(text.split()[:max_words])

        top_terms = sorted(scored, key=scored.get, reverse=True)[:top_n]

        # Try to find the single most informative sentence (contains a key term)
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        best_sentence = None
        best_score = -1
        for sent in sentences:
            sent_lower = sent.lower()
            score = sum(scored.get(t, 0) for t in top_terms if t in sent_lower)
            if score > best_score:
                best_score = score
                best_sentence = sent

        if best_sentence:
            words_in_sent = best_sentence.split()
            if len(words_in_sent) <= max_words:
                return best_sentence
            # Trim to max_words while keeping it readable
            return " ".join(words_in_sent[:max_words]) + "..."

        # Final fallback: just return key terms
        return " | ".join(top_terms)

    return summarize


# ==============================================================================
# PIPELINE A -- 311 ENRICHMENT
# Single output: 311_enriched.csv
# New columns: complaint_category, predicted_category, complaint_summary,
#              vader_compound, vader_pos, vader_neg, sentiment_label, severity,
#              textblob_polarity, textblob_subjectivity, top_ngrams
# ==============================================================================

def run_311(chunk_size=PREDICT_CHUNK_SIZE):
    print("\n" + "=" * 60)
    print("PIPELINE A -- 311 Enrichment  ->  311_enriched.csv")
    print("=" * 60)

    src = PROCESSED_DIR / "311_cleaned.csv"
    out = PROCESSED_DIR / "311_enriched.csv"

    if not src.exists():
        print(f"  [SKIP] Not found: {src}")
        return

    # ---- Phase A: fit classifiers on a training sample --------------------
    print(f"\n  Phase A -- fit  ({TRAIN_SAMPLE:,}-row sample)")
    train_df = load(src, sample=TRAIN_SAMPLE)

    text_col = next(
        (c for c in ["service_name", "subject", "status_notes", "service_notice"]
         if c in train_df.columns), None
    )
    if text_col is None:
        print("  [ERROR] No usable text column found in 311_cleaned.csv")
        return
    print(f"  Text column: '{text_col}'")

    train_df["_text"] = train_df[text_col].fillna("").astype(str).str.lower().str.strip()
    train_df["complaint_category"] = train_df["_text"].apply(
        lambda t: rule_label(t, COMPLAINT_RULES)
    )

    label_counts = train_df["complaint_category"].value_counts()
    print(f"\n  Label distribution (sample):\n{label_counts.to_string()}")

    valid = label_counts[label_counts >= 5].index
    fit_df = train_df[train_df["complaint_category"].isin(valid)].copy()

    vec311 = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2),
                              stop_words="english", sublinear_tf=True)
    X_fit = vec311.fit_transform(fit_df["_text"])
    le311 = LabelEncoder()
    y_fit = le311.fit_transform(fit_df["complaint_category"])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_fit, y_fit, test_size=0.2, random_state=42, stratify=y_fit
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "Linear SVM":          LinearSVC(max_iter=2000),
        "Random Forest":       RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                                      random_state=42),
    }

    best_name, best_model, best_f1 = None, None, -1.0
    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        report = classification_report(y_te, preds, target_names=le311.classes_,
                                       output_dict=True)
        macro_f1 = report["macro avg"]["f1-score"]
        print(f"\n  --- {name} ---")
        print(classification_report(y_te, preds, target_names=le311.classes_))
        if macro_f1 > best_f1:
            best_f1, best_name, best_model = macro_f1, name, model

    print(f"\n  Best classifier: {best_name}  (macro F1 = {best_f1:.3f})")

    # N-gram lookup: {category -> "term1 | term2 | ..."}
    print("\n  Computing n-gram summaries per category...")
    ngram_lookup = ngrams_per_category(fit_df, "_text", "complaint_category")
    for cat, terms in ngram_lookup.items():
        print(f"    {cat}: {terms}")

    # Build per-row summarizer fitted on the same training sample
    print("  Building per-row heuristic summarizer...")
    summarize_311 = build_summarizer(fit_df["_text"])

    # Sentiment tools (shared across chunks, instantiated once)
    analyzer_311 = SentimentIntensityAnalyzer()

    def vader_severity_311(score):
        # 311 has no star ratings, so severity comes purely from VADER.
        # High = strongly negative language (urgent complaint),
        # Low  = neutral/informational request.
        if score <= -0.2:
            return "High"
        elif score <= 0.05:
            return "Medium"
        return "Low"

    def sentiment_label_311(score):
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        return "Neutral"

    # ---- Phase B: stream full file, enrich each chunk, write one CSV ------
    print(f"\n  Phase B -- stream & enrich  (chunk = {chunk_size:,} rows)")
    total_in = total_out = 0
    first = True

    for idx, chunk in iter_csv_chunks(src, chunk_size):
        total_in += len(chunk)

        chunk["_text"] = (
            chunk[text_col].fillna("").astype(str).str.lower().str.strip()
        )

        # Task 1 -- classification
        chunk["complaint_category"] = chunk["_text"].apply(
            lambda t: rule_label(t, COMPLAINT_RULES)
        )
        X_chunk = vec311.transform(chunk["_text"])
        chunk["predicted_category"] = le311.inverse_transform(
            best_model.predict(X_chunk)
        )

        # Task 2 -- VADER + TextBlob sentiment on the raw (non-lowercased) text
        raw_text = chunk[text_col].fillna("").astype(str)
        vader_scores = raw_text.apply(analyzer_311.polarity_scores)
        chunk["vader_compound"]        = vader_scores.apply(lambda s: s["compound"])
        chunk["vader_pos"]             = vader_scores.apply(lambda s: s["pos"])
        chunk["vader_neg"]             = vader_scores.apply(lambda s: s["neg"])
        chunk["sentiment_label"]       = chunk["vader_compound"].apply(sentiment_label_311)
        chunk["severity"]              = chunk["vader_compound"].apply(vader_severity_311)
        tb = raw_text.apply(lambda t: TextBlob(t).sentiment)
        chunk["textblob_polarity"]     = tb.apply(lambda s: s.polarity)
        chunk["textblob_subjectivity"] = tb.apply(lambda s: s.subjectivity)

        # Task 3 -- per-row extractive summary + category-level n-gram lookup
        chunk["complaint_summary"] = raw_text.apply(summarize_311)
        chunk["top_ngrams"] = chunk["predicted_category"].map(ngram_lookup).fillna("")

        chunk = chunk.drop(columns=["_text"])
        total_out += len(chunk)
        append_csv(chunk, out, first)
        first = False
        print(f"    Chunk {idx + 1}: {len(chunk):,} rows  (total: {total_out:,})")

    print(f"\n  Done: {total_in:,} in -> {total_out:,} rows saved.")
    print(f"  Output -> {out}")
    print("  Columns added: complaint_category, predicted_category, complaint_summary,")
    print("                 vader_compound, vader_pos, vader_neg, sentiment_label, severity,")
    print("                 textblob_polarity, textblob_subjectivity, top_ngrams")


# ==============================================================================
# PIPELINE B -- YELP REVIEW ENRICHMENT
# Single output: yelp_reviews_enriched.csv
# New columns: review_aspect, vader_*, textblob_*, vader_severity,
#              ml_severity, top_ngrams, broad_category_rule,
#              cluster_id, cluster_label
# ==============================================================================

def run_yelp(chunk_size=PREDICT_CHUNK_SIZE):
    print("\n" + "=" * 60)
    print("PIPELINE B -- Yelp Review Enrichment  ->  yelp_reviews_enriched.csv")
    print("=" * 60)

    src_reviews  = PROCESSED_DIR / "yelp_review_cleaned.csv"
    src_business = PROCESSED_DIR / "yelp_business_cleaned.csv"
    out          = PROCESSED_DIR / "yelp_reviews_enriched.csv"

    if not src_reviews.exists():
        print(f"  [SKIP] Not found: {src_reviews}")
        return

    analyzer = SentimentIntensityAnalyzer()

    def vader_sentiment_label(score):
        # Readable label for how positive/negative the text SOUNDS.
        # Positive = good experience, Negative = bad experience.
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        return "Neutral"

    def stars_to_severity(s):
        # Complaint-oriented: High = problem (1-2 stars), Low = satisfied (4-5 stars).
        try:
            s = float(s)
        except (TypeError, ValueError):
            return None
        if s <= 2:
            return "High"
        elif s == 3:
            return "Medium"
        else:
            return "Low"

    def combined_severity(vader_compound, ml_sev):
        # VADER is the primary signal -- it reads the actual review words, so it
        # handles sarcasm, caveats, and mixed reviews better than a star-rating
        # proxy. The star-based ML model only breaks ties in the neutral band.
        #
        # Yelp reviews skew heavily positive (most are 4-5 stars), which causes
        # the ML model to over-predict "Low". So we only trust ML when VADER
        # is ambiguous (compound between -0.2 and +0.2).
        #
        # Severity is complaint-oriented: High = bad experience, Low = good.
        if vader_compound <= -0.2:
            # Clearly negative language -- always High regardless of star model
            return "High"
        elif vader_compound > 0.2:
            # Clearly positive language -- Low, unless ML strongly disagrees
            return "Low" if ml_sev != "High" else "Medium"
        else:
            # Ambiguous band: defer to the star-rating ML model
            return ml_sev if ml_sev else "Medium"

    def score_sentiment(chunk):
        # Add VADER + TextBlob columns. sentiment_label is the human-readable
        # Positive/Neutral/Negative label; raw scores are kept for analysis.
        chunk = chunk.copy()
        chunk["text"] = chunk["text"].fillna("").astype(str)
        vader_scores = chunk["text"].apply(analyzer.polarity_scores)
        chunk["vader_compound"]        = vader_scores.apply(lambda s: s["compound"])
        chunk["vader_pos"]             = vader_scores.apply(lambda s: s["pos"])
        chunk["vader_neg"]             = vader_scores.apply(lambda s: s["neg"])
        chunk["sentiment_label"]       = chunk["vader_compound"].apply(vader_sentiment_label)
        tb = chunk["text"].apply(lambda t: TextBlob(t).sentiment)
        chunk["textblob_polarity"]     = tb.apply(lambda s: s.polarity)
        chunk["textblob_subjectivity"] = tb.apply(lambda s: s.subjectivity)
        return chunk

    # ---- Phase A: fit models on a training sample -------------------------
    print(f"\n  Phase A -- fit  ({YELP_TRAIN_SAMPLE:,}-row sample)")
    train_df = load(src_reviews, sample=YELP_TRAIN_SAMPLE)

    if train_df is None or "text" not in train_df.columns:
        print("  [ERROR] 'text' column missing in yelp_review_cleaned.csv")
        return

    train_df["text"] = train_df["text"].fillna("").astype(str)

    # Task 2 -- severity classifier
    sev_vec = sev_clf = None
    if "stars" in train_df.columns:
        print("  Training ML severity classifier...")
        train_df["severity_label"] = train_df["stars"].apply(stars_to_severity)
        labeled = train_df.dropna(subset=["severity_label"]).copy()

        sev_vec = TfidfVectorizer(max_features=8_000, ngram_range=(1, 2),
                                  stop_words="english", sublinear_tf=True)
        X = sev_vec.fit_transform(labeled["text"])
        y = labeled["severity_label"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        # class_weight="balanced" compensates for the heavy skew toward
        # 4-5 star (Low severity) reviews in the Yelp dataset. Without it the
        # model learns to predict Low for almost everything.
        sev_clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        sev_clf.fit(X_tr, y_tr)
        print("\n  ML Severity report:")
        print(classification_report(y_te, sev_clf.predict(X_te)))
    else:
        print("  [WARN] 'stars' column not found -- skipping ML severity classifier")

    # Task 3 -- n-gram summaries per review aspect
    print("  Computing n-gram summaries per review aspect...")
    train_df["review_aspect"] = train_df["text"].apply(
        lambda t: rule_label(t, REVIEW_RULES)
    )
    ngram_lookup = ngrams_per_category(train_df, "text", "review_aspect")
    for aspect, terms in ngram_lookup.items():
        print(f"    {aspect}: {terms}")

    # Build per-row summarizer fitted on the review training sample
    print("  Building per-row heuristic summarizer...")
    summarize_yelp = build_summarizer(train_df["text"])

    # Free the review training sample before loading the business file.
    # Both are large; keeping them in RAM simultaneously causes OOM kills.
    del train_df
    try:
        del labeled
    except NameError:
        pass

    # Task 4 -- business category normalization
    # Build {business_id: {broad_category_rule, cluster_id, cluster_label}}
    # so we can join normalization columns onto each review chunk in O(1) per row.
    # The business file is sampled so it never fully loads into RAM.
    biz_lookup = {}

    if src_business.exists():
        print("\n  Task 4 -- normalizing Yelp business categories...")
        biz = load(src_business, sample=YELP_TRAIN_SAMPLE)
        cat_col = "categories_clean" if "categories_clean" in biz.columns else "categories"

        if cat_col in biz.columns:
            biz["raw_categories"] = biz[cat_col].fillna("").astype(str)

            def map_to_group(s):
                s = s.lower()
                for group, kws in CATEGORY_GROUPS.items():
                    if any(kw in s for kw in kws):
                        return group
                return "Other"

            biz["broad_category_rule"] = biz["raw_categories"].apply(map_to_group)
            print(f"\n  Rule-based distribution:\n"
                  f"{biz['broad_category_rule'].value_counts().to_string()}")

            # K-Means clustering
            mask = biz["raw_categories"].str.len() > 3
            sub  = biz[mask].copy()
            n_clusters = 10

            if len(sub) >= n_clusters:
                print(f"\n  K-Means (k={n_clusters}) on category TF-IDF...")
                biz_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5_000,
                                          stop_words="english")
                X_biz = biz_vec.fit_transform(sub["raw_categories"])
                km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                sub = sub.copy()
                sub["cluster_id"] = km.fit_predict(X_biz)

                feature_names = biz_vec.get_feature_names_out()
                order_centroids = km.cluster_centers_.argsort()[:, ::-1]
                cluster_labels = {}
                print("  Cluster top terms:")
                for i in range(n_clusters):
                    top = [feature_names[j] for j in order_centroids[i, :8]]
                    label = " / ".join(top[:3])
                    cluster_labels[i] = label
                    print(f"    Cluster {i:2d}: {top}")

                sub["cluster_label"] = sub["cluster_id"].map(cluster_labels)
                biz = biz.merge(
                    sub[["cluster_id", "cluster_label"]],
                    left_index=True, right_index=True, how="left"
                )
                biz["cluster_id"]    = biz["cluster_id"].fillna(-1).astype(int)
                biz["cluster_label"] = biz["cluster_label"].fillna("Unclustered")

                print("\n  Rule label vs. cluster cross-tab:")
                print(pd.crosstab(biz["broad_category_rule"],
                                  biz["cluster_label"]).to_string())

            if "business_id" in biz.columns:
                keep = ["business_id", "broad_category_rule"]
                if "cluster_id"    in biz.columns: keep.append("cluster_id")
                if "cluster_label" in biz.columns: keep.append("cluster_label")
                biz_lookup = biz[keep].set_index("business_id").to_dict(orient="index")
        else:
            print("  [WARN] No categories column in business file -- skipping normalization")
    else:
        print(f"  [WARN] Business file not found: {src_business}")

    # ---- Phase B: stream reviews, enrich each chunk, write single CSV -----
    print(f"\n  Phase B -- stream & enrich  (chunk = {chunk_size:,} rows)")
    total_in = total_out = 0
    first = True

    for idx, chunk in iter_csv_chunks(src_reviews, chunk_size):
        total_in += len(chunk)

        if "text" not in chunk.columns:
            print(f"    Chunk {idx + 1}: no 'text' column -- skipping")
            continue

        # Task 2a -- VADER + TextBlob scores
        chunk = score_sentiment(chunk)

        # Task 2b -- ML severity + unified severity
        if sev_vec is not None and sev_clf is not None:
            X_chunk = sev_vec.transform(chunk["text"])
            chunk["ml_severity"] = sev_clf.predict(X_chunk)
            # severity: single unified column combining VADER + ML
            # High = bad/complaint, Medium = mixed, Low = satisfied
            chunk["severity"] = chunk.apply(
                lambda r: combined_severity(r["vader_compound"], r["ml_severity"]),
                axis=1
            )
        else:
            # VADER-only fallback when no star labels available
            chunk["severity"] = chunk["vader_compound"].apply(
                lambda s: "High" if s <= -0.4 else ("Medium" if s <= -0.05 else "Low")
            )

        # Task 3 -- review aspect, per-row extractive summary, category n-grams
        chunk["review_aspect"] = chunk["text"].apply(
            lambda t: rule_label(t, REVIEW_RULES)
        )
        chunk["review_summary"] = chunk["text"].apply(summarize_yelp)
        chunk["top_ngrams"] = chunk["review_aspect"].map(ngram_lookup).fillna("")

        # Task 4 -- join business normalization columns via business_id
        if biz_lookup and "business_id" in chunk.columns:
            chunk["broad_category_rule"] = chunk["business_id"].map(
                lambda bid: biz_lookup.get(bid, {}).get("broad_category_rule", "Unknown")
            )
            chunk["cluster_id"] = chunk["business_id"].map(
                lambda bid: biz_lookup.get(bid, {}).get("cluster_id", -1)
            )
            chunk["cluster_label"] = chunk["business_id"].map(
                lambda bid: biz_lookup.get(bid, {}).get("cluster_label", "Unclustered")
            )

        total_out += len(chunk)
        append_csv(chunk, out, first)
        first = False
        print(f"    Chunk {idx + 1}: {len(chunk):,} rows  (total: {total_out:,})")

    print(f"\n  Done: {total_in:,} in -> {total_out:,} rows saved.")
    print(f"  Output -> {out}")
    added = ["review_aspect", "review_summary", "vader_compound", "vader_pos", "vader_neg",
             "sentiment_label", "textblob_polarity", "textblob_subjectivity",
             "top_ngrams", "severity"]
    if sev_clf:
        added.append("ml_severity")
    if biz_lookup:
        added += ["broad_category_rule", "cluster_id", "cluster_label"]
    print(f"  Columns added: {', '.join(added)}")


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classification Pipeline -- CS 4/5630 Project 1"
    )
    parser.add_argument(
        "--source",
        choices=["311", "yelp", "both"],
        default="both",
        help="Which pipeline to run (default: both)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=PREDICT_CHUNK_SIZE,
        metavar="N",
        help=(
            f"Rows per prediction batch (default: {PREDICT_CHUNK_SIZE:,}). "
            "Reduce if you encounter memory errors."
        ),
    )
    args = parser.parse_args()

    if args.source in ("311", "both"):
        run_311(chunk_size=args.chunk_size)
        # Explicitly trigger garbage collection before the Yelp pipeline starts
        # so the 311 fitted models and TF-IDF matrix are freed from RAM.
        import gc; gc.collect()

    if args.source in ("yelp", "both"):
        run_yelp(chunk_size=args.chunk_size)

    print("\n========== Classification Pipeline Complete ==========")