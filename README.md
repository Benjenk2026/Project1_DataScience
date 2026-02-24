# Data Cleaning Utilities

## Overview
This project includes data cleaning and processing utilities in `src/cleaning.py` to standardize and deduplicate data from multiple formats (JSON, Excel, CSV).

---
Pipeline Idea 
1. Clean the data using cleaning.py
  - standarize_data
  - deduplicate_records
  - handle_missing_values
  
2. Text and Category processing tasks
  - classify complaint descriptions (complaint_classifier.py)
  - estimate severity or sentiment (estimate_severity_sentiment.py)
  - Summarize complaint descriptions ()
  - normalize Yelp buisness catgories ()

3. Data integration
  - Geospatial integration
  - Hybrid integration

4. Analysis and Findings 
  - Complaint hotspots by neighborhood (e.g. heatmaps)
  - Relationship between business density and complaint frequency
  - Cluster of complaint types using classical clustering methods
  - Differences in complaint patterns near different business categories (e.g. resturants vs. retail vs. services)






## Data Cleaning Functions

### 1. Standardize Data (`standardize_data`)
Reads and standardizes data files, normalizing column names and format.

#### Normalization Rules Applied:
- **Snake case conversion:** Converts all column names to lowercase with underscores
- **Abbreviation expansion:** Common abbreviations are expanded to full names
  - `lat` → `latitude`
  - `lon`, `long` → `longitude`
  - `zip`, `postcode` → `postal_code`
  - `addr` → `address`
- **Column merging:** redundant columns with same normalized name are merged (first non-null value kept)
- **Semantic standardization:**
  - Geographic fields standardized to `latitude` and `longitude`
  - Count fields prefixed with `num_`
  - Boolean fields (0/1/true/false) prefixed with `is_`
- **Final cleanup:** Normalizes/Removes special characters, collapses underscores, ensures uniqueness

---

### 2. Deduplicate Records (`deduplicate_records`)
Identifies and removes duplicate records based on specified column(s), keeping the most relevant row based on your criteria.

#### Function Signature
```python
deduplicate_records(df, subset=None, keep='first', strategy='subset')
```

#### Parameters
- **df** (`pd.DataFrame`): Input DataFrame to deduplicate
- **subset** (`str`, `list`, or `None`, default `None`):
  - Columns to check for duplicates
  - `None`: checks all columns
  - `str`: single column (e.g., `'id'`)
  - `list`: multiple columns (e.g., `['id', 'name']`)
- **keep** (`{'first', 'last', 'most_complete'}`, default `'first'`):
  - `'first'`: keeps first occurrence of duplicate
  - `'last'`: keeps last occurrence of duplicate
  - `'most_complete'`: keeps row with fewest null values (best for data quality)
- **strategy** (`{'subset', 'approximate'}`, default `'subset'`):
  - `'subset'`: exact duplicate matching

#### Returns
- **dedup_df** (`pd.DataFrame`): Deduplicated DataFrame
- **stats** (`dict`): Summary statistics with:
  - `original_rows`: row count before deduplication
  - `final_rows`: row count after deduplication
  - `duplicates_removed`: number of rows removed
  - `duplicate_groups`: number of groups with duplicates

#### Usage Examples

**Remove rows with duplicate IDs:**
```python
from src.cleaning import deduplicate_records
import pandas as pd

df = pd.read_csv('data/processed/mydata_standardized.csv')
dedup_df, stats = deduplicate_records(df, subset='id')
print(f"Removed {stats['duplicates_removed']} duplicate rows")
```

**Remove rows with duplicate ID + Name combinations:**
```python
dedup_df, stats = deduplicate_records(df, subset=['id', 'name'])
```

**Keep the row with the most complete data (fewest nulls):**
```python
dedup_df, stats = deduplicate_records(df, subset='id', keep='most_complete')
```
---

---

### 3. Handle Missing Values (`handle_missing_values`)
Flexible handling for missing data: drop rows/columns or impute values using several strategies.

Function signature:
```python
handle_missing_values(df, drop_rows_subset=None, drop_cols_threshold=None, impute_strategy=None, impute_values=None)
```

Key options:
- `drop_rows_subset`: column or list of columns — drop rows where any of these are missing (useful for required fields like `id`).
- `drop_cols_threshold`: percentage (0-100) — drop columns with >= this percent missing (e.g., 80).
- `impute_strategy`: mapping `col -> strategy` where strategy is one of `mean`, `median`, `mode`, `forward_fill`, `backward_fill`.
- `impute_values`: mapping `col -> value` to fill missing values directly (takes precedence over strategies).

Returns `(clean_df, stats)` where `stats` reports rows/cols dropped and cells imputed.

Usage examples:
```python
from src.cleaning import handle_missing_values
clean_df, stats = handle_missing_values(df, drop_rows_subset='id')
clean_df, stats = handle_missing_values(df, drop_cols_threshold=80, impute_strategy={'age':'mean'})
clean_df, stats = handle_missing_values(df, impute_values={'department':'Unknown'})
```

### Notes on `standardize_data` and JSON parsing
- `standardize_data` accepts either a file path or a `pd.DataFrame` and now supports two useful flags:
  - `trim_whitespace=True` (default) — strips leading/trailing whitespace from string-like columns.
  - `casefold_values=False` (default) — when True applies `str.casefold()` for robust lowercasing.
- `openfile()` reads JSON files with `encoding='utf-8-sig'` to handle BOMs, tries `pd.read_json(..., lines=True)` first, and falls back to line-by-line `json.loads()` while warning and skipping malformed lines.


## Interactive CLI Usage

Run the data cleaning utility with:
```bash
python src/cleaning.py
```

### Menu Options:
1. **Standardize** - Load a file and standardize column names
2. **Deduplicate** - Remove duplicate records from a file
3. **Exit** - Close the utility

#### Deduplicate Workflow:
1. Select a data file (JSON, Excel, or CSV)
2. View available columns
3. Specify which column(s) to check for duplicates (comma-separated)
4. Choose keep strategy (first, last, or most_complete)
5. Review deduplication summary statistics
6. Optionally save deduplicated file to `data/processed/`

---

#### Example: Interactive missing-values flow
This short transcript shows the prompts you'll see when running option 3 (handle missing values):

```
$ python src/cleaning.py
Select an action: 1(standardize), 2(deduplicate), 3(handle missing values), 4(exit) 3
Selected file: data/raw/mydata.json

Available columns: id, name, age, dept, salary
Missing values summary:
  age: 12 (2.4%)
  dept: 50 (10.0%)

Enter column(s) where missing values should drop rows (comma-separated, or press Enter to skip): id
Enter threshold % to drop columns with missing values (0-100, or press Enter to skip): 80
Impute missing values? (y/n): y
Imputation strategies: mean, median, mode, forward_fill, backward_fill, or specific value
Enter column:strategy pairs (e.g., 'age:mean,dept:Unknown', leave blank to skip): age:mean,dept:Unknown

Missing Values Handling Summary:
  Original rows: 500, Final rows: 498 (dropped 2)
  Original columns: 12, Final columns: 12 (dropped 0)
  Cells imputed: 62

Save cleaned file? (y/n): y
Wrote cleaned file to: data/processed/mydata_handled_missing.csv
```

---

## Geospatial Integration (`data_integration.py`)

This module provides a simple geospatial join between 311 complaint records and Yelp business locations.

Key functionality:
- `integrate_geospatial(path_311, path_yelp, radius_m=300)` — finds the nearest Yelp business for each 311 complaint within `radius_m` meters and writes a joined CSV to `data/processed/integrated_dataset.csv`.
- Uses a Haversine distance (`haversine_m`) to compute distances in meters between latitude/longitude pairs.
- Performs standardization, missing-value handling, and deduplication (via `src/cleaning.py`) before matching.

Outputs:
- `data/processed/integrated_dataset.csv` — matched rows with fields from both sources plus `match_distance_m` and `radius_m`.
- The function also returns `(integrated_df, stats)` where `stats` includes row counts and match-rate.

Example usage (run from repo root):
```bash
python -m src.data_integration
```

---

## Text & Category Processing (`text_category_processing_full.py`)

### Overview

`text_category_processing_full.py` handles all of Phase 2's text and category enrichment tasks. It reads the cleaned CSVs produced by `cleaning.py` and produces **two unified output CSVs**, each containing every enrichment column in a single file — no intermediate files, no post-hoc merges.

---

### Pipeline Design

Each source file is processed in a **single streaming pass** split into two phases:

**Phase A — Fit (in-memory sample)**
- Load a manageable sample (`TRAIN_SAMPLE` rows) into RAM
- Fit TF-IDF vectorizer and train/evaluate all three classifiers (311)
- Fit TF-IDF and train/evaluate the severity LR classifier (Yelp reviews)
- Compute per-category n-gram summaries (stored as a lookup dict)
- Fit TF-IDF + K-Means on Yelp businesses (stored as a `business_id` lookup dict)

**Phase B — Stream & Enrich (chunked)**
- Read the full source CSV in `PREDICT_CHUNK_SIZE` batches — only one batch in RAM at a time
- Apply all fitted models and lookup dicts to each batch
- Append each enriched batch directly to the output CSV (low RAM footprint)

---

### Usage

```bash
python src/text_category_processing_full.py                     # run both pipelines
python src/text_category_processing_full.py --source 311        # only 311
python src/text_category_processing_full.py --source yelp       # only Yelp
python src/text_category_processing_full.py --chunk-size 25000  # smaller batches if memory is tight
```

#### Batch Size Controls

| Constant | Default | Purpose |
|---|---|---|
| `TRAIN_SAMPLE` | `200,000` | Rows loaded to fit 311 models |
| `YELP_TRAIN_SAMPLE` | `50,000` | Rows loaded to fit Yelp models (reviews are larger/denser) |
| `PREDICT_CHUNK_SIZE` | `50,000` | Rows per streaming prediction batch |

Reduce `--chunk-size` if you encounter memory errors during Phase B.

---

### Inputs & Outputs

**Inputs** (`data/processed/`):
- `311_cleaned.csv`
- `yelp_review_cleaned.csv`
- `yelp_business_cleaned.csv`

**Outputs** (`data/processed/`):
- `311_enriched.csv`
- `yelp_reviews_enriched.csv`

---

### Pipeline A — 311 Enrichment

Reads `311_cleaned.csv`, applies all four enrichment tasks, and writes `311_enriched.csv`.

#### Output Columns Added

| Column | Description |
|---|---|
| `complaint_category` | Rule-based seed label (e.g. Sanitation, Parking, Graffiti) |
| `predicted_category` | Best ML model prediction (winner of LR vs. SVM vs. RF by macro F1) |
| `complaint_summary` | Per-row extractive summary unique to each individual complaint text |
| `vader_compound` | VADER overall sentiment score (−1 to +1) |
| `vader_pos` | VADER positive sentiment component |
| `vader_neg` | VADER negative sentiment component |
| `sentiment_label` | `Positive` / `Neutral` / `Negative` derived from `vader_compound` |
| `severity` | `High` / `Medium` / `Low` — VADER-only since 311 has no star ratings |
| `textblob_polarity` | TextBlob polarity score (−1 to +1) |
| `textblob_subjectivity` | TextBlob subjectivity score (0 = objective, 1 = subjective) |
| `top_ngrams` | Most characteristic bigrams/trigrams for that predicted category |

#### Severity Thresholds (311)

Since 311 complaints have no star ratings, severity is derived purely from VADER:

| VADER compound | Severity |
|---|---|
| `≤ −0.20` | **High** — urgent/strongly negative language |
| `−0.20` to `+0.05` | **Medium** — mildly negative or ambiguous |
| `> +0.05` | **Low** — neutral or informational request |

#### Classifiers Evaluated

All three models are trained on rule-seeded labels and evaluated on a held-out 20% split. The best by macro F1 is used for Phase B prediction.

```
Logistic Regression  — max_iter=1000, C=1.0
Linear SVM           — max_iter=2000
Random Forest        — n_estimators=100, n_jobs=-1
```

---

### Pipeline B — Yelp Review Enrichment

Reads `yelp_review_cleaned.csv` and `yelp_business_cleaned.csv`, applies all four enrichment tasks, and writes `yelp_reviews_enriched.csv`.

#### Output Columns Added

| Column | Description |
|---|---|
| `review_aspect` | Rule-based aspect label (Food Quality, Service, Atmosphere, Price/Value, Wait Time, Cleanliness) |
| `review_summary` | Per-row extractive summary unique to each individual review |
| `vader_compound` | VADER overall sentiment score (−1 to +1) |
| `vader_pos` | VADER positive sentiment component |
| `vader_neg` | VADER negative sentiment component |
| `sentiment_label` | `Positive` / `Neutral` / `Negative` derived from `vader_compound` |
| `textblob_polarity` | TextBlob polarity score (−1 to +1) |
| `textblob_subjectivity` | TextBlob subjectivity score |
| `ml_severity` | LR classifier trained on star ratings with `class_weight="balanced"` |
| `severity` | Unified severity combining VADER (primary) + ML (tiebreaker in neutral band) |
| `top_ngrams` | Most characteristic bigrams/trigrams for that review aspect |
| `broad_category_rule` | Business group from string matching (e.g. Food & Dining, Retail, Health) |
| `cluster_id` | K-Means cluster ID assigned to the associated business |
| `cluster_label` | Human-readable label for that cluster (top 3 TF-IDF terms) |

#### Severity Logic (Yelp)

Yelp severity combines VADER (which reads the actual words) with an ML classifier trained on star ratings. VADER is the primary signal because the dataset skews heavily toward 4–5 star reviews, causing the ML model to over-predict "Low" without correction.

| Condition | Severity |
|---|---|
| `vader_compound ≤ −0.20` | **High** — negative language always wins |
| `vader_compound > +0.20` and `ml_severity != High` | **Low** |
| `vader_compound > +0.20` and `ml_severity == High` | **Medium** — ML overrides clearly positive VADER |
| `−0.20 < vader_compound ≤ +0.20` | Defer to `ml_severity` (neutral band) |

#### Business Category Normalization

Two strategies are applied to `yelp_business_cleaned.csv` and joined onto each review via `business_id`:

**Strategy A — String Matching:** Maps raw Yelp category strings to one of 8 broad groups using keyword rules.

| Group | Example Keywords |
|---|---|
| Food & Dining | restaurant, food, bar, cafe, pizza, sushi |
| Retail | shop, store, boutique, market, clothing |
| Health | medical, doctor, dentist, pharmacy, hospital |
| Beauty | salon, spa, nail, hair, barber |
| Automotive | auto, car, tire, mechanic |
| Services | plumber, electrician, contractor, cleaning |
| Entertainment | gym, fitness, yoga, movie, theatre |
| Education | school, tutor, university, college |

**Strategy B — K-Means Clustering:** Fits TF-IDF on raw category strings and clusters into 10 groups. Each cluster is labeled with its top 3 TF-IDF terms. A cross-tab of rule labels vs. cluster labels is printed for interpretability.

---

### Heuristic Summarizer (`build_summarizer`)

Returns a per-row summary function fitted on a training corpus. No LLMs — purely TF-IDF-based extraction.

**Algorithm:**
1. Fit TF-IDF on the training corpus to learn IDF weights (higher = more informative token)
2. For each row, score every word and bigram in the text by its IDF weight
3. Find the single sentence containing the highest concentration of top-scored terms
4. Return that sentence trimmed to 20 words, or fall back to top key terms joined with `|`

```python
summarize_311  = build_summarizer(fit_df["_text"])   # fitted on 311 training sample
summarize_yelp = build_summarizer(train_df["text"])  # fitted on Yelp review sample

```

## Hotspot Analysis (`src/hotspot.py`)

This module generates interactive hotspot maps from enriched complaint/review datasets in `data/processed`.

### Supported Inputs
- `311_enriched.csv`
- `yelp_reviews_enriched.csv`
- Or both combined in one run

### Run Commands
From the project root:

```bash
python src/hotspot.py --source 311
python src/hotspot.py --source yelp
python src/hotspot.py --source both
```

Optional:
```bash
python src/hotspot.py --source both --data-dir data/processed
```

### Output Files (for `--source both`)

- `heatmap_both.html`
  - **Purpose:** Shows a continuous density heatmap of all valid coordinates.
  - **Best use:** Quickly identifies broad geographic concentration zones (high vs. low intensity areas).

- `complaint_clusters_both.html`
  - **Purpose:** Shows point-based marker clusters that aggregate nearby records interactively as you zoom.
  - **Best use:** Inspects localized groupings and supports drill-down from city-level clusters to neighborhood-level points.

Both files are written to the project root directory and can be opened directly in a browser.

### Output Files (for `--source 311`)

- `heatmap_311.html`
  - **Purpose:** Density heatmap using 311 enriched complaint coordinates only.
  - **Best use:** Identifies broad 311 complaint concentration areas without Yelp review data mixed in.

- `complaint_clusters_311.html`
  - **Purpose:** Clustered point map of 311 complaint locations.
  - **Best use:** Explores local 311 complaint groupings and inspects neighborhood-level clusters.

### Output Files (for `--source yelp`)

- `heatmap_yelp.html`
  - **Purpose:** Density heatmap using Yelp review enriched coordinates only.
  - **Best use:** Visualizes where geocoded Yelp review activity is most concentrated.

- `complaint_clusters_yelp.html`
  - **Purpose:** Clustered point map of Yelp review locations.
  - **Best use:** Drills into localized Yelp review clusters by zoom level.