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



