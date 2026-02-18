# Data Cleaning Utilities

## Overview
This project includes data cleaning and processing utilities in `src/cleaning.py` to standardize and deduplicate data from multiple formats (JSON, Excel, CSV).

---

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
- **Final cleanup:** Removes special characters, collapses underscores, ensures uniqueness

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


