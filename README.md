# Project 1 — Urban Data Cleaning, Integration, and Enrichment with Python
**CS 4/5630 — Python for Computational and Data Sciences**
**BGSU — Dr. Arijit Khan**

---

## Project Overview

This project builds a complete Python pipeline that cleans, integrates, and analyzes two real-world urban datasets:

- **Philadelphia 311 Service Requests** — structured complaint data (CSV)
- **Yelp Open Dataset** — unstructured business and review data (JSON)

The final output is a cleaned, enriched, and integrated dataset supporting analysis of urban service complaints in relation to local business environments. All methods are classical data science and machine learning — no large language models.

---

## Repository Structure

```
Project1_DataScience/
├── data/
│   ├── Raw/                        # Original unmodified source files
│   │   ├── public_cases_fc.csv
│   │   ├── yelp_academic_dataset_business.json
│   │   ├── yelp_academic_dataset_review.json
│   │   ├── yelp_academic_dataset_checkin.json
│   │   ├── yelp_academic_dataset_tip.json
│   │   └── yelp_academic_dataset_user.json
│   ├── Full/
│   │   └── Processed/
│   │       ├── Phase 2/            # Cleaned and enriched outputs
│   │       └── Phase 3/            # Integrated outputs (all strategies)
│   └── Sample/
│       └── Processed/
│           ├── Phase 2/            # ~25MB samples of Phase 2 outputs
│           └── Phase 3/            # ~25MB samples of Phase 3 outputs
├── src/
│   ├── cleaning.py                 # Phase 2 — data cleaning pipeline
│   ├── processing_tasks.py         # Phase 2 — text & category enrichment
│   ├── strategy_A.py               # Phase 3 — geospatial integration
│   ├── data_integration.py         # Phase 3 — feature-based integration
│   ├── Hybrid_integration.py       # Phase 3 — hybrid integration
│   ├── hotspot.py                  # Phase 4 — complaint hotspot maps
│   ├── clusters_of_complaints.py   # Phase 4 — K-means complaint clustering
│   └── graph/
│       ├── business_density_vs_complaints.py
│       ├── business_category_complaints.py
│       ├── complaint_by_month.py
│       └── top_complaint_types_311.py
├── data/graphs/                    # All output charts and visualizations
├── analysis_findings/              # Cluster outputs and analysis results
├── documentation.md
├── requirements.txt
└── README.md
```

---

## Dataset Sources

| Dataset | Source | Format |
|---|---|---|
| Philadelphia 311 Requests | [OpenDataPhilly](https://opendataphilly.org/datasets/311-service-and-information-requests/) | CSV |
| Yelp Open Dataset | [yelp.com/dataset](https://www.yelp.com/dataset) | JSON |

Full raw and processed datasets are available via SharePoint (BGSU access required):
- **Raw & Matched Data:** https://falconbgsu-my.sharepoint.com/:f:/g/personal/rhannam_bgsu_edu/IgBM8vU9u0LBSqYM8M0BZXXiAbNAWcfRPygBp0kpCpOgH18?e=6awJXP
Sample versions (~25MB each) of all processed files are available in `data/Sample/` in this repository.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/Benjenk2026/Project1_DataScience.git
cd Project1_DataScience

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Phase 2 — Data Cleaning (`src/cleaning.py`)

Cleans all raw source files and produces standardized CSVs in `data/Full/Processed/Phase 2/`.

### What it does

- **Standardizes column names** to snake_case with common alias resolution (`lat` → `latitude`, `lon` → `longitude`, `zip` → `zipcode`)
- **Normalizes complaint types** using rule-based regex pattern matching (case folding, whitespace trimming, label consolidation)
- **Cleans location fields** — coerces lat/lon to numeric, nulls impossible values (lat > 90, lon > 180), extracts 5-digit ZIP codes
- **Handles missing values** — drops rows missing critical ID/timestamp fields, drops columns with ≥80% missing, imputes numeric fields with median where specified
- **Deduplicates records** — keeps the most complete row per unique ID (fewest null values)
- **Normalizes Yelp business categories** — maps raw category strings to broad groups (restaurants, retail, services, health, etc.) using keyword rules, assigns a `broad_category` column
- **Chunked processing** for large Yelp files (`yelp_review`, `yelp_user`) to avoid loading entire files into memory

### Run Commands

```bash
# Clean all files (auto-chunked for large Yelp files)
python src/cleaning.py --all

# Clean a single file
python src/cleaning.py --file 311
python src/cleaning.py --file yelp_business

# Clean a large file in chunks
python src/cleaning.py --file yelp_review --chunked
python src/cleaning.py --file yelp_user --chunked
```

### Outputs (`data/Full/Processed/Phase 2/`)

| File | Description |
|---|---|
| `311_cleaned.csv` | Cleaned Philadelphia 311 complaints |
| `311_enriched.csv` | 311 data with complaint categories, sentiment, severity, summaries |
| `yelp_business_cleaned.csv` | Cleaned Yelp business records with broad category labels |
| `yelp_review_cleaned.csv` | Cleaned Yelp reviews |
| `yelp_reviews_enriched.csv` | Reviews with sentiment, severity, aspect labels, cluster IDs |
| `yelp_checkin_cleaned.csv` | Cleaned check-in data |
| `yelp_tip_cleaned.csv` | Cleaned tip data |
| `yelp_user_cleaned.csv` | Cleaned user data |

---

## Phase 3 — Data Integration

Three integration strategies are implemented, each producing its own output. All scripts read from `data/Full/Processed/Phase 2/` and write to `data/Full/Processed/Phase 3/`.

### Strategy A — Geospatial Integration (`src/strategy_A.py`)

Matches 311 complaints to nearby Yelp businesses using the **Haversine formula**, which calculates real-world distance in meters between two lat/lon coordinate pairs. Each complaint is matched to any business within a defined radius threshold.

```bash
python src/strategy_A.py
```

**Outputs:**
- `311_yelp_strategyA_integrated.csv` — all complaints with matched business candidates
- `311_yelp_strategyA_integrated_matched.csv` — only rows with at least one match

---

### Strategy B — Feature-Based Integration (`src/data_integration.py`)

Matches complaints to businesses based on **TF-IDF text similarity** between complaint descriptions and business category strings. Similarity is computed using cosine similarity via scikit-learn vectorization. Optionally clusters complaints using K-means and maps clusters to business types.

```bash
python src/data_integration.py
```

**Outputs:**
- `311_yelp_strategyB_integrated.csv`
- `311_yelp_strategyB_integrated_Only_matches.csv`

---

### Strategy C — Hybrid Integration (`src/Hybrid_integration.py`)

Combines both strategies: first applies Haversine distance to filter businesses within a proximity radius, then ranks remaining candidates by TF-IDF category similarity to select the best match. This is the primary integration output used for analysis.

```bash
python src/Hybrid_integration.py
```

**Outputs:**
- `311_yelp_hybrid_integrated.csv`
- `311_yelp_hybrid_integrated_matched.csv`
- `311_yelp_hybrid_integrated_enriched.csv`
- `311_yelp_hybrid_integrated_enriched_matched.csv`

---

## Phase 4 — Analysis & Findings

### Complaint Hotspot Maps (`src/hotspot.py`)

Generates interactive HTML heatmaps and cluster maps using Folium.

```bash
python src/hotspot.py --source enriched
python src/hotspot.py --source matched
python src/hotspot.py --source both
```

Outputs HTML files to the project root, openable directly in any browser.

---

### Complaint Type Clustering (`src/clusters_of_complaints.py`)

Clusters complaint descriptions using K-means on TF-IDF features and exports cluster profiles and visualizations.

```bash
python src/clusters_of_complaints.py --source both
python src/clusters_of_complaints.py --source both --clusters 6 --sample-size 120000
```

Outputs saved to `analysis_findings/`.

---

### Graph Scripts (`src/graph/`)

| Script | Output |
|---|---|
| `business_density_vs_complaints.py` | Scatter plot — business count vs complaint count per ZIP code |
| `business_category_complaints.py` | Bar chart, heatmap, top complaints per business category |
| `complaint_by_month.py` | Time series of complaint volume |
| `top_complaint_types_311.py` | Top complaint types bar chart |

All graphs saved to `data/graphs/`.

```bash
python src/graph/business_density_vs_complaints.py
python src/graph/business_category_complaints.py
```

---
