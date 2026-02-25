"""
Hybrid_integration.py
Phase 3 - Strategy C (Hybrid Integration)
CS 4/5630 Project 1

Hybrid integration:
1) Restrict Yelp candidates to businesses within a geographic radius (meters)
2) Within those candidates, pick the best match using TF-IDF cosine similarity
   between a 311 text field (category/type) and Yelp business categories

Key improvements in this version:
- Uses BallTree (scikit-learn) to find nearby Yelp businesses efficiently
- Uses 311_enriched.csv and yelp_business_cleaned.csv
- Drops 311 rows WITHOUT valid coordinates from the final output
- Avoids TF-IDF empty vocabulary errors by skipping empty/stopword-only corpuses
- Writes TWO outputs:
  - Full integrated file (only rows with valid coordinates)
  - Matched-only file (subset where a business match exists)

Run:
  python src/Hybrid_integration.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

RADIUS_METERS = 1000
MIN_SIMILARITY = 0.05
TOPK_NEAREST = 80

EARTH_RADIUS_M = 6371000.0  # meters


def normalize_text(x):
    if pd.isna(x):
        return ""
    return (
        str(x)
        .lower()
        .replace("&", " ")
        .replace("/", " ")
        .replace(",", " ")
        .strip()
    )


def pick_311_text_col(df_311):
    # Keep this simple and predictable
    preferred = ["complaint_type", "service_name", "subject"]
    for col in preferred:
        if col in df_311.columns:
            return col
    raise ValueError("No suitable text column found in 311 dataframe.")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    in_311 = os.path.join(project_root, "data", "processed", "311_enriched.csv")
    in_yelp = os.path.join(project_root, "data", "processed", "yelp_business_cleaned.csv")

    out_path = os.path.join(project_root, "data", "processed", "311_yelp_hybrid_integrated_enriched.csv")
    out_matched_path = os.path.join(project_root, "data", "processed", "311_yelp_hybrid_integrated_enriched_matched.csv")

    df311 = pd.read_csv(in_311, low_memory=False)
    yelp = pd.read_csv(in_yelp, low_memory=False)

    # Validate Yelp columns
    required_yelp = {"business_id", "latitude", "longitude", "categories"}
    missing_yelp = required_yelp - set(yelp.columns)
    if missing_yelp:
        raise ValueError(f"Yelp file missing columns: {missing_yelp}")

    # Identify lat/lon columns in 311
    lat_col_311 = "latitude" if "latitude" in df311.columns else None
    lon_col_311 = "longitude" if "longitude" in df311.columns else None
    if lat_col_311 is None or lon_col_311 is None:
        for lat_c, lon_c in [("lat", "lon"), ("LATITUDE", "LONGITUDE"), ("Latitude", "Longitude")]:
            if lat_c in df311.columns and lon_c in df311.columns:
                lat_col_311, lon_col_311 = lat_c, lon_c
                break
    if lat_col_311 is None or lon_col_311 is None:
        raise ValueError("311 file missing latitude/longitude columns.")

    # Choose which 311 column becomes the matching text input
    text_col_311 = pick_311_text_col(df311)

    # Coerce coordinates to numeric
    df311 = df311.copy()
    yelp = yelp.copy()
    df311[lat_col_311] = pd.to_numeric(df311[lat_col_311], errors="coerce")
    df311[lon_col_311] = pd.to_numeric(df311[lon_col_311], errors="coerce")
    yelp["latitude"] = pd.to_numeric(yelp["latitude"], errors="coerce")
    yelp["longitude"] = pd.to_numeric(yelp["longitude"], errors="coerce")

    # Drop rows without coordinates BEFORE integration and keep them OUT of output
    df311_valid = df311.dropna(subset=[lat_col_311, lon_col_311]).reset_index(drop=True)
    yelp_valid = yelp.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    if yelp_valid.empty:
        raise ValueError("No Yelp rows with valid coordinates. Cannot run hybrid integration.")
    if df311_valid.empty:
        raise ValueError("No 311 rows with valid coordinates. Cannot run hybrid integration.")

    # Normalize text fields
    df311_valid["__text311__"] = df311_valid[text_col_311].map(normalize_text)
    yelp_valid["__cats__"] = yelp_valid["categories"].map(normalize_text)

    # Build BallTree using haversine on radians
    # BallTree expects points as [lat, lon] in radians for haversine metric
    yelp_points_rad = np.deg2rad(yelp_valid[["latitude", "longitude"]].to_numpy())
    tree = BallTree(yelp_points_rad, metric="haversine")

    radius_rad = RADIUS_METERS / EARTH_RADIUS_M

    matched_ids = [None] * len(df311_valid)
    matched_scores = [np.nan] * len(df311_valid)
    matched_dist_m = [np.nan] * len(df311_valid)

    for i, row in df311_valid.iterrows():
        query_text = row["__text311__"]
        if not isinstance(query_text, str) or not query_text.strip():
            continue

        lat = float(row[lat_col_311])
        lon = float(row[lon_col_311])

        q_rad = np.deg2rad([[lat, lon]])

        # Find Yelp neighbors within radius using BallTree
        ind, dist = tree.query_radius(q_rad, r=radius_rad, return_distance=True)

        idxs = ind[0]
        dists_rad = dist[0]

        if idxs.size == 0:
            continue

        # Convert distances to meters
        dists_m = dists_rad * EARTH_RADIUS_M

        # Limit to TOPK_NEAREST by distance (keeps it basic and faster)
        if idxs.size > TOPK_NEAREST:
            order = np.argsort(dists_m)[:TOPK_NEAREST]
            idxs = idxs[order]
            dists_m = dists_m[order]

        candidates = yelp_valid.iloc[idxs].copy()
        candidates["__dist_m__"] = dists_m

        # Build corpus: first item is query, rest are candidate category strings
        corpus = [query_text] + candidates["__cats__"].tolist()

        # Remove empties to avoid empty vocabulary errors
        corpus = [t for t in corpus if isinstance(t, str) and t.strip()]
        if len(corpus) < 2:
            continue

        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        try:
            X = vec.fit_transform(corpus)
        except ValueError:
            continue

        sims = linear_kernel(X[0:1], X[1:]).ravel()
        best_j = int(np.argmax(sims))
        best_sim = float(sims[best_j])

        if best_sim < MIN_SIMILARITY:
            continue

        best_business_id = candidates.iloc[best_j]["business_id"]
        best_dist = float(candidates.iloc[best_j]["__dist_m__"])

        matched_ids[i] = best_business_id
        matched_scores[i] = best_sim
        matched_dist_m[i] = best_dist

    # Attach match info to ONLY valid-coordinate rows
    out = df311_valid.copy()
    out["matched_business_id"] = matched_ids
    out["match_score"] = matched_scores
    out["match_distance_m"] = matched_dist_m

    # Add Yelp fields for matched businesses
    yelp_cols_to_add = [
        "business_id", "name", "address", "city", "state", "zipcode",
        "stars", "review_count", "is_open", "categories"
    ]
    yelp_cols_to_add = [c for c in yelp_cols_to_add if c in yelp.columns]

    out = out.merge(
        yelp[yelp_cols_to_add],
        left_on="matched_business_id",
        right_on="business_id",
        how="left",
        suffixes=("", "_yelp"),
    ).drop(columns=["business_id"], errors="ignore")

    # Save full output (valid coords only)
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)

    # Save matched-only output
    matched = out[out["matched_business_id"].notna()].copy()
    matched.to_csv(out_matched_path, index=False)
    print("Saved matched:", out_matched_path)

    print("Rows used (valid coords):", len(out))
    print("Matched rows:", len(matched))
    print("Match rate:", out["matched_business_id"].notna().mean())


if __name__ == "__main__":
    main()