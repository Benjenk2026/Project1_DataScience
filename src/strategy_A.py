"""
strategy_A.py
Phase 3 - Strategy A (Geospatial Integration)
CS 4/5630 Project 1

Geospatial integration:
- Match each 311 complaint to the nearest Yelp business within a radius (meters)
- Uses BallTree with haversine distance for speed

Key behavior:
- Drops 311 rows WITHOUT valid coordinates from the final output
- Drops Yelp rows WITHOUT valid coordinates
- Writes TWO outputs:
  - Full integrated file (only rows with valid coordinates)
  - Matched-only file (subset where a match exists)

Run:
  python src/strategy_A.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# CONFIG
RADIUS_METERS = 300
EARTH_RADIUS_M = 6_371_000  # meters


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    in_311 = os.path.join(project_root, "data", "processed", "311_cleaned.csv")
    in_yelp = os.path.join(project_root, "data", "processed", "yelp_business_cleaned.csv")

    out_path = os.path.join(project_root, "data", "processed", "311_yelp_strategyA_integrated.csv")
    out_matched_path = os.path.join(project_root, "data", "processed", "311_yelp_strategyA_integrated_matched.csv")

    # Load data
    df311 = pd.read_csv(in_311, low_memory=False)
    yelp = pd.read_csv(in_yelp, low_memory=False)

    # Coerce coords to numeric (in case they loaded as strings)
    df311["latitude"] = pd.to_numeric(df311.get("latitude"), errors="coerce")
    df311["longitude"] = pd.to_numeric(df311.get("longitude"), errors="coerce")
    yelp["latitude"] = pd.to_numeric(yelp.get("latitude"), errors="coerce")
    yelp["longitude"] = pd.to_numeric(yelp.get("longitude"), errors="coerce")

    # Drop rows without valid coordinates AND keep them out of output
    df311_valid = df311.dropna(subset=["latitude", "longitude"]).copy().reset_index(drop=True)
    yelp_valid = yelp.dropna(subset=["latitude", "longitude"]).copy().reset_index(drop=True)

    if df311_valid.empty:
        raise ValueError("No 311 rows with valid coordinates. Nothing to match.")
    if yelp_valid.empty:
        raise ValueError("No Yelp rows with valid coordinates. Cannot match.")

    # Convert to radians (BallTree haversine expects radians)
    df311_coords = np.radians(df311_valid[["latitude", "longitude"]].to_numpy())
    yelp_coords = np.radians(yelp_valid[["latitude", "longitude"]].to_numpy())

    # Build BallTree
    tree = BallTree(yelp_coords, metric="haversine")

    # Radius in radians
    radius_rad = RADIUS_METERS / EARTH_RADIUS_M

    # For each 311 point, get all yelp neighbors within radius
    neighbors = tree.query_radius(df311_coords, r=radius_rad, return_distance=True, sort_results=True)

    # Unpack results
    ind_array, dist_array = neighbors  # both are object arrays: one entry per 311 row

    matched_ids = [None] * len(df311_valid)
    matched_dist_m = [np.nan] * len(df311_valid)

    for i in range(len(df311_valid)):
        inds = ind_array[i]
        dists = dist_array[i]

        if len(inds) == 0:
            continue

        # Closest neighbor (sorted because sort_results=True)
        best_idx = int(inds[0])
        best_dist_m = float(dists[0]) * EARTH_RADIUS_M

        matched_ids[i] = yelp_valid.iloc[best_idx]["business_id"]
        matched_dist_m[i] = best_dist_m

    # Attach match columns
    df311_valid["matched_business_id"] = matched_ids
    df311_valid["match_distance_m"] = matched_dist_m

    # Merge Yelp info
    yelp_cols = [
        "business_id", "name", "address", "city",
        "state", "zipcode", "stars", "review_count", "is_open", "categories"
    ]
    yelp_cols = [c for c in yelp_cols if c in yelp_valid.columns]

    out = df311_valid.merge(
        yelp_valid[yelp_cols],
        left_on="matched_business_id",
        right_on="business_id",
        how="left",
        suffixes=("", "_yelp"),
    ).drop(columns=["business_id"], errors="ignore")

    # Save full file (valid coords only)
    out.to_csv(out_path, index=False)
    print("Saved full:", out_path)

    # Save matched only
    matched = out[out["matched_business_id"].notna()].copy()
    matched.to_csv(out_matched_path, index=False)

    print("Saved matched:", out_matched_path)
    print("Rows used (valid coords):", len(out))
    print("Matched rows:", out["matched_business_id"].notna().sum())
    print("Match rate:", out["matched_business_id"].notna().mean())


if __name__ == "__main__":
    main()