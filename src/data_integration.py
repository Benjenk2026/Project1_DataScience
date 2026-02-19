# src/data_integration.py
from pathlib import Path
import numpy as np
import pandas as pd

from .cleaning import openfile, standardize_data, handle_missing_values, deduplicate_records

EARTH_RADIUS_M = 6371000.0

def haversine_m(lat1, lon1, lat2, lon2):
    # expects numpy arrays in degrees
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))

def clean_coords(df, lat_col="latitude", lon_col="longitude"):
    df = df.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df[df[lat_col].between(-90, 90) & df[lon_col].between(-180, 180)]
    return df

def integrate_geospatial(
    path_311="data/raw/test_data.xlsx",
    path_yelp="data/raw/yelp_JSON_test.json",
    radius_m=300
):
    # Load + standardize
    df311_raw = openfile(path_311)
    dfy_raw = openfile(path_yelp)

    df311 = standardize_data(df311_raw, save=False)
    dfy = standardize_data(dfy_raw, save=False)

    # Ensure column names match what we expect (your doc shows lat/lon for 311) :contentReference[oaicite:3]{index=3}
    # cleaning.standardize_data already converts lat/lon -> latitude/longitude :contentReference[oaicite:4]{index=4}

    # Basic missing handling
    df311, _ = handle_missing_values(df311, drop_rows_subset=["latitude", "longitude"])
    dfy, _ = handle_missing_values(dfy, drop_rows_subset=["latitude", "longitude", "business_id"])

    # Optional: dedupe
    if "service_request_id" in df311.columns:
        df311, _ = deduplicate_records(df311, subset=["service_request_id"], keep="most_complete")
    if "business_id" in dfy.columns:
        dfy, _ = deduplicate_records(dfy, subset=["business_id"], keep="most_complete")

    # Coordinate validation
    df311 = clean_coords(df311)
    dfy = clean_coords(dfy)

    # Prepare output rows
    y_lat = dfy["latitude"].to_numpy()
    y_lon = dfy["longitude"].to_numpy()

    matched_rows = []
    for i, r in df311.iterrows():
        lat = r["latitude"]
        lon = r["longitude"]
        dists = haversine_m(lat, lon, y_lat, y_lon)

        j = int(np.argmin(dists))
        best_dist = float(dists[j])

        if best_dist <= radius_m:
            y = dfy.iloc[j]
            out = {
                # 311 fields
                "service_request_id": r.get("service_request_id", np.nan),
                "subject": r.get("subject", np.nan),
                "status": r.get("status", np.nan),
                "service_name": r.get("service_name", np.nan),
                "requested_datetime": r.get("requested_datetime", np.nan),
                "zipcode": r.get("zipcode", r.get("postal_code", np.nan)),
                "complaint_latitude": lat,
                "complaint_longitude": lon,

                # Yelp fields
                "business_id": y.get("business_id", np.nan),
                "business_name": y.get("name", np.nan),
                "business_categories": y.get("categories", np.nan),
                "business_stars": y.get("stars", np.nan),
                "business_review_count": y.get("review_count", np.nan),
                "business_latitude": y.get("latitude", np.nan),
                "business_longitude": y.get("longitude", np.nan),

                # Join info
                "match_distance_m": best_dist,
                "radius_m": radius_m,
            }
            matched_rows.append(out)

    integrated = pd.DataFrame(matched_rows)

    # Save
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "integrated_dataset.csv"
    integrated.to_csv(out_path, index=False, encoding="utf-8")

    # Stats for report
    stats = {
        "311_rows_after_cleaning": int(len(df311)),
        "yelp_rows_after_cleaning": int(len(dfy)),
        "matched_rows": int(len(integrated)),
        "match_rate": float(len(integrated) / max(1, len(df311))),
        "radius_m": radius_m,
        "distance_mean_m": float(integrated["match_distance_m"].mean()) if len(integrated) else None,
        "distance_median_m": float(integrated["match_distance_m"].median()) if len(integrated) else None,
    }
    return integrated, stats

if __name__ == "__main__":
    integrated, stats = integrate_geospatial()
    print("Integration complete:", stats)