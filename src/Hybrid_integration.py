import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

RADIUS_METERS = 300
MIN_SIMILARITY = 0.15
TOPK_NEAREST = 80


def haversine_meters(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).lower().replace("&", " ").replace("/", " ").replace(",", " ")


def pick_311_text_col(df_311):
    preferred = ["service_name", "service_type", "complaint_type", "subject", "category", "request_type", "title"]
    for col in preferred:
        if col in df_311.columns:
            return col

    bad_keywords = ("id", "key", "zip", "postal", "phone", "email", "url", "address", "street")
    obj_cols = [c for c in df_311.columns if df_311[c].dtype == "object"]
    obj_cols = [c for c in obj_cols if not any(k in c.lower() for k in bad_keywords)]

    if not obj_cols:
        raise ValueError("No suitable text column found in 311 dataframe.")
    return obj_cols[0]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    in_311 = os.path.join(project_root, "data", "processed", "311_cleaned.csv")
    in_yelp = os.path.join(project_root, "data", "processed", "yelp_business_cleaned.csv")
    out_path = os.path.join(project_root, "data", "processed", "311_yelp_hybrid_integrated.csv")

    df311 = pd.read_csv(in_311)
    yelp = pd.read_csv(in_yelp)

    required_yelp = {"business_id", "latitude", "longitude", "categories"}
    missing_yelp = required_yelp - set(yelp.columns)
    if missing_yelp:
        raise ValueError(f"Yelp file missing columns: {missing_yelp}")

    lat_col_311 = "latitude" if "latitude" in df311.columns else None
    lon_col_311 = "longitude" if "longitude" in df311.columns else None

    if lat_col_311 is None or lon_col_311 is None:
        for lat_c, lon_c in [("lat", "lon"), ("LATITUDE", "LONGITUDE"), ("Latitude", "Longitude")]:
            if lat_c in df311.columns and lon_c in df311.columns:
                lat_col_311, lon_col_311 = lat_c, lon_c
                break

    if lat_col_311 is None or lon_col_311 is None:
        raise ValueError("311 file missing latitude/longitude columns.")

    text_col_311 = pick_311_text_col(df311)

    df311 = df311.copy()
    yelp = yelp.copy()

    df311[lat_col_311] = pd.to_numeric(df311[lat_col_311], errors="coerce")
    df311[lon_col_311] = pd.to_numeric(df311[lon_col_311], errors="coerce")
    yelp["latitude"] = pd.to_numeric(yelp["latitude"], errors="coerce")
    yelp["longitude"] = pd.to_numeric(yelp["longitude"], errors="coerce")

    df311["__text311__"] = df311[text_col_311].map(normalize_text)
    yelp["__cats__"] = yelp["categories"].map(normalize_text)

    df311_valid = df311.dropna(subset=[lat_col_311, lon_col_311]).reset_index(drop=False)
    yelp_valid = yelp.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    yelp_lat = yelp_valid["latitude"].to_numpy()
    yelp_lon = yelp_valid["longitude"].to_numpy()

    matched_ids = [None] * len(df311_valid)
    matched_scores = [np.nan] * len(df311_valid)
    matched_dist_m = [np.nan] * len(df311_valid)

    for i, row in df311_valid.iterrows():
        lat = float(row[lat_col_311])
        lon = float(row[lon_col_311])
        query_text = row["__text311__"]

        if not query_text.strip():
            continue

        dists = haversine_meters(lat, lon, yelp_lat, yelp_lon)
        within = np.where(dists <= RADIUS_METERS)[0]
        if within.size == 0:
            continue

        if within.size > TOPK_NEAREST:
            nearest_order = np.argsort(dists[within])[:TOPK_NEAREST]
            within = within[nearest_order]

        candidates = yelp_valid.iloc[within].copy()
        candidates["__dist__"] = dists[within]

        corpus = [query_text] + candidates["__cats__"].tolist()
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        X = vec.fit_transform(corpus)

        sims = linear_kernel(X[0:1], X[1:]).ravel()
        best_j = int(np.argmax(sims))
        best_sim = float(sims[best_j])

        if best_sim < MIN_SIMILARITY:
            continue

        best_business_id = candidates.iloc[best_j]["business_id"]
        best_dist = float(candidates.iloc[best_j]["__dist__"])

        matched_ids[i] = best_business_id
        matched_scores[i] = best_sim
        matched_dist_m[i] = best_dist

    df311_valid["matched_business_id"] = matched_ids
    df311_valid["match_score"] = matched_scores
    df311_valid["match_distance_m"] = matched_dist_m

    out = df311.merge(
        df311_valid[["index", "matched_business_id", "match_score", "match_distance_m"]],
        left_index=True,
        right_on="index",
        how="left",
    ).drop(columns=["index"])

    yelp_cols_to_add = ["business_id", "name", "address", "city", "state", "zipcode", "stars", "review_count", "is_open", "categories"]
    yelp_cols_to_add = [c for c in yelp_cols_to_add if c in yelp.columns]

    out = out.merge(
        yelp[yelp_cols_to_add],
        left_on="matched_business_id",
        right_on="business_id",
        how="left",
        suffixes=("", "_yelp"),
    ).drop(columns=["business_id"], errors="ignore")

    out.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("Match rate:", out["matched_business_id"].notna().mean())


if __name__ == "__main__":
    main()