
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os
MIN_SIMILARITY = 0.15

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

    in_311 = os.path.join(project_root, "data", "processed", "311_processed.csv")
    in_yelp = os.path.join(project_root, "data", "processed", "yelp_academic_dataset_business_fully_cleaned.csv")
    out_path = os.path.join(project_root, "data", "processed", "311_yelp_Similarity_integrated.csv")
    out_path2 = os.path.join(project_root, "data", "processed", "311_yelp_Similarity_integrated_Only_matches.csv")

    df311 = pd.read_csv(in_311)
    yelp = pd.read_csv(in_yelp)

    required_yelp = {"business_id", "categories"}
    missing_yelp = required_yelp - set(yelp.columns)
    if missing_yelp:
        raise ValueError(f"Yelp file missing columns: {missing_yelp}")
    yelp = yelp[yelp["city"].str.lower().str.strip() == "philadelphia"].copy()
    text_col_311 = pick_311_text_col(df311)

    df311 = df311.copy()
    yelp = yelp.copy()


    df311["__text311__"] = df311[text_col_311].map(normalize_text)
    yelp["__cats__"] = yelp["categories"].map(normalize_text)

   
    matched_ids = [None] * len(df311)
    matched_scores = [np.nan] * len(df311)

   # Fit TF-IDF ONLY on Yelp categories
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    yelp_matrix = vec.fit_transform(yelp["__cats__"])

    for i, row in df311.iterrows():

        query_text = row["__text311__"]

        if not query_text.strip():
            continue

        query_vec = vec.transform([query_text])

        sims = linear_kernel(query_vec, yelp_matrix).ravel()

        best_j = int(np.argmax(sims))
        best_sim = float(sims[best_j])

        if best_sim < MIN_SIMILARITY:
            continue

        matched_ids[i] = yelp.iloc[best_j]["business_id"]
        matched_scores[i] = best_sim
    df311["matched_business_id"] = matched_ids
    df311["match_score"] = matched_scores

    out = df311.copy()

    yelp_cols_to_add = ["business_id", "name", "address", "city", "state", "postal_code", "stars", "num_reviews", "is_open", "categories"]
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
    out = out[out["matched_business_id"].notna()].copy()
    out.to_csv(out_path2, index=False)

if __name__ == "__main__":  
    main()