"""
Complaint Patterns Near Different Business Categories
CS 4/5630 - Project 1
Analyzes differences in 311 complaint types near Restaurants, Retail,
and Services using the hybrid-integrated dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MERGED_PATH = "/Users/reesehannam1/Downloads/CS 4360/Project 1/Project1_DataScience/data/processed/311_yelp_hybrid_integrated.csv"
OUTPUT_DIR  = "/Users/reesehannam1/Downloads/CS 4360/Project 1/Project1_DataScience/data/graphs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# How many top complaint types / business categories to show in charts
TOP_N_COMPLAINTS  = 10
TOP_N_CATEGORIES  = 3   # Restaurants, Retail, Services

# Mapping of raw Yelp category keywords → broad group
# Extend this if your data has other common categories
CATEGORY_MAP = {
    "Restaurants": [
        "restaurant", "food", "pizza", "sushi", "burger", "cafe", "coffee",
        "diner", "bbq", "bakery", "bar", "pub", "bistro", "grill", "deli",
        "seafood", "steakhouse", "taco", "sandwich", "noodle", "buffet"
    ],
    "Retail": [
        "retail", "shop", "store", "boutique", "clothing", "fashion", "grocery",
        "supermarket", "market", "hardware", "furniture", "electronics",
        "pharmacy", "drugstore", "convenience", "florist", "jewelry", "books"
    ],
    "Services": [
        "service", "salon", "spa", "gym", "fitness", "laundry", "dry clean",
        "repair", "auto", "mechanic", "bank", "financial", "insurance",
        "medical", "dental", "doctor", "clinic", "health", "hotel", "motel",
        "realty", "real estate", "law", "attorney", "accountant", "daycare"
    ],
}


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data(path):
    print("Loading merged dataset...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Rows: {len(df):,}  |  Columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────
# 2. FIND RELEVANT COLUMNS (flexible naming)
# ─────────────────────────────────────────────
def find_column(df, candidates, label):
    for col in candidates:
        if col in df.columns:
            print(f"  Using '{col}' as {label} column.")
            return col
    print(f"  WARNING: No {label} column found. Tried: {candidates}")
    print(f"           Available columns: {list(df.columns)}")
    return None


def detect_columns(df):
    complaint_col = find_column(df,
        ["complaint_type", "service_name", "subject", "type", "complaint",
         "issue_type", "category", "request_type"],
        "complaint type"
    )
    category_col = find_column(df,
        ["categories", "business_categories", "yelp_categories",
         "category", "biz_category", "business_type"],
        "business category"
    )
    return complaint_col, category_col


# ─────────────────────────────────────────────
# 3. MAP RAW YELP CATEGORIES → BROAD GROUPS
# ─────────────────────────────────────────────
def assign_broad_category(raw_text, category_map):
    """Return the first matching broad category, or 'Other'."""
    if pd.isna(raw_text):
        return "Other"
    text = str(raw_text).lower()
    for broad, keywords in category_map.items():
        if any(kw in text for kw in keywords):
            return broad
    return "Other"


def map_categories(df, category_col):
    print("\nMapping business categories to broad groups...")
    df = df.copy()
    df["broad_category"] = df[category_col].apply(
        lambda x: assign_broad_category(x, CATEGORY_MAP)
    )
    counts = df["broad_category"].value_counts()
    print(f"  Category distribution:\n{counts.to_string()}")
    return df


# ─────────────────────────────────────────────
# 4. CLEAN COMPLAINT TYPE COLUMN
# ─────────────────────────────────────────────
def clean_complaints(df, complaint_col):
    df = df.copy()
    df[complaint_col] = (
        df[complaint_col]
        .astype(str)
        .str.strip()
        .str.title()
    )
    df = df[df[complaint_col].notna() & (df[complaint_col] != "Nan")]
    return df


# ─────────────────────────────────────────────
# 5. FILTER TO MAIN 3 CATEGORIES
# ─────────────────────────────────────────────
def filter_main_categories(df):
    main_cats = list(CATEGORY_MAP.keys())   # Restaurants, Retail, Services
    filtered = df[df["broad_category"].isin(main_cats)]
    print(f"\n  Rows with Restaurants/Retail/Services: {len(filtered):,}")
    return filtered


# ─────────────────────────────────────────────
# 6. VISUALIZATIONS
# ─────────────────────────────────────────────

def plot_bar_chart(df, complaint_col, output_dir, top_n=TOP_N_COMPLAINTS):
    """Grouped bar chart: top complaint types per broad business category."""
    main_cats = list(CATEGORY_MAP.keys())

    # Get global top N complaint types across all categories
    top_complaints = (
        df[df["broad_category"].isin(main_cats)][complaint_col]
        .value_counts()
        .head(top_n)
        .index.tolist()
    )

    plot_data = (
        df[df["broad_category"].isin(main_cats) & df[complaint_col].isin(top_complaints)]
        .groupby(["broad_category", complaint_col])
        .size()
        .reset_index(name="count")
    )

    pivot = plot_data.pivot(index=complaint_col, columns="broad_category", values="count").fillna(0)
    # Reorder columns
    pivot = pivot[[c for c in main_cats if c in pivot.columns]]
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(pivot))
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for i, (cat, color) in enumerate(zip(pivot.columns, colors)):
        ax.bar(x + i * width, pivot[cat], width, label=cat, color=color, edgecolor="white", linewidth=0.6)

    ax.set_xticks(x + width)
    ax.set_xticklabels(pivot.index, rotation=40, ha="right", fontsize=9)
    ax.set_xlabel("Complaint Type", fontsize=11)
    ax.set_ylabel("Number of Complaints", fontsize=11)
    ax.set_title(f"Top {top_n} Complaint Types Near Each Business Category", fontsize=13, fontweight="bold")
    ax.legend(title="Business Category", fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()

    path = os.path.join(output_dir, "bar_complaints_by_category.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_heatmap(df, complaint_col, output_dir, top_n=TOP_N_COMPLAINTS):
    """Heatmap: complaint type (rows) vs business category (cols), normalized by column."""
    main_cats = list(CATEGORY_MAP.keys())

    top_complaints = (
        df[df["broad_category"].isin(main_cats)][complaint_col]
        .value_counts()
        .head(top_n)
        .index.tolist()
    )

    pivot = (
        df[df["broad_category"].isin(main_cats) & df[complaint_col].isin(top_complaints)]
        .groupby([complaint_col, "broad_category"])
        .size()
        .unstack(fill_value=0)
    )
    pivot = pivot[[c for c in main_cats if c in pivot.columns]]

    # Normalize each column so categories are comparable despite size differences
    pivot_norm = pivot.div(pivot.sum(axis=0), axis=1) * 100

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        pivot_norm,
        annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.5, linecolor="white",
        ax=ax, cbar_kws={"label": "% of complaints in category"}
    )
    ax.set_title(f"Complaint Type Distribution by Business Category\n(% within each category)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Business Category", fontsize=11)
    ax.set_ylabel("Complaint Type", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()

    path = os.path.join(output_dir, "heatmap_complaints_by_category.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_top_per_category(df, complaint_col, output_dir, top_n=8):
    """Side-by-side horizontal bar charts: top complaints for each category."""
    main_cats = list(CATEGORY_MAP.keys())
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, len(main_cats), figsize=(15, 6), sharey=False)

    for ax, cat, color in zip(axes, main_cats, colors):
        subset = df[df["broad_category"] == cat][complaint_col].value_counts().head(top_n)
        ax.barh(subset.index[::-1], subset.values[::-1], color=color, edgecolor="white")
        ax.set_title(cat, fontsize=12, fontweight="bold", color=color)
        ax.set_xlabel("Complaint Count", fontsize=9)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.tick_params(axis="y", labelsize=8)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle(f"Top {top_n} Complaints Near Each Business Category", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(output_dir, "top_complaints_per_category.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# 7. SAVE SUMMARY TABLE
# ─────────────────────────────────────────────
def save_summary(df, complaint_col, output_dir):
    main_cats = list(CATEGORY_MAP.keys())
    summary = (
        df[df["broad_category"].isin(main_cats)]
        .groupby(["broad_category", complaint_col])
        .size()
        .reset_index(name="count")
        .sort_values(["broad_category", "count"], ascending=[True, False])
    )
    path = os.path.join(output_dir, "complaint_patterns_by_category.csv")
    summary.to_csv(path, index=False)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    df = load_data(MERGED_PATH)

    complaint_col, category_col = detect_columns(df)

    if complaint_col is None or category_col is None:
        print("\nERROR: Could not find required columns. "
              "Please check the column names printed above and update the "
              "candidates lists in detect_columns().")
        return

    df = clean_complaints(df, complaint_col)
    df = map_categories(df, category_col)
    df = filter_main_categories(df)

    if len(df) < 20:
        print("\nWARNING: Very few rows matched. Check that category column "
              "contains recognizable Yelp category text.")
        return

    print("\nGenerating visualizations...")
    plot_bar_chart(df, complaint_col, OUTPUT_DIR)
    plot_heatmap(df, complaint_col, OUTPUT_DIR)
    plot_top_per_category(df, complaint_col, OUTPUT_DIR)
    save_summary(df, complaint_col, OUTPUT_DIR)

    print("\nDone! All outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()