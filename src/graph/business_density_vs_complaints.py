"""
Business Density vs. Complaint Frequency Analysis
CS 4/5630 - Project 1
Analyzes the relationship between Yelp business density and 311 complaint
frequency, grouped by ZIP code.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import pearsonr, spearmanr
import os

# ─────────────────────────────────────────────
# CONFIG — update these paths before running
# ─────────────────────────────────────────────
COMPLAINTS_PATH = "data/processed/311_cleaned.csv"
BUSINESSES_PATH = "data/processed/yelp_business_cleaned.csv"
OUTPUT_DIR      = "data/graphs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data(complaints_path, businesses_path):
    print("Loading data...")
    complaints  = pd.read_csv(complaints_path, low_memory=False)
    businesses  = pd.read_csv(businesses_path, low_memory=False)
    print(f"  311 complaints : {len(complaints):,} rows, columns: {list(complaints.columns)}")
    print(f"  Yelp businesses: {len(businesses):,} rows, columns: {list(businesses.columns)}")
    return complaints, businesses


# ─────────────────────────────────────────────
# 2. FIND ZIP CODE COLUMN (flexible)
# ─────────────────────────────────────────────
def find_zip_column(df, candidates):
    """Return the first matching column name from candidates list."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


# ─────────────────────────────────────────────
# 3. CLEAN & NORMALIZE ZIP CODES
# ─────────────────────────────────────────────
def clean_zip(series):
    """Normalize ZIP codes to 5-digit strings, drop invalid ones."""
    s = series.astype(str).str.strip()
    s = s.str.extract(r'(\d{5})')[0]   # keep first 5-digit sequence
    return s


# ─────────────────────────────────────────────
# 4. COMPUTE COMPLAINT FREQUENCY PER ZIP
# ─────────────────────────────────────────────
def complaint_frequency(complaints):
    zip_candidates = ["zipcode", "zip_code", "zip", "postal_code", "incident_zip"]
    zip_col = find_zip_column(complaints, zip_candidates)

    if zip_col is None:
        raise ValueError(
            f"No ZIP column found in 311 data. Available columns: {list(complaints.columns)}"
        )

    print(f"\n  Using '{zip_col}' as ZIP column for complaints.")
    complaints = complaints.copy()
    complaints["zip"] = clean_zip(complaints[zip_col])
    complaints = complaints.dropna(subset=["zip"])

    freq = (
        complaints.groupby("zip")
        .size()
        .reset_index(name="complaint_count")
    )
    print(f"  Complaint ZIP codes found: {len(freq)}")
    return freq


# ─────────────────────────────────────────────
# 5. COMPUTE BUSINESS DENSITY PER ZIP
# ─────────────────────────────────────────────
def business_density(businesses):
    zip_candidates = ["postal_code", "zipcode", "zip_code", "zip"]
    zip_col = find_zip_column(businesses, zip_candidates)

    if zip_col is None:
        raise ValueError(
            f"No ZIP column found in Yelp data. Available columns: {list(businesses.columns)}"
        )

    print(f"  Using '{zip_col}' as ZIP column for businesses.")
    businesses = businesses.copy()
    businesses["zip"] = clean_zip(businesses[zip_col])
    businesses = businesses.dropna(subset=["zip"])

    density = (
        businesses.groupby("zip")
        .size()
        .reset_index(name="business_count")
    )
    print(f"  Business ZIP codes found: {len(density)}")
    return density


# ─────────────────────────────────────────────
# 6. IQR OUTLIER REMOVAL
# ─────────────────────────────────────────────
def remove_outliers_iqr(df, columns, multiplier=1.5):
    """Remove rows where any of the given columns fall outside IQR bounds."""
    mask = pd.Series([True] * len(df), index=df.index)
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        mask &= df[col].between(lower, upper)
        print(f"  [{col}] IQR bounds: {lower:.1f} – {upper:.1f}  "
              f"(removed {(~df[col].between(lower, upper)).sum()} ZIP codes)")
    cleaned = df[mask]
    print(f"  ZIP codes before outlier removal: {len(df)}  →  after: {len(cleaned)}")
    return cleaned


# ─────────────────────────────────────────────
# 7. MERGE & FILTER
# ─────────────────────────────────────────────
def merge_data(freq, density, min_complaints=5, min_businesses=1):
    merged = pd.merge(freq, density, on="zip", how="inner")
    before = len(merged)
    merged = merged[
        (merged["complaint_count"] >= min_complaints) &
        (merged["business_count"]  >= min_businesses)
    ]
    print(f"\n  ZIP codes after merge      : {before}")
    print(f"  ZIP codes after filtering  : {len(merged)}")
    return merged


# ─────────────────────────────────────────────
# 7. CORRELATION ANALYSIS
# ─────────────────────────────────────────────
def correlation_analysis(merged):
    pearson_r,  pearson_p  = pearsonr(merged["business_count"], merged["complaint_count"])
    spearman_r, spearman_p = spearmanr(merged["business_count"], merged["complaint_count"])

    print("\n── Correlation Results ──────────────────────────")
    print(f"  Pearson  r = {pearson_r:.4f},  p = {pearson_p:.4e}")
    print(f"  Spearman r = {spearman_r:.4f},  p = {spearman_p:.4e}")

    if abs(pearson_r) >= 0.7:
        strength = "strong"
    elif abs(pearson_r) >= 0.4:
        strength = "moderate"
    else:
        strength = "weak"

    direction = "positive" if pearson_r > 0 else "negative"
    print(f"\n  Interpretation: {strength} {direction} linear correlation.")
    print("─────────────────────────────────────────────────")

    return {
        "pearson_r":  pearson_r,
        "pearson_p":  pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }


# ─────────────────────────────────────────────
# 8. VISUALIZATIONS
# ─────────────────────────────────────────────
def plot_scatter(merged, stats, output_dir):
    """Scatter plot with linear trend line."""
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        merged["business_count"],
        merged["complaint_count"],
        alpha=0.55, edgecolors="steelblue", facecolors="lightblue",
        s=60, linewidths=0.7
    )

    # Trend line
    m, b = np.polyfit(merged["business_count"], merged["complaint_count"], 1)
    x_line = np.linspace(merged["business_count"].min(), merged["business_count"].max(), 200)
    ax.plot(x_line, m * x_line + b, color="crimson", linewidth=1.8, label="Linear trend")

    ax.set_xlabel("Business Count per ZIP Code", fontsize=12)
    ax.set_ylabel("Complaint Count per ZIP Code", fontsize=12)
    ax.set_title("Business Density vs. Complaint Frequency by ZIP Code\n(Outliers removed via IQR method)", fontsize=13, fontweight="bold")

    annotation = (
        f"Pearson r = {stats['pearson_r']:.3f}  (p = {stats['pearson_p']:.2e})\n"
        f"Spearman r = {stats['spearman_r']:.3f}  (p = {stats['spearman_p']:.2e})"
    )
    ax.text(
        0.97, 0.05, annotation,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray", alpha=0.8)
    )

    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()

    path = os.path.join(output_dir, "scatter_density_vs_complaints.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Saved: {path}")


def plot_top_zips(merged, output_dir, top_n=20):
    """Bar chart: top ZIP codes by complaint count, colored by business density."""
    top = merged.nlargest(top_n, "complaint_count").copy()

    norm = plt.Normalize(top["business_count"].min(), top["business_count"].max())
    cmap = plt.cm.YlOrRd
    colors = cmap(norm(top["business_count"].values))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(top["zip"].astype(str), top["complaint_count"], color=colors, edgecolor="gray", linewidth=0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Business Count", fontsize=10)

    ax.set_xlabel("ZIP Code", fontsize=11)
    ax.set_ylabel("Complaint Count", fontsize=11)
    ax.set_title(f"Top {top_n} ZIP Codes by Complaint Count\n(Color = Business Density)", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()

    path = os.path.join(output_dir, "top_zips_complaints_density.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_density_bins(merged, output_dir, n_bins=5):
    """Box plot: complaint distribution across business density quantile bins."""
    merged = merged.copy()
    merged["density_bin"] = pd.qcut(
        merged["business_count"], q=n_bins,
        labels=[f"Q{i+1}" for i in range(n_bins)]
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    groups = [merged[merged["density_bin"] == q]["complaint_count"].values
              for q in merged["density_bin"].cat.categories]

    bp = ax.boxplot(groups, patch_artist=True, notch=False)
    colors_bp = plt.cm.Blues(np.linspace(0.3, 0.85, n_bins))
    for patch, color in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(color)

    ax.set_xticklabels(
        [f"Q{i+1}\n(Low→High)" if i == 0 else f"Q{i+1}" for i in range(n_bins)],
        fontsize=10
    )
    ax.set_xlabel("Business Density Quantile (Q1=Lowest, Q5=Highest)", fontsize=11)
    ax.set_ylabel("Complaint Count", fontsize=11)
    ax.set_title("Complaint Frequency Across Business Density Quantiles", fontsize=12, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "boxplot_density_bins.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# 9. SAVE SUMMARY TABLE
# ─────────────────────────────────────────────
def save_summary(merged, output_dir):
    path = os.path.join(output_dir, "density_complaint_summary.csv")
    merged.sort_values("complaint_count", ascending=False).to_csv(path, index=False)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    complaints, businesses = load_data(COMPLAINTS_PATH, BUSINESSES_PATH)

    freq    = complaint_frequency(complaints)
    density = business_density(businesses)
    merged  = merge_data(freq, density)

    print("\nRemoving outliers using IQR method (both axes)...")
    merged = remove_outliers_iqr(merged, columns=["complaint_count", "business_count"])

    if len(merged) < 10:
        print("\nWARNING: Very few ZIP codes matched. Check that ZIP columns are correct.")

    stats = correlation_analysis(merged)

    print("\nGenerating visualizations...")
    plot_scatter(merged, stats, OUTPUT_DIR)
    plot_top_zips(merged, OUTPUT_DIR)
    plot_density_bins(merged, OUTPUT_DIR)
    save_summary(merged, OUTPUT_DIR)

    print("\nDone! All outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()