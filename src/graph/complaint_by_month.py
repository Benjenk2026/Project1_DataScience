import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

in_path = Path("data/Full/Processed/Phase 3/311_yelp_hybrid_integrated_enriched_matched.csv")
out_dir = Path("data/graphs")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(in_path, low_memory=False)
df["requested_datetime"] = pd.to_datetime(df["requested_datetime"], errors="coerce")
df = df.dropna(subset=["requested_datetime"])

# ── find the business category column ────────────────────────────────────────
cat_col_candidates = ["broad_category", "business_category", "broad_category_rule", "category", "categories"]
cat_col = next((c for c in cat_col_candidates if c in df.columns), None)

if cat_col is None:
    print(f"WARNING: No business category column found. Available: {list(df.columns)}")
    print("Falling back to overall complaints only.")

# ── map raw Yelp category strings to broad groups ─────────────────────────────
if cat_col == "categories":
    def map_cat(val):
        val = str(val).lower()
        if any(k in val for k in ["restaurant", "food", "pizza", "cafe", "bar", "dining"]):
            return "Restaurants"
        if any(k in val for k in ["retail", "shop", "store", "market", "clothing", "grocery"]):
            return "Retail"
        if any(k in val for k in ["service", "repair", "salon", "spa", "cleaning", "contractor"]):
            return "Services"
        if any(k in val for k in ["health", "medical", "doctor", "dental", "pharmacy", "fitness", "gym"]):
            return "Health"
        if any(k in val for k in ["auto", "car", "mechanic", "tire"]):
            return "Auto"
        return "Other"
    df["_broad"] = df[cat_col].apply(map_cat)
    cat_col = "_broad"

# ── overall monthly trend ─────────────────────────────────────────────────────
monthly_total = df.set_index("requested_datetime").resample("ME").size()

fig, ax = plt.subplots(figsize=(13, 7))

# Plot overall total as a thicker background line
ax.plot(monthly_total.index, monthly_total.values,
        color="black", linewidth=2.5, label="All Complaints", zorder=5)

# ── per-category lines ────────────────────────────────────────────────────────
if cat_col is not None:
    # Clean up category values
    df[cat_col] = df[cat_col].astype(str).str.strip().str.title()
    df[cat_col] = df[cat_col].replace({"Nan": "Other", "None": "Other", "": "Other"})

    categories = ["Restaurants", "Retail", "Services"]

    colors = plt.cm.tab10.colors
    for i, cat in enumerate(categories):
        subset = df[df[cat_col] == cat].set_index("requested_datetime")
        monthly_cat = subset.resample("ME").size()
        ax.plot(monthly_cat.index, monthly_cat.values,
                color=colors[i % len(colors)],
                linewidth=1.4, alpha=0.85,
                label=cat, linestyle="--")

ax.set_title("311 Complaints per Month by Business Category", fontsize=14, fontweight="bold")
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Number of Complaints", fontsize=12)
ax.legend(title="Business Category", fontsize=9, title_fontsize=10,
          loc="upper left", framealpha=0.9)
ax.grid(axis="y", linestyle="--", alpha=0.4)

ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
plt.tight_layout()
out_path = out_dir / "311_complaints_per_month.png"
plt.savefig(out_path, dpi=200)
plt.close()
print(f"Saved: {out_path}")