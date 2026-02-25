import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

in_path = Path("data/processed/311_cleaned.csv")
out_dir = Path("data/graphs")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(in_path, low_memory=False)
df["requested_datetime"] = pd.to_datetime(df["requested_datetime"], errors="coerce")
df = df.dropna(subset=["requested_datetime"])

monthly_counts = df.set_index("requested_datetime").resample("ME").size()

plt.figure(figsize=(12, 6))
monthly_counts.plot()

plt.title("311 Complaints per Month")
plt.xlabel("Month")
plt.ylabel("Number of Complaints")

plt.tight_layout()
plt.savefig(out_dir / "311_complaints_per_month.png", dpi=200)
plt.show()