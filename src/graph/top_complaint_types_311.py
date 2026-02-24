import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/311_cleaned.csv")

# Count complaint types
counts = df["complaint_type"].value_counts()

# REMOVE "other"
counts = counts[counts.index != "other"]

# Take top 10
top10 = counts.head(10)

# Plot
plt.figure(figsize=(10, 6))
top10.sort_values().plot(kind="barh")

plt.title("Top Complaint Types (Excluding 'Other')")
plt.xlabel("Count")
plt.ylabel("Complaint Type")

plt.tight_layout()
plt.savefig("data/graphs/top_complaints_311.png")
plt.show()