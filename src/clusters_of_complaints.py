"""
Complaint Type Clustering
=========================
Builds complaint-type clusters from the hybrid integrated datasets using
K-means clustering and saves a presentation-ready cluster graph.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


def resolve_writable_path(path: Path) -> Path:
	try:
		path.parent.mkdir(parents=True, exist_ok=True)
		with open(path, "a", encoding="utf-8"):
			pass
		return path
	except PermissionError:
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		fallback = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
		print(f"Warning: {path} is locked. Writing to {fallback} instead.")
		return fallback


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Cluster complaint types with K-means.")
	parser.add_argument(
		"--source",
		choices=["enriched", "matched", "both"],
		default="both",
		help="Input dataset source to use.",
	)
	parser.add_argument(
		"--clusters",
		type=int,
		default=8,
		help="Number of clusters (default: 8).",
	)
	parser.add_argument(
		"--sample-size",
		type=int,
		default=40000,
		help="Maximum rows to cluster for speed/memory (default: 40000).",
	)
	parser.add_argument(
		"--data-dir",
		default="data/processed",
		help="Directory containing processed input CSV files.",
	)
	parser.add_argument(
		"--output-dir",
		default="analysis_findings",
		help="Directory to save cluster outputs.",
	)
	parser.add_argument(
		"--random-state",
		type=int,
		default=42,
		help="Random seed for sampling and k-means.",
	)
	return parser.parse_args()


def load_data(source: str, data_dir: str) -> pd.DataFrame:
	data_dir = Path(data_dir)
	source_files = {
		"enriched": data_dir / "311_yelp_hybrid_integrated_enriched.csv",
		"matched": data_dir / "311_yelp_hybrid_integrated_enriched_matched.csv",
	}

	requested = ["enriched", "matched"] if source == "both" else [source]
	frames: list[pd.DataFrame] = []

	for src in requested:
		path = source_files[src]
		if not path.exists():
			print(f"Warning: missing file for {src}: {path}")
			continue
		df_src = pd.read_csv(path, low_memory=False)
		df_src["source_dataset"] = src
		print(f"Loaded {len(df_src):,} rows from {path}")
		frames.append(df_src)

	if not frames:
		expected = ", ".join(str(source_files[s]) for s in requested)
		raise FileNotFoundError(f"No input data found. Expected one or more of: {expected}")

	if len(frames) == 1:
		return frames[0]

	combined = pd.concat(frames, ignore_index=True, sort=False)
	print(f"Combined rows: {len(combined):,}")
	return combined


def prepare_text_for_clustering(df: pd.DataFrame) -> pd.DataFrame:
	text_columns = ["complaint_type", "predicted_category", "complaint_summary", "__text311__"]
	present = [col for col in text_columns if col in df.columns]
	if not present:
		raise ValueError("No complaint text columns found for clustering.")

	prepared = df.copy()
	for col in present:
		prepared[col] = prepared[col].fillna("").astype(str)

	prepared["cluster_text"] = (
		prepared.get("complaint_type", "")
		+ " "
		+ prepared.get("predicted_category", "")
		+ " "
		+ prepared.get("complaint_summary", "")
		+ " "
		+ prepared.get("__text311__", "")
	).str.replace(r"\s+", " ", regex=True).str.strip()

	prepared = prepared[prepared["cluster_text"].str.len() > 0].copy()
	if prepared.empty:
		raise ValueError("All clustering text is empty after preprocessing.")

	return prepared


def maybe_sample(df: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
	if sample_size <= 0 or len(df) <= sample_size:
		return df
	return df.sample(n=sample_size, random_state=random_state)


def fit_vectorizer(text: pd.Series):
	vectorizer = TfidfVectorizer(
		max_features=7000,
		ngram_range=(1, 2),
		min_df=5,
		stop_words="english",
		sublinear_tf=True,
	)
	matrix = vectorizer.fit_transform(text)
	return vectorizer, matrix


def cluster_records(matrix, n_clusters: int, random_state: int):
	if n_clusters < 2:
		raise ValueError("--clusters must be at least 2")

	model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
	labels = model.fit_predict(matrix)
	return model, labels


def top_terms_per_cluster(vectorizer: TfidfVectorizer, matrix, labels: np.ndarray, top_n: int = 10) -> pd.DataFrame:
	terms = np.array(vectorizer.get_feature_names_out())
	rows = []
	for cluster_id in sorted(np.unique(labels)):
		cluster_mask = labels == cluster_id
		cluster_mean = np.asarray(matrix[cluster_mask].mean(axis=0)).ravel()
		top_idx = np.argsort(cluster_mean)[-top_n:][::-1]
		rows.append(
			{
				"cluster_id": int(cluster_id),
				"size": int(cluster_mask.sum()),
				"top_terms": ", ".join(terms[top_idx]),
			}
		)
	return pd.DataFrame(rows)


def save_cluster_plot(profiles: pd.DataFrame, out_path: Path, title: str, total_rows: int) -> Path:
	plot_df = profiles.sort_values("size", ascending=True).copy()
	plot_df["share_pct"] = (plot_df["size"] / total_rows) * 100
	plot_df["cluster_label"] = plot_df["cluster_id"].apply(lambda cid: f"Cluster {int(cid)}")

	fig, ax = plt.subplots(figsize=(13, 8), dpi=180)
	colors = plt.cm.tab20(np.linspace(0, 1, len(plot_df)))
	bars = ax.barh(plot_df["cluster_label"], plot_df["size"], color=colors, alpha=0.9)

	ax.set_title(title, fontsize=16, fontweight="bold")
	ax.set_xlabel("Number of Complaint Records")
	ax.set_ylabel("K-means Cluster")
	ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.4)
	ax.set_axisbelow(True)

	max_size = float(plot_df["size"].max()) if not plot_df.empty else 1.0
	ax.set_xlim(0, max_size * 1.35)

	for bar, (_, row) in zip(bars, plot_df.iterrows()):
		x = bar.get_width()
		y = bar.get_y() + bar.get_height() / 2
		ax.text(
			x + max_size * 0.01,
			y,
			f"{int(row['size']):,} ({row['share_pct']:.1f}%)",
			va="center",
			fontsize=9,
		)

	fig.text(
		0.01,
		0.01,
		f"Total clustered records: {total_rows:,}",
		fontsize=9,
		alpha=0.85,
	)

	fig.tight_layout(rect=[0, 0.03, 1, 1])
	output_path = resolve_writable_path(out_path)
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)
	return output_path


def main() -> None:
	args = parse_args()
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	print("\n" + "=" * 60)
	print("COMPLAINT TYPE CLUSTERING")
	print("=" * 60)
	print(f"Source: {args.source}")
	print("Method: kmeans")
	print(f"Clusters: {args.clusters}")

	df = load_data(args.source, args.data_dir)
	prepared = prepare_text_for_clustering(df)
	sampled = maybe_sample(prepared, args.sample_size, args.random_state)
	sampled = sampled.reset_index(drop=True)
	print(f"Rows used for clustering: {len(sampled):,}")

	vectorizer, matrix = fit_vectorizer(sampled["cluster_text"])
	print(f"TF-IDF shape: {matrix.shape[0]:,} rows x {matrix.shape[1]:,} features")

	_, labels = cluster_records(matrix, args.clusters, args.random_state)
	sampled["cluster_id"] = labels

	if len(np.unique(labels)) > 1:
		sil = silhouette_score(matrix, labels, sample_size=min(10000, len(labels)), random_state=args.random_state)
		print(f"Silhouette score: {sil:.4f}")
	else:
		sil = np.nan
		print("Silhouette score skipped (single cluster).")

	plot_file = output_dir / f"complaint_clusters_kmeans_{args.source}.png"
	profile_file = output_dir / f"complaint_cluster_profiles_kmeans_{args.source}.csv"
	sample_file = output_dir / f"complaint_cluster_assignments_kmeans_{args.source}.csv"
	metrics_file = output_dir / f"complaint_cluster_metrics_kmeans_{args.source}.txt"

	profiles = top_terms_per_cluster(vectorizer, matrix, labels)
	title = f"Complaint Type Cluster Sizes (KMeans | Source: {args.source})"
	plot_file = save_cluster_plot(profiles, plot_file, title, total_rows=len(sampled))

	profile_file = resolve_writable_path(profile_file)
	profiles.to_csv(profile_file, index=False)

	keep_columns = [
		c
		for c in [
			"service_request_id",
			"complaint_type",
			"predicted_category",
			"complaint_summary",
			"source_dataset",
			"cluster_id",
		]
		if c in sampled.columns
	]
	sample_file = resolve_writable_path(sample_file)
	sampled[keep_columns].to_csv(sample_file, index=False)

	metrics_file = resolve_writable_path(metrics_file)
	with open(metrics_file, "w", encoding="utf-8") as f:
		f.write("Complaint Type Clustering Metrics\n")
		f.write("=" * 40 + "\n")
		f.write(f"source={args.source}\n")
		f.write("method=kmeans\n")
		f.write(f"clusters={args.clusters}\n")
		f.write(f"rows_used={len(sampled)}\n")
		f.write(f"tfidf_features={matrix.shape[1]}\n")
		f.write(f"silhouette={sil}\n")

	print("\nOutput files:")
	print(f"  - {plot_file}")
	print(f"  - {profile_file}")
	print(f"  - {sample_file}")
	print(f"  - {metrics_file}")
	print("=" * 60 + "\n")


if __name__ == "__main__":
	main()
