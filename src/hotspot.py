"""
Analysis Module: Complaint Hotspot Visualization
=================================================
Visualizes 311 complaint densities by geographic location using interactive heatmaps
and density clustering to identify complaint hotspots in the city.
"""

import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
import os
import argparse
from pathlib import Path


def load_data(source="both", data_dir="data/processed"):
    """
    Load enriched analysis dataset(s) from processed data.
    
    Parameters
    ----------
    source : str
        Which source to load: '311', 'yelp', or 'both'
    data_dir : str
        Directory containing processed files
    
    Returns
    -------
    pd.DataFrame
        Loaded dataset with complaint coordinates and source labels
    """
    data_dir = Path(data_dir)
    source = source.lower()

    source_files = {
        "311": data_dir / "311_enriched.csv",
        "yelp": data_dir / "yelp_reviews_enriched.csv",
    }

    if source not in {"311", "yelp", "both"}:
        raise ValueError("source must be one of: '311', 'yelp', 'both'")

    requested_sources = ["311", "yelp"] if source == "both" else [source]
    frames = []

    for src in requested_sources:
        path = source_files[src]
        if not path.exists():
            print(f"Warning: {path} not found. Skipping {src} source.")
            continue
        df_src = pd.read_csv(path)
        df_src["source_dataset"] = src
        print(f"Loaded {len(df_src)} records from {path}")
        frames.append(df_src)

    if not frames:
        missing = ", ".join(str(source_files[s]) for s in requested_sources)
        raise FileNotFoundError(f"No input data found. Expected one or more of: {missing}")

    if len(frames) == 1:
        return frames[0]

    combined = pd.concat(frames, ignore_index=True, sort=False)
    print(f"Combined records: {len(combined)}")
    return combined


def extract_complaint_coordinates(df, lat_col="latitude", lon_col="longitude"):
    """
    Extract and validate complaint latitude/longitude coordinates.
    Removes records with missing or invalid coordinates.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with complaint records
    lat_col : str
        Column name for latitude (default: 'latitude')
    lon_col : str
        Column name for longitude (default: 'longitude')
    
    Returns
    -------
    pd.DataFrame
        Dataframe with valid coordinates only
    """
    # Identify the actual column names
    available_cols = df.columns.tolist()
    
    # Try to find latitude column
    if lat_col not in available_cols:
        lat_candidates = [c for c in available_cols if 'lat' in c.lower()]
        if lat_candidates:
            lat_col = lat_candidates[0]
            print(f"Using '{lat_col}' for latitude")
    
    # Try to find longitude column
    if lon_col not in available_cols:
        lon_candidates = [c for c in available_cols if 'lon' in c.lower()]
        if lon_candidates:
            lon_col = lon_candidates[0]
            print(f"Using '{lon_col}' for longitude")
    
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Coordinates columns not found. Available columns: {available_cols}")
    
    # Make a copy and ensure numeric types
    df = df.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    
    # Remove invalid coordinates
    invalid_count = df[[lat_col, lon_col]].isna().any(axis=1).sum()
    df_clean = df.dropna(subset=[lat_col, lon_col])
    
    # Validate geographic ranges
    df_clean = df_clean[
        (df_clean[lat_col].between(-90, 90)) & 
        (df_clean[lon_col].between(-180, 180))
    ]
    
    print(f"Extracted {len(df_clean)} valid coordinates ({invalid_count} records removed)")
    print(f"Latitude range: {df_clean[lat_col].min():.4f} to {df_clean[lat_col].max():.4f}")
    print(f"Longitude range: {df_clean[lon_col].min():.4f} to {df_clean[lon_col].max():.4f}")
    
    return df_clean, lat_col, lon_col


def create_density_heatmap(df, lat_col, lon_col, output_path="heatmap.html", 
                          radius=15, blur=25, max_zoom=13, title_prefix="Complaint"):
    """
    Create an interactive density heatmap of complaint locations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with complaint coordinates
    lat_col : str
        Latitude column name
    lon_col : str
        Longitude column name
    output_path : str
        Where to save the HTML map file
    radius : int
        Heatmap point radius in pixels
    blur : int
        Heatmap blur radius in pixels
    max_zoom : int
        Maximum heatmap intensity level (0-25)
    title_prefix : str
        Prefix shown in the map title (e.g., '311', 'Yelp Reviews', 'Combined')
    
    Returns
    -------
    folium.Map
        The generated folium map object
    """
    # Calculate map center
    center_lat = df[lat_col].mean()
    center_lon = df[lon_col].mean()
    
    # Create base map
    initial_zoom = 12
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=initial_zoom,
        tiles="OpenStreetMap"
    )
    
    # Prepare heatmap data: [[lat, lon, intensity], ...]
    # Use complaint density weight (normalize by count)
    heatmap_data = df[[lat_col, lon_col]].values.tolist()
    
    # Add heatmap layer
    HeatMap(
        heatmap_data,
        name="Complaint Density",
        radius=radius,
        blur=blur,
        max_zoom=max_zoom,
        min_opacity=0.3,
        gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add map title
    title_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 300px; height: 60px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:16px; font-weight: bold; padding: 10px;
                border-radius: 5px;">
        {title_prefix} Hotspots Heatmap
        <br/><small>Density-based visualization</small>
    </div>
    '''.format(title_prefix=title_prefix)
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save map
    m.save(output_path)
    print(f"\nHeatmap saved to {output_path}")
    
    return m


def create_cluster_map(df, lat_col, lon_col, output_path="complaint_clusters.html", title_prefix="Complaint"):
    """
    Create an interactive map with clustered complaint markers.
    Useful for identifying discrete complaint hotspot areas.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with complaint coordinates
    lat_col : str
        Latitude column name
    lon_col : str
        Longitude column name
    output_path : str
        Where to save the HTML map file
    title_prefix : str
        Prefix shown in the map title (e.g., '311', 'Yelp Reviews', 'Combined')
    
    Returns
    -------
    folium.Map
        The generated folium map object
    """
    # Calculate map center
    center_lat = df[lat_col].mean()
    center_lon = df[lon_col].mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="OpenStreetMap"
    )
    
    # Create marker cluster group
    marker_cluster = MarkerCluster(name="Complaint Clusters").add_to(m)
    
    # Add complaint markers
    for idx, row in df.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=4,
            popup=f"Complaint #{idx}",
            color="red",
            fill=True,
            fillColor="red",
            fillOpacity=0.6,
            weight=1
        ).add_to(marker_cluster)
    
    folium.LayerControl().add_to(m)
    
    # Add map title
    title_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 300px; height: 60px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:16px; font-weight: bold; padding: 10px;
                border-radius: 5px;">
        {title_prefix} Clusters
        <br/><small>Interactive cluster-based view</small>
    </div>
    '''.format(title_prefix=title_prefix)
    m.get_root().html.add_child(folium.Element(title_html))
    
    m.save(output_path)
    print(f"Cluster map saved to {output_path}")
    
    return m


def generate_hotspot_stats(df, lat_col, lon_col):
    """
    Generate summary statistics about complaint hotspots.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with complaint coordinates
    lat_col : str
        Latitude column name
    lon_col : str
        Longitude column name
    
    Returns
    -------
    dict
        Dictionary of hotspot statistics
    """
    stats = {
        "total_complaints": len(df),
        "center_lat": df[lat_col].mean(),
        "center_lon": df[lon_col].mean(),
        "lat_std": df[lat_col].std(),
        "lon_std": df[lon_col].std(),
        "lat_range": f"{df[lat_col].min():.4f} to {df[lat_col].max():.4f}",
        "lon_range": f"{df[lon_col].min():.4f} to {df[lon_col].max():.4f}",
    }
    
    print("\n" + "="*50)
    print("COMPLAINT HOTSPOT STATISTICS")
    print("="*50)
    print(f"Total Complaints: {stats['total_complaints']}")
    print(f"Center of Mass: ({stats['center_lat']:.4f}, {stats['center_lon']:.4f})")
    print(f"Latitude Std Dev: {stats['lat_std']:.4f}")
    print(f"Longitude Std Dev: {stats['lon_std']:.4f}")
    print(f"Latitude Range: {stats['lat_range']}")
    print(f"Longitude Range: {stats['lon_range']}")
    print("="*50)
    
    return stats

def business_density_and_complaint_frequency(df, lat_col, lon_col):
    """
    Calculate business density and complaint frequency for each business in the integrated dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Integrated dataset with business and complaint information
    lat_col : str
        Latitude column name
    lon_col : str
        Longitude column name
    
    Returns
    -------
    dict
        Dictionary of business density and complaint frequency statistics
    """
    if "business_id" not in df.columns:
        print("\nBusiness density analysis skipped: 'business_id' column not found.")
        return None

    # Count complaints per business_id (assuming each row is a complaint)
    complaint_counts = df.groupby("business_id").size()
    
    # Count businesses per location (density)
    business_counts = df.groupby([lat_col, lon_col]).size()
    
    stats = {
        "total_businesses": len(df["business_id"].unique()),
        "total_complaints": len(df),
        "complaints_per_business": complaint_counts.to_dict(),
        "business_density_per_location": business_counts.to_dict(),
        "average_complaints_per_business": float(complaint_counts.mean()) if len(complaint_counts) > 0 else 0,
        "max_complaints_per_business": int(complaint_counts.max()) if len(complaint_counts) > 0 else 0,
        "min_complaints_per_business": int(complaint_counts.min()) if len(complaint_counts) > 0 else 0,
    }
    
    print("\n" + "="*50)
    print("BUSINESS DENSITY AND COMPLAINT FREQUENCY")
    print("="*50)
    print(f"Total Businesses: {stats['total_businesses']}")
    print(f"Total Complaints: {stats['total_complaints']}")
    print(f"Average Complaints per Business: {stats['average_complaints_per_business']:.2f}")
    print(f"Max Complaints per Business: {stats['max_complaints_per_business']}")
    print(f"Min Complaints per Business: {stats['min_complaints_per_business']}")
    print("="*50)
    
    return stats


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Complaint hotspot analysis")
    parser.add_argument(
        "--source",
        choices=["311", "yelp", "both"],
        default="both",
        help="Input dataset source: 311_enriched, yelp_reviews_enriched, or both (default).",
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Directory containing processed input CSV files.",
    )
    return parser.parse_args()
    
def main():
    """
    Main workflow: Load data, extract coordinates, and create visualizations.
    """
    print("\n" + "="*50)
    print("COMPLAINT HOTSPOT ANALYSIS")
    print("="*50)

    args = parse_args()
    
    try:
        # Step 1: Load selected dataset(s)
        print(f"\n[1/5] Loading dataset source: {args.source}...")
        df = load_data(source=args.source, data_dir=args.data_dir)
        
        # Step 2: Extract valid coordinates
        print("\n[2/5] Extracting complaint coordinates...")
        df_coords, lat_col, lon_col = extract_complaint_coordinates(df)
        
        # Step 3: Generate statistics
        print("\n[3/5] Generating hotspot statistics...")
        stats = generate_hotspot_stats(df_coords, lat_col, lon_col)
        
        # Step 4: Business density and complaint frequency analysis
        print("\n[4/5] Analyzing business density and complaint frequency...")
        business_density_and_complaint_frequency(df, lat_col, lon_col)
        
        # Step 5: Create visualizations
        print("\n[5/5] Creating visualizations...")
        output_suffix = args.source.lower()
        title_lookup = {
            "311": "311 Complaint",
            "yelp": "Yelp Review",
            "both": "Combined",
        }
        title_prefix = title_lookup.get(output_suffix, "Complaint")
        heatmap_file = f"heatmap_{output_suffix}.html"
        cluster_file = f"complaint_clusters_{output_suffix}.html"

        create_density_heatmap(
            df_coords,
            lat_col,
            lon_col,
            output_path=heatmap_file,
            title_prefix=title_prefix,
        )
        create_cluster_map(
            df_coords,
            lat_col,
            lon_col,
            output_path=cluster_file,
            title_prefix=title_prefix,
        )
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print("Output files:")
        print(f"  - {heatmap_file} (density-based visualization)")
        print(f"  - {cluster_file} (cluster-based view)")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()