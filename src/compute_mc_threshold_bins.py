""" compute_mc_threshold_bins.py """

import json
import re
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np


def compute_kmeans_bins(thresholds, n_bins=4):
    """
    Returns sorted bin edges using K-Means clustering on thresholds.
    """
    thresholds = thresholds.reshape(-1, 1)

    kmeans = KMeans(n_clusters=n_bins, n_init="auto").fit(thresholds)
    centroids = np.sort(kmeans.cluster_centers_.flatten())

    # Compute boundaries (midpoints)
    edges = []
    for i in range(len(centroids) - 1):
        edges.append((centroids[i] + centroids[i + 1]) / 2)

    # Add 0 and 1 limits
    edges = [0.0] + edges + [1.0]
    return edges


def parse_filename(filename):
    """
    Extracts prefix and view from filenames like:
    mc_dropout_threshold_results_genus_caud.csv

    Returns: taxonomic rank, view
        Example: ("genus", "caud")
    """
    match = re.match(
        r"mc_dropout_threshold_results_(\w+)_(\w+)\.csv", filename
    )
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename}")
    return match.group(1), match.group(2)


def main():
    """
    Reads in the MC Dropout Treshold results from the 8 models and computes the
    confidence bins using the compute_kmeans_bins function(High Confidence,
    Medium Confidence, Low Confidence, and Uncertain). It then saves these
    computed bins to json files based on the taxonomic level(Genus or Species).
    """
    input_dir = Path("mc_dropout_results")
    output_dir = Path("port_inspector/beetle_detection/model_data")
    output_dir.mkdir(exist_ok=True)
    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSVs found in mc_dropout_results/")

    # Dictionary to store all results grouped by prefix
    grouped_output = {}

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        if "threshold" not in df.columns:
            raise ValueError(f"CSV {csv_file} missing 'threshold' column")

        thresholds = df["threshold"].to_numpy()

        prefix, view = parse_filename(csv_file.name)

        # Compute bins
        edges = compute_kmeans_bins(thresholds, n_bins=4)

        # Convert 4-bin edges -> 3 cutpoints
        # bins: [0, high, med, low, 1]
        high_cut = float(edges[1])
        med_cut = float(edges[2])
        low_cut = float(edges[3])

        # Organize into grouped dict
        if prefix not in grouped_output:
            grouped_output[prefix] = {}

        grouped_output[prefix][view] = {
            "high": round(high_cut, 6),
            "medium": round(med_cut, 6),
            "low": round(low_cut, 6)
        }

    # Save one file per prefix
    for prefix, data in grouped_output.items():
        json_out = output_dir / f"{prefix}_conf_thresholds.json"
        with open(json_out, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved to {json_out}")


if __name__ == "__main__":
    main()
