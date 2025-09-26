""" image_quality_counter.py """

import os
import pandas as pd
from collections import defaultdict, Counter

if __name__ == '__main__':
    dataset_dir = "dataset"

    # Nested dicts
    species_resolution_counts = defaultdict(Counter)
    genus_resolution_counts = defaultdict(Counter)

    for fname in os.listdir(dataset_dir):
        if not fname.lower().endswith(".jpg"):
            continue
        
        parts = fname.split()
        if len(parts) < 5:
            continue
        
        genus = parts[0]
        species = " ".join(parts[0:2])
        resolution = next((p for p in parts if "X" in p.upper()), "UNKNOWN")
        
        # Update nested counters
        species_resolution_counts[species][resolution] += 1
        genus_resolution_counts[genus][resolution] += 1

    # Convert to DataFrames
    species_df = pd.DataFrame.from_dict(species_resolution_counts, orient="index").fillna(0).astype(int)
    genus_df = pd.DataFrame.from_dict(genus_resolution_counts, orient="index").fillna(0).astype(int)

    # Save to CSV
    species_df.to_csv("species_resolution_counts.csv")
    genus_df.to_csv("genus_resolution_counts.csv")

    print("✅ Exported species and genus resolution counts to CSV")
    print("Species counts:")
    print(species_df.head())
    print("\nGenus counts:")
    print(genus_df.head())
