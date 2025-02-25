import pandas as pd
import os
from typing import List, Dict
import glob
import argparse

def get_latest_csv(algo_version: str) -> str:
    """
    Get the path of the most recent clustering results CSV file

    Args:
        algo_version: Version of the algorithm to use for directory path
    """
    csv_dir = f"clustering_results/{algo_version}"
    pattern = os.path.join(csv_dir, "cluster_data_*.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        raise FileNotFoundError(f"No clustering result CSV files found in {csv_dir}!")

    # Sort by modification time and get the latest
    latest_file = max(csv_files, key=os.path.getmtime)
    return latest_file

def analyze_clusters(csv_path: str, samples_per_cluster: int = 20) -> Dict[int, List[str]]:
    """
    Analyze clustering results and return sample file paths for each cluster

    Args:
        csv_path: Path to the clustering results CSV
        samples_per_cluster: Number of sample files to show per cluster

    Returns:
        Dictionary mapping cluster IDs to lists of sample file paths
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Get unique cluster IDs
    clusters = sorted(df['Cluster'].unique())

    # Dictionary to store results
    cluster_samples = {}

    # For each cluster, get sample file paths
    for cluster in clusters:
        cluster_df = df[df['Cluster'] == cluster]

        # Get basic cluster statistics
        cluster_size = len(cluster_df)
        day_percentage = (cluster_df['Is_Day'].mean() * 100)

        print(f"\nCluster {cluster} Statistics:")
        print(f"Total images: {cluster_size}")
        print(f"Day images: {day_percentage:.1f}%")
        print(f"Night images: {100 - day_percentage:.1f}%")
        print("\nSample file paths:")

        # Get sample files, trying to balance day and night if possible
        day_files = cluster_df[cluster_df['Is_Day']]['File_Path'].sample(
            min(samples_per_cluster // 2, sum(cluster_df['Is_Day']))
        ).tolist()

        night_files = cluster_df[~cluster_df['Is_Day']]['File_Path'].sample(
            min(samples_per_cluster // 2, sum(~cluster_df['Is_Day']))
        ).tolist()

        # If we need more samples, take them from either day or night
        remaining_samples = samples_per_cluster - len(day_files) - len(night_files)
        if remaining_samples > 0:
            additional_files = cluster_df['File_Path'].sample(remaining_samples).tolist()
            sample_files = day_files + night_files + additional_files
        else:
            sample_files = day_files + night_files

        # Print sample files
        for i, file_path in enumerate(sample_files, 1):
            file_name = os.path.basename(file_path)
            is_day = "day" if file_name.startswith("day") else "night"
            print(f"{i}. [{is_day}] {file_name}")

        cluster_samples[cluster] = sample_files

    return cluster_samples

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze clustering results')
    parser.add_argument('--algo_version', type=str, help='Algorithm version for the clustering results', default='',required=False)
    args = parser.parse_args()

    try:
        # Get the latest CSV file with the specified algorithm version
        latest_csv = get_latest_csv(args.algo_version)
        print(f"Analyzing clustering results from: {latest_csv}\n")

        # Analyze clusters and get sample files
        _cluster_samples = analyze_clusters(latest_csv)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
