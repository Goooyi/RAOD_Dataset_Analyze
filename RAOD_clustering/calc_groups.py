import argparse
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

BIT8 = 256
BIT16 = 65536
BIT24 = 1677721


@dataclass
class ImageFeatures:
    """Expanded container for image features used in clustering"""

    mean: float
    dynamic_range: float
    uniformity_score: float
    contrast_ratio: float
    gradient_magnitude: float
    g_r_ratio: float
    g_b_ratio: float

    # exposure features
    highlights_percentage: float  # % pixels above 90th percentile
    shadows_percentage: float  # % pixels below 10th percentile
    histogram_skewness: float
    histogram_kurtosis: float
    mid_tones_percentage: float  # % pixels in middle range

    # noise features
    flat_area_variance: float
    dark_noise_estimate: float
    bright_noise_estimate: float

    # color features
    local_color_variance: float
    color_cast_strength: float
    awb_confidence: float

    # tone mapping features
    zone_ratios: list  # Distribution across different exposure zones
    local_contrast: float  # Local contrast adaptation metric
    highlight_retention: float  # Measure of highlight detail preservation
    shadow_detail: float  # Measure of shadow detail preservation

    # New color features
    color_temp_estimate: float  # Estimated color temperature
    color_saturation: float  # Overall color saturation measure
    local_color_variance: float  # Spatial color consistency

    # TODO: New scene analysis features
    # edge_complexity: float  # Measure of edge distribution
    # spatial_frequency: float  # Frequency domain analysis
    # luminance_pattern: float  # Pattern-based scene analysis

    def to_array(self) -> np.ndarray:
        """Convert features to normalized numpy array"""
        base_features = [
            self.mean / BIT24,
            self.dynamic_range / BIT24,
            self.uniformity_score,
            self.contrast_ratio / 10.0,
            self.gradient_magnitude / BIT24,
            self.g_r_ratio / 3.0,
            self.g_b_ratio / 3.0,
            self.highlights_percentage,
            self.shadows_percentage,
            self.histogram_skewness / 5.0,  # Normalize based on typical range
            self.histogram_kurtosis / 20.0,
            self.mid_tones_percentage,
            self.flat_area_variance / BIT24,
            self.dark_noise_estimate / BIT24,
            self.bright_noise_estimate / BIT24,
            self.local_color_variance,
            self.color_cast_strength,
            self.awb_confidence,
        ]
        tone_features = [
            *[ratio for ratio in self.zone_ratios],  # Unpack zone ratios
            self.local_contrast / 10.0,
            self.highlight_retention,
            self.shadow_detail,
        ]

        color_features = [
            self.color_temp_estimate / 10000.0,  # Normalize to typical range
            self.color_saturation,
            self.local_color_variance,
        ]

        # # TODO
        # scene_features = [
        #     self.edge_complexity,
        #     self.spatial_frequency / BIT24,
        #     self.luminance_pattern,
        # ]

        return np.array(
            base_features + tone_features + color_features
        )  # + scene_features)


class ImageProcessor:
    def __init__(self, img_shape=(1, 1, 1856, 2880)):
        self.img_shape = img_shape
        self.lighting_analyzer = LightingAnalyzer()

    def read_raw_24b(self, file_path: str) -> np.ndarray:
        """Read 24-bit raw image data"""
        raw_data = np.fromfile(file_path, dtype=np.uint8)
        raw_data = raw_data[0::3] + raw_data[1::3] * BIT8 + raw_data[2::3] * BIT16
        return raw_data.reshape(self.img_shape).astype(np.float32)

    def analyze_tone_distribution(self, image: np.ndarray) -> dict:
        """Analyze tone distribution for HDR-aware processing"""
        # Calculate the actual dynamic range of the image
        min_val = np.min(image)
        max_val = np.max(image)

        # Create histogram with appropriate binning
        n_bins = 256  # Keep 256 bins for reasonable granularity
        hist, bin_edges = np.histogram(image, bins=n_bins, range=(min_val, max_val))

        # Define 8 zones (like photographic zones) plus highlights
        n_zones = 8
        zone_edges = (
            np.exp(np.linspace(np.log(min_val + 1), np.log(max_val + 1), n_zones + 1))
            - 1
        )

        # Calculate population in each zone
        zone_populations = []
        for i in range(len(zone_edges) - 1):
            # Find histogram bins that fall within this zone
            zone_start = zone_edges[i]
            zone_end = zone_edges[i + 1]

            # Find corresponding bin indices
            bin_start = np.searchsorted(bin_edges, zone_start)
            bin_end = np.searchsorted(bin_edges, zone_end)

            # Sum the populations
            zone_pop = np.sum(hist[bin_start:bin_end]) / np.sum(hist)
            zone_populations.append(float(zone_pop))

        # Calculate additional metrics
        highlight_retention = (
            1.0 - zone_populations[-1]
        )  # Less population in highest zone is better
        shadow_detail = (
            1.0 - zone_populations[0]
        )  # Less population in lowest zone is better

        return {
            "zone_ratios": zone_populations,
            "highlight_retention": float(highlight_retention),
            "shadow_detail": float(shadow_detail),
        }

    def estimate_color_temperature(self, image: np.ndarray) -> float:
        """Estimate color temperature based on RGB ratios"""
        r = image[0::2, 0::2]
        # g = (image[0::2, 1::2] + image[1::2, 0::2]) / 2
        b = image[1::2, 1::2]

        r_mean = np.mean(r)
        b_mean = np.mean(b)

        # Simple color temperature estimation
        # Lower ratio indicates warmer color temperature
        rb_ratio = b_mean / (r_mean + 1e-6)

        # Map ratio to approximate color temperature (1000K - 10000K range)
        estimated_temp = 2000 + (rb_ratio * 8000)
        return min(10000, max(1000, estimated_temp))

    def analyze_spatial_frequency(self, image: np.ndarray) -> float:
        """Analyze spatial frequency characteristics"""
        # Compute gradients
        gy, gx = np.gradient(image)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)

        # Analyze frequency distribution
        freq_metric = np.mean(gradient_magnitude) * np.std(gradient_magnitude)
        return freq_metric

    def estimate_local_noise(
        self, image: np.ndarray, patch_size: int = 16
    ) -> Tuple[float, float]:
        """Estimate noise in flat areas for dark and bright regions"""
        patches = self._extract_patches(image, patch_size)
        patch_means = np.mean(patches, axis=(1, 2))
        patch_vars = np.var(patches, axis=(1, 2))

        # Find flat patches (low variance)
        flat_threshold = np.percentile(patch_vars, 20)  # Bottom 20% variance
        flat_patches = patches[patch_vars < flat_threshold]
        flat_means = patch_means[patch_vars < flat_threshold]

        # Separate dark and bright regions
        dark_mask = flat_means < np.percentile(flat_means, 25)
        bright_mask = flat_means > np.percentile(flat_means, 75)

        dark_noise = np.mean(np.var(flat_patches[dark_mask], axis=(1, 2)))
        bright_noise = np.mean(np.var(flat_patches[bright_mask], axis=(1, 2)))

        return dark_noise, bright_noise

    def analyze_color_consistency(
        self, image: np.ndarray, patch_size: int = 32
    ) -> Tuple[float, float, float]:
        """Analyze local color consistency and detect color cast"""
        patches = self._extract_patches(image, patch_size)

        # Calculate color ratios for each patch
        r = patches[:, 0::2, 0::2]
        g = (patches[:, 0::2, 1::2] + patches[:, 1::2, 0::2]) / 2
        b = patches[:, 1::2, 1::2]

        gr_ratios = np.mean(g, axis=(1, 2)) / (np.mean(r, axis=(1, 2)) + 1e-6)
        gb_ratios = np.mean(g, axis=(1, 2)) / (np.mean(b, axis=(1, 2)) + 1e-6)

        # Calculate local color variance
        local_variance = np.std(gr_ratios) + np.std(gb_ratios)

        # Estimate color cast strength
        ideal_gr = 1.0  # Assuming neutral lighting
        ideal_gb = 1.0
        color_cast = np.sqrt(
            (np.mean(gr_ratios) - ideal_gr) ** 2 + (np.mean(gb_ratios) - ideal_gb) ** 2
        )

        # Calculate AWB confidence based on color consistency
        awb_confidence = 1.0 / (1.0 + local_variance + color_cast)

        return local_variance, color_cast, awb_confidence

    def calculate_features(self, image: np.ndarray) -> ImageFeatures:
        """Extract expanded set of features from image"""
        if image.ndim == 4:
            image = image[0, 0]

        # Original features
        mean = np.mean(image)
        sorted_vals = np.sort(image.flatten())
        p20 = np.percentile(sorted_vals, 20)
        p80 = np.percentile(sorted_vals, 80)
        dynamic_range = p80 - p20

        # Color ratios
        r = image[0::2, 0::2]
        g = (image[0::2, 1::2] + image[1::2, 0::2]) / 2
        b = image[1::2, 1::2]
        g_r_ratio = np.mean(g) / (np.mean(r) + 1e-6)
        g_b_ratio = np.mean(g) / (np.mean(b) + 1e-6)

        # Lighting pattern analysis
        lighting = self.lighting_analyzer.analyze_lighting_pattern(image)

        # New exposure features
        p90 = np.percentile(sorted_vals, 90)
        p10 = np.percentile(sorted_vals, 10)
        highlights_percentage = np.mean(image > p90)
        shadows_percentage = np.mean(image < p10)
        mid_tones_percentage = np.mean((image >= p20) & (image <= p80))

        # Histogram shape analysis
        hist_stats = stats.describe(image.flatten())
        histogram_skewness = hist_stats.skewness
        histogram_kurtosis = hist_stats.kurtosis

        # Noise estimation
        dark_noise, bright_noise = self.estimate_local_noise(image)

        # Color analysis
        local_color_var, color_cast, awb_conf = self.analyze_color_consistency(image)

        # tone mapping analysis
        tone_stats = self.analyze_tone_distribution(image)

        # color analysis
        color_temp = self.estimate_color_temperature(image)
        color_sat = np.std([np.mean(r), np.mean(g), np.mean(b)]) / mean

        # # TODO New scene analysis
        # spatial_freq = self.analyze_spatial_frequency(image)
        # edge_complexity = np.std(np.gradient(image)[0]) / mean

        # Local contrast analysis
        local_blocks = self.extract_blocks(image, 16)
        local_contrasts = np.array([np.max(b) - np.min(b) for b in local_blocks])
        local_contrast_metric = np.mean(local_contrasts) / dynamic_range

        return ImageFeatures(
            mean=mean,
            dynamic_range=dynamic_range,
            uniformity_score=lighting["uniformity_score"],
            contrast_ratio=lighting["contrast_ratio"],
            gradient_magnitude=lighting["gradient_magnitude"],
            g_r_ratio=g_r_ratio,
            g_b_ratio=g_b_ratio,
            highlights_percentage=highlights_percentage,
            shadows_percentage=shadows_percentage,
            histogram_skewness=histogram_skewness,
            histogram_kurtosis=histogram_kurtosis,
            mid_tones_percentage=mid_tones_percentage,
            flat_area_variance=np.mean([dark_noise, bright_noise]),
            dark_noise_estimate=dark_noise,
            bright_noise_estimate=bright_noise,
            local_color_variance=local_color_var,
            color_cast_strength=color_cast,
            awb_confidence=awb_conf,
            zone_ratios=tone_stats["zone_ratios"],
            local_contrast=local_contrast_metric,
            highlight_retention=tone_stats["highlight_retention"],
            shadow_detail=tone_stats["shadow_detail"],
            color_temp_estimate=color_temp,
            color_saturation=color_sat,
        )

    def extract_blocks(self, image: np.ndarray, block_size: int) -> np.ndarray:
        """Extract non-overlapping blocks from image"""
        h, w = image.shape
        blocks = []

        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = image[i : i + block_size, j : j + block_size]
                if block.shape == (block_size, block_size):
                    blocks.append(block)

        return np.array(blocks)

    def _extract_patches(self, image: np.ndarray, patch_size: int) -> np.ndarray:
        """Extract overlapping patches from image"""
        h, w = image.shape
        patches = []

        for y in range(0, h - patch_size, patch_size // 2):
            for x in range(0, w - patch_size, patch_size // 2):
                patch = image[y : y + patch_size, x : x + patch_size]
                if patch.shape == (patch_size, patch_size):
                    patches.append(patch)

        return np.array(patches)


class LightingAnalyzer:
    def __init__(self, grid_size=4):
        self.grid_size = grid_size

    def analyze_lighting_pattern(self, image: np.ndarray) -> Dict:
        """Analyze lighting pattern in image"""
        if image.ndim == 4:
            image = image[0, 0]

        # Calculate regional means
        region_height = image.shape[0] // self.grid_size
        region_width = image.shape[1] // self.grid_size
        region_means = np.zeros((self.grid_size, self.grid_size))

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                region = image[
                    y * region_height : (y + 1) * region_height,
                    x * region_width : (x + 1) * region_width,
                ]
                region_means[y, x] = np.mean(region)

        # Calculate lighting metrics
        uniformity_score = np.std(region_means) / (np.mean(region_means) + 1e-6)
        contrast_ratio = np.max(region_means) / (np.min(region_means) + 1e-6)

        # Calculate gradients
        gradient_y = np.mean(np.diff(region_means, axis=0))
        gradient_x = np.mean(np.diff(region_means, axis=1))
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        return {
            "uniformity_score": float(uniformity_score),
            "contrast_ratio": float(contrast_ratio),
            "gradient_magnitude": float(gradient_magnitude),
        }


class ClusterAnalyzer:
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.day_scaler = StandardScaler()
        self.night_scaler = StandardScaler()
        self.feature_names = [
            "Mean Brightness",
            "Dynamic Range",
            "Uniformity",
            "Contrast Ratio",
            "Gradient Magnitude",
            "G/R Ratio",
            "G/B Ratio",
            "Highlights %",
            "Shadows %",
            "Histogram Skewness",
            "Histogram Kurtosis",
            "Mid-tones %",
            "Flat Area Variance",
            "Dark Noise",
            "Bright Noise",
            "Color Cast Strength",
            "AWB Confidence",
            "zone_ratio_01",
            "zone_ratio_02",
            "zone_ratio_03",
            "zone_ratio_04",
            "zone_ratio_05",
            "zone_ratio_06",
            "zone_ratio_07",
            "zone_ratio_08",
            "zone_ratio_09",
            "local_contrast",
            "highlight_retention",
            "shadow_detail",
            "color_temp_estimate",
            "color_saturation",
        ]

        self.feature_groups = {
            "exposure": [0, 1, 7, 8, 11],
            "noise": [12, 13, 14],
            "color": [5, 6, 15, 16, 17],
            "texture": [2, 3, 4],
        }

    def analyze_dataset(self, dataset_path: str, processor: ImageProcessor) -> Dict:
        """Analyze dataset with separate day/night clustering"""
        # Extract features
        day_features = []
        night_features = []
        day_paths = []
        night_paths = []

        print("Extracting features...")
        for file_name in tqdm(os.listdir(dataset_path)):
            if not file_name.endswith(".raw"):
                continue

            file_path = os.path.join(dataset_path, file_name)
            is_night = file_name.startswith("night-")

            # Process image
            raw_data = processor.read_raw_24b(file_path)
            features = processor.calculate_features(raw_data)
            feature_array = features.to_array()

            # Separate day/night features
            if is_night:
                night_features.append(feature_array)
                night_paths.append(file_path)
            else:
                day_features.append(feature_array)
                day_paths.append(file_path)

        # Convert to arrays
        day_features_array = np.array(day_features)
        night_features_array = np.array(night_features)

        # Scale features separately
        day_scaled = self.day_scaler.fit_transform(day_features_array)
        night_scaled = self.night_scaler.fit_transform(night_features_array)

        # Cluster separately
        day_kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        night_kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)

        day_labels = day_kmeans.fit_predict(day_scaled)
        night_labels = night_kmeans.fit_predict(night_scaled)

        # Combine results
        all_features = np.vstack([day_features_array, night_features_array])
        all_scaled_features = np.vstack([day_scaled, night_scaled])
        all_paths = np.concatenate([day_paths, night_paths])
        all_labels = np.concatenate(
            [
                [f"day_{label}" for label in day_labels],
                [f"night_{label}" for label in night_labels],
            ]
        )
        all_centers = np.vstack(
            [day_kmeans.cluster_centers_, night_kmeans.cluster_centers_]
        )
        day_night_mask = np.concatenate(
            [
                np.ones(len(day_paths), dtype=bool),
                np.zeros(len(night_paths), dtype=bool),
            ]
        )

        # Create results dictionary
        results = {
            "cluster_labels": all_labels,
            "cluster_centers": all_centers,
            "scaled_features": all_scaled_features,
            "original_features": all_features,
            "file_paths": all_paths,
            "day_mask": day_night_mask,
            "feature_names": self.feature_names,
            "feature_groups": self.feature_groups,
            "inertia": day_kmeans.inertia_ + night_kmeans.inertia_,
            "n_clusters": self.n_clusters * 2,  # Total number of clusters (day + night)
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "day_night_info": {
                "day_inertia": day_kmeans.inertia_,
                "night_inertia": night_kmeans.inertia_,
                "day_centers": day_kmeans.cluster_centers_,
                "night_centers": night_kmeans.cluster_centers_,
                "day_labels": day_labels,
                "night_labels": night_labels,
            },
        }

        # Create DataFrame for detailed analysis
        df = pd.DataFrame(all_features, columns=self.feature_names)
        df["Cluster"] = all_labels
        df["Is_Day"] = day_night_mask
        df["File_Path"] = all_paths

        # Calculate cluster statistics
        cluster_stats = []
        for i in range(self.n_clusters):
            # Day cluster stats
            day_mask = day_labels == i
            day_info = {
                "cluster_id": f"day_{i}",
                "size": np.sum(day_mask),
                "feature_means": day_features_array[day_mask].mean(axis=0),
                "feature_stds": day_features_array[day_mask].std(axis=0),
            }
            cluster_stats.append(day_info)

            # Night cluster stats
            night_mask = night_labels == i
            night_info = {
                "cluster_id": f"night_{i}",
                "size": np.sum(night_mask),
                "feature_means": night_features_array[night_mask].mean(axis=0),
                "feature_stds": night_features_array[night_mask].std(axis=0),
            }
            cluster_stats.append(night_info)

        results["cluster_stats"] = cluster_stats

        # Save results
        save_dir = "clustering_results/v0.6"  # Keep consistent with original version
        os.makedirs(save_dir, exist_ok=True)

        # Save pickle file with all results
        timestamp = results["timestamp"]
        pickle_path = os.path.join(save_dir, f"cluster_results_{timestamp}.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(results, f)

        # Save CSV with detailed data
        csv_path = os.path.join(save_dir, f"cluster_data_{timestamp}.csv")
        df.to_csv(csv_path, index=False)

        return results


def process_single_image(
    file_path: str, processor: ImageProcessor
) -> Tuple[np.ndarray, str, bool]:
    """
    Process a single image file and return its features
    Returns: (feature_array, file_path, is_day)
    """
    is_night = os.path.basename(file_path).startswith("night-")
    raw_data = processor.read_raw_24b(file_path)
    features = processor.calculate_features(raw_data)
    return features.to_array(), file_path, not is_night


class ParallelClusterAnalyzer(ClusterAnalyzer):
    def __init__(self, n_clusters: int = 5, n_processes: int = None):
        super().__init__(n_clusters)
        self.n_processes = n_processes or max(1, cpu_count() - 1)  # Leave one core free

    def analyze_dataset(self, dataset_path: str, processor: ImageProcessor) -> Dict:
        """Analyze dataset with parallel processing"""
        # Get list of raw files
        raw_files = [
            os.path.join(dataset_path, f)
            for f in os.listdir(dataset_path)
            if f.endswith(".raw")
        ]

        print(
            f"Processing {len(raw_files)} images using {self.n_processes} processes..."
        )

        # Create a pool of workers
        with Pool(processes=self.n_processes) as pool:
            # Create a partial function with the processor argument
            process_func = partial(process_single_image, processor=processor)

            # Process images in parallel with progress bar
            results = list(
                tqdm(
                    pool.imap(process_func, raw_files),
                    total=len(raw_files),
                    desc="Processing images",
                )
            )

        day_features = []
        day_paths = []
        night_features = []
        night_paths = []

        # Sort results into day and night
        for feature_array, file_path, is_day in results:
            if is_day:
                day_features.append(feature_array)
                day_paths.append(file_path)
            else:
                night_features.append(feature_array)
                night_paths.append(file_path)

        # Convert to numpy arrays
        day_features = np.array(day_features)
        night_features = np.array(night_features)

        # Scale features separately
        day_scaled = self.day_scaler.fit_transform(day_features)
        night_scaled = self.night_scaler.fit_transform(night_features)

        # Cluster separately
        day_kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        night_kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)

        day_labels = day_kmeans.fit_predict(day_scaled)
        night_labels = night_kmeans.fit_predict(night_scaled)

        # Create separate DataFrames for day and night
        day_df = pd.DataFrame(day_features, columns=self.feature_names)
        day_df["Cluster"] = [f"day_{label}" for label in day_labels]
        day_df["Is_Day"] = True
        day_df["File_Path"] = day_paths

        night_df = pd.DataFrame(night_features, columns=self.feature_names)
        night_df["Cluster"] = [f"night_{label}" for label in night_labels]
        night_df["Is_Day"] = False
        night_df["File_Path"] = night_paths

        # Combine the DataFrames in the correct order
        df = pd.concat([day_df, night_df], axis=0, ignore_index=True)

        # Create results dictionary (keep your existing code)
        results = {
            "cluster_labels": df["Cluster"].values,
            "cluster_centers": np.vstack(
                [day_kmeans.cluster_centers_, night_kmeans.cluster_centers_]
            ),
            "scaled_features": np.vstack([day_scaled, night_scaled]),
            "original_features": np.vstack([day_features, night_features]),
            "file_paths": df["File_Path"].values,
            "day_mask": df["Is_Day"].values,
            "feature_names": self.feature_names,
            "feature_groups": self.feature_groups,
            "inertia": day_kmeans.inertia_ + night_kmeans.inertia_,
            "n_clusters": self.n_clusters * 2,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "day_night_info": {
                "day_inertia": day_kmeans.inertia_,
                "night_inertia": night_kmeans.inertia_,
                "day_centers": day_kmeans.cluster_centers_,
                "night_centers": night_kmeans.cluster_centers_,
                "day_labels": day_labels,
                "night_labels": night_labels,
            },
        }

        # Save results (keep your existing code)
        save_dir = "clustering_results/v0.6"
        os.makedirs(save_dir, exist_ok=True)

        # Save files
        timestamp = results["timestamp"]
        pickle_path = os.path.join(save_dir, f"cluster_results_{timestamp}.pkl")
        csv_path = os.path.join(save_dir, f"cluster_data_{timestamp}.csv")

        with open(pickle_path, "wb") as f:
            pickle.dump(results, f)
        df.to_csv(csv_path, index=False)

        return results


def main():
    parser = argparse.ArgumentParser(description="Analyze clustering results")
    parser.add_argument(
        "--dataset_path", type=str, help="path to ROAD dataset", required=True
    )
    parser.add_argument(
        "--n_processes", type=int, help="number of process", required=True
    )
    args = parser.parse_args()

    # Initialize components
    processor = ImageProcessor()
    analyzer = ClusterAnalyzer(n_clusters=5)

    # Create analyzer with custom number of processes (optional)
    n_processes = args.n_processes  # Adjust this based on your CPU

    # Process dataset
    dataset_path = args.dataset_path
    analyzer = ParallelClusterAnalyzer(n_clusters=5, n_processes=n_processes)
    _results = analyzer.analyze_dataset(dataset_path, processor)

    # TODO: for react project
    # with open("cluster_analysis.json", "w") as f:
    #     json.dump(results, f)

    print("Analysis complete!")


if __name__ == "__main__":
    main()
