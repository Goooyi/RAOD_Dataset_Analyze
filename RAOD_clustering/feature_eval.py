import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_cluster_results(results_path):
    """Load clustering results from pickle file."""
    with open(results_path, "rb") as f:
        return pickle.load(f)

def prepare_visualization_data(results):
    """Prepare data for visualization from the new clustering format."""
    scaled_features = results["scaled_features"]
    labels = results["cluster_labels"]
    centers = results["cluster_centers"]

    return scaled_features, labels, centers

def plot_dimensionality_reduction(scaled_features, labels, feature_names, save_dir):
    """Create PCA and t-SNE visualizations."""
    labels = np.array(labels)
    unique_labels = np.unique(labels)

    n_day = sum(1 for label in unique_labels if str(label).startswith('day'))
    n_night = sum(1 for label in unique_labels if str(label).startswith('night'))

    day_colors = plt.cm.YlOrRd(np.linspace(0.1, 0.9, n_day))
    night_colors = plt.cm.Blues(np.linspace(0.2, 0.8, n_night))
    colors = np.vstack([day_colors, night_colors])

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(scaled_features)
    plt.figure(figsize=(12, 8))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[colors[i]], label=label)

    plt.title("Clusters Visualized using PCA")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pca_visualization.png"))
    plt.close()

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(scaled_features)
    plt.figure(figsize=(12, 8))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=[colors[i]], label=label)

    plt.title("Clusters Visualized using t-SNE")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "tsne_visualization.png"))
    plt.close()

def plot_parallel_coordinates(scaled_features, labels, feature_names, save_dir):
    """Create parallel coordinates plot for day and night separately."""
    df = pd.DataFrame(scaled_features, columns=feature_names)
    df["Cluster"] = labels

    # Separate day and night
    day_df = df[df["Cluster"].astype(str).str.startswith("day")]
    night_df = df[df["Cluster"].astype(str).str.startswith("night")]

    # Plot day clusters
    plt.figure(figsize=(15, 8))
    pd.plotting.parallel_coordinates(day_df, "Cluster", colormap=plt.cm.YlOrRd)
    plt.title("Parallel Coordinates Plot - Day Clusters")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "parallel_coordinates_day.png"))
    plt.close()

    # Plot night clusters
    plt.figure(figsize=(15, 8))
    pd.plotting.parallel_coordinates(night_df, "Cluster", colormap=plt.cm.Blues)
    plt.title("Parallel Coordinates Plot - Night Clusters")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "parallel_coordinates_night.png"))
    plt.close()

def plot_cluster_centers(cluster_centers, feature_names, save_dir):
    """Create heatmap of cluster centers."""
    n_clusters = len(cluster_centers)
    centers_df = pd.DataFrame(cluster_centers, columns=feature_names)
    centers_df.index = [f'{"Day" if i < n_clusters//2 else "Night"} Cluster {i % (n_clusters//2)}'
                       for i in range(n_clusters)]

    plt.figure(figsize=(15, 10))
    sns.heatmap(centers_df, annot=True, cmap="coolwarm", center=0)
    plt.title("Cluster Centers Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cluster_centers_heatmap.png"))
    plt.close()

def plot_pairwise_features(scaled_features, labels, feature_names, save_dir):
    """Create pairwise feature plots for day and night separately."""
    df = pd.DataFrame(scaled_features, columns=feature_names)
    df["Cluster"] = labels

    # Separate day and night
    day_df = df[df["Cluster"].astype(str).str.startswith("day")]
    night_df = df[df["Cluster"].astype(str).str.startswith("night")]

    # Day clusters
    day_pairplot = sns.pairplot(day_df, hue="Cluster", diag_kind="kde",
                               palette="YlOrRd")
    plt.suptitle("Pairwise Feature Relationships - Day Clusters", y=1.02)
    day_pairplot.savefig(os.path.join(save_dir, "pairwise_features_day.png"))
    plt.close()

    # Night clusters
    night_pairplot = sns.pairplot(night_df, hue="Cluster", diag_kind="kde",
                                 palette="Blues")
    plt.suptitle("Pairwise Feature Relationships - Night Clusters", y=1.02)
    night_pairplot.savefig(os.path.join(save_dir, "pairwise_features_night.png"))
    plt.close()

def plot_radar(cluster_centers, feature_names, save_dir):
    """Create separate radar plots for day and night cluster centers."""
    n_features = len(feature_names)
    angles = [n / float(n_features) * 2 * np.pi for n in range(n_features)]
    angles += angles[:1]

    n_clusters = len(cluster_centers)
    day_centers = cluster_centers[:n_clusters//2]
    night_centers = cluster_centers[n_clusters//2:]

    # Day clusters
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection="polar"))
    for i, center in enumerate(day_centers):
        values = center.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle="solid", label=f"Day Cluster {i}")
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names)
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.title("Day Cluster Centers Radar Plot")
    plt.savefig(os.path.join(save_dir, "radar_plot_day.png"))
    plt.close()

    # Night clusters
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection="polar"))
    for i, center in enumerate(night_centers):
        values = center.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle="solid", label=f"Night Cluster {i}")
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names)
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.title("Night Cluster Centers Radar Plot")
    plt.savefig(os.path.join(save_dir, "radar_plot_night.png"))
    plt.close()

def plot_feature_importance(scaled_features, labels, feature_names, feature_groups, save_dir):
    """Create feature importance visualization based on cluster separation."""
    from sklearn.ensemble import RandomForestClassifier

    # Train a random forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(scaled_features, labels)

    # Get feature importance scores
    importance = rf.feature_importances_

    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })

    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=True)

    # Create group colors
    group_colors = {}
    for group, features in feature_groups.items():
        for idx in features:
            group_colors[feature_names[idx]] = group

    # Plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(importance_df)), importance_df['Importance'])

    # Color bars by feature group
    for i, bar in enumerate(bars):
        feature = importance_df.iloc[i]['Feature']
        if feature in group_colors:
            bar.set_color(plt.cm.Set3(list(feature_groups.keys()).index(group_colors[feature])))

    plt.yticks(range(len(importance_df)), importance_df['Feature'])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance in Cluster Separation')

    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=plt.cm.Set3(i))
                      for i in range(len(feature_groups))]
    plt.legend(legend_elements, feature_groups.keys(),
              loc='lower right', title='Feature Groups')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance.png'))
    plt.close()

def plot_zone_distribution(scaled_features, labels, feature_names, save_dir):
    """Plot zone distribution analysis for different clusters."""
    # Find zone ratio columns
    zone_cols = [i for i, name in enumerate(feature_names) if 'Zone' in name]

    if not zone_cols:
        return

    zone_data = scaled_features[:, zone_cols]

    # Plot separate figures for day and night
    for time in ['day', 'night']:
        mask = np.array([label.startswith(time) for label in labels])

        if not np.any(mask):
            continue

        plt.figure(figsize=(12, 6))

        # Plot average zone distribution for each cluster
        for cluster in np.unique(labels[mask]):
            cluster_mask = labels == cluster
            mean_dist = np.mean(zone_data[cluster_mask], axis=0)
            plt.plot(range(len(zone_cols)), mean_dist,
                    label=cluster, marker='o')

        plt.title(f'{time.capitalize()} Clusters Zone Distribution')
        plt.xlabel('Zone')
        plt.ylabel('Normalized Population')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'zone_distribution_{time}.png'))
        plt.close()

def plot_color_analysis(scaled_features, labels, feature_names, save_dir):
    """Create color-specific analysis plots."""
    # Find color-related features
    color_features = ['color_temp_estimate', 'color_saturation',
                     'local_color_variance', 'g_r_ratio', 'g_b_ratio']
    color_idx = [i for i, name in enumerate(feature_names)
                if any(cf in name.lower() for cf in color_features)]

    if not color_idx:
        return

    color_data = scaled_features[:, color_idx]
    color_names = [feature_names[i] for i in color_idx]

    # Create correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(np.corrcoef(color_data.T),
                xticklabels=color_names,
                yticklabels=color_names,
                annot=True, cmap='coolwarm')
    plt.title('Color Feature Correlations')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'color_correlations.png'))
    plt.close()

    # Create scatter plot matrix
    df = pd.DataFrame(color_data, columns=color_names)
    df['Cluster'] = labels

    # Separate day and night
    for time in ['day', 'night']:
        time_df = df[df['Cluster'].str.startswith(time)]
        if len(time_df) > 0:
            sns.pairplot(time_df, hue='Cluster', diag_kind='kde')
            plt.suptitle(f'{time.capitalize()} Clusters Color Analysis', y=1.02)
            plt.savefig(os.path.join(save_dir, f'color_analysis_{time}.png'))
            plt.close()

def main():
    # Specify the path to your clustering results
    # TODO: refactor
    results_path = "clustering_results/v0.7/cluster_results_20250224_161506.pkl"

    # Load results
    results = load_cluster_results(results_path)

    # Prepare data for visualization
    scaled_features, labels, centers = prepare_visualization_data(results)

    # Create visualization directory
    # TODO: refactor
    results_path = "clustering_results/v0.7/cluster_results_20250224_161506.pkl"
    viz_dir = "clustering_visualizations/v0.7"
    os.makedirs(viz_dir, exist_ok=True)

    # Create all visualizations
    plot_dimensionality_reduction(
        scaled_features,
        labels,
        results["feature_names"],
        viz_dir
    )
    print("PCA and t-SNE plots done!")

    plot_parallel_coordinates(
        scaled_features,
        labels,
        results["feature_names"],
        viz_dir
    )
    print("Parallel coordinates plots done!")

    plot_cluster_centers(
        centers,
        results["feature_names"],
        viz_dir
    )
    print("Cluster centers heatmap done!")

    plot_pairwise_features(
        scaled_features,
        labels,
        results["feature_names"],
        viz_dir
    )
    print("Pairwise feature plots done!")

    plot_radar(
        centers,
        results["feature_names"],
        viz_dir
    )
    print("Radar plots done!")

    # New ISP-focused visualizations
    plot_feature_importance(scaled_features, labels,
                          results["feature_names"],
                          results["feature_groups"],
                          viz_dir)
    print("Feature importance plot done!")

    plot_zone_distribution(scaled_features, labels,
                         results["feature_names"],
                         viz_dir)
    print("Zone distribution plots done!")

    plot_color_analysis(scaled_features, labels,
                       results["feature_names"],
                       viz_dir)
    print("Color analysis plots done!")

if __name__ == "__main__":
    main()
