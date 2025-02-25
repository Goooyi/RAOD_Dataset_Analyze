# Visualization and Clustering for RAOD dataset

[RAOD](https://github.com/huawei-noah/noah-research/tree/master/RAOD) dataset does not provide meta data to visualize the raw dataset, this repo aims to provide simple scripts to do that to help

1. simple visualization for eye-balling the the dataset.
2. Clustering RAOD dataset, so that ISP pipeline can be applied for further ablation study

## Setup

Install `pixi` then

```bash
pixi add
```

## Raw to Image Visualization

- examples in `RAOD2image.py`
- handle both day and night images in RAOD

## Clusetring and cluster feature viualization

### Clusetring v0.7

- defined feature in `calc_groups.py`
- added multi-processing

### Visualization for clusterings

run `feature_eval.py`

### Example Graph

1. Features importance visualization with `RandomForestClassifier`
![feature importance](./clustering_visualizations/feature_importance.png)

2. Cluster Visualization
![t-sne](./clustering_visualizations/tsne_visualization.png)
![pca](./clustering_visualizations/pca_visualization.png)
![cluster center heatmap](./clustering_visualizations/cluster_centers_heatmap.png)

3. feature visualizations
![Parallel Coordinates Plot Day](./clustering_visualizations/parallel_coordinates_day.png)
![pairwise_features_day](./clustering_visualizations/pairwise_features_day.png)
![radar_plot_day](./clustering_visualizations/radar_plot_day.png)
![radar_plot_night](./clustering_visualizations/radar_plot_night.png)
