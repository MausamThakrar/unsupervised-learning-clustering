# Unsupervised Learning: Clustering and Visualisation

This project demonstrates **unsupervised learning** techniques for clustering
synthetic datasets. It compares several clustering algorithms and visualises
their results.

## Algorithms Used

- **K-Means**
- **Agglomerative Clustering**
- **DBSCAN** (density-based clustering)

## Datasets

Two types of synthetic datasets are generated:

1. **Blobs**  
   - Well-separated Gaussian clusters in 2D  
   - Suitable for K-Means and Agglomerative Clustering

2. **Circles**  
   - Points arranged in concentric circles  
   - Non-linear structure, challenging for K-Means

Both datasets are generated using scikit-learn helper functions.

## Evaluation

Cluster quality is evaluated (when possible) using the **silhouette score**:

- Higher values (close to 1) indicate better-defined clusters.
- DBSCAN may mark some points as noise (`-1` label) and sometimes the silhouette
  score is not defined (e.g. only one cluster), which is handled in the code.

## Dimensionality Reduction and Visualisation

Even though the datasets are 2D, a simple **PCA** transformation to 2D is used
to illustrate how one might visualise higher-dimensional data.

For each dataset, the script:

- Runs K-Means, Agglomerative Clustering and DBSCAN
- Prints silhouette scores for each method (if defined)
- Plots clustering results side-by-side for visual comparison

## Project Structure

```text
unsupervised-learning-clustering/
│── src/
│   └── clustering_demo.py
│── requirements.txt
└── README.md
