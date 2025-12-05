from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


@dataclass
class DatasetConfig:
    n_samples: int = 500
    random_state: int = 42


def generate_blobs(config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a simple dataset of 2D Gaussian blobs.
    """
    X, y_true = make_blobs(
        n_samples=config.n_samples,
        centers=3,
        cluster_std=1.0,
        random_state=config.random_state,
    )
    return X, y_true


def generate_circles(config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset of points arranged in concentric circles (non-linear structure).
    """
    X, y_true = make_circles(
        n_samples=config.n_samples,
        factor=0.5,
        noise=0.05,
        random_state=config.random_state,
    )
    return X, y_true


def run_kmeans(X: np.ndarray, n_clusters: int = 3, random_state: int = 42) -> Tuple[np.ndarray, float]:
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    return labels, score


def run_agglomerative(X: np.ndarray, n_clusters: int = 3) -> Tuple[np.ndarray, float]:
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    return labels, score


def run_dbscan(X: np.ndarray, eps: float = 0.3, min_samples: int = 5) -> Tuple[np.ndarray, float | None]:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    # DBSCAN may label some points as -1 (noise). Silhouette requires >1 cluster.
    unique_labels = set(labels)
    if len(unique_labels) > 1 and -1 not in unique_labels:
        score = silhouette_score(X, labels)
    else:
        score = None
    return labels, score


def reduce_with_pca(X: np.ndarray, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X)
    return X_reduced


def plot_clusters(
    X_2d: np.ndarray,
    labels_dict: Dict[str, np.ndarray],
    title_prefix: str,
) -> None:
    """
    Plot clustering results from multiple algorithms side by side.
    """
    n_models = len(labels_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))

    if n_models == 1:
        axes = [axes]

    for ax, (name, labels) in zip(axes, labels_dict.items()):
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, s=20, edgecolor="k")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title_prefix)
    plt.tight_layout()
    plt.show()


def main():
    config = DatasetConfig(n_samples=500, random_state=42)

    # === 1. Blobs dataset (well-separated clusters) ===
    X_blobs, y_blobs_true = generate_blobs(config)
    X_blobs_2d = reduce_with_pca(X_blobs, n_components=2)

    kmeans_labels, kmeans_score = run_kmeans(X_blobs, n_clusters=3)
    agg_labels, agg_score = run_agglomerative(X_blobs, n_clusters=3)
    dbscan_labels, dbscan_score = run_dbscan(X_blobs, eps=0.6, min_samples=5)

    print("=== BLOBS DATASET (3 clusters) ===")
    print(f"K-Means Silhouette score:        {kmeans_score:.3f}")
    print(f"Agglomerative Silhouette score:  {agg_score:.3f}")
    if dbscan_score is not None:
        print(f"DBSCAN Silhouette score:         {dbscan_score:.3f}")
    else:
        print("DBSCAN Silhouette score:         not defined (noise or single cluster)")

    plot_clusters(
        X_blobs_2d,
        labels_dict={
            f"K-Means (k=3)": kmeans_labels,
            "Agglomerative (k=3)": agg_labels,
            "DBSCAN": dbscan_labels,
        },
        title_prefix="Blobs Dataset – Clustering Comparison",
    )

    # === 2. Circles dataset (non-linear structure) ===
    X_circles, y_circles_true = generate_circles(config)
    X_circles_2d = reduce_with_pca(X_circles, n_components=2)

    kmeans_labels_c, kmeans_score_c = run_kmeans(X_circles, n_clusters=2)
    agg_labels_c, agg_score_c = run_agglomerative(X_circles, n_clusters=2)
    dbscan_labels_c, dbscan_score_c = run_dbscan(X_circles, eps=0.2, min_samples=5)

    print("\n=== CIRCLES DATASET (non-linear structure) ===")
    print(f"K-Means Silhouette score:        {kmeans_score_c:.3f}")
    print(f"Agglomerative Silhouette score:  {agg_score_c:.3f}")
    if dbscan_score_c is not None:
        print(f"DBSCAN Silhouette score:         {dbscan_score_c:.3f}")
    else:
        print("DBSCAN Silhouette score:         not defined (noise or single cluster)")

    plot_clusters(
        X_circles_2d,
        labels_dict={
            f"K-Means (k=2)": kmeans_labels_c,
            "Agglomerative (k=2)": agg_labels_c,
            "DBSCAN": dbscan_labels_c,
        },
        title_prefix="Circles Dataset – Clustering Comparison",
    )


if __name__ == "__main__":
    main()
