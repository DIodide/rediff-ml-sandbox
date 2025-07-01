"""
Utility functions for vector machine learning operations.
"""

import numpy as np
import faiss
from typing import List, Tuple, Optional, Union
from sklearn.metrics.pairwise import cosine_similarity


def build_faiss_index(
    vectors: np.ndarray, index_type: str = "flat", nlist: int = 100
) -> faiss.Index:
    """
    Build FAISS index for efficient similarity search.

    Args:
        vectors: Array of vectors to index
        index_type: Type of index ('flat', 'ivf', 'hnsw')
        nlist: Number of clusters for IVF index

    Returns:
        FAISS index
    """
    vectors = vectors.astype("float32")
    d = vectors.shape[1]

    if index_type == "flat":
        index = faiss.IndexFlatL2(d)
    elif index_type == "ivf":
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        index.train(vectors)
    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(d, 32)
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    index.add(vectors)
    return index


def search_similar_vectors(
    index: faiss.Index, query_vectors: np.ndarray, k: int = 5, nprobe: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search for similar vectors using FAISS index.

    Args:
        index: FAISS index
        query_vectors: Query vectors
        k: Number of nearest neighbors
        nprobe: Number of clusters to search (for IVF)

    Returns:
        Tuple of (distances, indices)
    """
    query_vectors = query_vectors.astype("float32")

    # Set nprobe for IVF indices
    if hasattr(index, "nprobe"):
        index.nprobe = nprobe

    distances, indices = index.search(query_vectors, k)
    return distances, indices


def compute_similarities(
    vectors1: np.ndarray, vectors2: Optional[np.ndarray] = None, metric: str = "cosine"
) -> np.ndarray:
    """
    Compute similarity matrix between vectors.

    Args:
        vectors1: First set of vectors
        vectors2: Second set of vectors (if None, use vectors1)
        metric: Similarity metric ('cosine', 'dot', 'euclidean')

    Returns:
        Similarity matrix
    """
    if vectors2 is None:
        vectors2 = vectors1

    if metric == "cosine":
        return cosine_similarity(vectors1, vectors2)
    elif metric == "dot":
        return np.dot(vectors1, vectors2.T)
    elif metric == "euclidean":
        from scipy.spatial.distance import cdist

        distances = cdist(vectors1, vectors2, metric="euclidean")
        # Convert to similarity (higher = more similar)
        return 1 / (1 + distances)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def normalize_vectors(vectors: np.ndarray, norm: str = "l2") -> np.ndarray:
    """
    Normalize vectors.

    Args:
        vectors: Input vectors
        norm: Normalization type ('l2', 'l1', 'max')

    Returns:
        Normalized vectors
    """
    if norm == "l2":
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return vectors / norms
    elif norm == "l1":
        norms = np.sum(np.abs(vectors), axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms
    elif norm == "max":
        max_vals = np.max(np.abs(vectors), axis=1, keepdims=True)
        max_vals[max_vals == 0] = 1
        return vectors / max_vals
    else:
        raise ValueError(f"Unknown norm: {norm}")


def reduce_dimensions(
    vectors: np.ndarray, n_components: int, method: str = "pca"
) -> np.ndarray:
    """
    Reduce vector dimensions.

    Args:
        vectors: Input vectors
        n_components: Target number of dimensions
        method: Reduction method ('pca', 'tsne', 'umap')

    Returns:
        Reduced vectors
    """
    if method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=n_components)
    elif method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=n_components)
    elif method == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=n_components)
        except ImportError:
            raise ImportError(
                "UMAP not installed. Install with: pip install umap-learn"
            )
    else:
        raise ValueError(f"Unknown method: {method}")

    return reducer.fit_transform(vectors)


def find_outliers(
    vectors: np.ndarray, method: str = "isolation_forest", contamination: float = 0.1
) -> np.ndarray:
    """
    Find outlier vectors.

    Args:
        vectors: Input vectors
        method: Outlier detection method
        contamination: Expected proportion of outliers

    Returns:
        Boolean array indicating outliers
    """
    if method == "isolation_forest":
        from sklearn.ensemble import IsolationForest

        detector = IsolationForest(contamination=contamination, random_state=42)
    elif method == "local_outlier_factor":
        from sklearn.neighbors import LocalOutlierFactor

        detector = LocalOutlierFactor(contamination=contamination)
    else:
        raise ValueError(f"Unknown method: {method}")

    outliers = detector.fit_predict(vectors)
    return outliers == -1


def cluster_vectors(
    vectors: np.ndarray, n_clusters: int, method: str = "kmeans"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster vectors.

    Args:
        vectors: Input vectors
        n_clusters: Number of clusters
        method: Clustering method ('kmeans', 'gmm', 'spectral')

    Returns:
        Tuple of (cluster_labels, cluster_centers)
    """
    if method == "kmeans":
        from sklearn.cluster import KMeans

        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        labels = clusterer.fit_predict(vectors)
        centers = clusterer.cluster_centers_
    elif method == "gmm":
        from sklearn.mixture import GaussianMixture

        clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = clusterer.fit_predict(vectors)
        centers = clusterer.means_
    elif method == "spectral":
        from sklearn.cluster import SpectralClustering

        clusterer = SpectralClustering(n_clusters=n_clusters, random_state=42)
        labels = clusterer.fit_predict(vectors)
        # Compute cluster centers manually for spectral clustering
        centers = np.array(
            [vectors[labels == i].mean(axis=0) for i in range(n_clusters)]
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return labels, centers


def evaluate_retrieval(
    true_indices: np.ndarray, retrieved_indices: np.ndarray, k: Optional[int] = None
) -> dict:
    """
    Evaluate retrieval performance.

    Args:
        true_indices: Ground truth indices
        retrieved_indices: Retrieved indices
        k: Evaluate at top-k (if None, use all)

    Returns:
        Dictionary of evaluation metrics
    """
    if k is not None:
        retrieved_indices = retrieved_indices[:, :k]

    # Precision at k
    precision_at_k = []
    recall_at_k = []

    for i, (true_idx, retr_idx) in enumerate(zip(true_indices, retrieved_indices)):
        true_set = set(true_idx)
        retr_set = set(retr_idx)

        intersection = true_set.intersection(retr_set)

        precision = len(intersection) / len(retr_set) if len(retr_set) > 0 else 0
        recall = len(intersection) / len(true_set) if len(true_set) > 0 else 0

        precision_at_k.append(precision)
        recall_at_k.append(recall)

    f1_scores = [
        2 * p * r / (p + r) if (p + r) > 0 else 0
        for p, r in zip(precision_at_k, recall_at_k)
    ]

    return {
        "precision_at_k": np.mean(precision_at_k),
        "recall_at_k": np.mean(recall_at_k),
        "f1_at_k": np.mean(f1_scores),
        "precision_std": np.std(precision_at_k),
        "recall_std": np.std(recall_at_k),
        "f1_std": np.std(f1_scores),
    }
