"""Cosine similarity for single vectors and batched matrices."""

import numpy as np


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Cosine similarity of two equal-length vectors; returns 0.0 if either has zero norm."""
    if len(vec1) != len(vec2):
        raise ValueError(f"Vectors must have same length: {len(vec1)} != {len(vec2)}")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def batch_cosine_similarity(nd_array1: np.ndarray, nd_array2: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity matrix between two batches; output shape (N1, N2)."""
    if nd_array1.shape[1] != nd_array2.shape[1]:
        raise ValueError(
            f"Embedding dimensions must match: {nd_array1.shape[1]} != {nd_array2.shape[1]}",
        )

    dot_products = np.dot(nd_array1, nd_array2.T)
    norms1 = np.linalg.norm(nd_array1, axis=1)
    norms2 = np.linalg.norm(nd_array2, axis=1)
    norm_products = np.outer(norms1, norms2)
    # Guard against zero-norm rows to keep division finite.
    norm_products = np.where(norm_products == 0, 1e-10, norm_products)

    return dot_products / norm_products
