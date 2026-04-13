"""Vector similarity computation utilities.

Provides functions for calculating cosine similarity between vectors,
with support for both single vectors and batch operations using NumPy.
"""

import numpy as np


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate the cosine similarity between two numeric vectors.

    Cosine similarity measures the cosine of the angle between two vectors,
    returning a value between -1 (opposite) and 1 (identical direction).

    Args:
        vec1: First vector as a list of floats.
        vec2: Second vector as a list of floats.

    Returns:
        Cosine similarity value in range [-1.0, 1.0].
        Returns 0.0 if either vector has zero magnitude.

    Raises:
        ValueError: If vectors have different lengths.

    Examples:
        >>> cosine_similarity([1.0, 0.0], [1.0, 0.0])
        1.0
        >>> cosine_similarity([1.0, 0.0], [0.0, 1.0])
        0.0
        >>> cosine_similarity([1.0, 1.0], [-1.0, -1.0])
        -1.0
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"Vectors must have same length: {len(vec1)} != {len(vec2)}")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def batch_cosine_similarity(nd_array1: np.ndarray, nd_array2: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity matrix between two batches of vectors.

    Efficiently computes pairwise cosine similarities using matrix operations.

    Args:
        nd_array1: Matrix of shape (batch_size1, emb_size) representing
            the first batch of embedding vectors.
        nd_array2: Matrix of shape (batch_size2, emb_size) representing
            the second batch of embedding vectors.

    Returns:
        Similarity matrix of shape (batch_size1, batch_size2) where
        result[i, j] is the cosine similarity between nd_array1[i] and
        nd_array2[j]. Values are in range [-1.0, 1.0].

    Raises:
        ValueError: If embedding dimensions don't match between arrays.

    Examples:
        >>> import numpy as np
        >>> arr1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> arr2 = np.array([[1.0, 0.0], [1.0, 1.0]])
        >>> batch_cosine_similarity(arr1, arr2)
        array([[1.        , 0.70710678],
               [0.        , 0.70710678]])
    """
    if nd_array1.shape[1] != nd_array2.shape[1]:
        raise ValueError(
            f"Embedding dimensions must match: {nd_array1.shape[1]} != {nd_array2.shape[1]}"
        )

    # Compute dot products: (batch_size1, emb_size) @ (emb_size, batch_size2)
    # Result shape: (batch_size1, batch_size2)
    dot_products = np.dot(nd_array1, nd_array2.T)

    # Compute L2 norms for each vector
    norms1 = np.linalg.norm(nd_array1, axis=1)  # Shape: (batch_size1,)
    norms2 = np.linalg.norm(nd_array2, axis=1)  # Shape: (batch_size2,)

    # Compute outer product of norms: (batch_size1, 1) @ (1, batch_size2)
    # Result shape: (batch_size1, batch_size2)
    norm_products = np.outer(norms1, norms2)

    # Avoid division by zero
    norm_products = np.where(norm_products == 0, 1e-10, norm_products)

    # Compute cosine similarities
    return dot_products / norm_products
