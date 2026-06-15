"""Site-label derivation for the classifier output head.

For datasets with discrete sampling sites (popgen-canonical: multiple
individuals share each site centroid), derive integer site labels from
exact coord equivalence among training samples. The classifier head
trains on these labels and stores the matching centroid matrix for
inference.
"""

from __future__ import annotations

import math

import numpy as np


def derive_site_labels(
    locs: np.ndarray,
    min_samples_per_site: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Group training locations by exact coord equivalence into integer labels.

    :param locs: ``(n, 2)`` float array of training-sample coordinates
        (lon, lat or z-scored equivalents — any 2D real coords).
        Must contain no NaNs (caller is responsible for filtering
        prediction-target samples out before calling).
    :type locs: np.ndarray
    :param min_samples_per_site: Reject the partition when K (number of
        unique coord pairs) exceeds ``ceil(n / min_samples_per_site)``. Default
        2 means "every site must have at least 2 samples on average; sub-1.5
        ratios still pass via ceil rounding." Set to 1 to disable
        the check entirely.
    :type min_samples_per_site: int

    :returns: Tuple ``(labels, centroids)`` where ``labels`` is an
        ``(n,)`` int32 array assigning each sample to its site index,
        and ``centroids`` is a ``(K, 2)`` float64 array of unique coords
        in label order (so ``centroids[labels[i]] == locs[i]``).
    :rtype: tuple[np.ndarray, np.ndarray]

    :raises ValueError: If ``locs`` contains NaN rows, or if K exceeds
        ``ceil(n / min_samples_per_site)``.
    """
    if locs.ndim != 2 or locs.shape[1] != 2:
        raise ValueError(f"locs must have shape (n, 2); got {locs.shape}")
    if np.isnan(locs).any():
        raise ValueError(
            "derive_site_labels: locs contains NaN rows; filter prediction "
            "targets out before calling"
        )

    n = locs.shape[0]
    centroids, labels = np.unique(locs, axis=0, return_inverse=True)
    k = centroids.shape[0]

    # ceil avoids spuriously rejecting borderline cases like K=2, n=3, min=2:
    # float threshold 1.5 would reject a valid 2-site, 3-sample dataset.
    threshold = math.ceil(n / min_samples_per_site)
    if k > threshold:
        raise ValueError(
            f"derive_site_labels: K={k} unique coords for n={n} samples — "
            f"classification requires multiple samples per coord pair "
            f"(threshold: K <= ceil(n / min_samples_per_site) = {threshold}). "
            f"This is the elephant case where every individual has unique GPS. "
            f"Use prediction_mode='regress' instead, or pass "
            f"min_samples_per_site=1 to override."
        )

    return labels.astype(np.int32), centroids
