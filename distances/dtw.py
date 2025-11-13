"""
DTW Distance Calculator for sequence-aware baseline.

Implements Dynamic Time Warping distance computation with:
- Sakoe-Chiba band constraint for efficiency
- Path-normalized costs to avoid length bias
- Support for frame-level sequence alignment
"""

import logging
from typing import Optional, List, Union, Tuple
import numpy as np
from distances.base import DistanceCalculator

logger = logging.getLogger(__name__)

try:
    from tslearn.metrics import dtw_path
    TSLEARN_AVAILABLE = True
except ImportError:
    TSLEARN_AVAILABLE = False
    logger.warning("tslearn not available. Install with: pip install tslearn")

try:
    from dtaidistance import dtw as dtai_dtw
    DTAIDISTANCE_AVAILABLE = True
except ImportError:
    DTAIDISTANCE_AVAILABLE = False
    logger.warning("dtaidistance not available. Install with: pip install dtaidistance")


class DTWDistance(DistanceCalculator):
    """
    Computes pairwise DTW distances between sequences with Sakoe-Chiba band constraint.

    This implements a sequence-aware baseline that aligns frame-level embeddings
    before comparing, addressing the "sequence averaging" concern while remaining
    training-free.
    """

    def _initialize(
        self,
        radius_ratio: float = 0.1,
        local_distance: str = "euclidean",
        backend: str = "tslearn",
        normalize_by_path: bool = True,
        **kwargs
    ):
        """
        Initialize DTW distance calculator.

        Args:
            radius_ratio: Sakoe-Chiba band radius as ratio of max(Tx, Ty).
                         Default 0.1 (10% of sequence length).
            local_distance: Local frame-frame distance metric.
                           "euclidean" (default). For cosine-like distance,
                           L2-normalize sequences before passing to DTW.
            backend: Implementation to use - "tslearn" or "dtaidistance".
            normalize_by_path: If True, divide total cost by path length.

        Note:
            tslearn.metrics.dtw_path uses Euclidean distance internally.
            For cosine-like distance, ensure input sequences are L2-normalized,
            which makes Euclidean distance equivalent to cosine distance.
        """
        self.radius_ratio = radius_ratio
        self.local_distance = local_distance
        self.backend = backend
        self.normalize_by_path = normalize_by_path

        # Check backend availability
        if self.backend == "tslearn" and not TSLEARN_AVAILABLE:
            raise ImportError("tslearn backend requested but not installed. Run: pip install tslearn")
        elif self.backend == "dtaidistance" and not DTAIDISTANCE_AVAILABLE:
            raise ImportError("dtaidistance backend requested but not installed. Run: pip install dtaidistance")

        logger.info(
            f"DTW initialized: backend={backend}, radius_ratio={radius_ratio}, "
            f"local_distance={local_distance} (note: tslearn uses Euclidean; normalize seqs for cosine), "
            f"normalize_by_path={normalize_by_path}"
        )

    def _dtw_single_pair_tslearn(
        self,
        seq_a: np.ndarray,
        seq_b: np.ndarray
    ) -> float:
        """
        Compute DTW distance between two sequences using tslearn.

        Args:
            seq_a: First sequence [Ta, D]
            seq_b: Second sequence [Tb, D]

        Returns:
            DTW distance (normalized by path length if enabled)
        """
        if seq_a.shape[0] == 0 or seq_b.shape[0] == 0:
            logger.warning("Empty sequence encountered in DTW computation")
            return float('inf')

        # Calculate Sakoe-Chiba radius
        max_len = max(seq_a.shape[0], seq_b.shape[0])
        radius = int(self.radius_ratio * max_len)

        try:
            # Compute DTW path
            # Note: tslearn.metrics.dtw_path uses Euclidean distance by default
            # For cosine distance, sequences should be L2-normalized beforehand
            path, total_cost = dtw_path(
                seq_a,
                seq_b,
                global_constraint="sakoe_chiba",
                sakoe_chiba_radius=radius
            )

            # Normalize by path length to avoid length bias
            if self.normalize_by_path:
                cost = total_cost / max(1, len(path))
            else:
                cost = total_cost

            return float(cost)

        except Exception as e:
            logger.error(f"DTW computation failed: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')

    def _dtw_single_pair_dtaidistance(
        self,
        seq_a: np.ndarray,
        seq_b: np.ndarray
    ) -> float:
        """
        Compute DTW distance between two sequences using dtaidistance.

        Args:
            seq_a: First sequence [Ta, D]
            seq_b: Second sequence [Tb, D]

        Returns:
            DTW distance (normalized by path length if enabled)
        """
        if seq_a.shape[0] == 0 or seq_b.shape[0] == 0:
            logger.warning("Empty sequence encountered in DTW computation")
            return float('inf')

        # Calculate Sakoe-Chiba radius
        max_len = max(seq_a.shape[0], seq_b.shape[0])
        radius = int(self.radius_ratio * max_len)

        try:
            # dtaidistance expects 1D sequences for univariate
            # For multivariate, we need to compute distance per dimension and aggregate
            if seq_a.ndim == 1:
                # Univariate case
                cost = dtai_dtw.distance(
                    seq_a,
                    seq_b,
                    window=radius,
                    use_c=True
                )
            else:
                # Multivariate case - compute Euclidean distance per frame pair
                # This is a simplified implementation; tslearn is preferred for multivariate
                logger.warning("dtaidistance backend with multivariate sequences may be slow. Consider using tslearn.")

                # Flatten approach: treat as univariate concatenated sequence
                seq_a_flat = seq_a.flatten()
                seq_b_flat = seq_b.flatten()
                cost = dtai_dtw.distance(
                    seq_a_flat,
                    seq_b_flat,
                    window=radius,
                    use_c=True
                )

            # Note: dtaidistance doesn't return path, so path-normalization is approximate
            if self.normalize_by_path:
                # Approximate path length as average of sequence lengths
                approx_path_len = (seq_a.shape[0] + seq_b.shape[0]) / 2
                cost = cost / max(1, approx_path_len)

            return float(cost)

        except Exception as e:
            logger.error(f"DTW computation failed: {e}")
            return float('inf')

    def compute_pairwise(
        self,
        features_X: Union[List[np.ndarray], np.ndarray],
        features_Y: Optional[Union[List[np.ndarray], np.ndarray]] = None
    ) -> np.ndarray:
        """
        Compute pairwise DTW distance matrix between sequences.

        Args:
            features_X: List of sequences [N samples], each [Ti, D], or single sequence [T, D]
            features_Y: Optional second list of sequences [M samples], each [Tj, D]
                       If None, compute distances between X and X

        Returns:
            Distance matrix [N, M] or [N, N] if features_Y is None
        """
        # Handle single sequence input
        if isinstance(features_X, np.ndarray) and features_X.ndim == 2:
            features_X = [features_X]

        if features_Y is not None:
            if isinstance(features_Y, np.ndarray) and features_Y.ndim == 2:
                features_Y = [features_Y]

        # Validate input
        if not isinstance(features_X, list):
            raise TypeError("features_X must be a list of numpy arrays")

        n_samples_X = len(features_X)

        if features_Y is None:
            # Compute X vs X
            n_samples_Y = n_samples_X
            symmetric = True
            features_Y = features_X
        else:
            if not isinstance(features_Y, list):
                raise TypeError("features_Y must be a list of numpy arrays")
            n_samples_Y = len(features_Y)
            symmetric = False

        # Initialize distance matrix
        distance_matrix = np.zeros((n_samples_X, n_samples_Y), dtype=np.float32)

        # Select backend
        if self.backend == "tslearn":
            dtw_func = self._dtw_single_pair_tslearn
        else:
            dtw_func = self._dtw_single_pair_dtaidistance

        logger.info(
            f"Computing DTW distance matrix: {n_samples_X} x {n_samples_Y} "
            f"(symmetric={symmetric})"
        )

        # Compute pairwise distances
        total_comparisons = n_samples_X * n_samples_Y
        if symmetric:
            total_comparisons = n_samples_X * (n_samples_X - 1) // 2

        computed = 0
        for i in range(n_samples_X):
            # For symmetric case, only compute upper triangle
            start_j = i + 1 if symmetric else 0

            for j in range(start_j, n_samples_Y):
                dist = dtw_func(features_X[i], features_Y[j])
                distance_matrix[i, j] = dist

                if symmetric:
                    distance_matrix[j, i] = dist

                computed += 1
                if computed % 100 == 0:
                    logger.debug(f"DTW progress: {computed}/{total_comparisons} pairs")

        # Set diagonal to zero for symmetric case
        if symmetric:
            np.fill_diagonal(distance_matrix, 0.0)

        logger.info("DTW distance matrix computation complete")
        return distance_matrix
