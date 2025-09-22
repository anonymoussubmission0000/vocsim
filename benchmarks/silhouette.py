# In benchmarks/silhouette.py

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
import gc
from sklearn.metrics import silhouette_score, silhouette_samples

from benchmarks.base import Benchmark

logger = logging.getLogger(__name__)

class SilhouetteBenchmark(Benchmark):
    """
    Computes the Silhouette Score, a measure of cluster cohesion vs. separation.
    Operates on a pre-computed distance matrix and corresponding labels.
    """

    def _initialize(
        self,
        sample_size: Optional[int] = 5000,
        random_state: int = 42,
        metric: str = "precomputed",
        chunk_size: int = 2048,
        **kwargs
    ):
        """
        Initializes Silhouette benchmark parameters.

        Args:
            sample_size (Optional[int]): The number of samples to use for the calculation.
                If the dataset is larger, a random subset of this size is taken.
                If None, the entire dataset is used (may be slow or memory-intensive).
            random_state (int): Seed for the random sampler.
            metric (str): The metric for silhouette_score. Must be "precomputed" when
                a distance matrix is provided.
            chunk_size (int): Size of chunks for processing silhouette_samples if the
                full matrix is too large for memory.
            **kwargs: Additional keyword arguments.
        """
        self.sample_size = sample_size
        self.random_state = random_state
        if metric != "precomputed":
            raise ValueError("SilhouetteBenchmark with a distance matrix requires metric='precomputed'.")
        self.metric = metric
        self.chunk_size = chunk_size
        logger.info(
            f"Initialized SilhouetteBenchmark (sample_size={self.sample_size}, random_state={self.random_state})"
        )

    def evaluate(
        self,
        *,
        distance_matrix: Optional[Any] = None,
        labels: Optional[List[Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute the mean Silhouette Score.

        Args:
            distance_matrix (Optional[Any]): The distance matrix (numpy array or torch tensor). Required.
            labels (Optional[List[Any]]): A list of labels for each data point. Required.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Dict[str, Any]: A dictionary containing the 'silhouette_score' and an 'error' message if applicable.
        """
        start_time = time.time()
        default_return = {"silhouette_score": None, "error": None}

        if distance_matrix is None:
            return {**default_return, "error": "distance_matrix is required."}
        if labels is None:
            return {**default_return, "error": "labels are required."}

        try:
            if isinstance(distance_matrix, torch.Tensor):
                dist_mat_np = distance_matrix.cpu().numpy().astype(np.float32)
            else:
                dist_mat_np = np.array(distance_matrix, dtype=np.float32)

            if dist_mat_np.ndim != 2 or dist_mat_np.shape[0] != dist_mat_np.shape[1]:
                raise ValueError(f"distance_matrix must be square 2D, got shape {dist_mat_np.shape}")
            
            if not isinstance(labels, list):
                labels = list(labels)
            
            if len(labels) != dist_mat_np.shape[0]:
                 raise ValueError(f"Label count ({len(labels)}) != distance matrix dim ({dist_mat_np.shape[0]})")
            
            # --- Filtering for valid data ---
            valid_indices_mask = np.array([lbl is not None for lbl in labels])
            if not np.any(valid_indices_mask):
                raise ValueError("No valid (non-None) labels found.")
            
            labels_valid_str = np.array([str(lbl) for lbl in labels])[valid_indices_mask]
            
            # Identify classes with at least 2 members (required for silhouette)
            unique_labels, counts = np.unique(labels_valid_str, return_counts=True)
            valid_classes = {label for label, count in zip(unique_labels, counts) if count >= 2}

            if len(valid_classes) < 2:
                raise ValueError(f"Silhouette score requires at least 2 classes with >= 2 samples. Found {len(valid_classes)} valid classes.")

            final_filter_mask_in_valid = np.isin(labels_valid_str, list(valid_classes))
            original_indices_where_valid = np.where(valid_indices_mask)[0]
            final_indices = original_indices_where_valid[final_filter_mask_in_valid]

            if len(final_indices) == 0:
                raise ValueError("No samples remaining after filtering for min_class_size >= 2.")
            
            dist_mat_filtered = dist_mat_np[np.ix_(final_indices, final_indices)]
            labels_array_final = np.array(labels)[final_indices]
            
            n_samples_final = len(labels_array_final)
            logger.info(f"Filtered data for Silhouette. Kept {n_samples_final} samples across {len(valid_classes)} classes.")

            # --- Subsampling if necessary ---
            if self.sample_size is not None and n_samples_final > self.sample_size:
                logger.info(f"Subsampling {n_samples_final} samples down to {self.sample_size} for Silhouette calculation.")
                rng = np.random.RandomState(self.random_state)
                sample_indices = rng.choice(n_samples_final, self.sample_size, replace=False)
                
                dist_mat_sampled = dist_mat_filtered[np.ix_(sample_indices, sample_indices)]
                labels_sampled = labels_array_final[sample_indices]
                X_eval = dist_mat_sampled
                labels_eval = labels_sampled
            else:
                X_eval = dist_mat_filtered
                labels_eval = labels_array_final

            logger.info(f"Calculating Silhouette score on {len(labels_eval)} samples...")
            
            score = silhouette_score(X_eval, labels_eval, metric=self.metric)
            
            default_return["silhouette_score"] = float(score)

        except ValueError as ve:
            # Catch specific value errors from sklearn, which are informative
            default_return["error"] = f"Silhouette calculation failed: {ve}"
            logger.error(f"Silhouette calculation error: {ve}")
        except Exception as e:
            default_return["error"] = f"An unexpected error occurred during Silhouette calculation: {e}"
            logger.error(f"Silhouette calculation failed: {e}", exc_info=True)
        finally:
            gc.collect()

        elapsed_time = time.time() - start_time
        logger.info(f"Silhouette benchmark finished in {elapsed_time:.2f}s. Score: {default_return['silhouette_score']}")
        return default_return