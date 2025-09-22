import logging
from typing import Any, Dict, List, Optional
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import gc

from benchmarks.base import Benchmark

logger = logging.getLogger(__name__)
HDF5_DISTANCE_DATASET_NAME = "distance_matrix"

DEFAULT_METRICS_PREFIX = "P@"


class PrecisionAtK(Benchmark):
    def _initialize(self, k_values: List[int] = [1, 5], **kwargs):
        """
        Initializes the PrecisionAtK benchmark parameters.

        Args:
            k_values (List[int]): A list of positive integers representing the
                k values for which to compute precision. Defaults to [1, 5].
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        if not k_values or not all(isinstance(k, int) and k > 0 for k in k_values):
            raise ValueError("k_values must be non-empty list of positive integers.")
        self.k_values = sorted(k_values)
        self.max_k = max(self.k_values)
        self._default_metrics = {f"{DEFAULT_METRICS_PREFIX}{k}": None for k in self.k_values}
        logger.info(f"Initialized PrecisionAtK (Avg Prop Mode) with k_values: {self.k_values}")

    def evaluate(
        self,
        *,
        distance_matrix: Optional[Any] = None,
        distance_matrix_path: Optional[Path] = None,
        labels: Optional[List[Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute P@K scores (as average proportion).

        Calculates the average proportion of neighbors within the top K closest
        points that belong to the same class as the query point. Operates on
        a distance matrix and a corresponding list of labels. Handles filtering
        out points with None labels.

        Args:
            distance_matrix (Optional[Any]): The distance matrix (numpy array or torch tensor).
                If None, distance_matrix_path is used as a fallback (though direct loading
                from path within this method is simplified/removed). Required.
            distance_matrix_path (Optional[Path]): Path to the distance matrix file (for context, not used).
            labels (Optional[List[Any]]): A list of labels for each data point,
                corresponding to the rows/columns of the distance matrix. Required.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: A dictionary containing the P@K scores for each
                            specified k value, or default None values if
                            required inputs are missing or invalid.
        """
        default_return = self._default_metrics.copy()
        dist_mat = distance_matrix

        if dist_mat is None:
            logger.error("PrecisionAtK requires distance_matrix.")
            return default_return

        try:
            if isinstance(dist_mat, np.ndarray):
                dist_mat_tensor = torch.from_numpy(dist_mat).float()
            elif isinstance(dist_mat, torch.Tensor):
                dist_mat_tensor = dist_mat.float()
            else:
                raise TypeError(f"Unsupported distance_matrix type: {type(dist_mat)}")

            if dist_mat_tensor.ndim != 2 or dist_mat_tensor.shape[0] != dist_mat_tensor.shape[1]:
                raise ValueError(f"Distance matrix must be square 2D, got {dist_mat_tensor.shape}")
            n_samples = dist_mat_tensor.shape[0]
            logger.info(f"Using provided distance matrix for P@K. Shape: {dist_mat_tensor.shape}")

        except Exception as e:
            logger.error("Error processing distance matrix: %s", e, exc_info=True)
            gc.collect()
            return default_return

        if labels is None:
            logger.error("PrecisionAtK requires 'labels'.")
            gc.collect()
            return default_return
        if not isinstance(labels, list):
            logger.error("PrecisionAtK received non-list labels: %s.", type(labels))
            gc.collect()
            return default_return
        if len(labels) != n_samples:
            logger.error("Label count %d != matrix (%d).", len(labels), n_samples)
            gc.collect()
            return default_return
        try:
            valid_labels_with_indices = [(idx, str(lbl)) for idx, lbl in enumerate(labels) if lbl is not None]
            if not valid_labels_with_indices:
                logger.error("No valid (non-None) labels.")
                gc.collect()
                return default_return
            unique_labels = sorted(list(set(lbl for _, lbl in valid_labels_with_indices)))
            label_to_id = {label: i for i, label in enumerate(unique_labels)}
            label_ids = torch.full((n_samples,), -1, dtype=torch.long)
            valid_sample_indices = []
            for idx, label_str in valid_labels_with_indices:
                label_id = label_to_id.get(label_str, -1)
                if label_id != -1:
                    label_ids[idx] = label_id
                    valid_sample_indices.append(idx)
            num_valid_samples_for_eval = len(valid_sample_indices)
            if num_valid_samples_for_eval == 0:
                logger.warning("No samples with valid labels.")
                gc.collect()
                return default_return
        except Exception as e:
            logger.error("Label processing failed: %s", e, exc_info=True)
            gc.collect()
            return default_return

        results = {}
        try:
            target_device = torch.device("cpu")
            dist_mat_tensor = dist_mat_tensor.to(target_device)
            label_ids = label_ids.to(target_device)
            dist_mat_tensor.fill_diagonal_(float("inf"))
            logger.debug("Sorting distances for top %d neighbors...", self.max_k)
            sorted_indices = torch.argsort(dist_mat_tensor, dim=1)[:, : self.max_k]
            logger.debug("Sorting complete.")
            neighbor_labels = label_ids[sorted_indices]
            true_labels_expanded = label_ids.view(-1, 1).expand(-1, self.max_k)
            matches = neighbor_labels == true_labels_expanded
            logger.info("Calculating P@K (Avg Prop) for %d valid queries...", num_valid_samples_for_eval)
            valid_query_mask = label_ids != -1
            for k in self.k_values:
                num_correct_at_k = matches[:, :k].sum(dim=1)
                num_correct_valid_queries_at_k = num_correct_at_k[valid_query_mask]
                total_correct_neighbors = torch.sum(num_correct_valid_queries_at_k.float()).item()
                total_neighbors_considered = num_valid_samples_for_eval * k
                average_proportion_at_k = total_correct_neighbors / total_neighbors_considered if total_neighbors_considered > 0 else 0.0
                metric_name = f"{DEFAULT_METRICS_PREFIX}{k}"
                results[metric_name] = average_proportion_at_k
                logger.debug("%s: %.4f (on %d valid queries)", metric_name, average_proportion_at_k, num_valid_samples_for_eval)
            logger.info("P@K (Avg Prop) calculation finished.")
        except Exception as e:
            logger.error("Error during P@K calculation: %s", e, exc_info=True)
            results = default_return
        finally:
            del dist_mat_tensor, label_ids, sorted_indices, neighbor_labels, true_labels_expanded, matches
            if "valid_query_mask" in locals():
                del valid_query_mask
            if "num_correct_at_k" in locals():
                del num_correct_at_k
            if "num_correct_valid_queries_at_k" in locals():
                del num_correct_valid_queries_at_k
            gc.collect()
        return results