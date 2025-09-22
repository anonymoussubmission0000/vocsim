# In benchmarks/gsr.py

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import gc

from benchmarks.base import Benchmark

logger = logging.getLogger(__name__)


class GlobalSeparationRate(Benchmark):
    """
    Computes a point-wise Global Separation Rate (GSR) to measure class integrity.

    This metric provides a robust, instance-level measure of class separability.
    For each data point, it calculates a "local separation score" by comparing two
    key distances:
    1.  NID (Nearest Inter-class Distance): The distance to the closest point
        belonging to any *other* class.
    2.  Avg_ID (Average Intra-class Distance): The average distance to all
        other points within its *own* class.

    The local score for each point `i` is a ratio that quantifies its separation:
    Local_Score(i) = (NID(i) - Avg_ID(i)) / (NID(i) + Avg_ID(i) + epsilon)

    The final GSR score is the average of these local scores over all evaluable
    points, normalized to a [0, 1] range. A score of 1.0 indicates excellent
    separation, while a score of 0.5 suggests that inter-class and intra-class
    distances are, on average, equal for most points.
    """

    def _initialize(self, min_class_size: int = 2, epsilon: float = 1e-9, **kwargs):
        """
        Initializes the GlobalSeparationRate benchmark parameters.

        Args:
            min_class_size (int): The minimum number of samples a class must have
                to be included in the evaluation. Must be at least 2.
            epsilon (float): A small constant to prevent division by zero in the
                local score calculation.
            **kwargs: Additional keyword arguments, including 'device'.
        """
        if not isinstance(min_class_size, int) or min_class_size < 2:
            logger.warning(f"min_class_size must be >= 2. Setting to 2.")
            self.min_class_size = 2
        else:
            self.min_class_size = min_class_size
        self.epsilon = epsilon

        # FIX: The device needs to be initialized from the kwargs passed by the manager.
        device_str = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        
        logger.info(
            "Initialized GlobalSeparationRate (GSR) Benchmark "
            f"(min_class_size={self.min_class_size}, epsilon={self.epsilon}, device={self.device})"
        )

    def evaluate(
        self,
        *,
        distance_matrix: Optional[Any] = None,
        labels: Optional[List[Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Computes the robust GSR score from a distance matrix and corresponding labels.

        Args:
            distance_matrix (Optional[Any]): A pre-computed square distance matrix
                (numpy array or torch tensor).
            labels (Optional[List[Any]]): A list of class labels corresponding to the
                rows/columns of the distance matrix.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A dictionary containing the final normalized 'gsr_score', a 'raw_gsr_score'
            in the [-1, 1] range, the number of points evaluated, and an error
            message if applicable.
        """
        start_time = time.time()
        default_return = {"gsr_score": None, "raw_gsr_score": None, "error": None}

        # --- 1. Input Validation and Data Preparation ---
        if distance_matrix is None:
            return {**default_return, "error": "distance_matrix is required."}
        if labels is None:
            return {**default_return, "error": "labels are required."}
        
        try:
            dist_mat = torch.from_numpy(distance_matrix).float() if isinstance(distance_matrix, np.ndarray) else distance_matrix.float()
            
            if dist_mat.ndim != 2 or dist_mat.shape[0] != dist_mat.shape[1]:
                raise ValueError(f"Distance matrix must be a square 2D tensor, got {dist_mat.shape}")

            n_samples_orig = dist_mat.shape[0]
            if len(labels) != n_samples_orig:
                raise ValueError(f"Label count ({len(labels)}) does not match matrix dimension ({n_samples_orig})")

            # Filter out samples with None labels
            valid_indices_mask = np.array([lbl is not None for lbl in labels])
            if not np.any(valid_indices_mask):
                raise ValueError("No valid (non-None) labels found.")
            labels_valid_str = np.array([str(lbl) for lbl in labels])[valid_indices_mask]
            
            # Identify classes large enough for evaluation
            unique_labels, counts = np.unique(labels_valid_str, return_counts=True)
            valid_classes = {label for label, count in zip(unique_labels, counts) if count >= self.min_class_size}

            if len(valid_classes) < 2:
                raise ValueError(f"GSR requires at least 2 classes with size >= {self.min_class_size} samples. Found {len(valid_classes)} valid classes from {len(unique_labels)} total.")

            # Filter the distance matrix and labels to only include valid classes
            final_filter_mask_in_valid = np.isin(labels_valid_str, list(valid_classes))
            original_indices_where_valid = np.where(valid_indices_mask)[0]
            final_indices = original_indices_where_valid[final_filter_mask_in_valid]

            if len(final_indices) == 0:
                raise ValueError("No samples remaining after filtering for min_class_size.")

            dist_mat = dist_mat[final_indices][:, final_indices]
            labels_array = np.array(labels)[final_indices]
            
            # Factorize string labels to integer IDs for efficient tensor operations
            labels_tensor = torch.tensor(pd.factorize(labels_array, sort=True)[0])
            n_samples_final = dist_mat.shape[0]
            
            # Move data to the specified evaluation device (e.g., CUDA)
            dist_mat = dist_mat.to(self.device)
            labels_tensor = labels_tensor.to(self.device)

        except ValueError as ve:
            return {**default_return, "error": str(ve)}
        except Exception as e:
            return {**default_return, "error": f"Data preparation failed: {e}"}

        # --- 2. Core Point-wise Score Calculation ---
        local_separation_scores = []
        pbar = tqdm(range(n_samples_final), desc="Calculating GSR", leave=False)

        for i in pbar:
            anchor_label_id = labels_tensor[i]
            distances_from_i = dist_mat[i]

            # Create boolean masks to identify intra-class and inter-class neighbors
            same_label_mask = (labels_tensor == anchor_label_id)
            same_label_mask[i] = False  # Exclude the point itself from its own neighbors
            diff_label_mask = (labels_tensor != anchor_label_id)

            # A point can only be evaluated if it has at least one intra-class neighbor
            # and at least one inter-class point exists in the dataset.
            if not torch.any(same_label_mask) or not torch.any(diff_label_mask):
                continue

            # Calculate Average Intra-class Distance (Avg_ID) for this point
            avg_id = torch.mean(distances_from_i[same_label_mask])

            # Calculate Nearest Inter-class Distance (NID) for this point
            nid = torch.min(distances_from_i[diff_label_mask])

            # Calculate the local separation score
            denominator = nid + avg_id + self.epsilon
            if denominator > self.epsilon:
                local_score = (nid - avg_id) / denominator
                local_separation_scores.append(local_score.item())

        pbar.close()

        if not local_separation_scores:
            return {**default_return, "error": "No points could be evaluated."}

        # --- 3. Final Score Aggregation ---
        # The raw score is the average of all local scores (ranges from -1 to 1)
        raw_gsr = np.mean(local_separation_scores)
        
        # Normalize the score to a [0, 1] range for intuitive interpretation in tables
        gsr_score_normalized = (raw_gsr + 1.0) / 2.0

        elapsed_time = time.time() - start_time
        logger.info(f"GSR benchmark finished in {elapsed_time:.2f}s")
        gc.collect()

        return {
            "gsr_score": float(gsr_score_normalized),
            "raw_gsr_score": float(raw_gsr),
            "num_points_evaluated": len(local_separation_scores),
            "error": None,
        }