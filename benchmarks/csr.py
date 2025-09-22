import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
import gc

from benchmarks.base import Benchmark

logger = logging.getLogger(__name__)


class ClassSeparationRatio(Benchmark):
    """
    Computes the Class Separation Ratio (CSR) score.

    For each class, CSR compares the average distance to the nearest point of a
    *different* class (Avg NID) with the average distance to the *furthest*
    point of the *same* class (Avg MID).

    Steps:
    1. For each point i: calculate MID(i) and NID(i).
    2. For each class C: calculate Avg_MID(C) and Avg_NID(C) over points i in C.
    3. For each class C: calculate Class_Sep(C) = (Avg_NID(C) - Avg_MID(C)) / (Avg_NID(C) + Avg_MID(C) + epsilon). Ranges [-1, 1].
    4. Calculate overall score as the weighted average of Class_Sep(C) by class size.
    5. Normalize the overall score to [0, 1] using (score + 1) / 2.

    Requires min_class_size >= 2. Requires at least 2 valid classes.
    Score close to 1 indicates good separation (AvgNID >> AvgMID consistently).
    Score close to 0 indicates poor separation (AvgMID >> AvgNID consistently).
    Score around 0.5 indicates AvgNID ≈ AvgMID on average.
    """

    def _initialize(self, min_class_size: int = 2, epsilon: float = 1e-9, **kwargs):
        """
        Initializes CSR benchmark parameters.

        Args:
            min_class_size (int): Minimum size required for a class to be included
                in the calculation. Must be >= 2.
            epsilon (float): Small value added to the denominator to prevent
                division by zero.
            **kwargs: Additional keyword arguments.
        """
        if not isinstance(min_class_size, int) or min_class_size < 2:
            logger.warning(f"min_class_size must be >= 2. Setting to 2.")
            self.min_class_size = 2
        else:
            self.min_class_size = min_class_size
        self.epsilon = epsilon
        logger.info(
            "Initialized ClassSeparationRatio (CSR) Benchmark"
            f" (min_class_size={self.min_class_size}, epsilon={self.epsilon})"
        )

    def evaluate(
        self,
        *,
        distance_matrix: Optional[Any] = None,
        labels: Optional[List[Any]] = None,
        distance_matrix_path: Optional[Any] = None,
        item_id_map: Optional[Dict[str, Dict]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute the CSR score.

        Args:
            distance_matrix (Optional[Any]): The distance matrix (numpy array or torch tensor).
                Required for this benchmark.
            labels (Optional[List[Any]]): A list of labels for each data point,
                corresponding to the rows/columns of the distance matrix. Required.
            distance_matrix_path (Optional[Any]): Path to the distance matrix file (for context, not used).
            item_id_map (Optional[Dict[str, Dict]]): Mapping of item IDs (for context, not used).
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: A dictionary containing the 'csr_score', 'avg_class_separation',
                            'num_classes_evaluated', 'num_points_evaluated', 'per_class_metrics',
                            'params', and 'error' if applicable.
        """
        start_time = time.time()
        default_return = {
            "csr_score": None,
            "avg_class_separation": None,
            "num_classes_evaluated": 0,
            "num_points_evaluated": 0,
            "per_class_metrics": {},
            "params": {"min_class_size": self.min_class_size, "epsilon": self.epsilon},
            "error": None,
        }
        dist_mat_np = None
        labels_array = None
        dist_mat_filtered = None

        if distance_matrix is None:
            default_return["error"] = "distance_matrix is required."
            return default_return
        if labels is None:
            default_return["error"] = "labels are required."
            return default_return
        try:
            if isinstance(distance_matrix, torch.Tensor):
                dist_mat_np = distance_matrix.cpu().numpy().astype(np.float32)
            elif isinstance(distance_matrix, np.ndarray):
                dist_mat_np = distance_matrix.astype(np.float32)
            else:
                dist_mat_np = np.array(distance_matrix, dtype=np.float32)
            if dist_mat_np.ndim != 2 or dist_mat_np.shape[0] != dist_mat_np.shape[1]:
                raise ValueError(f"distance_matrix must be square 2D, got shape {dist_mat_np.shape}")
            n_samples_orig = dist_mat_np.shape[0]
            if np.any(dist_mat_np < 0):
                logger.warning("Distance matrix contains negative values. Taking absolute.")
                dist_mat_np = np.abs(dist_mat_np)
            np.fill_diagonal(dist_mat_np, 0)
            if not isinstance(labels, list):
                labels = list(labels)
            if len(labels) != n_samples_orig:
                raise ValueError(f"Label count ({len(labels)}) != distance matrix dim ({n_samples_orig})")
            original_indices_with_valid_labels = [i for i, lbl in enumerate(labels) if lbl is not None]
            filtered_labels_str = [str(labels[i]) for i in original_indices_with_valid_labels]
            if not filtered_labels_str:
                raise ValueError("No valid (non-None) labels found.")
            if len(original_indices_with_valid_labels) != n_samples_orig:
                dist_mat_valid_labels = dist_mat_np[np.ix_(original_indices_with_valid_labels, original_indices_with_valid_labels)]
                labels_valid = np.array(filtered_labels_str)
            else:
                dist_mat_valid_labels = dist_mat_np
                labels_valid = np.array(filtered_labels_str)
            n_samples_valid = dist_mat_valid_labels.shape[0]
            if n_samples_valid == 0:
                raise ValueError("No samples remaining after filtering None labels.")
            unique_labels, counts = np.unique(labels_valid, return_counts=True)
            valid_classes = {label for label, count in zip(unique_labels, counts) if count >= self.min_class_size}
            if len(valid_classes) < 2:
                raise ValueError(f"Need at least 2 classes with >= {self.min_class_size} samples. Found {len(valid_classes)} valid classes.")
            final_filter_mask = np.isin(labels_valid, list(valid_classes))
            final_valid_indices = np.where(final_filter_mask)[0]
            if len(final_valid_indices) == 0:
                raise ValueError("No samples remaining after filtering by min_class_size.")
            if len(final_valid_indices) == n_samples_valid:
                dist_mat_filtered = dist_mat_valid_labels
                labels_array = labels_valid
            else:
                dist_mat_filtered = dist_mat_valid_labels[np.ix_(final_valid_indices, final_valid_indices)]
                labels_array = labels_valid[final_valid_indices]
            n_samples_final = dist_mat_filtered.shape[0]
            unique_final_labels = np.unique(labels_array)
            logger.info(f"Final dataset size for CSR calculation: {n_samples_final} points across {len(unique_final_labels)} classes.")
        except ValueError as ve:
            default_return["error"] = str(ve)
            logger.error(f"Filtering error: {ve}")
            gc.collect()
            return default_return
        except Exception as e:
            default_return["error"] = f"Filtering failed: {e}"
            logger.error(f"Filtering failed: {e}", exc_info=True)
            gc.collect()
            return default_return

        per_class_mid_sums: Dict[str, float] = {lbl: 0.0 for lbl in unique_final_labels}
        per_class_nid_sums: Dict[str, float] = {lbl: 0.0 for lbl in unique_final_labels}
        per_class_counts: Dict[str, int] = {lbl: 0 for lbl in unique_final_labels}
        evaluated_points_total = 0

        try:
            pbar = tqdm(range(n_samples_final), desc="Calculating MID/NID per point", leave=False)
            for i in pbar:
                anchor_label = labels_array[i]
                mid_i = None
                nid_i = None

                same_label_mask = labels_array == anchor_label
                same_label_mask[i] = False
                if np.any(same_label_mask):
                    dists_same = dist_mat_filtered[i, same_label_mask]
                    if dists_same.size > 0:
                        mid_i = np.max(dists_same)

                diff_label_mask = labels_array != anchor_label
                if np.any(diff_label_mask):
                    dists_diff = dist_mat_filtered[i, diff_label_mask]
                    if dists_diff.size > 0:
                        nid_i = np.min(dists_diff)

                if mid_i is not None and nid_i is not None:
                    per_class_mid_sums[anchor_label] += mid_i
                    per_class_nid_sums[anchor_label] += nid_i
                    per_class_counts[anchor_label] += 1
                    evaluated_points_total += 1
            pbar.close()

            if evaluated_points_total == 0:
                raise ValueError("No points had both MID and NID calculated.")

        except Exception as e:
            default_return["error"] = f"MID/NID calculation loop failed: {e}"
            logger.error(f"MID/NID calculation loop failed: {e}", exc_info=True)
            gc.collect()
            return default_return

        class_separation_scores = []
        class_weights = []
        per_class_metrics_detail = {}

        for class_label in unique_final_labels:
            count = per_class_counts[class_label]
            if count == 0:
                logger.warning(f"Class '{class_label}' had 0 valid points for MID/NID, skipping.")
                continue

            avg_mid = per_class_mid_sums[class_label] / count
            avg_nid = per_class_nid_sums[class_label] / count
            denominator = avg_nid + avg_mid + self.epsilon

            if denominator < self.epsilon:
                class_sep = 0.0
                logger.warning(f"Class '{class_label}': AvgNID and AvgMID are both near zero. Setting ClassSep=0.")
            else:
                class_sep = (avg_nid - avg_mid) / denominator

            class_separation_scores.append(class_sep)
            class_weights.append(count)
            per_class_metrics_detail[class_label] = {
                "avg_mid": float(avg_mid),
                "avg_nid": float(avg_nid),
                "class_sep": float(class_sep),
                "count": count,
            }

        if not class_separation_scores:
            default_return["error"] = "No classes had valid separation scores."
            logger.error("CSR: No class separation scores could be calculated.")
            gc.collect()
            return default_return

        weighted_avg_class_sep = np.average(class_separation_scores, weights=class_weights)

        csr_score = (weighted_avg_class_sep + 1.0) / 2.0

        default_return["csr_score"] = float(csr_score)
        default_return["avg_class_separation"] = float(weighted_avg_class_sep)
        default_return["num_classes_evaluated"] = len(class_separation_scores)
        default_return["num_points_evaluated"] = evaluated_points_total
        default_return["per_class_metrics"] = per_class_metrics_detail

        logger.info(f"CSR Score: {csr_score:.4f}")
        logger.info(f"  Avg Class Separation (Weighted, -1 to 1): {weighted_avg_class_sep:.4f}")
        logger.info(f"  Evaluated {evaluated_points_total} points across {len(class_separation_scores)} classes.")

        elapsed_time = time.time() - start_time
        logger.info(f"CSR benchmark finished in {elapsed_time:.2f}s")
        gc.collect()
        return default_return