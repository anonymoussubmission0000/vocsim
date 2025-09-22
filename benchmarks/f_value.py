import logging
from typing import Any, Dict, List, Optional
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import gc
import h5py

from benchmarks.base import Benchmark

logger = logging.getLogger(__name__)
HDF5_DISTANCE_DATASET_NAME = "distance_matrix"
DEFAULT_MEMORY_LIMIT_GB = 4.0


class FValueBenchmark(Benchmark):
    """
    Computes the F-Value, averaged.
    For each ordered pair of distinct valid classes (C_i, C_j):
    1. Calculate AvgIntra(C_i).
    2. Calculate AvgInter(C_i, C_j).
    3. F_orig_ij = AvgInter(C_i, C_j) / AvgIntra(C_i).
    4. F_transformed_ij = 1 / (1 + F_orig_ij).
    The final reported F-value is the mean of these F_transformed_ij values
    over all M * (M-1) ordered pairs.
    Result is in [0, 1], where 0 indicates better separation on average.
    Operates directly on the provided distance matrix.
    """

    def _initialize(self, memory_limit_gb: float = DEFAULT_MEMORY_LIMIT_GB, **kwargs):
        """
        Initializes the Pairwise F-Value benchmark parameters.

        Args:
            memory_limit_gb (float): The maximum amount of memory in GB allowed
                for loading the distance matrix. If the matrix size exceeds this,
                the benchmark will be skipped.
            **kwargs: Additional keyword arguments.
        """
        self.memory_limit_gb = memory_limit_gb
        logger.info(f"Initialized PairwiseFValueBenchmark with Memory Limit: {self.memory_limit_gb} GB")

    def _estimate_memory_gb(self, num_samples: int) -> float:
        """Estimates the memory required for a square float32 matrix."""
        bytes_needed = num_samples * num_samples * 4
        return bytes_needed / (1024**3)

    def evaluate(
        self,
        *,
        distance_matrix: Optional[Any] = None,
        distance_matrix_path: Optional[Path] = None,
        labels: Optional[List[Any]] = None,
        min_class_size: int = 2,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Computes the mean pairwise transformed F-value.

        Args:
            distance_matrix (Optional[Any]): The distance matrix (numpy array or torch tensor).
                If None, distance_matrix_path must be provided and loadable within memory_limit_gb.
            distance_matrix_path (Optional[Path]): Path to an HDF5 file containing the distance matrix.
                Used if distance_matrix is None.
            labels (Optional[List[Any]]): A list of labels for each data point,
                corresponding to the rows/columns of the distance matrix. Required.
            min_class_size (int): Minimum size required for a class to be included
                in the calculation. Must be >= 2.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: A dictionary containing the 'pairwise_f_value' and
                            an error message if applicable ('error').
        """
        default_return = {"pairwise_f_value": None, "error": None}
        raw_dist_mat_input = distance_matrix
        dist_mat_tensor_for_calc = None
        n_samples_orig = 0

        if raw_dist_mat_input is None:
            if distance_matrix_path is None or not distance_matrix_path.exists():
                default_return["error"] = "Dist matrix path missing/invalid."
                return default_return
            try:
                with h5py.File(distance_matrix_path, "r") as f:
                    if HDF5_DISTANCE_DATASET_NAME not in f:
                        raise ValueError("HDF5 dataset name not found")
                    dset = f[HDF5_DISTANCE_DATASET_NAME]
                    matrix_shape = dset.shape
                    if len(matrix_shape) != 2 or matrix_shape[0] != matrix_shape[1]:
                        raise ValueError("Matrix not square 2D")
                    n_samples_orig = matrix_shape[0]
                estimated_gb = self._estimate_memory_gb(n_samples_orig)
                if estimated_gb > self.memory_limit_gb:
                    error_msg = f"Skipped: Est. memory ({estimated_gb:.2f}GB) > limit ({self.memory_limit_gb:.2f}GB)"
                    default_return["error"] = error_msg
                    return default_return
            except Exception as e:
                default_return["error"] = f"Failed read dist matrix meta: {e}"
                return default_return
            logger.warning("PairwiseFValue: distance_matrix not provided directly. Attempting load from path.")
            try:
                with h5py.File(distance_matrix_path, "r") as f:
                    raw_dist_mat_input = f[HDF5_DISTANCE_DATASET_NAME][()]
            except Exception as e:
                default_return["error"] = f"Failed to load dist matrix from path: {e}"
                return default_return

        elif isinstance(raw_dist_mat_input, (np.ndarray, torch.Tensor)):
            if raw_dist_mat_input.ndim != 2 or raw_dist_mat_input.shape[0] != raw_dist_mat_input.shape[1]:
                default_return["error"] = f"Passed distance_matrix not square 2D: {raw_dist_mat_input.shape}"
                return default_return
            n_samples_orig = raw_dist_mat_input.shape[0]
            estimated_gb = self._estimate_memory_gb(n_samples_orig)
            if estimated_gb > self.memory_limit_gb:
                default_return["error"] = f"Passed matrix too large ({estimated_gb:.2f}GB > limit)."
                return default_return
        else:
            default_return["error"] = "Invalid distance_matrix type."
            return default_return

        if labels is None:
            default_return["error"] = "Labels missing."
            return default_return
        if not isinstance(labels, list):
            try:
                labels = list(labels)
            except TypeError:
                default_return["error"] = f"Labels must be list, got {type(labels)}."
                return default_return
        if len(labels) != n_samples_orig:
            logger.warning(f"Label count {len(labels)} != matrix dim {n_samples_orig}. Adjusting.")
            labels = labels[:n_samples_orig] if len(labels) > n_samples_orig else labels + [None] * (n_samples_orig - len(labels))

        try:
            if isinstance(raw_dist_mat_input, np.ndarray):
                dist_mat_tensor_for_calc = torch.from_numpy(raw_dist_mat_input).float()
            elif isinstance(raw_dist_mat_input, torch.Tensor):
                dist_mat_tensor_for_calc = raw_dist_mat_input.float()
            else:
                raise TypeError(f"Unsupported matrix type: {type(raw_dist_mat_input)}")
            if dist_mat_tensor_for_calc.shape[0] != n_samples_orig:
                raise ValueError(f"Matrix shape {dist_mat_tensor_for_calc.shape} mismatch")
            dist_mat_tensor_for_calc.fill_diagonal_(0)
        except Exception as e:
            default_return["error"] = f"Dist matrix processing failed: {e}"
            gc.collect()
            return default_return

        original_indices_with_valid_labels = [i for i, lbl in enumerate(labels) if lbl is not None and 0 <= i < n_samples_orig]
        filtered_labels = [str(labels[i]) for i in original_indices_with_valid_labels]
        if not filtered_labels:
            default_return["error"] = "No valid labels found."
            gc.collect()
            return default_return

        if len(original_indices_with_valid_labels) != n_samples_orig:
            try:
                valid_torch_indices = torch.tensor(original_indices_with_valid_labels, dtype=torch.long, device=dist_mat_tensor_for_calc.device)
                dist_mat_tensor_for_calc = dist_mat_tensor_for_calc[valid_torch_indices][:, valid_torch_indices]
            except Exception as e:
                default_return["error"] = f"Error filtering dist matrix: {e}"
                gc.collect()
                return default_return

        n_samples_valid = dist_mat_tensor_for_calc.shape[0]
        if n_samples_valid != len(filtered_labels):
            default_return["error"] = "Internal filter mismatch."
            gc.collect()
            return default_return

        results = default_return.copy()
        try:
            unique_label_list = sorted(list(set(filtered_labels)))
            label_to_id_map = {lbl: i for i, lbl in enumerate(unique_label_list)}
            labels_numeric = np.array([label_to_id_map[lbl] for lbl in filtered_labels], dtype=int)

            valid_class_data = {}
            all_numeric_ids = np.unique(labels_numeric)

            for num_label_id in all_numeric_ids:
                indices_this_class = np.where(labels_numeric == num_label_id)[0]
                if len(indices_this_class) >= min_class_size:
                    intra_class_block = dist_mat_tensor_for_calc[indices_this_class][:, indices_this_class]
                    if len(indices_this_class) > 1:
                        intra_mask = ~torch.eye(len(indices_this_class), dtype=torch.bool, device=intra_class_block.device)
                        valid_intra_distances = intra_class_block[intra_mask]
                        if valid_intra_distances.numel() > 0:
                            avg_intra_dist = torch.mean(valid_intra_distances).item()
                            valid_class_data[num_label_id] = {
                                "indices": torch.tensor(indices_this_class, device=dist_mat_tensor_for_calc.device),
                                "avg_intra_dist": avg_intra_dist,
                            }

            valid_class_ids = list(valid_class_data.keys())
            num_valid_classes = len(valid_class_ids)

            if num_valid_classes < 2:
                results["error"] = f"Need >= 2 classes (size >= {min_class_size}) for Pairwise F-value."
                return results

            logger.info(f"Calculating Pairwise F-values for {num_valid_classes} valid classes.")

            all_pairwise_transformed_f_values = []
            epsilon = 1e-9

            pbar_outer = tqdm(range(num_valid_classes), desc="Pairwise F-Value (Class i)", leave=False)
            for i_idx in pbar_outer:
                class_id_i = valid_class_ids[i_idx]
                data_i = valid_class_data[class_id_i]
                indices_i = data_i["indices"]
                avg_intra_dist_i = data_i["avg_intra_dist"]

                for j_idx in range(num_valid_classes):
                    if i_idx == j_idx:
                        continue

                    class_id_j = valid_class_ids[j_idx]
                    data_j = valid_class_data[class_id_j]
                    indices_j = data_j["indices"]

                    inter_block_ij = dist_mat_tensor_for_calc[indices_i][:, indices_j]
                    if inter_block_ij.numel() == 0:
                        logger.warning(f"Empty inter-block C{class_id_i}-C{class_id_j}. Skipping.")
                        continue
                    avg_inter_dist_ij = torch.mean(inter_block_ij).item()

                    f_orig_ij: Optional[float] = None
                    f_transformed_ij: Optional[float] = None

                    if abs(avg_intra_dist_i) < epsilon:
                        if abs(avg_inter_dist_ij) >= epsilon:
                            f_orig_ij = float("inf")
                            f_transformed_ij = 0.0
                        else:
                            f_orig_ij = 1.0
                            f_transformed_ij = 0.5
                    elif abs(avg_inter_dist_ij) < epsilon and abs(avg_intra_dist_i) >= epsilon:
                        f_orig_ij = 0.0
                        f_transformed_ij = 1.0
                    else:
                        f_orig_ij = avg_inter_dist_ij / avg_intra_dist_i
                        f_transformed_ij = 1.0 / (1.0 + f_orig_ij)

                    all_pairwise_transformed_f_values.append(f_transformed_ij)
            pbar_outer.close()

            if not all_pairwise_transformed_f_values:
                results["error"] = "No pairwise F-values could be calculated."
            else:
                final_pairwise_f_value = np.mean(all_pairwise_transformed_f_values)
                logger.info(
                    "Mean Pairwise Transformed F-Value (0=best):"
                    f" {final_pairwise_f_value:.4f} (from {len(all_pairwise_transformed_f_values)} pairs)"
                )
                results = {"pairwise_f_value": final_pairwise_f_value, "error": None}

        except Exception as e:
            logger.error(f"Pairwise F-Value calculation failed: {e}", exc_info=True)
            results["error"] = str(e)
        finally:
            del dist_mat_tensor_for_calc
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        return results