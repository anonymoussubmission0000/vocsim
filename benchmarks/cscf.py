import logging
from typing import Any, Dict, List, Optional
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import gc

from benchmarks.base import Benchmark

logger = logging.getLogger(__name__)


class CSCFBenchmark(Benchmark):
    """
    Computes a Class Separation Confusion Fraction (PCCF).
    """

    def _initialize(self, **kwargs):
        """
        Initializes the benchmark.
        """
        pass

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
        Evaluates the Class Separation Confusion Fraction (PCCF).

        Args:
            distance_matrix (Optional[Any]): Directly provided distance matrix data.
            distance_matrix_path (Optional[Path]): Path to HDF5 file containing the distance matrix (not used).
            labels (Optional[List[Any]]): A list of labels corresponding to original item indices.
            min_class_size (int): Minimum size a true class must have to be included
                in the PCCF calculation. Must be >= 2.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            Dict[str, Any]: A dictionary containing the PCCF score ('pccf') and
                            an error message if applicable ('error').
        """
        default_return = {"pccf": None, "error": None}
        raw_dist_mat_input = distance_matrix
        n_samples_orig = 0
        dist_mat_tensor_for_calc = None

        if not isinstance(min_class_size, int) or min_class_size < 2:
            logger.warning(f"min_class_size must be >= 2. Setting to 2.")
            min_class_size = 2

        if raw_dist_mat_input is None:
            default_return["error"] = "distance_matrix not provided."
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

        try:
            if isinstance(raw_dist_mat_input, np.ndarray):
                dist_mat_tensor_for_calc = torch.from_numpy(raw_dist_mat_input).float()
            elif isinstance(raw_dist_mat_input, torch.Tensor):
                dist_mat_tensor_for_calc = raw_dist_mat_input.float()
            else:
                raise TypeError(f"Unsupported distance_matrix: {type(raw_dist_mat_input)}")

            if dist_mat_tensor_for_calc.ndim != 2 or dist_mat_tensor_for_calc.shape[0] != dist_mat_tensor_for_calc.shape[1]:
                raise ValueError(f"dist_mat not square 2D: {dist_mat_tensor_for_calc.shape}")
            n_samples_orig = dist_mat_tensor_for_calc.shape[0]
            dist_mat_tensor_for_calc.fill_diagonal_(0)
            logger.info(f"Using provided distance matrix for PCCF. Shape: {dist_mat_tensor_for_calc.shape}")

        except Exception as e:
            default_return["error"] = f"Distance matrix processing failed: {e}"
            gc.collect()
            return default_return

        if len(labels) != n_samples_orig:
            logger.warning(f"Label count {len(labels)} != matrix dim {n_samples_orig}. Adjusting.")
            if len(labels) > n_samples_orig:
                labels = labels[:n_samples_orig]
            else:
                labels = labels + [None] * (n_samples_orig - len(labels))

        original_indices_with_valid_labels = [i for i, lbl in enumerate(labels) if lbl is not None and 0 <= i < n_samples_orig]
        filtered_labels = [str(labels[i]) for i in original_indices_with_valid_labels]

        if not filtered_labels:
            default_return["error"] = "No valid labels found."
            gc.collect()
            return default_return

        if len(original_indices_with_valid_labels) != n_samples_orig:
            logger.info(f"Filtering distance matrix for {len(filtered_labels)} valid labels.")
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
            label_to_id = {lbl: i for i, lbl in enumerate(unique_label_list)}
            labels_numeric = np.array([label_to_id[lbl] for lbl in filtered_labels], dtype=int)

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
                        else:
                            logger.warning(f"Class ID {num_label_id} had no valid intra-pairs despite size {len(indices_this_class)}. Skipping.")

            valid_class_ids = list(valid_class_data.keys())
            num_valid_classes = len(valid_class_ids)

            if num_valid_classes < 2:
                logger.warning(f"PCCF requires at least 2 classes meeting min_class_size. Found {num_valid_classes}.")
                results["error"] = f"Need >= 2 classes with size >= {min_class_size} for PCCF."
                return results

            logger.info(f"Calculating PCCF based on {num_valid_classes} valid classes.")

            total_pairwise_comparisons = 0
            total_confusion_events = 0

            pbar_outer = tqdm(range(num_valid_classes), desc="PCCF Outer Loop (Class i)", leave=False)
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
                    avg_intra_dist_j = data_j["avg_intra_dist"]

                    inter_block_ij = dist_mat_tensor_for_calc[indices_i][:, indices_j]
                    if inter_block_ij.numel() == 0:
                        logger.warning(f"Empty inter-block between class {class_id_i} and {class_id_j}. Skipping this pair.")
                        continue
                    avg_inter_dist_ij = torch.mean(inter_block_ij).item()

                    total_pairwise_comparisons += 1
                    if avg_inter_dist_ij < avg_intra_dist_i:
                        total_confusion_events += 1

            pbar_outer.close()

            if total_pairwise_comparisons == 0:
                logger.warning("No pairwise class comparisons were made.")
                results["error"] = "No pairwise comparisons possible."
            else:
                pccf = total_confusion_events / total_pairwise_comparisons
                logger.info(
                    f"PCCF: Confusions={total_confusion_events}, Comparisons={total_pairwise_comparisons}, PCCF={pccf:.4f}"
                )
                results = {"pccf": pccf, "error": None}

        except Exception as e:
            logger.error(f"PCCF calculation failed: {e}", exc_info=True)
            results["pccf"] = None
            results["error"] = f"PCCF calculation failed: {e}"
        finally:
            del dist_mat_tensor_for_calc
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"CUDA cache clear failed: {e}")
        return results