import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
from collections import Counter
import time
from pathlib import Path
import gc
from sklearn.cluster import HDBSCAN
from sklearn.metrics.cluster import contingency_matrix
import umap

from benchmarks.base import Benchmark
from utils.file_utils import load_hdf5, read_hdf5_metadata

logger = logging.getLogger(__name__)
HDF5_FEATURE_DATASET_NAME = "features"
HDF5_DISTANCE_DATASET_NAME = "distance_matrix"

DEFAULT_CP_METRICS = {
    "weighted_purity": None,
    "num_clusters_found": None,
    "num_noise_points": None,
    "error": None,
}


class ClusteringPurity(Benchmark):
    """
    Evaluates feature separation using UMAP projection and HDBSCAN clustering.
    Loads required features or distance matrix from HDF5 file internally.
    """

    def _initialize(
        self,
        umap_n_components: int = 2,
        umap_metric: str = "cosine",
        umap_low_memory: bool = True,
        umap_random_state: int = 42,
        hdbscan_min_cluster_size: int = 5,
        hdbscan_min_samples: Optional[int] = None,
        hdbscan_metric: str = "euclidean",
        use_distance_matrix_for_umap: bool = False,
        **kwargs,
    ):
        """
        Initializes parameters for UMAP, HDBSCAN, and data loading.

        Args:
            umap_n_components (int): Number of dimensions for UMAP projection.
            umap_metric (str): Metric to use for UMAP ('cosine', 'euclidean', etc.).
            umap_low_memory (bool): Whether to use UMAP's low_memory mode.
            umap_random_state (int): Random state for UMAP.
            hdbscan_min_cluster_size (int): Minimum size of clusters for HDBSCAN.
            hdbscan_min_samples (Optional[int]): Minimum samples in neighborhood for HDBSCAN.
            hdbscan_metric (str): Metric for HDBSCAN.
            use_distance_matrix_for_umap (bool): If True, loads distance matrix; otherwise loads features.
            **kwargs: Additional keyword arguments.
        """
        self.use_dist_matrix = use_distance_matrix_for_umap
        _umap_metric = "precomputed" if self.use_dist_matrix else umap_metric
        self.umap_params = {
            "n_components": umap_n_components,
            "metric": _umap_metric,
            "low_memory": umap_low_memory,
            "random_state": umap_random_state,
            "verbose": False,
            "init": "random",
        }
        self.hdbscan_params = {
            "min_cluster_size": hdbscan_min_cluster_size,
            "min_samples": hdbscan_min_samples if hdbscan_min_samples else hdbscan_min_cluster_size,
            "metric": hdbscan_metric,
            "n_jobs": -1,
        }

    def evaluate(
        self,
        *,
        distance_matrix_path: Optional[Path] = None,
        feature_hdf5_path: Optional[Path] = None,
        distance_matrix: Optional[Any] = None,
        features: Optional[Any] = None,
        dataset: Optional[Any] = None,
        labels: Optional[List[Any]] = None,
        item_id_map: Optional[Dict[str, Dict]] = None,
        min_class_size_for_purity: int = 1,
        feature_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Performs UMAP dimensionality reduction, HDBSCAN clustering on the projection,
        and calculates the weighted purity of the clusters against true labels.

        Loads data (either features or a distance matrix) from HDF5 files or
        uses directly provided data. Filters data and labels to ensure correspondence
        and handle missing labels.

        Args:
            distance_matrix_path (Optional[Path]): Path to HDF5 file containing the distance matrix.
            feature_hdf5_path (Optional[Path]): Path to HDF5 file containing the features.
            distance_matrix (Optional[Any]): Directly provided distance matrix data.
            features (Optional[Any]): Directly provided features data.
            dataset (Optional[Any]): Contextual dataset object (not directly used).
            labels (Optional[List[Any]]): A list of labels corresponding to original item indices.
            item_id_map (Optional[Dict[str, Dict]]): Mapping from original item IDs (not directly used).
            min_class_size_for_purity (int): Minimum size a true class must have to be included
                in the purity calculation.
            feature_config (Optional[Dict[str, Any]]): Contextual feature configuration (not directly used).
            **kwargs: Additional keyword arguments (not used).

        Returns:
            Dict[str, Any]: A dictionary containing weighted purity, number of clusters found,
                            number of noise points, sample count after filtering, timing information,
                            and an error message if applicable.
        """
        input_data = None
        input_data_source = None
        n_samples = 0
        filtered_labels = []
        metadata = None
        data_load_success = False

        logger.info(f"ClusteringPurity: use_distance_matrix={self.use_dist_matrix}")
        if self.use_dist_matrix:
            if distance_matrix is not None:
                logger.debug("Using directly provided distance_matrix.")
                input_data = distance_matrix
                input_data_source = "direct_distance_matrix"
                data_load_success = True
            elif distance_matrix_path is not None and distance_matrix_path.exists():
                logger.debug(f"Loading distance matrix from HDF5: {distance_matrix_path}")
                input_data = load_hdf5(distance_matrix_path, HDF5_DISTANCE_DATASET_NAME, "Distance Matrix")
                input_data_source = str(distance_matrix_path)
                if input_data is not None:
                    metadata = read_hdf5_metadata(distance_matrix_path, HDF5_DISTANCE_DATASET_NAME)
                    data_load_success = True
            if not data_load_success:
                return {**DEFAULT_CP_METRICS, "error": "Distance matrix required but not provided/loaded"}
            try:
                if not isinstance(input_data, np.ndarray):
                    input_data = np.array(input_data, dtype=np.float32)
                if input_data.ndim != 2 or input_data.shape[0] != input_data.shape[1]:
                    raise ValueError(f"Loaded distance matrix invalid shape: {input_data.shape}")
                n_samples = input_data.shape[0]
            except Exception as e:
                return {**DEFAULT_CP_METRICS, "error": f"Invalid distance matrix format: {e}"}
            if self.umap_params.get("metric") != "precomputed":
                logger.warning("Using distance matrix but UMAP metric is not 'precomputed'. Forcing.")
                self.umap_params["metric"] = "precomputed"

        else:
            if features is not None:
                logger.debug("Using directly provided features.")
                input_data = features
                input_data_source = "direct_features"
                data_load_success = True
            elif feature_hdf5_path is not None and feature_hdf5_path.exists():
                logger.debug(f"Loading features from HDF5: {feature_hdf5_path}")
                input_data = load_hdf5(feature_hdf5_path, HDF5_FEATURE_DATASET_NAME, "Features")
                input_data_source = str(feature_hdf5_path)
                if input_data is not None:
                    metadata = read_hdf5_metadata(feature_hdf5_path, HDF5_FEATURE_DATASET_NAME)
                    data_load_success = True
            if not data_load_success:
                return {**DEFAULT_CP_METRICS, "error": "Features required but not provided/loaded"}
            if self.umap_params.get("metric") == "precomputed":
                logger.warning("Using features but UMAP metric is 'precomputed'. Changing to 'cosine'.")
                self.umap_params["metric"] = "cosine"

            try:
                if not isinstance(input_data, np.ndarray):
                    input_data = np.array(input_data, dtype=np.float32)
                n_samples = input_data.shape[0]
                if input_data.ndim > 2:
                    logger.info(f"Input features >2D ({input_data.shape}). Flattening non-sample dims for UMAP/HDBSCAN.")
                    input_data = input_data.reshape(n_samples, -1)
                elif input_data.ndim == 1:
                    logger.warning(f"Input features are 1D ({input_data.shape}). Reshaping to (N, 1).")
                    input_data = input_data.reshape(-1, 1)
                elif input_data.ndim == 0:
                    raise ValueError("Cannot use scalar features")
                elif input_data.ndim != 2:
                    raise ValueError(f"Unexpected feature dimensionality: {input_data.ndim}")
            except Exception as e:
                return {**DEFAULT_CP_METRICS, "error": f"Invalid feature format/flattening failed: {e}"}

        logger.info(f"Loaded data for ClusteringPurity. Shape: {input_data.shape}, Source: {input_data_source}")

        if labels is None:
            logger.error("ClusteringPurity requires 'labels'.")
            del input_data
            gc.collect()
            return {**DEFAULT_CP_METRICS, "error": "Labels missing"}
        if not isinstance(labels, list):
            try:
                labels = list(labels)
            except TypeError:
                del input_data
                gc.collect()
                return {**DEFAULT_CP_METRICS, "error": "Labels not list-convertible"}

        original_indices = None
        if metadata and "original_indices" in metadata:
            original_indices = metadata["original_indices"]
            if len(original_indices) != n_samples:
                logger.warning(
                    f"Metadata original_indices length ({len(original_indices)}) != data samples ({n_samples}). Trusting data shape."
                )
                original_indices = list(range(n_samples))
        else:
            logger.warning("No 'original_indices' in metadata. Assuming data corresponds to first N labels.")
            original_indices = list(range(n_samples))

        valid_indices_map = {}
        filtered_indices_in_input_data = []

        for i, orig_idx in enumerate(original_indices):
            if 0 <= orig_idx < len(labels) and labels[orig_idx] is not None:
                valid_indices_map[i] = str(labels[orig_idx])
                filtered_indices_in_input_data.append(i)

        if not valid_indices_map:
            logger.error("No valid labels found corresponding to the loaded data.")
            del input_data
            gc.collect()
            return {**DEFAULT_CP_METRICS, "error": "No valid labels for data"}

        if len(filtered_indices_in_input_data) != n_samples:
            logger.info(f"Filtering input data ({n_samples}) for {len(filtered_indices_in_input_data)} valid labels.")
            if self.use_dist_matrix:
                input_data = input_data[np.ix_(filtered_indices_in_input_data, filtered_indices_in_input_data)]
            else:
                input_data = input_data[filtered_indices_in_input_data]
            n_samples = input_data.shape[0]
            filtered_labels = [valid_indices_map[i] for i in filtered_indices_in_input_data]
            logger.info(f"Data filtered. New shape: {input_data.shape}")
        else:
            filtered_labels = [str(labels[orig_idx]) for orig_idx in original_indices]

        if n_samples != len(filtered_labels):
            logger.error(f"CRITICAL: Filter mismatch: Data samples ({n_samples}) != Filtered labels ({len(filtered_labels)}).")
            del input_data
            gc.collect()
            return {**DEFAULT_CP_METRICS, "error": "Internal data/label filter mismatch"}

        if n_samples < self.umap_params.get("n_components", 2) or n_samples < self.hdbscan_params.get("min_cluster_size", 2):
            logger.error(f"Not enough valid samples ({n_samples}) for UMAP/HDBSCAN. Skipping.")
            del input_data
            gc.collect()
            return {**DEFAULT_CP_METRICS, "error": "Insufficient valid samples after filtering"}

        if np.isnan(input_data).any() or np.isinf(input_data).any():
            logger.warning(
                f"NaN/Inf detected in filtered input data (shape {input_data.shape}) before UMAP/HDBSCAN. Replacing with finite min/max."
            )
            finite_mask = np.isfinite(input_data)
            max_finite = 0.0
            min_finite = 0.0
            if np.any(finite_mask):
                finite_values = input_data[finite_mask]
                if finite_values.size > 0:
                    max_finite = np.max(finite_values)
                    min_finite = np.min(finite_values)
                logger.debug(f"Replacing non-finite: Max finite = {max_finite}, Min finite = {min_finite}")
            else:
                logger.warning("Input data contains only NaN/Inf values after filtering! Replacing with 0.0.")
            input_data = np.where(np.isnan(input_data), max_finite, input_data)
            input_data = np.where(np.isposinf(input_data), max_finite, input_data)
            input_data = np.where(np.isneginf(input_data), min_finite, input_data)
            if np.isnan(input_data).any() or np.isinf(input_data).any():
                logger.error("CRITICAL: Non-finite values still present after replacement! Aborting.")
                return {**DEFAULT_CP_METRICS, "error": "Failed to replace non-finite values"}
            if self.use_dist_matrix:
                logger.debug("Re-zeroing diagonal of distance matrix after NaN replacement.")
                np.fill_diagonal(input_data, 0)

        embeddings = None
        umap_time = -1.0
        logger.info("Starting UMAP projection...")
        start_time = time.time()
        try:
            reducer = umap.UMAP(**self.umap_params)
            embeddings = reducer.fit_transform(input_data)
            umap_time = time.time() - start_time
            logger.info(f"UMAP done ({umap_time:.2f}s). Embedding Shape: {embeddings.shape}")
            if embeddings.shape[0] != n_samples:
                raise RuntimeError(f"UMAP output samples ({embeddings.shape[0]}) != expected ({n_samples})")
            if np.isnan(embeddings).any() or np.isinf(embeddings).any():
                raise ValueError("UMAP output contains NaN/Inf.")
        except Exception as e:
            logger.error(f"UMAP failed: {e}. Input shape: {input_data.shape}, UMAP params: {self.umap_params}", exc_info=True)
            if input_data.size < 100:
                logger.error(f"Sample of input data to UMAP:\n{input_data}")
            del input_data
            gc.collect()
            return {**DEFAULT_CP_METRICS, "error": f"UMAP fail: {e}"}

        cluster_labels = None
        num_clusters = 0
        num_noise = 0
        hdbscan_time = -1.0
        logger.info("Starting HDBSCAN clustering...")
        start_time = time.time()
        try:
            clusterer = HDBSCAN(**self.hdbscan_params)
            cluster_labels = clusterer.fit_predict(embeddings)
            hdbscan_time = time.time() - start_time
            num_clusters = len(set(cluster_labels) - {-1})
            num_noise = np.sum(cluster_labels == -1)
            logger.info(f"HDBSCAN done ({hdbscan_time:.2f}s). Found {num_clusters} clusters, {num_noise} noise points.")
            if len(cluster_labels) != n_samples:
                raise RuntimeError(f"HDBSCAN output labels ({len(cluster_labels)}) != expected ({n_samples})")
        except Exception as e:
            logger.error(f"HDBSCAN failed: {e}", exc_info=True)
            del input_data, embeddings
            gc.collect()
            return {
                **DEFAULT_CP_METRICS,
                "num_clusters_found": 0,
                "num_noise_points": n_samples,
                "error": f"HDBSCAN fail: {e}",
            }

        weighted_purity = None
        logger.info("Calculating weighted purity...")
        try:
            true_label_counts = Counter(filtered_labels)
            _min_class_size = max(1, min_class_size_for_purity)
            valid_class_mask = np.array([true_label_counts[lbl] >= _min_class_size for lbl in filtered_labels])
            if not np.any(valid_class_mask):
                logger.warning(f"No classes >= {_min_class_size} samples found. Purity is None.")
                weighted_purity = None
            else:
                labels_for_purity = np.array(filtered_labels)[valid_class_mask]
                preds_for_purity = cluster_labels[valid_class_mask]
                if len(labels_for_purity) != len(preds_for_purity):
                    raise RuntimeError(f"Purity length mismatch: Labels ({len(labels_for_purity)}) vs Preds ({len(preds_for_purity)}).")
                cont_mat = contingency_matrix(labels_for_purity, preds_for_purity)
                cluster_sizes = np.sum(cont_mat, axis=0)
                majority_counts = np.max(cont_mat, axis=0)
                unique_preds, cluster_indices = np.unique(preds_for_purity, return_inverse=True)
                actual_cluster_sizes = cluster_sizes
                actual_majority_counts = majority_counts
                cluster_purities = np.zeros(len(actual_cluster_sizes), dtype=float)
                valid_cluster_mask_purity = actual_cluster_sizes > 0
                if np.any(valid_cluster_mask_purity):
                    cluster_purities[valid_cluster_mask_purity] = actual_majority_counts[valid_cluster_mask_purity] / actual_cluster_sizes[valid_cluster_mask_purity]
                if np.any(valid_cluster_mask_purity):
                    total_weight = np.sum(actual_cluster_sizes[valid_cluster_mask_purity])
                    if total_weight > 0:
                        weighted_purity = np.average(cluster_purities[valid_cluster_mask_purity], weights=actual_cluster_sizes[valid_cluster_mask_purity])
                    else:
                        weighted_purity = 0.0
                else:
                    weighted_purity = 0.0
                logger.info(
                    "Weighted Purity calc details:"
                    f" Valid mask sum={np.sum(valid_class_mask)}, Labels={len(labels_for_purity)}, Preds={len(preds_for_purity)},"
                    f" ContMat Shape={cont_mat.shape}, Unique Preds={len(unique_preds)}"
                )
                logger.info(f"Weighted Purity: {weighted_purity:.4f}")
        except Exception as e:
            logger.error(f"Purity calculation failed: {e}", exc_info=True)
            weighted_purity = None

        del input_data, embeddings, cluster_labels, filtered_labels
        if "labels_for_purity" in locals():
            del labels_for_purity
        if "preds_for_purity" in locals():
            del preds_for_purity
        gc.collect()

        return {
            "weighted_purity": weighted_purity,
            "num_clusters_found": num_clusters,
            "num_noise_points": num_noise,
            "samples_after_filtering": n_samples,
            "umap_time_seconds": round(umap_time, 2) if umap_time >= 0 else None,
            "hdbscan_time_seconds": round(hdbscan_time, 2) if hdbscan_time >= 0 else None,
            "error": None,
        }