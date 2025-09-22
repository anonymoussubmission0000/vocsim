import gc
import importlib
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from tqdm.auto import tqdm
import h5py

from distances.base import DistanceCalculator
from utils.file_utils import (
    HDF5_DATASET_NAME,
    get_cache_path,
    find_cache_path,
    read_hdf5_metadata,
)
from .feature_manager import FeatureManager
from h5py import Dataset, File


logger = logging.getLogger(__name__)
HDF5_DISTANCE_DATASET_NAME = "distance_matrix"
HDF5_FEATURE_DATASET_NAME = "features"


class DistanceManager:
    """
    Manages computation and caching of pairwise distance matrices.
    Reads features chunk-by-chunk. Writes distances chunk-by-chunk.
    Uses find_cache_path for loading (loose match fallback) and get_cache_path for saving.
    """

    def __init__(self, config: Dict[str, Any], base_features_cache_dir: Path, device: torch.device):
        """
        Initializes the DistanceManager.

        Args:
            config (Dict[str, Any]): The configuration dictionary.
            base_features_cache_dir (Path): The base directory for caching features.
            device (torch.device): The device to use for computation.
        """
        self.cfg = config
        self.base_features_dir = base_features_cache_dir
        self.device = device
        self.distance_configs = config.get("distances", [])
        self._calculator_instances_cache: Dict[str, DistanceCalculator] = {}
        self.gpu_block_size = config.get("distance_gpu_block_size", 1024)
        # Ensure raw_distance_configs_map is initialized even if empty in config
        self.raw_distance_configs_map = {dc["name"]: dc for dc in config.get("distances", []) if "name" in dc}
        logger.info("Initialized DistanceManager with GPU block size (Distance Matrix Blocks): %d", self.gpu_block_size)

    def _get_class_from_module(self, module_name: str, class_name: str) -> Type:
        """
        Dynamically imports a class from a module path.

        Args:
            module_name (str): The dotted module path.
            class_name (str): The name of the class within the module.

        Returns:
            Type: The imported class object.

        Raises:
            ImportError: If the module or class cannot be found/imported.
        """
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except Exception as e:
            logger.error("Failed load class '%s' from '%s': %s", class_name, module_name, e, exc_info=True)
            raise ImportError(f"Failed import {class_name} from {module_name}.") from e

    def _get_distance_calculator(self, config: Dict[str, Any]) -> DistanceCalculator:
        """
        Gets or instantiates a distance calculator.

        Args:
            config (Dict[str, Any]): The configuration dictionary for the distance calculator.

        Returns:
            DistanceCalculator: An instance of the distance calculator.

        Raises:
            ValueError: If the configuration is invalid.
            ImportError: If the class cannot be instantiated.
        """
        name = config.get("name")
        if not name:
            raise ValueError("Distance configuration requires a 'name' field.")
        if name in self._calculator_instances_cache:
            return self._calculator_instances_cache[name]
        params = config.get("params", {})
        dist_name_lower = name.lower()
        class_name = config.get("class")
        if not class_name:
            if dist_name_lower == "cosine":
                class_name = "CosineDistance"
            elif dist_name_lower == "euclidean":
                class_name = "EuclideanDistance"
            elif dist_name_lower == "spearman":
                class_name = "SpearmanDistance"
            else:
                class_name = f"{name.capitalize()}Distance"
        module_path = config.get("module", f"distances.{dist_name_lower}")
        try:
            calculator_class = self._get_class_from_module(module_path, class_name)
            instance = calculator_class(**params)
            self._calculator_instances_cache[name] = instance
            logger.info("Instantiated distance calculator '%s' (Config Name: %s).", class_name, name)
            return instance
        except Exception as e:
            logger.error("Failed instantiate distance calculator '%s' (Config Name: %s) from module '%s': %s", class_name, name, module_path, e, exc_info=True)
            raise ImportError(f"Failed instantiate {class_name} from {module_path}: {e}") from e

    def _clear_calculator_cache(self) -> None:
        """Clears cached distance calculator instances."""
        logger.debug("Clearing distance calculator instance cache.")
        self._calculator_instances_cache.clear()
        gc.collect()

    def _compute_distances_block_gpu_hdf5(
        self,
        calculator: DistanceCalculator,
        h5_features: "Dataset",
        distance_h5_path: Path,
        n_samples: int,
        distance_dataset_name: str = HDF5_DISTANCE_DATASET_NAME,
    ) -> bool:
        """
        Computes pairwise distances chunk-by-chunk on GPU using data from HDF5 and saves to a NEW HDF5 file.

        Args:
            calculator (DistanceCalculator): The distance calculator instance.
            h5_features (Dataset): The h5py Dataset object containing features.
            distance_h5_path (Path): Path to save the new distance matrix HDF5 file.
            n_samples (int): The total number of samples (rows) in the feature dataset.
            distance_dataset_name (str): Name of the dataset to create in the distance HDF5 file.

        Returns:
            bool: True if the distance matrix was computed and saved successfully, False otherwise.
        """
        calc_name = calculator.__class__.__name__
        feature_h5_path_str = h5_features.file.filename
        if n_samples <= 0:
            logger.warning("[%s] n_samples is 0, skipping.", calc_name)
            try:
                distance_h5_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            try:
                with h5py.File(distance_h5_path, "w") as f_dist:
                    f_dist.create_dataset(distance_dataset_name, shape=(0, 0), dtype=np.float32)
                logger.info("Created empty distance file: %s", distance_h5_path.name)
                return True
            except Exception as e:
                logger.error("Failed write empty distance matrix: %s", e)
                return False
        if n_samples == 1:
            logger.info("[%s] n_samples is 1.", calc_name)
            try:
                distance_h5_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            try:
                with h5py.File(distance_h5_path, "w") as f_dist:
                    f_dist.create_dataset(distance_dataset_name, shape=(1, 1), dtype=np.float32, data=[[0.0]])
                return True
            except Exception as e:
                logger.error("Failed write single-sample distance matrix: %s", e)
                return False

        logger.info("Starting HDF5->HDF5 Block Distance Calc (%s) for N=%d, Block=%d", calc_name, n_samples, self.gpu_block_size)
        start_time = time.time()
        block_size = self.gpu_block_size
        num_blocks = math.ceil(n_samples / block_size)
        try:
            distance_h5_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(distance_h5_path, "w") as f_dist:
                if h5_features.shape[0] != n_samples:
                    logger.error("Feature count mismatch (%d vs %d)", h5_features.shape[0], n_samples)
                    return False
                h5_distances = f_dist.create_dataset(distance_dataset_name, shape=(n_samples, n_samples), dtype=np.float32, compression="gzip", compression_opts=4, shuffle=True, fletcher32=True)
                total_block_pairs = num_blocks * (num_blocks + 1) // 2
                pbar = tqdm(total=total_block_pairs, desc=f"Dist Blocks [{calc_name}]", leave=False)
                for i in range(num_blocks):
                    start_i = i * block_size
                    end_i = min((i + 1) * block_size, n_samples)
                    try:
                        block_i_cpu = h5_features[start_i:end_i]
                    except Exception as read_err:
                        logger.error("Read block i=%d fail: %s", i, read_err)
                        pbar.update(num_blocks - i)
                        continue
                    block_i_flat = block_i_cpu.reshape(block_i_cpu.shape[0], -1).astype(np.float32)
                    block_i_gpu = torch.from_numpy(block_i_flat).to(self.device)
                    del block_i_cpu, block_i_flat
                    gc.collect()
                    for j in range(i, num_blocks):
                        start_j = j * block_size
                        end_j = min((j + 1) * block_size, n_samples)
                        try:
                            try:
                                block_j_cpu = h5_features[start_j:end_j]
                            except Exception as read_err:
                                logger.error("Read block j=%d fail: %s", j, read_err)
                                pbar.update(1)
                                continue
                            block_j_flat = block_j_cpu.reshape(block_j_cpu.shape[0], -1).astype(np.float32)
                            block_j_gpu = torch.from_numpy(block_j_flat).to(self.device)
                            del block_j_cpu, block_j_flat
                            gc.collect()
                            dist_ij_gpu = calculator.compute_pairwise(block_i_gpu, block_j_gpu)
                            if dist_ij_gpu is None:
                                raise RuntimeError(f"Calculator {calc_name} returned None for block ({i},{j})")
                            dist_block = dist_ij_gpu.cpu().numpy().astype(np.float32)
                            h5_distances[start_i:end_i, start_j:end_j] = dist_block
                            if i != j:
                                h5_distances[start_j:end_j, start_i:end_i] = dist_block.T
                            del dist_ij_gpu, block_j_gpu, dist_block
                            if self.device.type == "cuda":
                                torch.cuda.empty_cache()
                        except Exception as e_inner:
                            logger.error("Error block (%d vs %d) (%s): %s", i, j, calc_name, e_inner, exc_info=True)
                        finally:
                            if "block_j_gpu" in locals() and block_j_gpu is not None:
                                del block_j_gpu
                            gc.collect()
                        pbar.update(1)
                    del block_i_gpu
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                pbar.close()
                logger.debug("Zeroing diagonal...")
                diag_chunk_size = min(n_samples, max(block_size * 4, 4096))
                for k in range(0, n_samples, diag_chunk_size):
                    end_k = min(k + diag_chunk_size, n_samples)
                    if k >= end_k:
                        continue
                    try:
                        block_data = h5_distances[k:end_k, k:end_k]
                        np.fill_diagonal(block_data, 0.0)
                        h5_distances[k:end_k, k:end_k] = block_data
                    except Exception as diag_err:
                        logger.error("Error zeroing diag %d:%d: %s", k, end_k, diag_err)
                logger.debug("Diagonal zeroing done.")
        except Exception as e:
            logger.error(
                "HDF5 distance calculation failed (%s) for %s -> %s: %s",
                calc_name,
                feature_h5_path_str,
                distance_h5_path,
                e,
                exc_info=True,
            )
            try:
                if distance_h5_path.exists():
                    distance_h5_path.unlink()
                    logger.info("Deleted incomplete distance file: %s", distance_h5_path.name)
            except OSError as unlink_err:
                logger.error("Failed delete incomplete dist file: %s", unlink_err)
            return False
        elapsed = time.time() - start_time
        logger.info("HDF5 Block Distance Calc (%s) done in %.2fs. Output: %s", calc_name, elapsed, distance_h5_path.name)
        return True

    def process_subset_distances(
        self,
        dataset_cache_id: str,
        current_subset_key: str,
        feature_manager: "FeatureManager",
        subset_features_dir: Path,
        item_id_map: Dict[str, Dict],
        feature_paths: Dict[str, Path],
        run_steps: List[str],
    ) -> Dict[Tuple[str, str], Path]:
        """
        Computes or finds existing distance matrices for the given features.

        Args:
            dataset_cache_id (str): Unique identifier for this dataset subset/split.
            current_subset_key (str): Name of the current subset.
            feature_manager (FeatureManager): The FeatureManager instance.
            subset_features_dir (Path): Directory containing cached feature files for this subset.
            item_id_map (Dict[str, Dict]): Map from original item IDs to metadata/indices.
            feature_paths (Dict[str, Path]): Dictionary mapping feature names to paths of cached feature files.
            run_steps (List[str]): List of steps being run in the pipeline.

        Returns:
            Dict[Tuple[str, str], Path]: A dictionary mapping (feature_name, distance_name) tuples to
                                        paths of distance matrix files that were successfully processed or found.
        """
        logger.info("--- Distance Computation (Steps: %s) for Subset: %s ---", run_steps, current_subset_key)
        distance_file_paths: Dict[Tuple[str, str], Path] = {}
        n_samples_expected = len(item_id_map)
        if n_samples_expected == 0:
            logger.warning("No items in %s. Skip dist.", current_subset_key)
            return distance_file_paths
        self._clear_calculator_cache()

        compute_distances_step = "distances" in run_steps

        feature_iter = tqdm(feature_paths.items(), desc=f"Distances [{current_subset_key}]", leave=False)
        for feature_name, feature_path in feature_iter:
            feature_iter.set_postfix_str(f"{feature_name}")

            if not feature_path or not feature_path.exists():
                logger.warning("Input feature file missing for '%s' at %s. Skipping distance calculation.", feature_name, feature_path)
                continue

            raw_feature_conf = feature_manager.all_feature_configs_map.get(feature_name)
            if not raw_feature_conf:
                logger.error("Raw feature config missing for %s. Skip dist.", feature_name)
                continue
            effective_config = feature_manager._get_effective_feature_config(raw_feature_conf, current_subset_key)

            allowed_dists = effective_config.get("compute_distances_for")
            if allowed_dists is not None and not isinstance(allowed_dists, list):
                logger.warning("Invalid 'compute_distances_for'. Using all.")
                allowed_dists = None

            distances_to_process = [dc for dc in self.distance_configs if dc.get("name") and (allowed_dists is None or dc.get("name") in allowed_dists)]
            if not distances_to_process:
                continue

            feature_metadata = read_hdf5_metadata(feature_path, HDF5_FEATURE_DATASET_NAME)
            if not feature_metadata:
                logger.error("Metadata missing for feature file '%s'. Skip dist.", feature_path.name)
                continue

            is_base_feature = not effective_config.get("base_extractor")
            item_count_attr = "num_items_processed" if is_base_feature else "num_items"
            n_samples_feat = feature_metadata.get(item_count_attr)

            if n_samples_feat is None:
                fallback_attr = "num_items" if is_base_feature else "num_items_processed"
                n_samples_feat = feature_metadata.get(fallback_attr)
                if n_samples_feat is not None:
                    logger.warning("Primary item count attr '%s' missing for '%s'. Using fallback '%s'.", item_count_attr, feature_path.name, fallback_attr)
                else:
                    logger.error("Item count missing (checked '%s', '%s') in metadata for '%s'. Skip dist.", item_count_attr, fallback_attr, feature_path.name)
                    continue

            n_samples_feat = int(n_samples_feat)
            if n_samples_feat <= 0:
                logger.warning("Feature '%s' has %d samples. Skipping distances.", feature_name, n_samples_feat)
                continue

            dist_iter = tqdm(distances_to_process, desc=f"Distances for {feature_name}", leave=False)
            for dist_conf in dist_iter:
                dist_name = dist_conf.get("name")
                if not dist_name:
                    continue
                dist_iter.set_postfix_str(f"{dist_name}")
                matrix_key = (feature_name, dist_name)

                combined_config_for_path = {"feature_config": effective_config, "distance_config": dist_conf, "name": f"{feature_name}_{dist_name}"}
                found_path = find_cache_path(subset_features_dir, f"distances_{dist_name}", dataset_cache_id, combined_config_for_path)

                cache_status = "MISSING"
                if found_path and found_path.is_file():
                    logger.debug("Found potential match for dist '%s/%s': %s", dist_name, feature_name, found_path.name)
                    try:
                        with h5py.File(found_path, "r") as f_dist_check:
                            if HDF5_DISTANCE_DATASET_NAME in f_dist_check and f_dist_check[HDF5_DISTANCE_DATASET_NAME].shape == (n_samples_feat, n_samples_feat):
                                cache_status = "VALID"
                            else:
                                cache_status = f"INVALID_SHAPE/DATASET ({found_path.name})"
                    except Exception as e:
                        cache_status = f"INVALID_READ ({found_path.name})"
                        logger.warning("Error reading cache %s: %s", found_path.name, e)

                if cache_status == "VALID" and found_path:
                    logger.info("Using valid cache for dist '%s/%s': %s", dist_name, feature_name, found_path.name)
                    distance_file_paths[matrix_key] = found_path
                elif compute_distances_step:
                    logger.info("Computing distance matrix for '%s' on '%s' (Cache Status: %s).", dist_name, feature_name, cache_status)

                    expected_save_path = get_cache_path(subset_features_dir, f"distances_{dist_name}", dataset_cache_id, combined_config_for_path)

                    try:
                        with h5py.File(feature_path, "r") as f_feat:
                            if HDF5_FEATURE_DATASET_NAME not in f_feat:
                                logger.error("Feature dataset missing in %s. Cannot compute distance.", feature_path.name)
                                continue
                            h5_features_dataset = f_feat[HDF5_FEATURE_DATASET_NAME]
                            if h5_features_dataset.shape[0] != n_samples_feat:
                                logger.error("Feature file shape mismatch during compute for %s. Skip.", feature_path.name)
                                continue

                            calculator = self._get_distance_calculator(dist_conf)
                            success = self._compute_distances_block_gpu_hdf5(
                                calculator=calculator,
                                h5_features=h5_features_dataset,
                                distance_h5_path=expected_save_path,
                                n_samples=n_samples_feat,
                                distance_dataset_name=HDF5_DISTANCE_DATASET_NAME,
                            )

                        if success:
                            distance_file_paths[matrix_key] = expected_save_path
                            try:
                                with h5py.File(expected_save_path, "a") as f_dist_meta:
                                    if HDF5_DISTANCE_DATASET_NAME in f_dist_meta:
                                        dset = f_dist_meta[HDF5_DISTANCE_DATASET_NAME]
                                        dist_meta = {"feature_name": feature_name, "distance_name": dist_name, "n_samples": n_samples_feat, "creation_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                                        for k, v in dist_meta.items():
                                            dset.attrs[k] = str(v)
                            except Exception as meta_e:
                                logger.error("Failed write metadata to %s: %s", expected_save_path, meta_e)
                            logger.info("Distance matrix computed and saved for '%s/%s'.", dist_name, feature_name)
                        else:
                            logger.error("Distance calculation failed for %s.", matrix_key)

                    except Exception as compute_err:
                        logger.error("Error during distance computation for %s: %s", matrix_key, compute_err, exc_info=True)
                    finally:
                        gc.collect()
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                else:
                    logger.info("Cache MISS for dist '%s/%s' and 'distances' step skipped. Cannot compute.", dist_name, feature_name)

            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self._clear_calculator_cache()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("--- Distance Computation Completed for Subset: %s ---", current_subset_key)
        return distance_file_paths