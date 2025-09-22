import gc
import importlib
import json
import logging
from itertools import chain, islice
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Type, Union, Callable

import h5py
import numpy as np
import torch
from sklearn.decomposition import PCA, IncrementalPCA
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import joblib
from datasets import Dataset, IterableDataset

from features.base import FeatureExtractor
from utils.feature_utils import apply_averaging
from utils.file_utils import (
    NpEncoder,
    get_cache_path,
    load_hdf5,
    load_pca_model,
    read_hdf5_metadata,
    save_pca_model,
)

logger = logging.getLogger(__name__)

HDF5_DATASET_NAME = "features"
HDF5_INDICES_NAME = "original_indices"


class FeatureManager:
    """
    Manages feature extraction, averaging/flattening, and PCA.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        base_features_cache_dir: Path,
        device: torch.device,
        base_models_dir: Path,
    ):
        """
        Initializes the FeatureManager.

        Args:
            config (Dict[str, Any]): The configuration dictionary.
            base_features_cache_dir (Path): The base directory for caching features.
            device (torch.device): The device to use for computation.
            base_models_dir (Path): The base directory for caching models (like PCA).
        """
        self.cfg = config
        self.base_features_dir = base_features_cache_dir
        self.base_models_dir = base_models_dir
        self.device = device
        self.all_feature_configs = config.get("feature_extractors", [])
        self.all_feature_configs_map = {fc["name"]: fc for fc in self.all_feature_configs if "name" in fc}
        self.whisperseg_param_map = self._create_subset_to_params_map(config, "whisperseg_subset_params")
        self.pca_model_cache: Dict[str, Union[IncrementalPCA, PCA]] = {}
        self._extractor_instances_cache: Dict[str, FeatureExtractor] = {}
        self.processed_feature_metadata: Dict[str, Dict[str, Any]] = {}
        self.default_pca_load_chunks = config.get("pca_load_chunks", 0)
        logger.info(
            "FeatureManager initialized (HDF5, Single-Scan, Default PCA Load Chunks: %d).", self.default_pca_load_chunks
        )
        logger.info("Features Cache Dir: %s", self.base_features_dir)
        logger.info("Models (PCA) Dir: %s", self.base_models_dir)

    def _create_subset_to_params_map(self, config: Dict[str, Any], map_key: str) -> Dict[str, Dict[str, Any]]:
        """
        Creates a map from subset key to dynamic parameters for specific extractors.

        Args:
            config (Dict[str, Any]): The full configuration dictionary.
            map_key (str): The key in the config containing the subset parameter map.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary mapping subset names to parameter dictionaries.
        """
        subset_param_map = {}
        category_config = config.get(map_key, {})
        if not isinstance(category_config, dict):
            logger.warning("'%s' in config is not a dictionary.", map_key)
            return subset_param_map
        for subset_name, subset_data in category_config.items():
            if isinstance(subset_data, dict) and "params" in subset_data:
                if isinstance(subset_data["params"], dict):
                    subset_param_map[subset_name] = subset_data["params"].copy()
                else:
                    logger.warning("Params for '%s' in '%s' is not a dict.", subset_name, map_key)
            else:
                logger.debug("Invalid format or missing 'params' for '%s' in '%s'. Skipping.", subset_name, map_key)
        if subset_param_map:
            logger.info("Created dynamic param map for '%s': %s", map_key, list(subset_param_map.keys()))
        return subset_param_map

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
        except ModuleNotFoundError:
            logger.error("Module not found: %s", module_name)
            raise
        except AttributeError:
            logger.error("Class '%s' not found in module '%s'", class_name, module_name)
            raise
        except Exception as e:
            logger.error("Failed load class '%s' from '%s': %s", class_name, module_name, e, exc_info=True)
            raise ImportError(f"Failed import {class_name} from {module_name}.") from e

    def _get_feature_extractor(self, config: Dict[str, Any]) -> Optional[FeatureExtractor]:
        """
        Gets or instantiates a feature extractor.

        Args:
            config (Dict[str, Any]): The configuration dictionary for the feature extractor.

        Returns:
            Optional[FeatureExtractor]: An instance of the FeatureExtractor class, or None if instantiation fails.
        """
        extractor_name = config.get("name")
        if not extractor_name:
            logger.error("Feature config needs 'name'.")
            return None

        if extractor_name in self._extractor_instances_cache:
            return self._extractor_instances_cache[extractor_name]

        class_name = config.get("class", extractor_name)
        module_path = config.get("module")
        if not module_path:
            module_name_guess = class_name.replace("Extractor", "").lower()
            if "whisperseg" in module_name_guess:
                module_path = "features.whisperseg.extractor"
            elif "paperautoencoder" in module_name_guess:
                module_path = "reproducibility.features.autoencoder"
            elif "papervae" in module_name_guess:
                module_path = "reproducibility.features.vae"
            else:
                module_path = f"features.{module_name_guess}"
            logger.debug("Inferred module path for %s: %s", class_name, module_path)

        try:
            extractor_class = self._get_class_from_module(module_path, class_name)
            params = config.get("params", {}).copy()
            params["device"] = self.device
            if "Paper" in class_name:
                params["base_models_dir"] = self.base_models_dir

            instance = extractor_class(**params)
            self._extractor_instances_cache[extractor_name] = instance
            logger.info("Instantiated extractor '%s' (Config: %s).", class_name, extractor_name)
            return instance
        except Exception as e:
            logger.error("Failed instantiate extractor '%s' (Config: %s): %s", class_name, extractor_name, e, exc_info=True)
            return None

    def _clear_extractor_cache(self) -> None:
        """Clears the cache of instantiated feature extractors."""
        logger.debug("Clearing feature extractor instance cache...")
        for instance in list(self._extractor_instances_cache.values()):
            del instance
        self._extractor_instances_cache.clear()
        gc.collect()
        if self.device.type == "cuda":
            try:
                torch.cuda.empty_cache()
                logger.debug("Cleared CUDA cache.")
            except Exception as cuda_err:
                logger.warning("Could not clear CUDA cache: %s", cuda_err)

    def _get_effective_feature_config(self, feature_config: Dict[str, Any], subset_key: Optional[str]) -> Dict[str, Any]:
        """
        Applies dynamic subset parameters to a feature configuration if configured.

        Args:
            feature_config (Dict[str, Any]): The base feature configuration.
            subset_key (Optional[str]): The key of the current dataset subset.

        Returns:
            Dict[str, Any]: The feature configuration with dynamic parameters applied.
        """
        effective_config = feature_config.copy()
        is_whisperseg = "whisperseg" in effective_config.get("module", "").lower() or "WhisperSeg" in effective_config.get("name", "")

        if is_whisperseg and subset_key:
            subset_params = self.whisperseg_param_map.get(subset_key)
            if subset_params:
                if "params" not in effective_config:
                    effective_config["params"] = {}
                effective_config["params"] = {**effective_config.get("params", {}), **subset_params}
                logger.debug("Applied dynamic params for '%s' on subset '%s': %s", effective_config.get("name"), subset_key, subset_params)
            elif self.whisperseg_param_map:
                logger.debug("No dynamic params found for subset '%s'. Using defaults for '%s'.", subset_key, effective_config.get("name"))
        return effective_config

    def get_feature_file_path(self, feature_name: str, dataset_cache_id: str, subset_features_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Determines the expected path for a cached feature file (base, intermediate, or final PCA).

        Args:
            feature_name (str): The name of the feature configuration.
            dataset_cache_id (str): Unique identifier for the dataset subset/split.
            subset_features_dir (Optional[Path]): The directory specific to the current subset's features.

        Returns:
            Optional[Path]: The expected Path object, or None if the feature configuration is not found.
        """
        raw_config = self.all_feature_configs_map.get(feature_name)
        if not raw_config:
            logger.error("Configuration for feature '%s' not found.", feature_name)
            return None

        is_base = not raw_config.get("base_extractor")
        needs_intermediate = raw_config.get("averaging") is not None
        needs_pca = raw_config.get("pca") is not None

        final_config = raw_config

        final_prefix = "features"
        if needs_intermediate and not needs_pca:
            final_prefix = "intermediate"

        cache_dir_to_use = subset_features_dir if subset_features_dir else self.base_features_dir

        return get_cache_path(cache_dir=cache_dir_to_use, prefix=final_prefix, dataset_cache_id=dataset_cache_id, config_dict=final_config, extra_suffix=None)

    def _extract_and_save_features_hdf5(
        self,
        extractor: FeatureExtractor,
        dataset: Union[Dataset, IterableDataset],
        cache_path: Path,
        feature_config: Dict[str, Any],
        item_id_map: Optional[Dict] = None,
        batch_size: int = 32,
    ) -> Optional[Dict[str, Any]]:
        """
        Extracts features using a two-pass scan-then-write process and saves to HDF5.

        Args:
            extractor (FeatureExtractor): The feature extractor instance.
            dataset (Union[Dataset, IterableDataset]): The dataset to extract from.
            cache_path (Path): Path to save the new feature HDF5 file.
            feature_config (Dict[str, Any]): The configuration dictionary for the feature extractor.
            item_id_map (Optional[Dict]): Map from item IDs to metadata, used to get original indices.
            batch_size (int): Batch size for processing during extraction and writing.

        Returns:
            Optional[Dict[str, Any]]: Metadata dictionary of the saved file, or None on failure.
        """
        feature_name = feature_config.get("name", "unknown_feature")
        extractor_name = extractor.__class__.__name__
        logger.info("Starting HDF5 extraction for '%s' using '%s' to %s.", feature_name, extractor_name, cache_path.name)
        params = feature_config.get("params", {})

        # --- Pass 1: Scan for valid items and determine maximum feature shape ---
        logger.info("Step 1: Scanning dataset for valid items and max feature shape...")
        max_dims: List[int] = []
        final_ndim: Optional[int] = None
        final_dtype: Optional[np.dtype] = None
        valid_item_original_indices: List[int] = []
        valid_item_internal_indices: List[int] = []
        processed_indices_set: Set[int] = set()
        skipped_scan_count = 0
        is_iterable = not hasattr(dataset, "__len__")
        total_items_scan = None if is_iterable else len(dataset)

        loader_scan = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=lambda x: x)
        pbar_scan = tqdm(loader_scan, desc=f"Scan [{feature_name}]", total=total_items_scan, leave=False)
        global_idx_scan = 0

        for batch in pbar_scan:
            for item in batch:
                current_internal_idx = global_idx_scan
                audio_info = item.get("audio")
                is_valid_item = isinstance(audio_info, dict) and audio_info.get("array") is not None and "sampling_rate" in audio_info

                if is_valid_item:
                    try:
                        features = extractor.extract(audio_data=audio_info["array"], sample_rate=audio_info["sampling_rate"], **params)
                        features_np = features.cpu().numpy() if isinstance(features, torch.Tensor) else np.array(features)

                        if features_np.size == 0 or np.isnan(features_np).any() or np.isinf(features_np).any():
                            is_valid_item = False
                        else:
                            if final_ndim is None:
                                final_ndim = features_np.ndim
                                final_dtype = np.float32
                                max_dims = list(features_np.shape)
                            elif features_np.ndim != final_ndim:
                                logger.warning("Inconsistent feature dimensionality at item %d. Expected %dD, got %dD. Skipping.", current_internal_idx, final_ndim, features_np.ndim)
                                is_valid_item = False
                            else:
                                for i in range(final_ndim):
                                    max_dims[i] = max(max_dims[i], features_np.shape[i])
                    except Exception as e:
                        logger.warning("Error extracting feature for item %d during scan: %s. Skipping.", current_internal_idx, e)
                        is_valid_item = False

                if is_valid_item:
                    valid_item_original_indices.append(current_internal_idx)
                    valid_item_internal_indices.append(current_internal_idx)
                    processed_indices_set.add(current_internal_idx)
                else:
                    skipped_scan_count += 1
                global_idx_scan += 1
            pbar_scan.set_postfix({"Valid": len(valid_item_internal_indices), "Skipped": skipped_scan_count, "MaxShape": str(tuple(max_dims))})
        pbar_scan.close()

        processed_count = len(valid_item_internal_indices)
        if processed_count == 0:
            logger.error("No valid audio items found for '%s'. Aborting.", feature_name)
            return None
        max_shape = tuple(max_dims)
        logger.info("Scan complete. Found %d valid items. Max feature shape: %s", processed_count, max_shape)

        # --- Pass 2: Create HDF5 file and write features ---
        metadata_for_attrs = {
            "feature_config_name": feature_name,
            "num_items_processed": processed_count,
            "num_items_skipped": skipped_scan_count,
            "target_padded_shape": list(max_shape),
            "feature_ndim": final_ndim,
            "config": json.loads(json.dumps(feature_config, cls=NpEncoder)),
        }
        final_returned_metadata = metadata_for_attrs.copy()
        final_returned_metadata[HDF5_INDICES_NAME] = valid_item_original_indices

        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(cache_path, "w") as h5_file:
                h5_dataset = h5_file.create_dataset(
                    HDF5_DATASET_NAME, shape=(processed_count, *max_shape), dtype=final_dtype,
                    compression="gzip", compression_opts=4, shuffle=True, fletcher32=True)
                for k, v in metadata_for_attrs.items():
                    h5_dataset.attrs[k] = json.dumps(v, cls=NpEncoder)
                h5_file.create_dataset(HDF5_INDICES_NAME, data=np.array(valid_item_original_indices, dtype=np.int64), compression="gzip", compression_opts=4)

                logger.info("Step 2: Writing %d features to HDF5 file '%s'...", processed_count, cache_path.name)
                write_idx = 0
                error_occurred_write = False
                loader_write = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=lambda x: x)
                pbar_write = tqdm(loader_write, desc=f"Write [{feature_name}]", total=int(total_items_scan / batch_size), leave=False)
                global_idx_write = 0
                for batch in pbar_write:
                    features_to_write_list = []
                    for item in batch:
                        current_internal_idx = global_idx_write
                        global_idx_write += 1
                        if current_internal_idx not in processed_indices_set:
                            continue
                        try:
                            features = extractor.extract(audio_data=item["audio"]["array"], sample_rate=item["audio"]["sampling_rate"], **params)
                            features_np = features.cpu().numpy() if isinstance(features, torch.Tensor) else np.array(features)
                            pad_width = [(0, max_shape[i] - features_np.shape[i]) for i in range(final_ndim)]
                            padded_feature = np.pad(features_np, pad_width, mode="constant", constant_values=0.0) if any(p[1] > 0 for p in pad_width) else features_np
                            features_to_write_list.append(padded_feature)
                        except Exception as e:
                            logger.error("Processing item internal_idx %d failed during write pass: %s", current_internal_idx, e)
                            error_occurred_write = True
                    if features_to_write_list:
                        batch_array = np.stack(features_to_write_list).astype(final_dtype)
                        h5_dataset[write_idx : write_idx + len(features_to_write_list)] = batch_array
                        write_idx += len(features_to_write_list)
                    pbar_write.set_postfix({"Written": write_idx, "TotalValid": processed_count})
                pbar_write.close()

                if write_idx != processed_count or error_occurred_write:
                    raise RuntimeError(f"Write failed. Written: {write_idx}/{processed_count}. Errors: {error_occurred_write}")

        except Exception as e:
            logger.error("HDF5 extraction/write failed for '%s': %s", feature_name, e, exc_info=True)
            if cache_path.exists(): cache_path.unlink(missing_ok=True)
            return None

        logger.info("Successfully wrote %d features to HDF5: %s", write_idx, cache_path)
        return final_returned_metadata

    def _process_intermediate_features(
        self,
        base_feature_path: Path,
        intermediate_feature_path: Path,
        feature_config: Dict[str, Any],
        chunk_size: int = 1024,
    ) -> Optional[Dict[str, Any]]:
        """
        Processes base features (averaging/flattening) -> intermediate HDF5.

        Args:
            base_feature_path (Path): Path to READ base feature HDF5 file.
            intermediate_feature_path (Path): Path to SAVE intermediate HDF5 file.
            feature_config (Dict[str, Any]): Effective configuration for the intermediate step.
            chunk_size (int): Size of chunks to read from the base HDF5 file.

        Returns:
            Optional[Dict[str, Any]]: Metadata of the intermediate file, or None on failure.
        """
        feature_name = feature_config.get("name", "unknown_intermediate")
        averaging_method = feature_config.get("averaging")
        logger.info("Processing intermediate features for '%s' (Method: %s) -> %s", feature_name, averaging_method, intermediate_feature_path.name)

        if not base_feature_path.exists():
            logger.error("Base feature file not found: %s. Cannot create intermediate.", base_feature_path)
            return None

        base_metadata = read_hdf5_metadata(base_feature_path, HDF5_DATASET_NAME)
        if base_metadata is None:
            logger.error("Could not read metadata from base HDF5: %s. Cannot create intermediate.", base_feature_path)
            return None
        base_original_indices = load_hdf5(base_feature_path, HDF5_INDICES_NAME, "Base Indices")

        item_count_attr = "num_items_processed"
        n_items = base_metadata.get(item_count_attr)
        if n_items is None:
            try:
                with h5py.File(base_feature_path, "r") as f_check:
                    n_items = f_check[HDF5_DATASET_NAME].shape[0]
                logger.warning("Base metadata missing 'num_items_processed'. Used shape[0]: %d", n_items)
            except Exception:
                logger.error("Could not determine item count from base file: %s.", base_feature_path)
                return None
        n_items = int(n_items)
        if n_items == 0:
            logger.warning("Base file has 0 items: %s. Skipping intermediate creation.", base_feature_path)
            return None

        intermediate_shape_per_item: Optional[Tuple[int, ...]] = None
        final_dtype = np.float32
        write_errors = 0
        successful_indices: List[int] = []
        h5_intermediate_file = None
        h5_base_file = None

        try:
            h5_base_file = h5py.File(base_feature_path, "r")
            if HDF5_DATASET_NAME not in h5_base_file:
                raise ValueError(f"Dataset '{HDF5_DATASET_NAME}' missing in {base_feature_path}")
            base_dset = h5_base_file[HDF5_DATASET_NAME]
            if base_dset.shape[0] != n_items:
                logger.warning("Base shape %d != meta count %d. Using shape[0].", base_dset.shape[0], n_items)
                n_items = base_dset.shape[0]
            if n_items == 0:
                logger.warning("Base file confirmed 0 items. Skipping intermediate creation.")
                return None

            first_item = base_dset[0]
            processed_first_item = apply_averaging(first_item, averaging_method)
            if processed_first_item is None:
                raise ValueError("Averaging returned None for first item.")
            if processed_first_item.ndim != 1:
                raise ValueError(f"Intermediate shape must be 1D after processing, got {processed_first_item.ndim}D.")
            intermediate_shape_per_item = processed_first_item.shape
            logger.info("Intermediate feature dimension for '%s': %d", feature_name, intermediate_shape_per_item[0])

            intermediate_feature_path.parent.mkdir(parents=True, exist_ok=True)
            h5_intermediate_file = h5py.File(intermediate_feature_path, "w")
            intermediate_dset = None

            pbar = tqdm(range(0, n_items, chunk_size), desc=f"Intermediate [{feature_name}]", leave=False)
            all_processed_batches = []
            items_processed_count = 0

            for i in pbar:
                start_idx = i
                end_idx = min(i + chunk_size, n_items)
                if start_idx >= end_idx:
                    continue
                try:
                    chunk_data = base_dset[start_idx:end_idx]
                    current_chunk_processed_items = []
                    current_chunk_success_indices = []

                    for j in range(chunk_data.shape[0]):
                        item_idx_in_base = start_idx + j
                        processed_item = apply_averaging(chunk_data[j], averaging_method)
                        if processed_item is not None and processed_item.size > 0 and not np.isnan(processed_item).any() and not np.isinf(processed_item).any():
                            if processed_item.shape != intermediate_shape_per_item:
                                logger.warning("Item %d processed shape %s != expected %s. Skipping.", item_idx_in_base, processed_item.shape, intermediate_shape_per_item)
                                write_errors += 1
                            else:
                                current_chunk_processed_items.append(processed_item.astype(final_dtype))
                                if base_original_indices is not None:
                                    current_chunk_success_indices.append(base_original_indices[item_idx_in_base])
                                else:
                                    current_chunk_success_indices.append(item_idx_in_base)

                        else:
                            logger.warning("Skipping item %d due to processing error/invalid output (avg='%s').", item_idx_in_base, averaging_method)
                            write_errors += 1

                    if current_chunk_processed_items:
                        batch_array = np.stack(current_chunk_processed_items)
                        all_processed_batches.append(batch_array)
                        successful_indices.extend(current_chunk_success_indices)
                        items_processed_count += batch_array.shape[0]

                    del chunk_data, current_chunk_processed_items, current_chunk_success_indices
                    gc.collect()

                except Exception as chunk_err:
                    logger.error("Error processing intermediate chunk %d-%d: %s", start_idx, end_idx, chunk_err, exc_info=True)
                    write_errors += end_idx - start_idx
            pbar.close()

            if not all_processed_batches:
                logger.error("No items processed successfully for intermediate feature '%s'.", feature_name)
                raise RuntimeError("Intermediate processing failed for all items.")

            final_intermediate_data = np.concatenate(all_processed_batches, axis=0)
            if final_intermediate_data.shape[0] != items_processed_count:
                raise RuntimeError("Mismatch between processed count and concatenated array size.")

            intermediate_dset = h5_intermediate_file.create_dataset(
                HDF5_DATASET_NAME,
                data=final_intermediate_data,
                dtype=final_dtype,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                fletcher32=True,
            )
            logger.info("Written %d items to intermediate dataset.", final_intermediate_data.shape[0])

            intermediate_metadata_attrs = {
                "feature_config_name": feature_name,
                "base_feature_path": str(base_feature_path),
                "averaging_method": averaging_method,
                "num_items": items_processed_count,
                "feature_dim": list(intermediate_shape_per_item),
                "feature_ndim": 1,
                "num_processing_errors": write_errors,
            }
            serializable_attrs = json.loads(json.dumps(intermediate_metadata_attrs, cls=NpEncoder))
            for k, v in serializable_attrs.items():
                try:
                    intermediate_dset.attrs[k] = v
                except TypeError:
                    intermediate_dset.attrs[k] = str(v)

            if successful_indices:
                indices_array = np.array(successful_indices, dtype=np.int64)
                if indices_array.shape[0] == items_processed_count:
                    h5_intermediate_file.create_dataset(HDF5_INDICES_NAME, data=indices_array, compression="gzip", compression_opts=4)
                    logger.info("Saved corresponding original indices (%d items).", len(successful_indices))
                else:
                    logger.error("Mismatch between successful indices count and items written. Indices not saved.")
                    intermediate_metadata_attrs["indices_save_error"] = "Count mismatch"

            final_intermediate_metadata = intermediate_metadata_attrs.copy()
            if successful_indices and "indices_save_error" not in final_intermediate_metadata:
                final_intermediate_metadata[HDF5_INDICES_NAME] = successful_indices
            else:
                final_intermediate_metadata[HDF5_INDICES_NAME] = None
            return final_intermediate_metadata

        except Exception as e:
            logger.error("Failed processing intermediate features for '%s': %s", feature_name, e, exc_info=True)
            try:
                intermediate_feature_path.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        finally:
            if h5_base_file:
                h5_base_file.close()
            if h5_intermediate_file:
                h5_intermediate_file.close()
            gc.collect()

    def _process_pca_features(
        self,
        intermediate_feature_path: Path,
        pca_feature_path: Path,
        feature_config: Dict[str, Any],
        dataset_cache_id: str,
        subset_features_dir: Path,
        chunk_size: int = 1024,
    ) -> Optional[Dict[str, Any]]:
        """
        Applies PCA (standard or incremental) based on config.

        Args:
            intermediate_feature_path (Path): Path to READ intermediate feature HDF5 file.
            pca_feature_path (Path): Path to SAVE final PCA feature HDF5 file.
            feature_config (Dict[str, Any]): Effective configuration for the PCA step.
            dataset_cache_id (str): Unique identifier for the dataset subset/split.
            subset_features_dir (Path): Directory for saving PCA model cache files.
            chunk_size (int): Size of chunks to read from the intermediate HDF5 file.

        Returns:
            Optional[Dict[str, Any]]: Metadata of the final PCA feature file, or None on failure.
        """
        feature_name = feature_config.get("name", "unknown_pca")
        pca_components = feature_config.get("pca")
        if not pca_components:
            logger.error("PCA components not specified for '%s'.", feature_name)
            return None
        pca_components = int(pca_components)
        pca_load_chunks = feature_config.get("pca_load_chunks", self.default_pca_load_chunks)
        pca_model_base_dir = self.base_models_dir / "pca_models"
        pca_model_base_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Processing PCA for '%s' (Load Chunks Config: %s) -> %s", feature_name, pca_load_chunks, pca_feature_path.name)

        if not intermediate_feature_path.exists():
            logger.error("Intermediate file not found: %s. Cannot apply PCA.", intermediate_feature_path)
            return None

        intermediate_metadata = read_hdf5_metadata(intermediate_feature_path, HDF5_DATASET_NAME)
        if not intermediate_metadata:
            logger.error("Could not read metadata from intermediate file: %s", intermediate_feature_path)
            return None

        intermediate_original_indices = load_hdf5(intermediate_feature_path, HDF5_INDICES_NAME, "Intermediate Indices")

        n_items = intermediate_metadata.get("num_items")
        if n_items is None:
            logger.error("Item count missing from intermediate metadata: %s", intermediate_feature_path)
            return None
        n_items = int(n_items)
        if n_items == 0:
            logger.warning("Intermediate file has 0 items. Skipping PCA processing.")
            return None

        pca_model: Optional[Union[PCA, IncrementalPCA]] = None
        h5_pca_file = None
        h5_intermediate_file_fit = None
        h5_intermediate_file_transform = None
        final_shape_per_item: Tuple[int, ...] = (pca_components,)
        final_dtype = np.float32
        write_errors = 0
        intermediate_dim = intermediate_metadata.get("feature_dim", [0])[0]

        try:
            pca_model_path = get_cache_path(pca_model_base_dir, "pca_model", dataset_cache_id, feature_config)

            if str(pca_model_path) in self.pca_model_cache:
                pca_model = self.pca_model_cache[str(pca_model_path)]
                logger.info("Using cached PCA model instance for '%s'.", feature_name)
            elif pca_model_path.exists():
                pca_model = load_pca_model(pca_model_path)
                if pca_model:
                    logger.info("Loaded existing PCA model file for '%s' from %s", feature_name, pca_model_path.name)
                    self.pca_model_cache[str(pca_model_path)] = pca_model
                else:
                    logger.warning("Failed to load PCA model from %s. Will refit.", pca_model_path.name)
                    try:
                        pca_model_path.unlink(missing_ok=True)
                    except OSError:
                        pass

            if pca_model is None:
                logger.info("Fitting PCA model for '%s'...", feature_name)
                h5_intermediate_file_fit = h5py.File(intermediate_feature_path, "r")
                intermediate_dset_fit = h5_intermediate_file_fit[HDF5_DATASET_NAME]

                if pca_components > intermediate_dim:
                    logger.warning("Requested PCA components (%d) > intermediate dim (%d). Using %d.", pca_components, intermediate_dim, intermediate_dim)
                    pca_components = intermediate_dim
                    final_shape_per_item = (pca_components,)

                if pca_load_chunks == -1:
                    logger.info("Using standard PCA (loading all %d intermediate items)...", n_items)
                    all_data = None
                    try:
                        all_data = intermediate_dset_fit[:].astype(np.float32)
                        logger.info("Fitting standard PCA (n=%d) on data shape %s...", pca_components, all_data.shape)
                        pca_model = PCA(n_components=pca_components)
                        pca_model.fit(all_data)
                        explained_var = np.sum(pca_model.explained_variance_ratio_) if hasattr(pca_model, "explained_variance_ratio_") else np.nan
                        logger.info("Standard PCA fit complete. Explained variance: %.4f", explained_var)
                    except MemoryError:
                        logger.error(
                            "MemoryError loading all %d items (%d dims) for standard PCA. Try setting 'pca_load_chunks: 0' for '%s'.",
                            n_items,
                            intermediate_dim,
                            feature_name,
                        )
                        raise
                    except Exception as std_pca_err:
                        logger.error("Standard PCA fitting failed: %s", std_pca_err, exc_info=True)
                        raise
                    finally:
                        if all_data is not None:
                            del all_data
                        gc.collect()
                else:
                    pca_model = IncrementalPCA(n_components=pca_components, batch_size=chunk_size)
                    num_chunks_to_fit = float("inf") if pca_load_chunks == 0 else pca_load_chunks
                    fit_desc = "all chunks" if num_chunks_to_fit == float("inf") else f"first {int(num_chunks_to_fit)} chunk(s)"
                    logger.info("Using IncrementalPCA (n=%d, fitting on %s)...", pca_components, fit_desc)
                    pbar_fit = tqdm(range(0, n_items, chunk_size), desc=f"Fit IncPCA [{feature_name}]", leave=False)
                    chunks_fitted = 0
                    total_samples_fitted = 0
                    for i in pbar_fit:
                        if chunks_fitted >= num_chunks_to_fit:
                            break
                        start_idx, end_idx = i, min(i + chunk_size, n_items)
                        if start_idx >= end_idx:
                            continue
                        try:
                            chunk_data = intermediate_dset_fit[start_idx:end_idx].astype(np.float32)
                            if chunk_data.shape[0] > 0:
                                pca_model.partial_fit(chunk_data)
                                chunks_fitted += 1
                                total_samples_fitted += chunk_data.shape[0]
                                pbar_fit.set_postfix({"Fitted Chunks": chunks_fitted, "Samples": total_samples_fitted})
                            del chunk_data
                            gc.collect()
                        except Exception as inc_fit_err:
                            logger.error("IncrementalPCA fit failed chunk %d-%d: %s", start_idx, end_idx, inc_fit_err, exc_info=True)
                            raise
                    pbar_fit.close()
                    if chunks_fitted == 0:
                        raise RuntimeError("IncrementalPCA fitted 0 chunks.")
                    fitted_components_inc = getattr(pca_model, "n_components_", 0)
                    explained_var_inc = np.sum(pca_model.explained_variance_ratio_) if hasattr(pca_model, "explained_variance_ratio_") and pca_model.explained_variance_ratio_ is not None else np.nan
                    logger.info("IncrementalPCA fit complete. Fitted %d components on %d samples. Explained var: %.4f", fitted_components_inc, total_samples_fitted, explained_var_inc)

                h5_intermediate_file_fit.close()
                h5_intermediate_file_fit = None
                if pca_model:
                    save_pca_model(pca_model, pca_model_path)
                    logger.info("PCA model fitted and saved to %s", pca_model_path.name)
                    self.pca_model_cache[str(pca_model_path)] = pca_model
                else:
                    raise RuntimeError("PCA fitting finished without a model.")
            else:
                loaded_components = getattr(pca_model, "n_components_", 0)
                if loaded_components > intermediate_dim:
                    logger.error("Loaded PCA model has %d components, but intermediate data only has %d dimensions. Cannot use this model.", loaded_components, intermediate_dim)
                    return None
                if loaded_components < pca_components:
                    logger.warning("Loaded PCA model has %d components, less than requested %d. Final features will have %d dimensions.", loaded_components, pca_components, loaded_components)
                    final_shape_per_item = (loaded_components,)
                elif loaded_components > pca_components:
                    logger.info("Loaded PCA model has %d components. Will use only the first %d as requested.", loaded_components, pca_components)

            logger.info("Applying PCA transform (Output Dim: %d) and writing to: %s", final_shape_per_item[0], pca_feature_path.name)
            pca_feature_path.parent.mkdir(parents=True, exist_ok=True)
            h5_intermediate_file_transform = h5py.File(intermediate_feature_path, "r")
            intermediate_dset_transform = h5_intermediate_file_transform[HDF5_DATASET_NAME]
            h5_pca_file = h5py.File(pca_feature_path, "w")
            pca_dset = h5_pca_file.create_dataset(
                HDF5_DATASET_NAME,
                shape=(n_items, *final_shape_per_item),
                dtype=final_dtype,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                fletcher32=True,
            )

            explained_var_ratio = pca_model.explained_variance_ratio_[:final_shape_per_item[0]] if hasattr(pca_model, "explained_variance_ratio_") else None
            pca_metadata_attrs = {
                "feature_config_name": feature_name,
                "source_feature_path": str(intermediate_feature_path),
                "pca_model_path": str(pca_model_path),
                "pca_components_requested": int(pca_components),
                "pca_components_actual": final_shape_per_item[0],
                "pca_load_chunks_config": pca_load_chunks,
                "num_items": n_items,
                "feature_dim": list(final_shape_per_item),
                "feature_ndim": 1,
                "explained_variance_ratio_sum": float(np.sum(explained_var_ratio)) if explained_var_ratio is not None else None,
            }
            serializable_attrs = json.loads(json.dumps(pca_metadata_attrs, cls=NpEncoder))
            for k, v in serializable_attrs.items():
                try:
                    pca_dset.attrs[k] = v
                except TypeError:
                    pca_dset.attrs[k] = str(v)

            if intermediate_original_indices is not None:
                indices_array = np.array(intermediate_original_indices, dtype=np.int64)
                if indices_array.shape[0] == n_items:
                    h5_pca_file.create_dataset(HDF5_INDICES_NAME, data=indices_array, compression="gzip", compression_opts=4)
                else:
                    logger.warning("Indices length mismatch (%d) vs item count (%d). Indices not saved.", indices_array.shape[0], n_items)
                    pca_metadata_attrs["indices_save_error"] = "Count mismatch"

            pbar_transform = tqdm(range(0, n_items, chunk_size), desc=f"Transform PCA [{feature_name}]", leave=False)
            write_idx = 0
            for i in pbar_transform:
                start_idx, end_idx = i, min(i + chunk_size, n_items)
                if start_idx >= end_idx:
                    continue
                try:
                    chunk_data = intermediate_dset_transform[start_idx:end_idx].astype(np.float32)
                    transformed_chunk = pca_model.transform(chunk_data)
                    if transformed_chunk.shape[1] > final_shape_per_item[0]:
                        transformed_chunk = transformed_chunk[:, :final_shape_per_item[0]]

                    transformed_chunk = transformed_chunk.astype(final_dtype)

                    current_chunk_size = transformed_chunk.shape[0]
                    actual_end_write_idx = write_idx + current_chunk_size
                    expected_slice_shape = (current_chunk_size, *final_shape_per_item)

                    if transformed_chunk.shape != expected_slice_shape:
                        raise ValueError(f"Transformed chunk shape {transformed_chunk.shape} mismatch expected {expected_slice_shape}")
                    if actual_end_write_idx > n_items:
                        raise IndexError(f"PCA write index {actual_end_write_idx} exceeds total {n_items}")

                    pca_dset[write_idx:actual_end_write_idx] = transformed_chunk
                    write_idx = actual_end_write_idx

                except Exception as chunk_pca_err:
                    logger.error("Error transforming/writing PCA chunk %d-%d: %s", start_idx, end_idx, chunk_pca_err, exc_info=True)
                    write_errors += end_idx - start_idx
            pbar_transform.close()

            if write_idx != n_items or write_errors > 0:
                raise RuntimeError(f"PCA feature write failed. Written: {write_idx}/{n_items}. Errors: {write_errors}.")

            final_pca_metadata = pca_metadata_attrs.copy()
            if intermediate_original_indices is not None and "indices_save_error" not in final_pca_metadata:
                final_pca_metadata[HDF5_INDICES_NAME] = intermediate_original_indices
            else:
                final_pca_metadata[HDF5_INDICES_NAME] = None
            return final_pca_metadata

        except Exception as e:
            logger.error("Failed PCA processing for '%s': %s", feature_name, e, exc_info=True)
            try:
                pca_feature_path.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        finally:
            if h5_intermediate_file_fit:
                h5_intermediate_file_fit.close()
            if h5_intermediate_file_transform:
                h5_intermediate_file_transform.close()
            if h5_pca_file:
                h5_pca_file.close()
            if "pca_model" in locals() and pca_model is not None:
                del pca_model
            gc.collect()

    def _check_cache(
        self,
        prefix: str,
        cache_dir: Path,
        dataset_cache_id: str,
        effective_config: Dict[str, Any],
        expected_item_count: Optional[int] = None,
        check_attr: str = "num_items",
    ) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
        """
        Checks cache using the exact expected path, validates metadata.

        Args:
            prefix (str): Prefix for the cache file (e.g., 'features', 'intermediate').
            cache_dir (Path): The directory where the cache file is expected.
            dataset_cache_id (str): Unique identifier for the dataset.
            effective_config (Dict[str, Any]): The effective feature configuration used for path generation.
            expected_item_count (Optional[int]): The expected number of items in the HDF5 file.
            check_attr (str): The name of the attribute in the HDF5 file to check the item count against.

        Returns:
            Tuple[Optional[Path], Optional[Dict[str, Any]]]: A tuple containing the path to the valid cache file
                                                            and its metadata, or (None, None) if cache is missing,
                                                            invalid, or does not match expected count.
        """
        feature_name = effective_config.get("name", "unknown")
        expected_path = self.get_feature_file_path(feature_name, dataset_cache_id, cache_dir)
        if expected_path is None:
            logger.warning("Could not determine expected cache path for '%s'.", feature_name)
            return None, None

        file_type = prefix.split("_")[0].upper()

        if not expected_path.is_file():
            logger.debug("%s cache MISS for '%s': %s", file_type, feature_name, expected_path.name)
            return None, None

        logger.debug("Found potential %s match for '%s': %s", file_type, feature_name, expected_path.name)
        metadata = read_hdf5_metadata(expected_path, HDF5_DATASET_NAME)

        if not metadata:
            logger.warning("Could not read metadata from %s cache %s. Discarding.", file_type, expected_path.name)
            return None, None

        if expected_item_count is not None and expected_item_count >= 0:
            item_count_from_meta = metadata.get(check_attr)
            if item_count_from_meta is None:
                logger.warning("%s cache %s for '%s' missing item count attr '%s'. Discarding.", file_type, expected_path.name, feature_name, check_attr)
                return None, None
            try:
                item_count = int(item_count_from_meta)
                if item_count != expected_item_count:
                    logger.warning(
                        "%s cache %s for '%s' has mismatched count (%d vs %d). Discarding.",
                        file_type,
                        expected_path.name,
                        feature_name,
                        item_count,
                        expected_item_count,
                    )
                    return None, None
            except (ValueError, TypeError):
                logger.warning("Invalid item count attribute '%s' value '%s' in %s. Discarding.", check_attr, item_count_from_meta, expected_path.name)
                return None, None

        indices = load_hdf5(expected_path, HDF5_INDICES_NAME, f"{feature_name} indices")
        if indices is not None:
            metadata[HDF5_INDICES_NAME] = indices

        logger.info("Using valid %s cache for '%s': %s", file_type, feature_name, expected_path.name)
        return expected_path, metadata

    def process_subset_features(
        self,
        subset_dataset_obj: Any,
        dataset_cache_id: str,
        current_subset_key: str,
        subset_features_dir: Path,
        item_id_map: Optional[Dict],
        run_steps: List[str],
    ) -> Dict[str, Path]:
        """
        Processes all configured features for a given dataset subset.

        Args:
            subset_dataset_obj (Any): The actual dataset subset object.
            dataset_cache_id (str): Unique identifier for the dataset subset/split.
            current_subset_key (str): Name of the current subset.
            subset_features_dir (Path): Directory for storing feature cache files for this subset.
            item_id_map (Optional[Dict]): Map from original item IDs to metadata.
            run_steps (List[str]): List of steps being run in the pipeline.

        Returns:
            Dict[str, Path]: A dictionary mapping feature names to paths of processed feature files.
                             Only includes features successfully processed or found in cache.
        """
        logger.info("--- Feature Processing (Steps: %s) for Subset: %s ---", run_steps, current_subset_key)
        final_feature_paths: Dict[str, Path] = {}
        available_base_paths: Dict[str, Path] = {}
        available_intermediate_paths: Dict[str, Path] = {}

        self.pca_model_cache.clear()
        self._clear_extractor_cache()
        self.processed_feature_metadata.clear()
        total_items_in_subset: Optional[int] = None
        if hasattr(subset_dataset_obj, "__len__"):
            try:
                total_items_in_subset = len(subset_dataset_obj)
            except TypeError:
                logger.warning("Dataset object claims no length.")

        compute_features_step = "features" in run_steps

        subset_features_dir.mkdir(parents=True, exist_ok=True)

        configs_to_process = self.all_feature_configs

        logger.info("--- Feature Pass 1: Base Features ---")
        for feature_config in configs_to_process:
            feature_name = feature_config.get("name")
            if not feature_name or feature_config.get("base_extractor"):
                continue

            effective_config = self._get_effective_feature_config(feature_config, current_subset_key)
            path_to_use, cached_metadata = self._check_cache("features", subset_features_dir, dataset_cache_id, effective_config, total_items_in_subset, check_attr="num_items_processed")

            if path_to_use and cached_metadata:
                available_base_paths[feature_name] = path_to_use
                self.processed_feature_metadata[feature_name] = cached_metadata
                if effective_config.get("benchmark_this", True) and not effective_config.get("averaging") and not effective_config.get("pca"):
                    final_feature_paths[feature_name] = path_to_use
            elif compute_features_step:
                logger.info("Base cache MISS for '%s'. Computing...", feature_name)
                expected_save_path = get_cache_path(subset_features_dir, "features", dataset_cache_id, effective_config)
                extractor = self._get_feature_extractor(effective_config)
                if extractor:
                    computed_metadata = self._extract_and_save_features_hdf5(
                        extractor=extractor,
                        dataset=subset_dataset_obj,
                        cache_path=expected_save_path,
                        feature_config=effective_config,
                        item_id_map=item_id_map,
                        batch_size=self.cfg.get("extraction_batch_size", 32),
                    )
                    if feature_name in self._extractor_instances_cache:
                        del self._extractor_instances_cache[feature_name]
                    del extractor
                    gc.collect()

                    if computed_metadata:
                        available_base_paths[feature_name] = expected_save_path
                        self.processed_feature_metadata[feature_name] = computed_metadata
                        if effective_config.get("benchmark_this", True) and not effective_config.get("averaging") and not effective_config.get("pca"):
                            final_feature_paths[feature_name] = expected_save_path
                        logger.info("Base features computed for '%s'.", feature_name)
                    else:
                        logger.error("Base computation FAILED for '%s'.", feature_name)
                else:
                    logger.error("Could not get extractor for base feature '%s'.", feature_name)
            else:
                logger.info("Base cache MISS for '%s' and 'features' step skipped.", feature_name)

        logger.info("--- Feature Pass 2: Intermediate Features ---")
        for feature_config in configs_to_process:
            feature_name = feature_config.get("name")
            base_feature_name = feature_config.get("base_extractor")
            needs_intermediate = feature_config.get("averaging") is not None
            if not base_feature_name or not needs_intermediate:
                continue

            effective_config = self._get_effective_feature_config(feature_config, current_subset_key)
            
            # For intermediate, we expect the count to match the base feature's count.
            base_meta = self.processed_feature_metadata.get(base_feature_name)
            expected_count = int(base_meta.get("num_items_processed", -1)) if base_meta else -1
            
            path_to_use, cached_metadata = self._check_cache("intermediate", subset_features_dir, dataset_cache_id, effective_config, expected_count, check_attr="num_items")

            if path_to_use and cached_metadata:
                available_intermediate_paths[feature_name] = path_to_use
                self.processed_feature_metadata[feature_name] = cached_metadata
                if effective_config.get("benchmark_this", True) and not effective_config.get("pca"):
                    final_feature_paths[feature_name] = path_to_use
            elif compute_features_step:
                base_feature_path = available_base_paths.get(base_feature_name)
                if not base_feature_path:
                    logger.error("Cannot compute intermediate '%s': Base feature '%s' not available.", feature_name, base_feature_name)
                    continue
                logger.info("Intermediate cache MISS for '%s'. Computing...", feature_name)
                expected_inter_save_path = get_cache_path(subset_features_dir, "intermediate", dataset_cache_id, effective_config)
                inter_meta = self._process_intermediate_features(base_feature_path=base_feature_path, intermediate_feature_path=expected_inter_save_path, feature_config=effective_config, chunk_size=self.cfg.get("extraction_batch_size", 1024))
                if inter_meta:
                    available_intermediate_paths[feature_name] = expected_inter_save_path
                    self.processed_feature_metadata[feature_name] = inter_meta
                    if effective_config.get("benchmark_this", True) and not effective_config.get("pca"):
                        final_feature_paths[feature_name] = expected_inter_save_path
                    logger.info("Intermediate features computed for '%s'.", feature_name)
                else:
                    logger.error("Intermediate computation FAILED for '%s'.", feature_name)
            else:
                logger.info("Intermediate cache MISS for '%s' and 'features' step skipped.", feature_name)

        logger.info("--- Feature Pass 3: PCA Features ---")
        for feature_config in configs_to_process:
            feature_name = feature_config.get("name")
            base_feature_name = feature_config.get("base_extractor")
            needs_pca = feature_config.get("pca") is not None
            if not base_feature_name or not needs_pca:
                continue

            effective_config = self._get_effective_feature_config(feature_config, current_subset_key)
            path_to_use, cached_metadata = self._check_cache("features", subset_features_dir, dataset_cache_id, effective_config, -1, check_attr="num_items") # Count check is tricky here

            if path_to_use and cached_metadata:
                self.processed_feature_metadata[feature_name] = cached_metadata
                if effective_config.get("benchmark_this", True):
                    final_feature_paths[feature_name] = path_to_use
            elif compute_features_step:
                # Determine the name of the feature that serves as input to PCA
                pca_input_feature_name = base_feature_name
                
                pca_input_path = available_intermediate_paths.get(pca_input_feature_name)
                if not pca_input_path:
                    pca_input_path = available_base_paths.get(pca_input_feature_name)

                if not pca_input_path:
                    logger.error("Cannot compute PCA '%s': Input feature '%s' (base or intermediate) not available.", feature_name, pca_input_feature_name)
                    continue

                logger.info("PCA cache MISS for '%s'. Computing...", feature_name)
                expected_pca_save_path = get_cache_path(subset_features_dir, "features", dataset_cache_id, effective_config)
                pca_meta = self._process_pca_features(
                    intermediate_feature_path=pca_input_path,
                    pca_feature_path=expected_pca_save_path,
                    feature_config=effective_config,
                    dataset_cache_id=dataset_cache_id,
                    subset_features_dir=subset_features_dir,
                    chunk_size=self.cfg.get("extraction_batch_size", 1024),
                )
                if pca_meta:
                    self.processed_feature_metadata[feature_name] = pca_meta
                    if effective_config.get("benchmark_this", True):
                        final_feature_paths[feature_name] = expected_pca_save_path
                    logger.info("PCA features computed for '%s'.", feature_name)
                else:
                    logger.error("PCA computation FAILED for '%s'.", feature_name)
            else:
                logger.info("PCA cache MISS for '%s' and 'features' step skipped.", feature_name)

        self.pca_model_cache.clear()
        self._clear_extractor_cache()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("--- Feature Processing Completed for Subset: %s ---", current_subset_key)
        return final_feature_paths