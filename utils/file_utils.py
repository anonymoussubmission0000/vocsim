import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA, IncrementalPCA
import h5py
import os


logger = logging.getLogger(__name__)
HDF5_DATASET_NAME = "features"
HDF5_INDICES_NAME = "original_indices"


class NpEncoder(json.JSONEncoder):
    """Helper class for JSON encoding NumPy types, Path, Tensors etc."""

    def default(self, obj):
        """
        Encodes various object types into JSON-serializable formats.

        Args:
            obj: The object to encode.

        Returns:
            JSON-serializable representation of the object.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj) if np.isfinite(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (torch.Tensor)):
            return obj.detach().cpu().numpy().tolist()
        if isinstance(obj, Path):
            return str(obj)
        if obj is None:
            return None
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


def ensure_dir_exists(path_obj: Union[str, Path]):
    """
    Checks if a directory exists at the given path, and creates it if not.

    Args:
        path_obj: The path to the directory (string or Path object).

    Raises:
        OSError: If the path exists but is a file, or if directory creation fails.
    """
    if not isinstance(path_obj, Path):
        path_obj = Path(path_obj)

    try:
        if path_obj.exists():
            if not path_obj.is_dir():
                error_msg = f"Path exists but is not a directory: {path_obj}"
                logger.error(error_msg)
                raise OSError(error_msg)
        else:
            logger.info("Creating directory: %s", path_obj)
            path_obj.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error("Failed to ensure directory exists at %s: %s", path_obj, e, exc_info=True)
        raise


def _generate_config_hash(config: Dict[str, Any], length: int = 10) -> str:
    """
    Generates a deterministic hash for a configuration dictionary.

    Args:
        config (Dict[str, Any]): The configuration dictionary.
        length (int): The desired length of the hash string.

    Returns:
        str: The configuration hash.
    """
    hasher = hashlib.md5()
    relevant_keys = [
        "name",
        "module",
        "class",
        "base_extractor",
        "averaging",
        "pca",
        "pca_load_chunks",
        "distance_name",
        "feature_config",
        "distance_config",
        "benchmark_config",
        "params",
    ]
    relevant_config = {}
    for key in relevant_keys:
        value = config.get(key)
        if value is not None:
            if isinstance(value, dict):
                try:
                    relevant_config[key] = json.dumps(value, sort_keys=True, cls=NpEncoder)
                except TypeError:
                    relevant_config[key] = repr(sorted(value.items()))
            elif isinstance(value, (list, tuple, set)):
                try:
                    try:
                        processed_list = sorted(list(value)) if isinstance(value, set) else sorted(value)
                        relevant_config[key] = json.dumps(processed_list, cls=NpEncoder)
                    except TypeError:
                        relevant_config[key] = repr(value)
                except TypeError:
                    relevant_config[key] = repr(value)
            else:
                relevant_config[key] = value

    try:
        serialized = json.dumps(relevant_config, sort_keys=True, cls=NpEncoder).encode("utf-8")
        hasher.update(serialized)
    except TypeError as e:
        logger.warning("JSON hash failed: %s. Using repr.", e)
        hasher.update(repr(sorted(relevant_config.items())).encode("utf-8"))

    return hasher.hexdigest()[:length]


def _get_safe_path_part(name: Optional[str], default="unknown") -> str:
    """
    Sanitizes a string to be safe for use in file paths.

    Args:
        name (Optional[str]): The input string.
        default (str): The default string to use if the input is None or becomes empty after sanitization.

    Returns:
        str: The sanitized string.
    """
    if name is None:
        name = default
    name = str(name)
    chars_to_replace = r'<>:"/\|?*' + "".join(map(chr, range(32)))
    safe_name = name
    for char in chars_to_replace:
        safe_name = safe_name.replace(char, "_")
    safe_name = safe_name.strip(" .")
    return safe_name or default


def get_cache_path(cache_dir: Path, prefix: str, dataset_cache_id: str, config_dict: Dict[str, Any], extra_suffix: Optional[str] = None) -> Path:
    """
    Generates the deterministic cache file path based on configuration details and prefix.

    Args:
        cache_dir (Path): The base directory for caching.
        prefix (str): Prefix indicating data type (e.g., 'features', 'distances_cosine').
        dataset_cache_id (str): Unique identifier for the dataset subset/split.
        config_dict (Dict[str, Any]): Configuration dictionary for hash generation and name extraction.
        extra_suffix (Optional[str]): Optional extra string used in the filename.

    Returns:
        Path: The generated cache file path.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_dataset_id = _get_safe_path_part(dataset_cache_id, "unknown_dataset")
    config_hash = _generate_config_hash(config_dict)

    if prefix.startswith("distances_"):
        feature_conf = config_dict.get("feature_config", {})
        item_name = feature_conf.get("name", "unknown_feature")
        safe_item_name = _get_safe_path_part(item_name)
        base_name_parts = [prefix, safe_dataset_id, safe_item_name]
        suffix = ".h5"
    elif prefix == "features":
        item_name = config_dict.get("name", "unknown_feature")
        safe_item_name = _get_safe_path_part(item_name)
        base_name_parts = [prefix, safe_dataset_id, safe_item_name]
        suffix = ".h5"
    elif prefix == "intermediate":
        item_name = config_dict.get("name", "unknown_intermediate")
        safe_item_name = _get_safe_path_part(item_name)
        base_name_parts = [prefix, safe_dataset_id, safe_item_name]
        suffix = ".h5"
    elif prefix == "pca_model":
        item_name = config_dict.get("name", "unknown_pcamodel")
        safe_item_name = _get_safe_path_part(item_name)
        base_name_parts = [prefix, safe_dataset_id, safe_item_name]
        suffix = ".pkl"
    elif prefix == "bench_item":
        item_name = config_dict.get("name", "unknown_benchitem")
        safe_item_name = _get_safe_path_part(item_name)
        base_name_parts = [prefix, safe_dataset_id, safe_item_name]
        suffix = ".json"
    elif prefix == "bench_results_summary":
        summary_name = f"{prefix}_{safe_dataset_id}"
        if extra_suffix:
            summary_name += f"_{_get_safe_path_part(extra_suffix)}"
        return (cache_dir / summary_name).with_suffix(".csv")
    elif prefix == "bench_results":
        results_name = f"{prefix}_{safe_dataset_id}"
        if extra_suffix:
            results_name += f"_{_get_safe_path_part(extra_suffix)}"
        return (cache_dir / results_name).with_suffix(".json")
    else:
        logger.warning("Using fallback path construction for unknown prefix: '%s'", prefix)
        item_name = config_dict.get("name", prefix)
        safe_item_name = _get_safe_path_part(item_name)
        base_name_parts = [prefix, safe_dataset_id]
        if item_name != prefix and safe_item_name:
            base_name_parts.append(safe_item_name)
        suffix = ".dat"

    if extra_suffix:
        base_name_parts.append(_get_safe_path_part(extra_suffix))

    base_name_parts.append(config_hash)
    base_name = "_".join(filter(None, base_name_parts))

    return (cache_dir / base_name).with_suffix(suffix)


def find_cache_path(cache_dir: Path, prefix: str, dataset_cache_id: str, config_dict: Dict[str, Any], extra_suffix: Optional[str] = None) -> Optional[Path]:
    """
    Finds a cache file path, first trying the exact path with hash,
    then falling back to a looser match based on names (without hash).

    Args:
        cache_dir: The base directory for caching.
        prefix: Prefix indicating data type (e.g., 'features', 'distances_cosine').
        dataset_cache_id: Unique identifier for the dataset subset/split.
        config_dict: Configuration dictionary for hash generation and name extraction.
                     For distances, expects 'feature_config'.
        extra_suffix: Optional extra string used in the filename.

    Returns:
        The Path object if a matching file is found, otherwise None.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        logger.debug("Cache directory does not exist: %s", cache_dir)
        return None

    exact_path = get_cache_path(cache_dir, prefix, dataset_cache_id, config_dict, extra_suffix)

    if exact_path.is_file():
        logger.debug("Found exact cache match: %s", exact_path.name)
        return exact_path

    logger.debug("Exact path %s not found. Trying loose match...", exact_path.name)

    safe_dataset_id = _get_safe_path_part(dataset_cache_id, "unknown_dataset")
    item_name = "unknown"
    base_name_parts_loose = []
    suffix = ".h5"

    if prefix.startswith("distances_"):
        feature_conf = config_dict.get("feature_config", {})
        item_name = feature_conf.get("name", "unknown_feature")
        safe_item_name = _get_safe_path_part(item_name)
        base_name_parts_loose = [prefix, safe_dataset_id, safe_item_name]
        suffix = ".h5"
    elif prefix == "features":
        item_name = config_dict.get("name", "unknown_feature")
        safe_item_name = _get_safe_path_part(item_name)
        base_name_parts_loose = [prefix, safe_dataset_id, safe_item_name]
        suffix = ".h5"
    elif prefix == "intermediate":
        item_name = config_dict.get("name", "unknown_intermediate")
        safe_item_name = _get_safe_path_part(item_name)
        base_name_parts_loose = [prefix, safe_dataset_id, safe_item_name]
        suffix = ".h5"
    elif prefix == "pca_model":
        item_name = config_dict.get("name", "unknown_pcamodel")
        safe_item_name = _get_safe_path_part(item_name)
        base_name_parts_loose = [prefix, safe_dataset_id, safe_item_name]
        suffix = ".pkl"
    elif prefix == "bench_item":
        item_name = config_dict.get("name", "unknown_benchitem")
        safe_item_name = _get_safe_path_part(item_name)
        base_name_parts_loose = [prefix, safe_dataset_id, safe_item_name]
        suffix = ".json"
    else:
        logger.warning("Using fallback construction for unknown prefix: '%s'", prefix)
        item_name = config_dict.get("name", prefix)
        safe_item_name = _get_safe_path_part(item_name)
        base_name_parts_loose = [prefix, safe_dataset_id, safe_item_name]
        suffix = ".dat"

    if extra_suffix:
        base_name_parts_loose.append(_get_safe_path_part(extra_suffix))

    loose_filename_base = "_".join(filter(None, base_name_parts_loose))
    loose_filename_pattern = f"{loose_filename_base}_*{suffix}"
    

    search_pattern = str(cache_dir / loose_filename_pattern)

    logger.debug("Searching for loose match with pattern: %s", loose_filename_pattern)

    relative_pattern = loose_filename_pattern
    found_files = list(cache_dir.glob(relative_pattern))

    if found_files:
        found_files.sort(key=os.path.getmtime, reverse=True)
        selected_file = found_files[0]

        if len(found_files) > 1:
            logger.warning("Found %d loose matches for %s/%s. Using the most recent: %s", len(found_files), prefix, item_name, selected_file.name)
        else:
            logger.info("Found loose cache match: %s", selected_file.name)
        return selected_file

    logger.info("No cache file found (exact or loose) for %s/%s", prefix, item_name)
    return None

def load_pickle(filepath: Path, log_prefix: str = "Data") -> Optional[Any]:
    """
    Loads data from a pickle file if it exists.

    Args:
        filepath (Path): Path to the pickle file.
        log_prefix (str): Prefix for logging messages.

    Returns:
        Optional[Any]: The loaded data, or None if loading fails.
    """
    if not filepath.is_file():
        logger.debug("%s pickle cache file not found: %s", log_prefix, filepath.name)
        return None
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        logger.info("Loaded %s from cache: %s", log_prefix.lower(), filepath.name)
        return data
    except (EOFError, pickle.UnpicklingError, ImportError, ModuleNotFoundError, AttributeError) as e:
        logger.error("Corrupt/incompatible pickle file %s: %s. Deleting.", filepath.name, e)
        try:
            filepath.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    except Exception as e:
        logger.error("Failed load %s from %s: %s", log_prefix.lower(), filepath.name, e)
        return None


def load_hdf5(filepath: Path, dataset_name: str = "data", log_prefix: str = "Data") -> Optional[np.ndarray]:
    """
    Loads a NumPy array from an HDF5 file if it exists.

    Args:
        filepath (Path): Path to the HDF5 file.
        dataset_name (str): Name of the dataset within the HDF5 file.
        log_prefix (str): Prefix for logging messages.

    Returns:
        Optional[np.ndarray]: The loaded array, or None if loading fails.
    """
    if not filepath.is_file():
        logger.debug("%s HDF5 cache file not found: %s", log_prefix, filepath.name)
        return None
    try:
        with h5py.File(filepath, "r") as f:
            if dataset_name not in f:
                logger.error("Dataset '%s' not found in HDF5 file %s.", dataset_name, filepath.name)
                return None
            data = f[dataset_name][:]
            logger.info("Loaded %s from HDF5 cache: %s (Shape: %s)", log_prefix.lower(), filepath.name, data.shape)
            return data
    except OSError as e:
        logger.error("Error opening/reading HDF5 %s: %s.", filepath.name, e)
        return None
    except KeyError:
        logger.error("Dataset '%s' not found in HDF5 file %s.", dataset_name, filepath.name)
        return None
    except Exception as e:
        logger.error("Failed load %s from HDF5 %s: %s", log_prefix.lower(), filepath.name, e)
        return None


def load_json_results(filepath: Path, log_prefix: str = "Benchmark item") -> Optional[Dict]:
    """
    Loads dictionary results from a JSON file if it exists.

    Args:
        filepath (Path): Path to the JSON file.
        log_prefix (str): Prefix for logging messages.

    Returns:
        Optional[Dict]: The loaded dictionary, or None if loading fails.
    """
    if not filepath.is_file():
        logger.debug("%s JSON cache file not found: %s", log_prefix, filepath.name)
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        if not isinstance(loaded_data, dict):
            logger.error("Invalid data type in JSON cache %s.", filepath.name)
            return None
        logger.info("Loaded %s from JSON cache: %s", log_prefix.lower(), filepath.name)
        return loaded_data
    except json.JSONDecodeError as e:
        logger.error("Corrupt JSON cache file %s: %s. Deleting.", filepath.name, e)
        try:
            filepath.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    except Exception as e:
        logger.error("Failed load JSON cache %s: %s", filepath.name, e)
        return None


def read_hdf5_metadata(filepath: Path, dataset_name: str = HDF5_DATASET_NAME) -> Optional[Dict[str, Any]]:
    """
    Reads metadata attributes and the original_indices dataset from an HDF5 file.

    Args:
        filepath (Path): Path to the HDF5 file.
        dataset_name (str): Name of the main dataset to read attributes from.

    Returns:
        Optional[Dict[str, Any]]: Dictionary containing metadata, including 'original_indices'
                                  if present as a dataset or attribute. Returns None if file
                                  not found or error occurs.
    """
    if not filepath.is_file():
        logger.debug("HDF5 metadata file not found: %s", filepath.name)
        return None
    metadata: Dict[str, Any] = {}
    loaded_indices: Optional[List[int]] = None
    try:
        with h5py.File(filepath, "r") as f:
            main_dset = f.get(dataset_name)
            if isinstance(main_dset, h5py.Dataset):
                metadata.update(dict(main_dset.attrs))
                metadata.pop(HDF5_INDICES_NAME, None)
            else:
                metadata.update(dict(f.attrs))
                if not metadata:
                    logger.warning("Main dataset '%s' not found and no root attributes in %s.", dataset_name, filepath.name)

            indices_dset = f.get(HDF5_INDICES_NAME)
            if isinstance(indices_dset, h5py.Dataset):
                try:
                    loaded_indices = [int(x) for x in indices_dset[:]]
                    logger.debug("Read '%s' from dataset.", HDF5_INDICES_NAME)
                except Exception as idx_err:
                    logger.error("Failed read '%s' dataset: %s. Check attrs.", HDF5_INDICES_NAME, idx_err)
            elif loaded_indices is None:
                attr_indices_val = None
                if isinstance(main_dset, h5py.Dataset) and HDF5_INDICES_NAME in main_dset.attrs:
                    attr_indices_val = main_dset.attrs[HDF5_INDICES_NAME]
                    logger.warning("Found '%s' as attribute on main dataset (old format?).", HDF5_INDICES_NAME)
                elif HDF5_INDICES_NAME in f.attrs:
                    attr_indices_val = f.attrs[HDF5_INDICES_NAME]
                    logger.warning("Found '%s' as attribute on file root.", HDF5_INDICES_NAME)

                if attr_indices_val is not None:
                    try:
                        if isinstance(attr_indices_val, np.ndarray):
                            loaded_indices = [int(x) for x in attr_indices_val]
                        elif isinstance(attr_indices_val, (list, tuple)):
                            loaded_indices = [int(x) for x in attr_indices_val]
                        elif isinstance(attr_indices_val, str):
                            loaded_indices = [int(x) for x in json.loads(attr_indices_val)]
                        else:
                            loaded_indices = [int(x) for x in list(attr_indices_val)]
                    except (TypeError, ValueError, json.JSONDecodeError) as attr_err:
                        logger.error("Could not convert '%s' attribute: %s.", HDF5_INDICES_NAME, attr_err)

            if loaded_indices is not None:
                metadata[HDF5_INDICES_NAME] = loaded_indices

    except OSError as e:
        logger.error("Error opening/reading HDF5 metadata %s: %s", filepath.name, e)
        return None
    except Exception as e:
        logger.error("Unexpected error reading HDF5 metadata %s: %s", filepath.name, e, exc_info=True)
        return None
    return metadata


def save_pickle(data: Any, filepath: Path, log_prefix: str = "Data") -> bool:
    """
    Saves data to a pickle file.

    Args:
        data (Any): The data to save.
        filepath (Path): Path to save the file.
        log_prefix (str): Prefix for logging messages.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("%s saved to %s", log_prefix, filepath.name)
        return True
    except Exception as e:
        logger.error("Failed save pickle %s: %s", filepath.name, e, exc_info=True)
        try:
            filepath.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def save_hdf5(data: np.ndarray, filepath: Path, dataset_name: str = "data", attributes: Optional[Dict[str, Any]] = None, log_prefix: str = "Data") -> bool:
    """
    Saves a NumPy array to an HDF5 file with attributes.

    Args:
        data (np.ndarray): The NumPy array to save.
        filepath (Path): Path to save the HDF5 file.
        dataset_name (str): Name of the dataset to create within the HDF5 file.
        attributes (Optional[Dict[str, Any]]): Dictionary of attributes to save with the dataset.
        log_prefix (str): Prefix for logging messages.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(filepath, "w") as f:
            dset = f.create_dataset(dataset_name, data=data, compression="gzip", compression_opts=4, shuffle=True, fletcher32=True)
            if attributes:
                serializable_attrs = json.loads(json.dumps(attributes, cls=NpEncoder))
                for k, v in serializable_attrs.items():
                    try:
                        dset.attrs[k] = v
                    except TypeError as te:
                        logger.warning("HDF5 Attr Warn %s: %s. String fallback.", k, te)
                        dset.attrs[k] = str(v)
        logger.info("%s saved to HDF5 file %s (Dataset: '%s')", log_prefix, filepath.name, dataset_name)
        return True
    except Exception as e:
        logger.error("Failed save HDF5 %s: %s", filepath.name, e, exc_info=True)
        try:
            filepath.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def save_json_results(results: Dict[str, Any], filepath: Path, log_prefix: str = "Benchmark item") -> bool:
    """
    Saves dictionary results to a JSON file.

    Args:
        results (Dict[str, Any]): The dictionary to save.
        filepath (Path): Path to save the JSON file.
        log_prefix (str): Prefix for logging messages.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, cls=NpEncoder)
        logger.info("Saved %s JSON results to %s", log_prefix.lower(), filepath.name)
        return True
    except Exception as e:
        logger.error("Failed save JSON %s: %s", filepath.name, e, exc_info=True)
        try:
            filepath.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def save_stacked_features(features: np.ndarray, metadata: Dict[str, Any], cache_dir: Path, dataset_cache_id: str, feature_config: Dict[str, Any]) -> bool:
    """
    Saves stacked features and metadata to HDF5.

    Args:
        features (np.ndarray): Stacked features array.
        metadata (Dict[str, Any]): Metadata associated with the features.
        cache_dir (Path): Base caching directory.
        dataset_cache_id (str): Identifier for the dataset.
        feature_config (Dict[str, Any]): Configuration of the feature extractor.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    cache_path = get_cache_path(cache_dir, "features", dataset_cache_id, feature_config)
    log_prefix = f"Stacked features ({feature_config.get('name', '?')})"
    metadata.setdefault("num_items_processed", features.shape[0])
    metadata.setdefault("feature_ndim", features.ndim - 1 if features.ndim > 1 else features.ndim)
    metadata.setdefault("target_padded_shape", list(features.shape[1:]) if features.ndim > 1 else [])
    indices = metadata.pop(HDF5_INDICES_NAME, None)
    success = save_hdf5(features, cache_path, dataset_name=HDF5_DATASET_NAME, attributes=metadata, log_prefix=log_prefix)
    if success and indices is not None:
        try:
            with h5py.File(cache_path, "a") as f:
                indices_array = np.array(indices, dtype=np.int64)
                f.create_dataset(HDF5_INDICES_NAME, data=indices_array, compression="gzip", compression_opts=4)
            logger.debug("Appended '%s' dataset to %s", HDF5_INDICES_NAME, cache_path.name)
        except Exception as e:
            logger.error("Failed to append indices dataset to %s: %s", cache_path.name, e)
    return success



def load_stacked_features(filepath: Path) -> Optional[np.ndarray]:
    """
    Loads stacked features from HDF5 using the core load_hdf5 function.

    Args:
        filepath (Path): Path to the HDF5 file.

    Returns:
        Optional[np.ndarray]: The loaded features array, or None if loading fails.
    """
    return load_hdf5(filepath=filepath, dataset_name=HDF5_DATASET_NAME, log_prefix=f"Stacked features ({filepath.name})")


def save_pca_model(pca_model: Union[PCA, IncrementalPCA], filepath: Path) -> bool:
    """
    Saves a PCA model (standard or incremental) to a Pickle file.

    Args:
        pca_model (Union[PCA, IncrementalPCA]): The PCA model to save.
        filepath (Path): Path to save the file.

    Returns:
        bool: True if successful, False otherwise.
    """
    return save_pickle(pca_model, filepath, log_prefix=f"PCA model ({filepath.name})")


def load_pca_model(filepath: Path) -> Optional[Union[PCA, IncrementalPCA]]:
    """
    Loads a PCA model (standard or incremental) using the core load_pickle function.

    Args:
        filepath (Path): Path to the Pickle file.

    Returns:
        Optional[Union[PCA, IncrementalPCA]]: The loaded PCA model, or None if loading fails or type is invalid.
    """
    log_prefix = f"PCA model ({filepath.name})"
    model = load_pickle(filepath=filepath, log_prefix=log_prefix)
    if model is not None and not isinstance(model, (PCA, IncrementalPCA)):
        logger.warning("Loaded PCA model has invalid type (%s). Discarding.", type(model))
        try:
            filepath.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    return model


def save_distance_matrix(matrix: np.ndarray, filepath: Path, feature_name: str, distance_name: str) -> bool:
    """
    Saves a distance matrix to an HDF5 file.

    Args:
        matrix (np.ndarray): The distance matrix array.
        filepath (Path): Path to save the HDF5 file.
        feature_name (str): Name of the feature used to compute distances.
        distance_name (str): Name of the distance metric.

    Returns:
        bool: True if successful, False otherwise.
    """
    log_prefix = f"Distance matrix ({distance_name} for {feature_name})"
    attributes = {"feature_name": feature_name, "distance_name": distance_name}
    return save_hdf5(matrix, filepath, dataset_name="distance_matrix", attributes=attributes, log_prefix=log_prefix)


def load_distance_matrix(filepath: Path) -> Optional[np.ndarray]:
    """
    Loads a distance matrix from HDF5 using the core load_hdf5 function.

    Args:
        filepath (Path): Path to the HDF5 file.

    Returns:
        Optional[np.ndarray]: The loaded distance matrix array, or None if loading fails.
    """
    return load_hdf5(filepath=filepath, dataset_name="distance_matrix", log_prefix=f"Distance matrix ({filepath.name})")


def load_benchmark_item_results(filepath: Path) -> Optional[Dict]:
    """
    Loads individual benchmark item JSON results using core load_json_results.

    Args:
        filepath (Path): Path to the JSON file.

    Returns:
        Optional[Dict]: The loaded results dictionary, or None if loading fails.
    """
    return load_json_results(filepath=filepath, log_prefix=f"Benchmark item result ({filepath.name})")


def save_results(results: Dict[str, Any], output_dir: Union[str, Path], filename_prefix: str):
    """
    Saves final benchmark results to JSON and a summary CSV.

    Args:
        results (Dict[str, Any]): The dictionary containing all benchmark results.
        output_dir (Union[str, Path]): Directory to save the files.
        filename_prefix (str): Prefix for the output filenames. Expected to contain dataset/run identifiers.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dataset_id_str = "unknown_dataset"
    run_id_suffix = "unknown_run"
    if filename_prefix:
        parts = filename_prefix.split("_")
        if len(parts) >= 3:
            base_dataset_id_str = parts[1]
            run_id_suffix = "_".join(parts[2:])
        elif len(parts) == 2:
            base_dataset_id_str = parts[0]
            run_id_suffix = parts[1]
        else:
            run_id_suffix = filename_prefix

    json_path = output_dir / f"bench_results_{base_dataset_id_str}_{run_id_suffix}.json"
    csv_path = output_dir / f"bench_results_summary_{base_dataset_id_str}_{run_id_suffix}.csv"

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, cls=NpEncoder)
        logger.info("Final results saved to %s", json_path.name)
    except Exception as e:
        logger.error("Failed save results JSON: %s", e, exc_info=True)

    try:
        all_records = []
        is_multi_subset = False
        if results and isinstance(results, dict):
            first_val = next(iter(results.values()), None)
            if isinstance(first_val, dict) and first_val and isinstance(next(iter(first_val.values()), None), dict):
                is_multi_subset = True

        subset_results = results if is_multi_subset else {base_dataset_id_str: results}

        for subset_key, subset_data in subset_results.items():
            if not isinstance(subset_data, dict):
                continue
            for feature_key, feature_data in subset_data.items():
                if not isinstance(feature_data, dict):
                    if feature_key == "error":
                        record = {"subset": subset_key, "feature": feature_key, "metric_type": "error", "distance": None, "benchmark": "Extraction/Processing", "error": metric_data}
                        all_records.append(record)
                    continue

                for metric_type_key, metric_data in feature_data.items():
                    if not isinstance(metric_data, dict):
                        if metric_type_key == "error":
                            record = {"subset": subset_key, "feature": feature_key, "metric_type": metric_type_key, "distance": None, "benchmark": None, "error": str(metric_data)}
                            all_records.append(record)
                        continue

                    base_info = {"subset": subset_key, "feature": feature_key, "metric_type": metric_type_key}
                    if metric_type_key == "distance_based":
                        for distance_key, bench_data in metric_data.items():
                            if not isinstance(bench_data, dict):
                                continue
                            for benchmark_key, scores in bench_data.items():
                                record = base_info.copy()
                                record["distance"] = distance_key
                                record["benchmark"] = benchmark_key
                                if isinstance(scores, dict):
                                    record.update({k: v for k, v in scores.items() if isinstance(v, (str, int, float, bool, type(None)))})
                                    record.update({k: f"{v[0]:.4f} - {v[1]:.4f}" if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (float, int, np.number)) for x in v) else str(v) for k, v in scores.items() if isinstance(v, (list, tuple, set))})
                                else:
                                    record["value"] = scores
                                all_records.append(record)
                    elif metric_type_key == "feature_based":
                        for benchmark_key, scores in metric_data.items():
                            record = base_info.copy()
                            record["distance"] = None
                            record["benchmark"] = benchmark_key
                            if isinstance(scores, dict):
                                record.update({k: v for k, v in scores.items() if isinstance(v, (str, int, float, bool, type(None)))})
                                record.update({k: f"{v[0]:.4f} - {v[1]:.4f}" if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (float, int, np.number)) for x in v) else str(v) for k, v in scores.items() if isinstance(v, (list, tuple, set))})
                            else:
                                record["value"] = scores
                            all_records.append(record)
                    elif metric_type_key == "error":
                        record = base_info.copy()
                        record["distance"] = metric_data.get("distance")
                        record["benchmark"] = metric_data.get("benchmark", "Unknown")
                        record["error"] = metric_data.get("error", str(metric_data))
                        all_records.append(record)

        if all_records:
            df = pd.DataFrame(all_records)
            cols_order = ["subset", "feature", "metric_type", "distance", "benchmark"]
            existing_cols_ordered = [c for c in cols_order if c in df.columns]
            other_cols = sorted([c for c in df.columns if c not in existing_cols_ordered])
            if "value" in other_cols and len(other_cols) > 1:
                other_cols.remove("value")
                other_cols.append("value")
            if "error" in other_cols:
                other_cols.remove("error")
                other_cols.append("error")
            final_cols = existing_cols_ordered + other_cols
            try:
                df = df[final_cols]
            except KeyError as e:
                logger.warning("Col reorder fail CSV: %s. Cols: %s", e, list(df.columns))
            df.to_csv(csv_path, index=False, encoding="utf-8")
            logger.info("Summary results saved to %s", csv_path.name)
        else:
            logger.debug("No records for CSV summary.")
    except Exception as e:
        logger.warning("Failed create/save summary CSV: %s", e, exc_info=True)


def read_hdf5_slice(filepath: Path, start_idx: int, end_idx: int, dataset_name: str = "data") -> Optional[np.ndarray]:
    """
    Reads a slice of a dataset from an HDF5 file.

    Args:
        filepath (Path): Path to the HDF5 file.
        start_idx (int): Starting index of the slice (inclusive).
        end_idx (int): Ending index of the slice (exclusive).
        dataset_name (str): Name of the dataset within the HDF5 file.

    Returns:
        Optional[np.ndarray]: The requested slice of data, or None if loading fails.
                             Returns an empty array if start_idx >= end_idx or slice is out of bounds.
    """
    if not filepath.is_file():
        logger.error("HDF5 slice file not found: %s", filepath.name)
        return None
    try:
        with h5py.File(filepath, "r") as f:
            if dataset_name not in f:
                logger.error("Dataset '%s' not found in %s.", dataset_name, filepath.name)
                return None
            dset = f[dataset_name]
            actual_start_idx = max(0, start_idx)
            actual_end_idx = min(end_idx, dset.shape[0])
            if actual_start_idx >= actual_end_idx:
                return np.array([], dtype=dset.dtype)
            data_slice = dset[actual_start_idx:actual_end_idx]
            return data_slice
    except OSError as e:
        logger.error("Error reading slice HDF5 %s: %s", filepath.name, e)
        return None
    except Exception as e:
        logger.error("Failed read slice HDF5 %s: %s", filepath.name, e, exc_info=True)
        return None