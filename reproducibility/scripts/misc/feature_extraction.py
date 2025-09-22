import datetime
import gc
import logging
import sys
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[3]
    expected_vocsim_dir = project_root / "vocsim"
    if not expected_vocsim_dir.is_dir() and not (project_root / "vocsim" / "runner.py").exists():
        project_root = Path.cwd()
        print(f"WARNING: Assuming CWD project root: {project_root}")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"INFO: Added project root: {project_root}")
except NameError:
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    print(f"INFO: Assuming CWD project root: {project_root}")

from vocsim.managers.dataset_manager import DatasetManager
from vocsim.managers.feature_manager import FeatureManager
from utils.config_loader import load_config
from utils.feature_utils import apply_averaging
from utils.logging_utils import setup_logging
from utils.torch_utils import get_device

TARGET_SUBSET_KEY = "BS1"
CONFIG_NAME = "vocsim_paper.yaml"
BASE_CONFIG_NAME = "base.yaml"
CONFIG_DIR = project_root / "reproducibility" / "configs"
BASE_CONFIG_DIR = project_root / "configs"
OUTPUT_DIR_BASE = project_root / "reproducibility_outputs" / "sample_features"

log_dir = project_root / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
setup_logging(
    {
        "level": "INFO",
        "log_dir": str(log_dir),
        "log_file": "extract_FIRST_sample.log",
    }
)
logger = logging.getLogger(__name__)


def save_npy_feature(feature_data: Optional[np.ndarray], filename: str, output_dir: Path):
    """
    Saves a numpy array to a .npy file, handles None or empty input.

    Args:
        feature_data (Optional[np.ndarray]): The numpy array containing the feature data.
        filename (str): The base filename (without extension).
        output_dir (Path): The directory where the file should be saved.

    Returns:
        bool: True if the save was successful, False otherwise.
    """
    output_path = output_dir / f"{filename}.npy"
    if feature_data is None:
        logger.warning(f"Feature data is None for: {filename}. Skipping save.")
        return False
    if feature_data.size == 0:
        logger.warning(f"Feature data is empty for: {filename}. Skipping save.")
        return False

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_path, feature_data)
        logger.info(f"Saved feature: {output_path.name} (Shape: {feature_data.shape}, Dtype: {feature_data.dtype})")
        return True
    except Exception as e:
        logger.error(f"Failed to save feature '{filename}.npy': {e}", exc_info=True)
        return False


def main():
    """
    Main function to extract features from the first valid audio sample
    in a target dataset subset based on configuration.
    """
    logger.info("--- Starting Single Sample Feature Extraction Script (First Item) ---")

    config_path = CONFIG_DIR / CONFIG_NAME
    base_config_path = BASE_CONFIG_DIR / BASE_CONFIG_NAME
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)
    cfg = load_config(
        config_path,
        base_config_path=base_config_path if base_config_path.exists() else None,
    )
    logger.info("Loaded config from %s", config_path.name)

    device = get_device(cfg.get("force_cpu", False))

    dataset_manager = DatasetManager(cfg)
    if not dataset_manager.load_full_dataset():
        logger.error("Failed to load dataset.")
        sys.exit(1)

    selected_item_data = None
    selected_item_index = -1
    logger.info("Searching for the FIRST valid audio item in subset '%s'...", TARGET_SUBSET_KEY)
    try:
        if not hasattr(dataset_manager.full_dataset_obj, "__iter__"):
            raise TypeError("Dataset object is not iterable.")

        for idx, item in enumerate(dataset_manager.full_dataset_obj):
            if item is None:
                continue
            item_subset = item.get("subset")
            if item_subset != TARGET_SUBSET_KEY:
                continue

            audio_info = item.get("audio")
            if isinstance(audio_info, dict) and audio_info.get("array") is not None and "sampling_rate" in audio_info:
                audio_array_check = audio_info["array"]
                is_valid_audio = False
                if isinstance(audio_array_check, np.ndarray) and audio_array_check.size > 0:
                    is_valid_audio = True
                elif isinstance(audio_array_check, list) and len(audio_array_check) > 0:
                    is_valid_audio = True
                elif isinstance(audio_array_check, torch.Tensor) and audio_array_check.numel() > 0:
                    is_valid_audio = True

                if is_valid_audio:
                    selected_item_data = item
                    selected_item_index = idx
                    logger.info("Found first valid item at index %d.", selected_item_index)
                    break

        if selected_item_data is None:
            logger.error("Could not find any valid audio item in subset '%s'. Cannot proceed.", TARGET_SUBSET_KEY)
            sys.exit(1)

    except Exception as e:
        logger.error("Error while searching for the first item: %s", e, exc_info=True)
        sys.exit(1)

    try:
        item_identifier = f"item_idx_{selected_item_index}"
        for id_field in ["original_name", "id", "filename"]:
            if id_field in selected_item_data and selected_item_data[id_field]:
                safe_name = Path(str(selected_item_data[id_field])).stem
                safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in safe_name)
                item_identifier = f"{id_field}_{safe_name}"
                break

        audio_array = selected_item_data["audio"]["array"]
        sample_rate = selected_item_data["audio"]["sampling_rate"]

        logger.info("Processing selected item from subset '%s':", TARGET_SUBSET_KEY)
        logger.info("  Identifier: %s", item_identifier)
        logger.info("  Original Index: %d", selected_item_index)
        logger.info("  Audio Array Shape: %s", audio_array.shape if isinstance(audio_array, np.ndarray) else type(audio_array))
        logger.info("  Sample Rate: %d", sample_rate)
    except Exception as e:
        logger.error("Failed to process selected item data: %s", e, exc_info=True)
        sys.exit(1)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = OUTPUT_DIR_BASE / f"{TARGET_SUBSET_KEY}_{item_identifier}_{timestamp}"
    output_subdir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving extracted features to: %s", output_subdir)

    feature_manager = FeatureManager(cfg, Path("./dummy_feature_cache"), device)

    feature_configs = cfg.get("feature_extractors", [])
    logger.info("Found %d feature configurations to process.", len(feature_configs))

    extracted_base_features: Dict[str, Optional[np.ndarray]] = {}

    logger.info("\n--- PASS 1: Extracting and Saving BASE Features ---")
    for feature_config in feature_configs:
        feature_name = feature_config.get("name")
        if not feature_name:
            continue

        is_base = not feature_config.get("base_extractor")

        if is_base:
            logger.info("Processing BASE feature: %s", feature_name)
            effective_config = feature_manager._get_effective_feature_config(feature_config, TARGET_SUBSET_KEY)
            extractor = None
            base_features_np: Optional[np.ndarray] = None
            try:
                extractor = feature_manager._get_feature_extractor(effective_config)
                params = effective_config.get("params", {})
                logger.debug("Extracting with %s using params: %s", extractor.__class__.__name__, params)
                base_features_raw = extractor.extract(
                    audio_data=audio_array,
                    sample_rate=sample_rate,
                    **params,
                )

                if isinstance(base_features_raw, torch.Tensor):
                    base_features_np = base_features_raw.cpu().numpy()
                elif isinstance(base_features_raw, np.ndarray):
                    base_features_np = base_features_raw
                elif base_features_raw is None:
                    logger.warning("Extractor returned None.")
                    base_features_np = None
                else:
                    logger.warning("Extractor returned unexpected type: %s. Treating as failure.", type(base_features_raw))
                    base_features_np = None

                if base_features_np is not None:
                    if base_features_np.size == 0:
                        logger.warning("Extractor returned empty features.")
                        base_features_np = None
                    elif np.isnan(base_features_np).any() or np.isinf(base_features_np).any():
                        logger.error("Features contain NaN/Inf! Discarding.")
                        base_features_np = None

                if base_features_np is not None:
                    save_npy_feature(base_features_np, f"{feature_name}_BASE", output_subdir)
                else:
                    logger.error("Failed to get valid base features for %s.", feature_name)

                extracted_base_features[feature_name] = base_features_np

            except Exception as e:
                logger.error("Error processing BASE feature %s: %s", feature_name, e, exc_info=True)
                extracted_base_features[feature_name] = None
            finally:
                gc.collect()

    logger.info("\n--- PASS 2: Processing and Saving INTERMEDIATE Features (No PCA) ---")
    for feature_config in feature_configs:
        feature_name = feature_config.get("name")
        if not feature_name:
            continue

        base_extractor_name = feature_config.get("base_extractor")
        averaging_method = feature_config.get("averaging")
        has_pca = "pca" in feature_config

        if not base_extractor_name or has_pca or averaging_method is None:
            continue

        logger.info("Processing INTERMEDIATE feature: %s", feature_name)
        logger.info("  (Base: %s, Avg: %s)", base_extractor_name, averaging_method)

        base_features_np = extracted_base_features.get(base_extractor_name)

        if base_features_np is None:
            logger.error(
                "Base features for '%s' were not successfully extracted or are missing. Cannot generate intermediate feature '%s'.",
                base_extractor_name,
                feature_name,
            )
            continue

        intermediate_features_np: Optional[np.ndarray] = None
        try:
            logger.debug("Applying averaging '%s' to base features of shape %s", averaging_method, base_features_np.shape)
            intermediate_features_np = apply_averaging(base_features_np, averaging_method)

            if intermediate_features_np is not None:
                if np.isnan(intermediate_features_np).any() or np.isinf(intermediate_features_np).any():
                    logger.error("Intermediate features for %s contain NaN/Inf! Skipping save.", feature_name)
                elif intermediate_features_np.size == 0:
                    logger.warning("Intermediate features for %s are empty. Skipping save.", feature_name)
                else:
                    save_npy_feature(intermediate_features_np, f"{feature_name}_INTERMEDIATE", output_subdir)
            else:
                logger.error(
                    "Averaging method '%s' failed for base features of '%s' (shape: %s). Cannot generate intermediate feature '%s'.",
                    averaging_method,
                    base_extractor_name,
                    base_features_np.shape,
                    feature_name,
                )

        except Exception as e:
            logger.error("Error processing INTERMEDIATE feature %s: %s", feature_name, e, exc_info=True)
        finally:
            if intermediate_features_np is not None:
                del intermediate_features_np
            gc.collect()

    feature_manager._clear_extractor_cache()
    del extracted_base_features
    gc.collect()
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    logger.info("--- Single Sample Feature Extraction Script Finished ---")


if __name__ == "__main__":
    main()