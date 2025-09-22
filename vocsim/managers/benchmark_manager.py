from collections import Counter
import gc
import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import h5py
from tqdm import tqdm
from benchmarks.base import Benchmark
from benchmarks.classification import ClassificationBenchmark
from benchmarks.clustering import ClusteringPurity
from benchmarks.cscf import CSCFBenchmark
from benchmarks.f_value import FValueBenchmark
from benchmarks.perceptual import PerceptualAlignment
from benchmarks.precision import PrecisionAtK
from benchmarks.csr import ClassSeparationRatio
from benchmarks.gsr import GlobalSeparationRate  
from benchmarks.silhouette import SilhouetteBenchmark
from utils.file_utils import get_cache_path, load_distance_matrix, save_json_results, load_benchmark_item_results


logger = logging.getLogger(__name__)

HDF5_FEATURE_DATASET_NAME = "features"
HDF5_DISTANCE_DATASET_NAME = "distance_matrix"


class BenchmarkManager:
    """
    Orchestrates benchmark execution and caching for a dataset subset.
    """

    def __init__(self, config: Dict[str, Any], device: torch.device, root_dir: Path):
        """
        Initializes the BenchmarkManager.

        Args:
            config (Dict[str, Any]): The configuration dictionary.
            device (torch.device): The device to use for benchmark computation.
            root_dir (Path): The project root directory.
        """
        self.cfg = config
        self.device = device
        self.root_dir = root_dir
        self.benchmark_configs = config.get("benchmarks", [])
        self.raw_feature_configs_map = {fc["name"]: fc for fc in config.get("feature_extractors", []) if "name" in fc}
        self.raw_distance_configs_map = {dc["name"]: dc for dc in config.get("distances", []) if "name" in dc}
        self.base_results_dir = Path(config.get("results_dir", root_dir / "results")).resolve()
        logger.info("BenchmarkManager initialized.")

    def _get_class_from_module(self, module_name: str, class_name: str) -> Type:
        """
        Dynamically imports a class from a module path.

        Args:
            module_name (str): The dotted module path (e.g., 'benchmarks.precision').
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

    def _get_benchmark(self, config: Dict[str, Any]) -> Benchmark:
        """
        Instantiates a benchmark class from its configuration.

        Args:
            config (Dict[str, Any]): The configuration dictionary for the benchmark.

        Returns:
            Benchmark: An instance of the benchmark class.

        Raises:
            ValueError: If the configuration is invalid or instantiation fails.
        """
        name = config.get("name")
        if not name:
            raise ValueError("Benchmark configuration requires a 'name' field.")
        params = config.get("params", {})
        module_path = config.get("module")
        if not module_path:
            module_name_base = name.lower().replace("benchmark", "")
            module_mappings = {"fvalue": "f_value", "precisionatk": "precision", "clusteringpurity": "clustering", "cscf": "cscf", "classseparationratio": "csr", "perceptualalignment": "perceptual", "classification": "classification"}
            module_name_base = module_mappings.get(module_name_base, module_name_base)
            module_path = f"benchmarks.{module_name_base}"
            logger.debug("Inferred module path for %s: %s", name, module_path)
        try:
            target_class = self._get_class_from_module(module_path, name)
            instance = target_class(**params)
            logger.debug("Instantiated benchmark '%s'.", name)
            return instance
        except Exception as e:
            logger.error("Failed instantiate benchmark '%s' from '%s': %s", name, module_path, e, exc_info=True)
            raise ValueError(f"Failed init benchmark '{name}': {e}") from e

    def _prepare_evaluate_kwargs(
        self, base_data: Dict, item_id_map: Optional[Dict], benchmark_config: Optional[Dict] = None
    ) -> Tuple[Dict, bool]:
        """
        Prepares keyword arguments for benchmark.evaluate(), including labels derived from item_id_map if necessary.

        Args:
            base_data (Dict): Base dictionary of evaluation arguments.
            item_id_map (Optional[Dict]): Map from original item IDs to metadata/indices.
            benchmark_config (Optional[Dict]): The specific configuration for the benchmark being prepared for.

        Returns:
            Tuple[Dict, bool]: A tuple containing:
                - Dict: The prepared keyword arguments dictionary.
                - bool: True if labels were successfully prepared or already present, False otherwise.
        """
        evaluate_kwargs = base_data.copy()
        labels_prepared_successfully = True

        if "item_id_map" not in evaluate_kwargs and item_id_map:
            evaluate_kwargs["item_id_map"] = item_id_map
        elif "item_id_map" not in evaluate_kwargs:
            logger.warning("Item ID map is missing and not provided.")

        if "labels" not in evaluate_kwargs and evaluate_kwargs.get("item_id_map"):
            item_map = evaluate_kwargs["item_id_map"]
            if isinstance(item_map, dict):
                logger.debug("Auto-preparing labels list from item_id_map...")
                indices = [v.get("index") for v in item_map.values() if isinstance(v.get("index"), int)]
                if not indices:
                    logger.warning("No valid integer indices found in item_id_map.")
                    labels_prepared_successfully = False
                else:
                    num_items = max(indices) + 1
                    ordered_labels = [None] * num_items
                    labels_found_count = 0

                    label_key_to_use = "label"
                    if benchmark_config:
                        label_key_from_bench = benchmark_config.get("params", {}).get("label_source_key")
                        if label_key_from_bench:
                            label_key_to_use = label_key_from_bench
                            logger.info("Using benchmark-specified label_source_key: '%s'", label_key_to_use)
                        else:
                            if any("label" in v for v in item_map.values() if isinstance(v, dict)):
                                label_key_to_use = "label"
                                logger.debug("Using detected label key: '%s' (benchmark default)", label_key_to_use)
                            elif any("speaker" in v for v in item_map.values() if isinstance(v, dict)):
                                label_key_to_use = "speaker"
                                logger.debug("Using detected label key: '%s' (benchmark default, 'label' not found)", label_key_to_use)
                            else:
                                logger.warning("Neither benchmark key, 'label', nor 'speaker' found. Defaulting to '%s'.", label_key_to_use)
                    else:
                        if any("label" in v for v in item_map.values() if isinstance(v, dict)):
                            label_key_to_use = "label"
                        elif any("speaker" in v for v in item_map.values() if isinstance(v, dict)):
                            label_key_to_use = "speaker"
                        logger.warning("Benchmark config not passed to label prep. Guessed label key: '%s'.", label_key_to_use)

                    num_skipped_index_bounds = 0
                    num_skipped_missing_key = 0
                    num_skipped_none_value = 0
                    for item_key, item_data in item_map.items():
                        idx = item_data.get("index")
                        label_val = item_data.get(label_key_to_use)

                        if not isinstance(idx, int):
                            continue

                        if 0 <= idx < num_items:
                            if label_key_to_use not in item_data:
                                num_skipped_missing_key += 1
                                ordered_labels[idx] = None
                            elif label_val is not None:
                                ordered_labels[idx] = str(label_val)
                                labels_found_count += 1
                            else:
                                num_skipped_none_value += 1
                                ordered_labels[idx] = None
                        else:
                            num_skipped_index_bounds += 1

                    if num_skipped_index_bounds > 0:
                        logger.warning("Skipped %d items due to out-of-bounds index during label prep.", num_skipped_index_bounds)
                    if num_skipped_missing_key > 0:
                        logger.warning("Label key '%s' was missing for %d items.", label_key_to_use, num_skipped_missing_key)
                    if num_skipped_none_value > 0:
                        logger.warning("Label key '%s' had None value for %d items.", label_key_to_use, num_skipped_none_value)

                    if labels_found_count > 0:
                        evaluate_kwargs["labels"] = ordered_labels
                        if labels_found_count < len(item_map):
                            logger.debug("Label coverage partial (%d/%d items).", labels_found_count, len(item_map))
                    else:
                        logger.error("No valid labels found in item_id_map using key '%s'.", label_key_to_use)
                        labels_prepared_successfully = False
            else:
                logger.error("Cannot auto-prepare labels: item_id_map missing or invalid type.")
                labels_prepared_successfully = False
        elif "labels" in evaluate_kwargs:
            logger.debug("Using pre-provided 'labels' list.")
            if not isinstance(evaluate_kwargs["labels"], list):
                try:
                    evaluate_kwargs["labels"] = list(evaluate_kwargs["labels"])
                except TypeError:
                    logger.error("Provided labels not list-convertible (%s).", type(evaluate_kwargs["labels"]))
                    labels_prepared_successfully = False
            if labels_prepared_successfully:
                evaluate_kwargs["labels"] = [(str(label) if label is not None else None) for label in evaluate_kwargs["labels"]]
                logger.debug("Ensured pre-provided labels are list/str/None (len: %d)", len(evaluate_kwargs["labels"]))

        return evaluate_kwargs, labels_prepared_successfully

    def run_subset_benchmarks(
        self,
        subset_features_dir: Path,
        subset_dataset_obj: Any,
        item_id_map: Dict[str, Dict],
        dataset_cache_id: str,
        current_subset_key: str,
        subset_cache_dir: Path,
        feature_paths: Dict[str, Path],
        distance_paths: Dict[Tuple[str, str], Path],
    ) -> Dict[str, Any]:
        """
        Runs benchmarks using available feature and distance paths.

        Args:
            subset_features_dir (Path): Directory containing cached feature files for this subset.
            subset_dataset_obj (Any): The actual dataset subset object (e.g., Hugging Face Dataset).
            item_id_map (Dict[str, Dict]): Map from original item IDs to metadata/indices.
            dataset_cache_id (str): Unique identifier for this dataset subset/split.
            current_subset_key (str): Name of the current subset.
            subset_cache_dir (Path): Directory for storing benchmark item cache files for this subset.
            feature_paths (Dict[str, Path]): Dictionary mapping feature names to paths of cached feature files.
            distance_paths (Dict[Tuple[str, str], Path]): Dictionary mapping (feature_name, distance_name) tuples to paths of cached distance matrix files.

        Returns:
            Dict[str, Any]: A nested dictionary containing benchmark results for the subset.
        """
        logger.info("--- Starting Benchmarking for Subset: %s ---", current_subset_key)
        logger.debug("Received distance_paths dict:\n%s", distance_paths)

        subset_results: Dict[str, Dict[str, Any]] = {}
        base_evaluate_data: Dict[str, Any] = {}
        if subset_dataset_obj is not None:
            base_evaluate_data["dataset"] = subset_dataset_obj
        if not item_id_map:
            logger.warning("Item ID map missing for subset %s, benchmarks needing labels may fail.", current_subset_key)

        feature_iter = tqdm(feature_paths.items(), desc=f"Benching Features [{current_subset_key}]", leave=False)
        for feature_name, feature_hdf5_path in feature_iter:
            if not feature_hdf5_path or not feature_hdf5_path.exists():
                logger.warning("Feature path for '%s' (%s) is invalid. Skipping benchmarks.", feature_name, feature_hdf5_path)
                subset_results.setdefault(feature_name, {})["error"] = f"Input feature file missing: {feature_hdf5_path}"
                continue

            feature_iter.set_postfix_str(f"{feature_name}")
            raw_feature_conf = self.raw_feature_configs_map.get(feature_name)
            if not raw_feature_conf:
                logger.error("Config missing for available feature '%s'. Skipping benchmarks.", feature_name)
                subset_results.setdefault(feature_name, {})["error"] = f"Config missing for feature {feature_name}"
                continue
            effective_feature_config = raw_feature_conf

            logger.info("-- Benchmarking Feature: %s (using file %s) --", feature_name, feature_hdf5_path.name)
            subset_results.setdefault(feature_name, {})

            for bench_conf in self.benchmark_configs:
                benchmark_name = bench_conf.get("name")
                if not benchmark_name:
                    continue

                target_features = bench_conf.get("target_features")
                if target_features and isinstance(target_features, list) and feature_name not in target_features:
                    continue

                try:
                    benchmark = self._get_benchmark(bench_conf)
                except Exception as e:
                    logger.error("Skip bench %s: Init fail: %s", benchmark_name, e)
                    continue

                needs_features = isinstance(benchmark, ClassificationBenchmark) or \
               (isinstance(benchmark, ClusteringPurity) and not getattr(benchmark, "use_dist_matrix", False)) or \
               (isinstance(benchmark, SilhouetteBenchmark) and not getattr(benchmark, "use_distance_matrix", False))                
                needs_distances = isinstance(benchmark, (PerceptualAlignment, PrecisionAtK, FValueBenchmark, CSCFBenchmark, ClassSeparationRatio, GlobalSeparationRate)) or \
                (isinstance(benchmark, ClusteringPurity) and getattr(benchmark, "use_dist_matrix", True)) or \
                (isinstance(benchmark, SilhouetteBenchmark) and getattr(benchmark, "use_distance_matrix", True))
                if needs_distances:
                    metric_key = "distance_based"
                    subset_results[feature_name].setdefault(metric_key, {})
                    target_distances = bench_conf.get("target_distances")

                    processed_any_distance = False
                    for dist_name, dist_conf in self.raw_distance_configs_map.items():
                        if target_distances and isinstance(target_distances, list) and dist_name not in target_distances:
                            continue

                        matrix_key = (feature_name, dist_name)
                        logger.debug("Checking distance_paths for key: %s (Type: %s, %s)", matrix_key, type(matrix_key[0]), type(matrix_key[1]))

                        dist_matrix_path = distance_paths.get(matrix_key)

                        if dist_matrix_path is None:
                            found_variation = False
                            for k_feat, k_dist in distance_paths.keys():
                                if k_feat.lower() == feature_name.lower() and k_dist.lower() == dist_name.lower():
                                    logger.warning("Key %s NOT FOUND, but found case variation: (%s, %s)", matrix_key, k_feat, k_dist)
                                    found_variation = True
                                    break
                            if not found_variation:
                                logger.debug("Key %s NOT FOUND in distance_paths.", matrix_key)
                        else:
                            logger.debug("Key %s FOUND in distance_paths: %s", matrix_key, dist_matrix_path)

                        if not dist_matrix_path:
                            feature_expects_dist = raw_feature_conf.get("compute_distances_for") is None or dist_name in raw_feature_conf.get("compute_distances_for", [])

                            if feature_expects_dist:
                                logger.warning("Distance matrix path missing for %s. Cannot run '%s'.", matrix_key, benchmark_name)
                            else:
                                logger.debug("Skipping distance '%s' for feature '%s' as not specified in 'compute_distances_for'.", dist_name, feature_name)
                                continue

                            subset_results[feature_name][metric_key].setdefault(dist_name, {})[benchmark_name] = {"error": f"Required distance matrix for {dist_name} unavailable."}
                            continue
                        if not dist_matrix_path.exists():
                            logger.error("Distance matrix file missing at expected path %s. Cannot run '%s'.", dist_matrix_path, benchmark_name)
                            subset_results[feature_name][metric_key].setdefault(dist_name, {})[benchmark_name] = {"error": f"Distance matrix file missing: {dist_matrix_path.name}"}
                            continue

                        processed_any_distance = True
                        logger.debug("Prep D-Bench '%s' | Feat: %s | Dist: %s (using %s)", benchmark_name, feature_name, dist_name, dist_matrix_path.name)
                        combined_config_bench = {"feature_config": effective_feature_config, "distance_config": dist_conf, "benchmark_config": bench_conf, "name": f"{feature_name}_{dist_name}_{benchmark_name}"}

                        bench_cache_path = get_cache_path(subset_cache_dir, "bench_item", dataset_cache_id, combined_config_bench)
                        bench_res = load_benchmark_item_results(bench_cache_path)

                        if bench_res is None:
                            logger.info("Running D-Bench '%s' | Feat: %s | Dist: %s", benchmark_name, feature_name, dist_name)
                            loaded_dist_matrix = load_distance_matrix(dist_matrix_path)

                            if loaded_dist_matrix is None:
                                logger.error("Failed load dist matrix %s for %s", dist_matrix_path.name, benchmark_name)
                                bench_res = {"error": f"Dist matrix load fail: {dist_matrix_path.name}"}
                            else:
                                evaluate_kwargs, labels_ok = self._prepare_evaluate_kwargs(base_evaluate_data, item_id_map, bench_conf)
                                evaluate_kwargs["distance_matrix"] = loaded_dist_matrix
                                evaluate_kwargs["distance_matrix_path"] = dist_matrix_path
                                evaluate_kwargs["feature_config"] = effective_feature_config

                                requires_labels_or_map = any(isinstance(benchmark, cls) for cls in [PerceptualAlignment, PrecisionAtK, FValueBenchmark, CSCFBenchmark, ClassSeparationRatio]) or (isinstance(benchmark, ClusteringPurity) and getattr(benchmark, "use_dist_matrix", True))

                                run_this_benchmark = True
                                if requires_labels_or_map and not labels_ok:
                                    if not evaluate_kwargs.get("item_id_map"):
                                        logger.error("[%s] Skip %s/%s: Label prep failed and item_id_map missing.", benchmark_name, feature_name, dist_name)
                                        bench_res = {"error": "Label preparation failed and item_id_map missing."}
                                        run_this_benchmark = False
                                    else:
                                        logger.warning("[%s] Label prep failed for %s/%s, but item_id_map is present. Proceeding, benchmark must handle map.", benchmark_name, feature_name, dist_name)

                                if run_this_benchmark:
                                    try:
                                        results = benchmark.evaluate(**evaluate_kwargs)
                                        bench_res = results
                                    except Exception as e:
                                        logger.error("Error D-Bench %s (%s/%s): %s", benchmark_name, feature_name, dist_name, e, exc_info=True)
                                        bench_res = {"error": str(e)}

                                del loaded_dist_matrix
                                gc.collect()
                                if self.device.type == "cuda":
                                    torch.cuda.empty_cache()

                            save_json_results(bench_res, bench_cache_path, f"{benchmark_name} result")

                        subset_results[feature_name][metric_key].setdefault(dist_name, {})[benchmark_name] = bench_res

                    if not processed_any_distance:
                        logger.debug("No applicable/available distance matrices found for D-Bench '%s' on feature '%s'.", benchmark_name, feature_name)

                if needs_features:
                    metric_key = "feature_based"
                    subset_results[feature_name].setdefault(metric_key, {})
                    logger.debug("Prep F-Bench '%s' | Feat: %s (using %s)", benchmark_name, feature_name, feature_hdf5_path.name)
                    combined_config_bench = {"feature_config": effective_feature_config, "benchmark_config": bench_conf, "name": f"{feature_name}_{benchmark_name}"}

                    bench_cache_path = get_cache_path(subset_cache_dir, "bench_item", dataset_cache_id, combined_config_bench)
                    bench_res = load_benchmark_item_results(bench_cache_path)

                    if bench_res is None:
                        logger.info("Running F-Bench '%s' | Feature: %s", benchmark_name, feature_name)
                        evaluate_kwargs, labels_ok = self._prepare_evaluate_kwargs(base_evaluate_data, item_id_map, bench_conf)
                        evaluate_kwargs["feature_hdf5_path"] = feature_hdf5_path
                        evaluate_kwargs["feature_config"] = effective_feature_config

                        requires_labels = isinstance(benchmark, ClassificationBenchmark) or (isinstance(benchmark, ClusteringPurity) and not getattr(benchmark, "use_dist_matrix", False))

                        run_this_benchmark = True
                        if requires_labels and not labels_ok:
                            if not evaluate_kwargs.get("item_id_map"):
                                logger.error("[%s] Skip %s: Label prep failed and item_id_map missing.", benchmark_name, feature_name)
                                bench_res = {"error": "Label prep failed and item_id_map missing."}
                                run_this_benchmark = False
                            else:
                                logger.warning("[%s] Label prep failed for %s, but item_id_map is present. Proceeding, benchmark must handle map.", benchmark_name, feature_name)

                        if run_this_benchmark:
                            try:
                                results = benchmark.evaluate(**evaluate_kwargs)
                                bench_res = results
                            except FileNotFoundError:
                                logger.error("Feature file %s missing during F-Bench %s execution.", feature_hdf5_path.name, benchmark_name)
                                bench_res = {"error": f"Feature file missing: {feature_hdf5_path.name}"}
                            except Exception as e:
                                logger.error("Error F-Bench %s (%s): %s", benchmark_name, feature_name, e, exc_info=True)
                                bench_res = {"error": str(e)}
                            finally:
                                gc.collect()
                                if self.device.type == "cuda":
                                    torch.cuda.empty_cache()

                        save_json_results(bench_res, bench_cache_path, f"{benchmark_name} result")

                    subset_results[feature_name][metric_key][benchmark_name] = bench_res

                if "benchmark" in locals():
                    del benchmark
                gc.collect()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("--- Benchmarking Completed for Subset: %s ---", current_subset_key)
        return subset_results