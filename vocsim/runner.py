import datetime
import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Callable
import os


import torch
import pandas as pd
import numpy as np
import json


from vocsim.managers import (
    BenchmarkManager,
    DatasetManager,
    DistanceManager,
    FeatureManager,
    TrainerManager,
)

from utils.config_loader import load_config
from utils.file_utils import get_cache_path, find_cache_path, save_results, _get_safe_path_part
from utils.logging_utils import setup_logging
from utils.torch_utils import get_device

logger = logging.getLogger(__name__)


class NpEncoder(json.JSONEncoder):
    """Helper JSON Encoder class for NumPy types, Path, Tensors, etc."""

    def default(self, obj):
        """
        Encodes various object types into JSON-serializable formats.

        Args:
            obj: The object to encode.

        Returns:
            JSON-serializable representation of the object.
        """
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            if np.isnan(obj):
                return None
            if np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return [self.default(el) for el in obj.tolist()]
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, torch.Tensor):
            return self.default(obj.detach().cpu().numpy())
        elif isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            logger.warning("NpEncoder encountered unhandled type %s, converting to string.", type(obj))
            return str(obj)


class PipelineRunner:
    """
    Manages the execution of the VocSim benchmark pipeline based on specified steps.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the PipelineRunner with configuration and manager instances.

        Args:
            config: Configuration dictionary for the pipeline.
        """
        self.cfg = config
        project_root_str = config.get("project_root", ".")
        self.root_dir = Path(project_root_str).resolve()

        cfg_results_dir = config.get("results_dir")
        self.base_results_dir = Path(cfg_results_dir).resolve() if cfg_results_dir and Path(cfg_results_dir).is_absolute() else (self.root_dir / (cfg_results_dir or "results")).resolve()

        cfg_features_dir = config.get("features_dir")
        self.base_features_dir = Path(cfg_features_dir).resolve() if cfg_features_dir and Path(cfg_features_dir).is_absolute() else (self.root_dir / (cfg_features_dir or "features_cache")).resolve()

        cfg_models_dir = config.get("models_dir")
        self.base_models_dir = Path(cfg_models_dir).resolve() if cfg_models_dir and Path(cfg_models_dir).is_absolute() else (self.root_dir / (cfg_models_dir or "models")).resolve()

        self.device = get_device(config.get("force_cpu", False))
        self._setup_base_directories()

        self.dataset_manager = DatasetManager(config)
        self.feature_manager = FeatureManager(config, self.base_features_dir, self.device, self.base_models_dir)
        self.distance_manager = DistanceManager(config, self.base_features_dir, self.device)
        self.benchmark_manager = BenchmarkManager(config, self.device, self.root_dir)
        self.trainer_manager = TrainerManager(config, self.base_models_dir, self.device)
        logger.info("PipelineRunner initialized.")

    def _setup_base_directories(self) -> None:
        """
        Creates base directories for results, features, and models if they do not exist.
        """
        self.base_results_dir.mkdir(parents=True, exist_ok=True)
        self.base_features_dir.mkdir(parents=True, exist_ok=True)
        self.base_models_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Results directory: %s", self.base_results_dir)
        logger.info("Features cache directory: %s", self.base_features_dir)
        logger.info("Models directory: %s", self.base_models_dir)

    def _run_subset_loop(self, steps: List[str]) -> None:
        """
        Internal method to loop through dataset subsets and run selected evaluation steps.

        Args:
            steps (List[str]): The list of pipeline steps requested.
        """
        overall_results: Dict[str, Dict[str, Any]] = {}
        run_id_template = self.cfg.get("run_id", "run_${now:%Y%m%d_%H%M%S}")
        run_id = datetime.datetime.now().strftime(run_id_template.replace("${now:%Y%m%d_%H%M%S}", "%Y%m%d_%H%M%S"))
        logger.info("Using Run ID: %s", run_id)

        needs_eval = any(step in steps for step in ["features", "distances", "benchmarks"])
        if needs_eval and not self.dataset_manager.full_dataset_obj:
            if not self.dataset_manager.load_full_dataset():
                logger.error("Failed to load dataset for evaluation steps.")
                return
            logger.info("Full dataset loaded for evaluation steps.")

        subsets_to_run_config = self.cfg.get("dataset", {}).get("subsets_to_run")
        if subsets_to_run_config is None:
            top_level_subset = self.cfg.get("dataset", {}).get("subset")
            subsets_to_run = [top_level_subset] if top_level_subset else ["all"]
        elif isinstance(subsets_to_run_config, str):
            subsets_to_run = [subsets_to_run_config]
        elif isinstance(subsets_to_run_config, list):
            subsets_to_run = subsets_to_run_config
        else:
            logger.warning("Invalid 'subsets_to_run' type: %s. Defaulting to ['all'].", type(subsets_to_run_config))
            subsets_to_run = ["all"]

        log_subset_msg = "'all' (full dataset)" if subsets_to_run == ["all"] else f"subsets: {subsets_to_run}"
        logger.info("Processing %s for steps: %s", log_subset_msg, steps)

        for subset_key in subsets_to_run:
            start_time = time.time()
            logger.info("\n===== Processing Subset: %s =====", subset_key)

            subset_features_dir = self.base_features_dir / _get_safe_path_part(subset_key)
            subset_bench_cache_dir = subset_features_dir / "benchmark_cache"
            subset_results_dir = self.base_results_dir / _get_safe_path_part(subset_key)
            for directory in (subset_features_dir, subset_bench_cache_dir, subset_results_dir):
                directory.mkdir(parents=True, exist_ok=True)
            logger.debug("Dirs - Feat: %s, BenchCache: %s, Results: %s", subset_features_dir, subset_bench_cache_dir, subset_results_dir)

            subset_dataset: Optional[Any] = None
            dataset_cache_id: Optional[str] = None
            item_id_map: Dict[str, Dict] = {}
            if needs_eval:
                subset_info = self.dataset_manager.get_subset_dataset(subset_key)
                if subset_info is None:
                    logger.warning("Skipping subset '%s' due to dataset loading/filtering failure.", subset_key)
                    continue
                subset_dataset, dataset_cache_id = subset_info
                item_id_map = self.dataset_manager.get_current_item_map()
                if not item_id_map:
                    logger.warning("Empty item ID map for subset '%s'. Benchmarks requiring labels might fail.", subset_key)
            else:
                dataset_id_for_filename = _get_safe_path_part(Path(self.cfg.get("dataset", {}).get("id", "unknown_dataset")).name)
                split_name = self.cfg.get("dataset", {}).get("split", "train")
                dataset_cache_id = f"{dataset_id_for_filename}_{_get_safe_path_part(subset_key)}_{split_name}"
                logger.info("No evaluation steps requested. Constructed approximate dataset_cache_id: %s", dataset_cache_id)

            feature_paths: Dict[str, Path] = {}
            distance_paths: Dict[Tuple[str, str], Path] = {}
            subset_results: Dict[str, Any] = {}

            try:
                if "features" in steps:
                    logger.info("--- Running Feature Processing for Subset: %s ---", subset_key)
                    feature_paths = self.feature_manager.process_subset_features(
                        subset_dataset_obj=subset_dataset,
                        dataset_cache_id=dataset_cache_id,
                        current_subset_key=subset_key,
                        subset_features_dir=subset_features_dir,
                        item_id_map=item_id_map,
                        run_steps=steps,
                    )
                    if not feature_paths:
                        logger.warning("Feature processing yielded no valid paths for subset '%s'. Subsequent steps may fail if cache is missing.", subset_key)
                elif any(step in steps for step in ["distances", "benchmarks"]):
                    logger.info("--- Locating Existing Feature Paths Only for Subset: %s ---", subset_key)
                    for feat_conf in self.cfg.get("feature_extractors", []):
                        feat_name = feat_conf.get("name")
                        if not feat_name or not feat_conf.get("benchmark_this", True):
                            continue
                        expected_path = self.feature_manager.get_feature_file_path(feat_name, dataset_cache_id, subset_features_dir)
                        if expected_path and expected_path.is_file():
                            feature_paths[feat_name] = expected_path
                        else:
                            eff_conf = self.feature_manager._get_effective_feature_config(feat_conf, subset_key)
                            path_to_use = find_cache_path(
                                cache_dir=subset_features_dir, prefix="features", dataset_cache_id=dataset_cache_id, config_dict=eff_conf, extra_suffix=None
                            )
                            if path_to_use and path_to_use.is_file():
                                logger.debug("Found feature '%s' using find_cache_path: %s", feat_name, path_to_use.name)
                                feature_paths[feat_name] = path_to_use
                            else:
                                logger.debug("No cached feature file found (using find_cache_path or expected path) for '%s' when feature step was skipped.", feat_name)
                    if not feature_paths:
                        logger.warning("Feature step skipped and no cached feature files found. Subsequent steps will likely fail.")

                if "distances" in steps:
                    if not feature_paths:
                        logger.warning("Skipping distances step for subset '%s' as no feature paths are available.", subset_key)
                    elif not self.cfg.get("distances"):
                        logger.info("No distance metrics configured, skipping distance computation for '%s'.", subset_key)
                    else:
                        logger.info("--- Running Distance Computation for Subset: %s ---", subset_key)
                        distance_paths = self.distance_manager.process_subset_distances(
                            dataset_cache_id=dataset_cache_id,
                            current_subset_key=subset_key,
                            feature_manager=self.feature_manager,
                            subset_features_dir=subset_features_dir,
                            item_id_map=item_id_map,
                            feature_paths=feature_paths,
                            run_steps=steps,
                        )
                        if not distance_paths:
                            logger.warning("Distance processing yielded no valid paths for subset '%s'. Subsequent steps may fail if cache is missing.", subset_key)
                elif "benchmarks" in steps and self.cfg.get("distances"):
                    logger.info("--- Locating Existing Distance Paths Only for Subset: %s ---", subset_key)
                    for f_name in feature_paths.keys():
                        raw_feat_conf = self.feature_manager.all_feature_configs_map.get(f_name)
                        if not raw_feat_conf:
                            continue
                        eff_feat_conf = self.feature_manager._get_effective_feature_config(raw_feat_conf, subset_key)
                        allowed_dists = eff_feat_conf.get("compute_distances_for")

                        for dist_conf in self.cfg.get("distances", []):
                            d_name = dist_conf.get("name")
                            if not d_name or (allowed_dists is not None and d_name not in allowed_dists):
                                continue
                            combined_config_for_path_check = {"feature_config": eff_feat_conf, "distance_config": dist_conf, "name": f"{f_name}_{d_name}"}
                            expected_path = get_cache_path(
                                cache_dir=subset_features_dir, prefix=f"distances_{d_name}", dataset_cache_id=dataset_cache_id, config_dict=combined_config_for_path_check, extra_suffix=None
                            )
                            if expected_path and expected_path.is_file():
                                distance_paths[(f_name, d_name)] = expected_path
                            else:
                                combined_config = {"feature_config": eff_feat_conf, "distance_config": dist_conf, "name": f"{f_name}_{d_name}"}
                                found_dist_path = find_cache_path(
                                    cache_dir=subset_features_dir, prefix=f"distances_{d_name}", dataset_cache_id=dataset_cache_id, config_dict=combined_config
                                )
                                if found_dist_path and found_dist_path.is_file():
                                    logger.debug("Found distance '%s/%s' using find_cache_path: %s", f_name, d_name, found_dist_path.name)
                                    distance_paths[(f_name, d_name)] = found_dist_path
                                else:
                                    logger.debug("No cached distance file found (using find_cache_path or expected path) for '%s/%s' when distance step was skipped.", f_name, d_name)

                if "benchmarks" in steps:
                    if not self.cfg.get("benchmarks"):
                        logger.info("No benchmarks configured, skipping benchmark run for '%s'.", subset_key)
                    elif not feature_paths:
                        logger.warning("Skipping benchmarks step for subset '%s' as no feature paths are available (step skipped or compute failed).", subset_key)
                    else:
                        logger.info("--- Running Benchmarks for Subset: %s ---", subset_key)
                        subset_results = self.benchmark_manager.run_subset_benchmarks(
                            subset_features_dir=subset_features_dir,
                            subset_dataset_obj=subset_dataset,
                            item_id_map=item_id_map,
                            dataset_cache_id=dataset_cache_id,
                            current_subset_key=subset_key,
                            subset_cache_dir=subset_bench_cache_dir,
                            feature_paths=feature_paths,
                            distance_paths=distance_paths,
                        )
                        overall_results[subset_key] = subset_results

                        if subset_results:
                            dataset_id_for_filename = _get_safe_path_part(Path(self.cfg.get("dataset", {}).get("id", "unknown_dataset")).name)
                            filename_prefix = f"{_get_safe_path_part(subset_key)}_{dataset_id_for_filename}_{run_id}_results"
                            save_results(subset_results, subset_results_dir, filename_prefix)
                else:
                    logger.info("Skipping benchmark step for subset '%s'.", subset_key)

            except Exception as e:
                logger.error("Error processing subset '%s': %s", subset_key, e, exc_info=True)
            finally:
                if subset_dataset is not None:
                    del subset_dataset
                if "item_id_map" in locals():
                    del item_id_map
                if "feature_paths" in locals():
                    del feature_paths
                if "distance_paths" in locals():
                    del distance_paths
                if "subset_results" in locals():
                    del subset_results
                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception as cuda_err:
                        logger.warning("CUDA cache clear failed: %s", cuda_err)
                logger.info("Memory cleared after processing subset '%s'.", subset_key)

            elapsed = time.time() - start_time
            logger.info("Subset '%s' completed in %.2fs.", subset_key, elapsed)
            logger.info("===== Subset %s Complete =====", subset_key)

        if len(subsets_to_run) > 1 and overall_results:
            logger.info("Saving combined results for all processed subsets.")
            dataset_id_for_filename = _get_safe_path_part(Path(self.cfg.get("dataset", {}).get("id", "unknown_dataset")).name)
            combined_prefix = f"COMBINED_{dataset_id_for_filename}_{run_id}"
            save_results(overall_results, self.base_results_dir, combined_prefix)
        elif overall_results:
            logger.info("Single subset processed. Results saved under subset name (see above).")
        else:
            logger.info("No overall benchmark results generated.")

    def run(self, steps: List[str]) -> None:
        """
        Executes the specified pipeline steps based on the provided list.

        Args:
            steps: List of steps to run (e.g., ["train", "features", "distances", "benchmarks"]).
                   Steps are executed in a fixed logical order if present.
        """
        logger.info("--- Executing Pipeline Stages: %s ---", steps)

        if "train" in steps:
            if self.cfg.get("train"):
                logger.info("--- Stage: Training ---")
                if not self.dataset_manager.full_dataset_obj:
                    if not self.dataset_manager.load_full_dataset():
                        logger.error("Failed to load dataset for training stage. Aborting.")
                        return

                if not self.trainer_manager.run_all_training_jobs(self.dataset_manager):
                    logger.error("Training stage encountered errors.")
                else:
                    logger.info("--- Training Stage Complete ---")
            else:
                logger.info("Skipping training stage: No 'train' jobs defined in config.")
        else:
            logger.info("Skipping training stage (not requested).")

        evaluation_steps_requested = [s for s in steps if s in ["features", "distances", "benchmarks"]]
        if evaluation_steps_requested:
            logger.info("--- Starting Evaluation Stages: %s ---", evaluation_steps_requested)
            self._run_subset_loop(steps)
            logger.info("--- Evaluation Stages Complete ---")
        else:
            logger.info("Skipping all evaluation stages (features, distances, benchmarks) as none were requested.")

        logger.info("--- Pipeline Runner Finished ---")