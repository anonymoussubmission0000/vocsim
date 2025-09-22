import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type
import torch
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
import h5py
import gc
import time
import itertools

from sklearn.metrics import make_scorer, top_k_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics._scorer import _BaseScorer

from benchmarks.base import Benchmark
from utils.file_utils import read_hdf5_slice, read_hdf5_metadata

logger = logging.getLogger(__name__)
HDF5_FEATURE_DATASET_NAME = "features"
DEFAULT_CHUNK_SIZE = 1024


class ClassificationBenchmark(Benchmark):
    def _initialize(
        self,
        n_splits: int = 5,
        classifiers: Optional[List[str]] = None,
        classifier_params: Optional[Dict[str, Dict]] = None,
        random_state: int = 42,
        eval_metrics: Optional[List[str]] = None,
        top_k: Optional[int] = 5,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        label_source_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initializes the ClassificationBenchmark parameters and sets up classifier configurations.

        Args:
            n_splits (int): Number of splits for cross-validation.
            classifiers (Optional[List[str]]): List of classifier types to run
                ('knn', 'rf', 'mlp'). If None, defaults to all.
            classifier_params (Optional[Dict[str, Dict]]): Dictionary of parameter
                grids or values for each classifier type. Merged with defaults.
            random_state (int): Random state for reproducibility.
            eval_metrics (Optional[List[str]]): List of standard scikit-learn
                scoring metric names (e.g., 'accuracy', 'f1_macro').
                If None, defaults to ['accuracy'].
            top_k (Optional[int]): The k value for top-k accuracy scoring. If None
                or <= 1, top-k scoring is disabled.
            chunk_size (int): Size of chunks to read features (not directly used in
                current CV approach, but kept for potential future uses).
            label_source_key (Optional[str]): Key used by BenchmarkManager to
                retrieve the list of labels.
            **kwargs: Additional keyword arguments.
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.eval_metrics = eval_metrics or ["accuracy"]
        self.top_k = top_k if top_k and top_k > 1 else None
        self.chunk_size = chunk_size
        self.label_source_key = label_source_key
        default_classifiers = ["knn", "rf", "mlp"]
        self.classifiers_to_run = classifiers or default_classifiers
        default_params = {
            "knn": {"n_neighbors": [3, 10, 30], "n_jobs": [-1]},
            "rf": {"max_depth": [10, 20], "random_state": [self.random_state], "n_jobs": [-1]},
            "mlp": {
                "alpha": [0.1, 0.01, 0.001],
                "random_state": [self.random_state],
                "max_iter": [500],
                "solver": ["adam"],
                "activation": ["relu"],
                "hidden_layer_sizes": [(100,), [400], [200, 200]],
                "batch_size": ["auto", 256],
                "learning_rate_init": [0.001],
                "learning_rate": ["adaptive"],
                "tol": [1e-4],
                "early_stopping": [True],
            },
        }
        self.classifier_configs = {}
        user_params = classifier_params or {}
        config_counter = 0

        logger.debug("Generating classifier configurations...")
        for clf_type in self.classifiers_to_run:
            base_params = default_params.get(clf_type, {})
            user_type_params = user_params.get(clf_type, {})
            param_grid = {**base_params, **user_type_params}
            param_grid = {k: (v if isinstance(v, list) else [v]) for k, v in param_grid.items()}

            keys, values = zip(*param_grid.items())
            product_count = 0
            for bundle_values in itertools.product(*values):
                product_count += 1
                bundle = dict(zip(keys, bundle_values))

                def format_val(v):
                    if isinstance(v, tuple):
                        return f"({','.join(map(str, v))}{',' if len(v)==1 else ''})"
                    if isinstance(v, list):
                        return f"[{','.join(map(str, v))}]"
                    return repr(v)

                param_strs = []
                for k, v_loop in sorted(bundle.items()):
                    param_strs.append(f"{k}={format_val(v_loop)}")
                desc_string = f"{clf_type}(" + ", ".join(param_strs) + ")"
                config_key = f"{clf_type}_{config_counter}"
                config_counter += 1
                self.classifier_configs[config_key] = {
                    "type": clf_type,
                    "params": bundle,
                    "description": desc_string,
                }
            logger.debug(f"Generated {product_count} configurations for {clf_type}.")
        logger.info(
            f"Initialized ClassificationBenchmark. N_Splits={n_splits}, TopK={self.top_k}, LabelSourceKey='{self.label_source_key}'"
        )
        logger.info(
            f"Classifiers to run ({len(self.classifier_configs)} configs): {[v['description'] for v in self.classifier_configs.values()]}"
        )

    def _get_classifier_instance(self, clf_type: str, params: Dict) -> Optional[Any]:
        """
        Creates an instance of a scikit-learn classifier based on type and parameters.

        Args:
            clf_type (str): The type of classifier ('knn', 'rf', 'mlp').
            params (Dict): The parameters for the classifier constructor.

        Returns:
            Optional[Any]: An instantiated classifier object, or None if
                           instantiation fails (especially for MLP with bad params).
        """
        if clf_type == "knn":
            return KNeighborsClassifier(**params)
        elif clf_type == "rf":
            return RandomForestClassifier(**params)
        elif clf_type == "mlp":
            params_copy = params.copy()
            hls_param = params_copy.get("hidden_layer_sizes")
            final_hls_tuple = None
            if isinstance(hls_param, (list, tuple)):
                if hls_param and all(isinstance(i, int) and i > 0 for i in hls_param):
                    final_hls_tuple = tuple(hls_param)
                else:
                    logger.error(f"Invalid elements in hidden_layer_sizes list/tuple: {hls_param}")
                    return None
            elif isinstance(hls_param, int):
                if hls_param > 0:
                    final_hls_tuple = (hls_param,)
                else:
                    logger.error(f"Invalid non-positive integer for hidden_layer_sizes: {hls_param}")
                    return None
            else:
                logger.error(f"Invalid or missing 'hidden_layer_sizes'. Expected list/tuple/int, got {type(hls_param)}.")
                return None
            params_copy["hidden_layer_sizes"] = final_hls_tuple
            try:
                logger.debug(f"Instantiating MLPClassifier with final params: {params_copy}")
                return MLPClassifier(**params_copy)
            except Exception as mlp_init_err:
                logger.error(f"Error instantiating MLPClassifier: {mlp_init_err}. Params: {params_copy}", exc_info=True)
                return None
        else:
            raise ValueError(f"Unsupported classifier type: {clf_type}")

    def evaluate(
        self, feature_hdf5_path: Optional[Path] = None, labels: Optional[List[Any]] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Runs the classification benchmark.

        Loads features from the specified HDF5 file, prepares and encodes labels
        based on HDF5 metadata indices, and performs cross-validation using
        the configured classifiers and scoring metrics.

        Args:
            feature_hdf5_path (Optional[Path]): Path to the HDF5 file containing features.
            labels (Optional[List[Any]]): A list of labels provided by the
                BenchmarkManager, corresponding to the original item indices.
            **kwargs: Additional keyword arguments (not used by this method).

        Returns:
            Dict[str, Any]: A dictionary containing the results for each classifier
                           configuration. Each key is the configuration description,
                           and the value is a dictionary of metric results (mean and std)
                           or an error message.
        """
        start_eval_time = time.time()
        if not self.enabled:
            return {"error": "scikit-learn not installed"}
        if feature_hdf5_path is None:
            return {"error": "feature_hdf5_path is required for ClassificationBenchmark."}
        if not feature_hdf5_path.exists():
            return {"error": f"Feature HDF5 file not found: {feature_hdf5_path}"}
        if labels is None:
            logger.error(
                "ClassificationBenchmark requires 'labels' list. This list should be populated by BenchmarkManager using 'label_source_key'."
            )
            return {"error": "Labels missing"}

        overall_results = {}
        logger.info(f"Starting Classification Benchmark on: {feature_hdf5_path.name}")
        logger.info(f"Received labels list length: {len(labels) if labels else 'None'}")
        if labels and len(labels) > 5:
            logger.info(f"Sample of first 5 received labels: {labels[:5]}")

        try:
            feature_metadata = read_hdf5_metadata(feature_hdf5_path, HDF5_FEATURE_DATASET_NAME)
            if not feature_metadata:
                logger.error(f"Could not read feature metadata from {feature_hdf5_path} for dataset {HDF5_FEATURE_DATASET_NAME}")
                raise ValueError("Could not read feature metadata.")

            n_samples_hdf5 = feature_metadata.get("num_items_processed")
            if n_samples_hdf5 is None:
                n_samples_hdf5 = feature_metadata.get("num_items")
                attr_used = "'num_items' (fallback)"
            else:
                attr_used = "'num_items_processed'"
            if n_samples_hdf5 is None:
                logger.error(
                    f"Could not determine number of samples from HDF5 metadata ('num_items_processed' or 'num_items'). File: {feature_hdf5_path.name}"
                )
                raise ValueError("Could not determine number of samples from metadata.")
            n_samples_hdf5 = int(n_samples_hdf5)
            logger.info(f"HDF5 metadata indicates {n_samples_hdf5} samples (using {attr_used}).")
            if n_samples_hdf5 <= 0:
                return {"error": f"Feature file contains {n_samples_hdf5} samples."}

            original_indices = feature_metadata.get("original_indices")

            if not isinstance(labels, list):
                try:
                    labels = list(labels)
                except TypeError:
                    return {"error": f"Labels object (type {type(labels)}) is not list-convertible."}

            if original_indices is not None:
                logger.info(f"Found 'original_indices' in HDF5 metadata (length: {len(original_indices)}). Mapping labels...")
                if not isinstance(original_indices, (list, np.ndarray)):
                    raise TypeError("original_indices in metadata must be a list or numpy array.")
                if len(original_indices) != n_samples_hdf5:
                    logger.error(
                        f"Metadata mismatch: len(original_indices)={len(original_indices)} != n_samples_hdf5={n_samples_hdf5}. Aborting."
                    )
                    return {"error": "Metadata mismatch: indices vs sample count."}
                hdf5_idx_map = {int(orig_idx): hdf5_pos for hdf5_pos, orig_idx in enumerate(original_indices)}
                logger.debug(f"Built hdf5_idx_map from original_indices. Size: {len(hdf5_idx_map)}")
            else:
                logger.warning("No 'original_indices' in metadata. Assuming HDF5 data corresponds to first N labels.")
                if len(labels) < n_samples_hdf5:
                    logger.error(
                        f"Labels list length ({len(labels)}) is less than HDF5 sample count ({n_samples_hdf5}) and no 'original_indices' provided."
                    )
                    return {"error": f"Labels list length ({len(labels)}) < HDF5 sample count ({n_samples_hdf5}) and no indices provided."}
                hdf5_idx_map = {i: i for i in range(n_samples_hdf5)}
                logger.debug(f"Built hdf5_idx_map assuming direct correspondence. Size: {len(hdf5_idx_map)}")

            y_ordered = [None] * n_samples_hdf5
            valid_label_count = 0
            indices_in_hdf5_with_valid_label = []
            max_label_index = len(labels) - 1
            num_skipped_due_to_none_label = 0
            num_skipped_due_to_idx_out_of_bounds = 0

            for orig_idx_key, hdf5_pos_val in hdf5_idx_map.items():
                orig_idx = int(orig_idx_key)
                hdf5_pos = int(hdf5_pos_val)

                if 0 <= orig_idx <= max_label_index:
                    label_value = labels[orig_idx]
                    if label_value is not None:
                        y_ordered[hdf5_pos] = str(label_value)
                        valid_label_count += 1
                        indices_in_hdf5_with_valid_label.append(hdf5_pos)
                    else:
                        num_skipped_due_to_none_label += 1
                else:
                    num_skipped_due_to_idx_out_of_bounds += 1

            logger.debug(
                f"Label mapping: Skipped {num_skipped_due_to_none_label} (None label),"
                f" {num_skipped_due_to_idx_out_of_bounds} (idx OOB)."
            )

            if valid_label_count == 0:
                logger.error("No valid labels found corresponding to features in HDF5 file after mapping.")
                return {"error": "No valid labels found corresponding to features in HDF5 file."}

            y_str = np.array([y_ordered[hdf5_pos] for hdf5_pos in indices_in_hdf5_with_valid_label])
            valid_hdf5_indices = np.array(indices_in_hdf5_with_valid_label, dtype=int)

            if valid_hdf5_indices.size == 0 or y_str.size == 0 or valid_hdf5_indices.size != y_str.size:
                logger.error("Internal error: Mismatch between valid indices and labels after filtering.")
                return {"error": "Internal error: Label/index mismatch after filtering."}

            logger.info(f"Encoding {len(y_str)} string labels into numerical format...")
            le = LabelEncoder()
            try:
                y = le.fit_transform(y_str)
                encoded_classes = le.classes_
                logger.info(f"Label encoding successful. Found {len(encoded_classes)} unique classes.")
                if not np.issubdtype(y.dtype, np.integer):
                    logger.warning(f"LabelEncoder output dtype is {y.dtype}, ensuring integer.")
                    y = y.astype(int)
            except Exception as le_err:
                logger.error(f"LabelEncoder failed: {le_err}", exc_info=True)
                return {"error": f"Label encoding failed: {le_err}"}

            logger.info(f"Prepared {len(y)} numerical labels for classification.")
            if len(y) > 0:
                unique_y_numeric, counts_y_numeric = np.unique(y, return_counts=True)
                logger.info(f"Final numerical label distribution (count: {len(unique_y_numeric)}): {dict(zip(unique_y_numeric, counts_y_numeric))}")

            if len(y) < self.n_splits:
                logger.error(f"Insufficient valid labeled samples ({len(y)}) for {self.n_splits}-fold CV.")
                return {"error": f"Insufficient valid labeled samples ({len(y)}) for {self.n_splits}-fold CV"}

            label_counts = Counter(y)
            min_samples_per_class = min(label_counts.values()) if label_counts else 0
            logger.info(f"Min samples per class: {min_samples_per_class}, n_splits: {self.n_splits}")

            if min_samples_per_class < self.n_splits:
                error_msg = (
                    f"Smallest class ({min_samples_per_class}) < n_splits ({self.n_splits}). "
                    f"Distribution: {dict(label_counts)}"
                )
                logger.error(error_msg)
                return {"error": error_msg}

            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            scorers = {metric: metric for metric in self.eval_metrics}
            unique_y_labels_numeric = np.unique(y)
            if self.top_k is not None and len(unique_y_labels_numeric) > 1:
                try:
                    scorers[f"top_{self.top_k}_accuracy"] = make_scorer(
                        score_func=top_k_accuracy_score,
                        response_method=("decision_function", "predict_proba"),
                        greater_is_better=True,
                        k=self.top_k,
                        labels=unique_y_labels_numeric,
                    )
                    logger.info(f"Successfully created top-{self.top_k} accuracy scorer.")
                except Exception as scorer_err:
                    logger.error(f"Failed to create top-{self.top_k} scorer: {scorer_err}", exc_info=True)
                    scorers.pop(f"top_{self.top_k}_accuracy", None)

        except Exception as e:
            logger.error(f"Data/Label preparation failed: {e}", exc_info=True)
            return {"error": f"Data prep/CV setup failed: {e}"}

        num_configs = len(self.classifier_configs)
        for config_idx, (config_key, config_data) in enumerate(self.classifier_configs.items()):
            config_start_time = time.time()
            clf_type = config_data["type"]
            params = config_data["params"]
            config_description = config_data["description"]

            logger.info(f"--- Evaluating Config {config_idx+1}/{num_configs}: {config_description} ---")
            clf_results = {}
            X = None

            if clf_type in ["knn", "rf", "mlp"]:
                logger.debug(f"Loading relevant feature data for {config_description}...")
                try:
                    if valid_hdf5_indices.size == 0:
                        raise ValueError("No valid indices to load features from.")
                    min_hdf5_idx = valid_hdf5_indices.min()
                    max_hdf5_idx = valid_hdf5_indices.max()

                    block_features = read_hdf5_slice(
                        feature_hdf5_path, int(min_hdf5_idx), int(max_hdf5_idx) + 1, HDF5_FEATURE_DATASET_NAME
                    )
                    if block_features is None:
                        raise IOError(f"Failed to read HDF5 feature block [{min_hdf5_idx}-{max_hdf5_idx+1}]")

                    relative_indices = valid_hdf5_indices - min_hdf5_idx
                    if np.any(relative_indices < 0) or np.any(relative_indices >= block_features.shape[0]):
                        raise IndexError(
                            "Calculated relative indices out of bounds for loaded block."
                            f" MinRel={relative_indices.min()}, MaxRel={relative_indices.max()}, BlockShape={block_features.shape[0]}"
                        )

                    X_raw = block_features[relative_indices]

                    feature_dim_orig = X_raw.shape[1:]
                    if X_raw.ndim > 2:
                        X = X_raw.reshape(X_raw.shape[0], -1)
                    elif X_raw.ndim == 1:
                        X = X_raw.reshape(-1, 1)
                    elif X_raw.ndim == 2:
                        X = X_raw
                    else:
                        raise ValueError(f"Invalid feature dimensionality {X_raw.ndim}")
                    feature_dim_flat = X.shape[1]

                    if not np.isfinite(X).all():
                        nan_count = np.sum(np.isnan(X))
                        inf_count = np.sum(np.isinf(X))
                        logger.warning(f"NaN/Inf found in features for CV (NaNs: {nan_count}, Infs: {inf_count}). Replacing with 0.")
                        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

                    if X.shape[0] != len(y):
                        raise RuntimeError(f"Feature samples ({X.shape[0]}) != label count ({len(y)}).")
                    logger.info(f"Loaded features for CV (orig dim: {feature_dim_orig}, flat dim: {feature_dim_flat}). Shape: {X.shape}")

                    classifier = self._get_classifier_instance(clf_type, params)
                    if classifier is None:
                        logger.error(f"Could not instantiate classifier for {config_description}. Skipping.")
                        clf_results = {"error": "Classifier instantiation failed (check logs for parameter issues)."}
                        overall_results[config_description] = clf_results
                        continue

                    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])
                    current_scorers = scorers.copy()

                    clf_needs_proba = any(
                        isinstance(s, _BaseScorer)
                        and (s._response_method == "predict_proba" or s._response_method == ("decision_function", "predict_proba"))
                        for s in current_scorers.values()
                    )
                    clf_supports_proba = hasattr(classifier, "predict_proba")
                    clf_supports_decision = hasattr(classifier, "decision_function")

                    if clf_needs_proba and not (clf_supports_proba or clf_supports_decision):
                        logger.warning(
                            f"Classifier {config_description} doesn't support predict_proba or decision_function,"
                            f" removing scorers requiring them (e.g., top_k_accuracy)."
                        )
                        scorers_to_remove = [
                            name
                            for name, scorer in current_scorers.items()
                            if isinstance(scorer, _BaseScorer) and scorer._response_method != "predict"
                        ]
                        current_scorers = {k: v for k, v in current_scorers.items() if k not in scorers_to_remove}

                        if not current_scorers:
                            logger.warning(f"No scorers left for {config_description}. Skipping.")
                            clf_results = {"warning": "Skipped, no valid scorers."}
                            overall_results[config_description] = clf_results
                            continue

                    logger.debug(f"Running standard cross_validate for {config_description}...")
                    cv_results = cross_validate(pipeline, X, y, cv=skf, scoring=current_scorers, n_jobs=1, error_score="raise")

                    for metric in current_scorers.keys():
                        score_key = f"test_{metric}"
                        mean_score = np.nan
                        std_score = np.nan
                        if score_key in cv_results and cv_results[score_key] is not None:
                            scores_array = np.asarray(cv_results[score_key])
                            valid_scores = scores_array[~np.isnan(scores_array)]
                            if len(valid_scores) > 0:
                                mean_score = np.mean(valid_scores) * 100
                                std_score = np.std(valid_scores) * 100
                            else:
                                logger.warning(f"No valid scores found for metric '{metric}' for {config_description}.")
                        else:
                            logger.warning(
                                f"Metric '{metric}' (key: {score_key}) not found or None in cv_results for {config_description}. Keys: {list(cv_results.keys())}"
                            )

                        clf_results[f"{metric}_mean"] = mean_score
                        clf_results[f"{metric}_std"] = std_score
                        if not np.isnan(mean_score):
                            logger.info(f"  {metric}: {mean_score:.2f}% (+/- {std_score:.2f}%)")
                        else:
                            logger.info(f"  {metric}: NaN (Check logs for errors during CV)")

                except Exception as e:
                    logger.error(f"Failed CV Pipeline for {config_description}: {e}", exc_info=True)
                    clf_results = {f"{metric}_mean": np.nan for metric in scorers}
                    clf_results.update({f"{metric}_std": np.nan for metric in scorers})
                    clf_results["error"] = f"CV Fail: {str(e)}"
                finally:
                    if X is not None:
                        del X
                    if "pipeline" in locals():
                        del pipeline
                    if "classifier" in locals():
                        del classifier
                    if "block_features" in locals():
                        del block_features
                    if "X_raw" in locals():
                        del X_raw
                    gc.collect()
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

            else:
                logger.warning(f"Skipping unsupported classifier type: {clf_type}")
                clf_results = {"error": f"Classifier type '{clf_type}' not handled."}

            overall_results[config_description] = clf_results
            config_elapsed = time.time() - config_start_time
            logger.info(f"--- Config {config_description} finished in {config_elapsed:.2f}s ---")

        eval_elapsed = time.time() - start_eval_time
        logger.info(f"Classification benchmark finished in {eval_elapsed:.2f}s.")
        return overall_results