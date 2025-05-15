"""
Benchmark for evaluating embedding alignment with avian perceptual judgments.

This module defines the PerceptualAlignment benchmark, which assesses how well
the distance metric of a given feature embedding aligns with zebra finch
perceptual judgments collected using probe (AXB) and derived triplet tasks.

Reference:
  Zandberg L, Morfi V, George JM, Clayton DF, Stowell D, Lachlan RF (2024)
  Bird song comparison using deep learning trained from avian perceptual judgments.
  PLOS Computational Biology 20(8): e1012329.
  https://doi.org/10.1371/journal.pcbi.1012329
  Dataset: https://doi.org/10.5281/zenodo.5545872
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import h5py
import gc
from scipy.stats import binomtest, kendalltau, spearmanr
from tqdm import tqdm
from sklearn.utils import resample

from benchmarks.base import Benchmark

logger = logging.getLogger(__name__)
HDF5_DISTANCE_DATASET_NAME = "distance_matrix"


class PerceptualAlignment(Benchmark):
    """
    Evaluates alignment with perceptual judgments using probe and triplet tasks.
    """

    def _initialize(
        self,
        probe_csv_path: str,
        triplet_csv_path: str,
        probe_consistency_threshold: Optional[float] = 0.7,
        bootstrap_ci: bool = True,
        n_bootstraps: int = 1000,
        **kwargs,
    ):
        """
        Initializes PerceptualAlignment benchmark parameters.

        Returns:
            None
        """
        self.probe_csv_path_str = probe_csv_path
        self.triplet_csv_path_str = triplet_csv_path
        self.probe_consistency_threshold = probe_consistency_threshold
        self.bootstrap_ci = bootstrap_ci
        self.n_bootstraps = n_bootstraps
        self.probe_csv_path: Optional[Path] = None
        self.triplet_csv_path: Optional[Path] = None
        logger.info("Initialized PerceptualAlignment config:")
        logger.info(f"  Probes: {self.probe_csv_path_str}")
        logger.info(f"  Triplets: {self.triplet_csv_path_str} (will evaluate 'all' and 'high' margin types)")
        logger.info(f"  Probe Consistency Threshold: {self.probe_consistency_threshold}")
        logger.info(f"  Bootstrap CI: {self.bootstrap_ci}, N_Bootstraps: {self.n_bootstraps}")

    def _load_and_prepare_data(
        self, item_id_map: Optional[Dict[str, Dict]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
        """
        Loads perceptual data, performs filtering, creates name-to-index map.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]: Raw probes dataframe,
            filtered probes dataframe, raw triplets dataframe, and name-to-index map.
        """
        if self.probe_csv_path is None or self.triplet_csv_path is None:
            raise ValueError("Probe/Triplet CSV paths must be set before calling _load_and_prepare_data.")
        if not self.probe_csv_path.exists():
            raise FileNotFoundError(f"Probe CSV not found: {self.probe_csv_path}")
        if not self.triplet_csv_path.exists():
            raise FileNotFoundError(f"Triplet CSV not found: {self.triplet_csv_path}")
        if not item_id_map:
            raise ValueError("item_id_map is required to map names to matrix indices.")

        try:
            probes_df = pd.read_csv(self.probe_csv_path)
            required_probe_cols = {"sound_id", "left", "right", "decision"}
            if not required_probe_cols.issubset(probes_df.columns):
                missing = required_probe_cols - set(probes_df.columns)
                raise ValueError(f"Probe CSV missing columns: {missing}")
            for col in required_probe_cols:
                probes_df[col] = probes_df[col].astype(str)
            logger.info(f"Loaded {len(probes_df)} probe judgments from {self.probe_csv_path.name}.")
        except Exception as e:
            logger.error("Failed to load probe CSV file: %s", e, exc_info=True)
            raise

        triplet_required_cols = {"Anchor", "Positive", "Negative", "Margin_Type"}
        try:
            triplets_df_full = pd.read_csv(self.triplet_csv_path, usecols=lambda c: c in triplet_required_cols)
            if not triplet_required_cols.issubset(triplets_df_full.columns):
                missing_cols = triplet_required_cols - set(triplets_df_full.columns)
                raise ValueError(f"Triplet CSV missing required columns: {missing_cols}")
            for col in triplet_required_cols:
                triplets_df_full[col] = triplets_df_full[col].astype(str)
            logger.info(f"Loaded {len(triplets_df_full)} derived triplet entries from {self.triplet_csv_path.name}.")
        except Exception as e:
            logger.error("Failed to load or process triplet CSV file: %s", e, exc_info=True)
            raise

        name_to_idx: Dict[str, int] = {}
        map_build_errors = []
        logger.info("Building name_to_idx from item_id_map...")
        found_indices = set()
        for item_id, meta in item_id_map.items():
            try:
                original_name = meta.get("original_name")
                index = meta.get("index")
                if original_name is not None and index is not None:
                    original_name_str = str(original_name)
                    current_index = int(index)
                    if original_name_str in name_to_idx and name_to_idx[original_name_str] != current_index:
                        logger.warning(
                            "Duplicate original_name '%s' in item_id_map with different indices. Using first found (%s).",
                            original_name_str,
                            name_to_idx[original_name_str],
                        )
                    elif original_name_str not in name_to_idx:
                        name_to_idx[original_name_str] = current_index
                        found_indices.add(current_index)
                else:
                    map_build_errors.append(f"{item_id}:missing_keys")
            except (TypeError, ValueError) as e:
                map_build_errors.append(f"{item_id}:error({e})")

        if map_build_errors:
            logger.warning("%d items had issues during name_to_idx build: %s...", len(map_build_errors), map_build_errors[:5])
        if not name_to_idx:
            raise ValueError("Failed to build any name_to_idx entries.")
        logger.info(f"Built name_to_idx map covering {len(name_to_idx)} unique names and {len(found_indices)} unique indices.")
        self.max_index_from_map = max(found_indices) if found_indices else -1

        initial_triplet_count = len(triplets_df_full)
        triplets_df_full.dropna(subset=["Anchor", "Positive", "Negative"], inplace=True)
        dropped_nan_count = initial_triplet_count - len(triplets_df_full)
        if dropped_nan_count > 0:
            logger.info(f"Dropped {dropped_nan_count} triplets with missing A/P/N.")
        triplets_df_full = triplets_df_full.drop_duplicates(subset=["Anchor", "Positive", "Negative"], keep="first")
        dropped_dup_count = (initial_triplet_count - dropped_nan_count) - len(triplets_df_full)
        if dropped_dup_count > 0:
            logger.warning("Removed %d duplicate triplets (A,P,N). Kept first.", dropped_dup_count)
        logger.info(f"Triplet processing complete. Keeping {len(triplets_df_full)} unique triplets.")

        filtered_probes_df = probes_df.copy()
        initial_probe_count = len(probes_df)
        probes_df.dropna(subset=["sound_id", "left", "right", "decision"], inplace=True)
        if len(probes_df) < initial_probe_count:
            logger.info("Dropped %d probes with missing values.", initial_probe_count - len(probes_df))
        probes_df["decision_lower"] = probes_df["decision"].str.lower()
        probes_df = probes_df[probes_df["decision_lower"].isin(["left", "right"])].copy()
        if self.probe_consistency_threshold is not None:
            logger.info("Applying probe consistency filter (threshold: %s)...", self.probe_consistency_threshold)
            decision_counts = probes_df.groupby("sound_id")["decision_lower"].value_counts().unstack(fill_value=0)
            if "left" not in decision_counts.columns:
                decision_counts["left"] = 0
            if "right" not in decision_counts.columns:
                decision_counts["right"] = 0
            decision_counts["total"] = decision_counts["left"] + decision_counts["right"]
            decision_counts["majority_choice"] = decision_counts.apply(
                lambda row: "left" if row["left"] >= row["right"] else "right", axis=1
            )
            decision_counts["majority_count"] = decision_counts.apply(lambda row: row[row["majority_choice"]], axis=1)
            decision_counts["consistency"] = decision_counts["majority_count"] / decision_counts["total"]
            consistent_sound_ids = decision_counts[decision_counts["consistency"] >= self.probe_consistency_threshold].index
            logger.info("Found %d sound IDs meeting consistency threshold.", len(consistent_sound_ids))
            probes_df_merged = probes_df.merge(decision_counts[["majority_choice"]], left_on="sound_id", right_index=True)
            filtered_probes_df = probes_df_merged[
                (probes_df_merged["sound_id"].isin(consistent_sound_ids))
                & (probes_df_merged["decision_lower"] == probes_df_merged["majority_choice"])
            ].copy()
            filtered_probes_df = filtered_probes_df.drop_duplicates(subset=["sound_id"], keep="first")
            logger.info("Filtered probes based on consistency. Kept %d probes.", len(filtered_probes_df))
        else:
            logger.info("Skipping probe consistency filter.")
            filtered_probes_df = probes_df.drop_duplicates(subset=["sound_id"], keep="first").copy()
            logger.info("Filtered probes (example: unique sound_id). Kept %d probes.", len(filtered_probes_df))

        probes_df.drop(columns=["decision_lower"], inplace=True, errors="ignore")
        if "decision_lower" in filtered_probes_df.columns:
            filtered_probes_df.drop(columns=["decision_lower"], inplace=True, errors="ignore")
        if "majority_choice" in filtered_probes_df.columns:
            filtered_probes_df.drop(columns=["majority_choice"], inplace=True, errors="ignore")

        logger.info(f"Data loading complete. Probes: {len(probes_df)} raw, {len(filtered_probes_df)} filtered. Triplets: {len(triplets_df_full)} total unique.")
        return probes_df, filtered_probes_df, triplets_df_full, name_to_idx

    def _compute_correlations(self, model_dists: List[float], bird_dists: List[float], task_name: str) -> Dict:
        """
        Computes Spearman and Kendall correlations with optional CIs.

        Returns:
            Dict: Dictionary containing correlation results and related metrics.
        """
        results = {
            f"{task_name}_spearman_rho": None,
            f"{task_name}_spearman_p": None,
            f"{task_name}_spearman_ci95": [None, None],
            f"{task_name}_kendall_tau": None,
            f"{task_name}_kendall_p": None,
            f"{task_name}_kendall_ci95": [None, None],
            f"{task_name}_num_comparisons": 0,
        }
        if not model_dists or not bird_dists or len(model_dists) != len(bird_dists):
            logger.warning("Cannot compute correlations for %s: Invalid input lists.", task_name)
            return results
        try:
            model_dists_arr = np.array(model_dists, dtype=np.float64)
            bird_dists_arr = np.array(bird_dists, dtype=np.float64)
            valid_mask = np.isfinite(model_dists_arr) & np.isfinite(bird_dists_arr)
            valid_count = int(np.sum(valid_mask))
            if valid_count < 2:
                logger.warning("Cannot compute correlations for %s: < 2 valid pairs (%d).", task_name, valid_count)
                results[f"{task_name}_num_comparisons"] = valid_count
                return results

            model_dists_valid = model_dists_arr[valid_mask]
            bird_dists_valid = bird_dists_arr[valid_mask]
            results[f"{task_name}_num_comparisons"] = valid_count
            std_model = np.std(model_dists_valid)
            std_bird = np.std(bird_dists_valid)
            variance_threshold = 1e-9
            if std_model < variance_threshold or std_bird < variance_threshold:
                logger.warning(
                    "Cannot compute correlations for %s: Zero variance detected (std_model=%.2e, std_bird=%.2e).",
                    task_name,
                    std_model,
                    std_bird,
                )
                for k in results:
                    results[k] = np.nan if not isinstance(results[k], list) else [np.nan, np.nan]
                results[f"{task_name}_num_comparisons"] = valid_count
                return results

            try:
                rho_s, p_s = spearmanr(model_dists_valid, bird_dists_valid)
                results[f"{task_name}_spearman_rho"] = float(rho_s) if np.isfinite(rho_s) else None
                results[f"{task_name}_spearman_p"] = float(p_s) if np.isfinite(p_s) else None
            except Exception as e:
                logger.error("Spearman failed for %s: %s", task_name, e)
            try:
                rho_k, p_k = kendalltau(model_dists_valid, bird_dists_valid, variant="b")
                results[f"{task_name}_kendall_tau"] = float(rho_k) if np.isfinite(rho_k) else None
                results[f"{task_name}_kendall_p"] = float(p_k) if np.isfinite(p_k) else None
            except Exception as e:
                logger.error("Kendall failed for %s: %s", task_name, e)

            if self.bootstrap_ci and valid_count >= 10:
                spear_boot_rhos, kend_boot_taus = [], []
                logger.debug("Running %d bootstrap samples for %s correlations...", self.n_bootstraps, task_name)
                indices = np.arange(valid_count)
                for i in range(self.n_bootstraps):
                    try:
                        boot_idx = resample(indices, replace=True, n_samples=valid_count, random_state=i)
                        if len(boot_idx) < 2:
                            continue
                        boot_model = model_dists_valid[boot_idx]
                        boot_bird = bird_dists_valid[boot_idx]
                        if np.std(boot_model) < variance_threshold or np.std(boot_bird) < variance_threshold:
                            continue
                        try:
                            rho_s_boot, _ = spearmanr(boot_model, boot_bird)
                            spear_boot_rhos.append(rho_s_boot) if np.isfinite(rho_s_boot) else None
                        except ValueError:
                            pass
                        try:
                            rho_k_boot, _ = kendalltau(boot_model, boot_bird, variant="b")
                            kend_boot_taus.append(rho_k_boot) if np.isfinite(rho_k_boot) else None
                        except ValueError:
                            pass
                    except Exception as boot_err:
                        logger.warning("Bootstrap sample %d error for %s: %s", i, task_name, boot_err)
                if spear_boot_rhos:
                    try:
                        ci_s = np.percentile(spear_boot_rhos, [2.5, 97.5])
                        results[f"{task_name}_spearman_ci95"] = [float(x) for x in ci_s] if np.all(np.isfinite(ci_s)) else [np.nan, np.nan]
                    except Exception as ci_err:
                        logger.error("Spearman CI failed for %s: %s", task_name, ci_err)
                else:
                    logger.warning("No valid Spearman rhos for CI in %s.", task_name)
                if kend_boot_taus:
                    try:
                        ci_k = np.percentile(kend_boot_taus, [2.5, 97.5])
                        results[f"{task_name}_kendall_ci95"] = [float(x) for x in ci_k] if np.all(np.isfinite(ci_k)) else [np.nan, np.nan]
                    except Exception as ci_err:
                        logger.error("Kendall CI failed for %s: %s", task_name, ci_err)
                else:
                    logger.warning("No valid Kendall taus for CI in %s.", task_name)
            elif self.bootstrap_ci and valid_count < 10:
                logger.warning("Skipping bootstrap CI for %s: Need >= 10 valid points.", task_name)
        except Exception as e:
            logger.error("Correlation calculation failed for %s: %s", task_name, e, exc_info=True)
            num_comp = results[f"{task_name}_num_comparisons"]
            for k in results:
                results[k] = np.nan if not isinstance(results[k], list) else [np.nan, np.nan]
            results[f"{task_name}_num_comparisons"] = num_comp
        return results

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
        probe_csv_path: Optional[str] = None,
        triplet_csv_path: Optional[str] = None,
        feature_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluates alignment using distances read selectively from HDF5.

        Returns:
            Dictionary containing evaluation results.
        """
        logger.info("Starting PerceptualAlignment evaluation (HDF5 Mode)...")
        final_results = {}
        h5_file = None

        if item_id_map is None:
            return {"error": "item_id_map is required."}

        current_probe_path = Path(probe_csv_path if probe_csv_path else self.probe_csv_path_str)
        current_triplet_path = Path(triplet_csv_path if triplet_csv_path else self.triplet_csv_path_str)
        self.probe_csv_path = current_probe_path
        self.triplet_csv_path = current_triplet_path
        if not current_probe_path.is_file():
            return {"error": f"Probe CSV not found: {current_probe_path}"}
        if not current_triplet_path.is_file():
            return {"error": f"Triplet CSV not found: {current_triplet_path}"}

        dist_mat = None
        dist_mat_source_desc = "N/A"
        if distance_matrix is not None:
            logger.warning("Using directly provided distance_matrix instead of HDF5 path.")
            dist_mat = distance_matrix
            dist_mat_source_desc = "direct_matrix"
            if not isinstance(dist_mat, np.ndarray):
                try:
                    dist_mat = np.array(dist_mat, dtype=np.float32)
                except Exception as e:
                    return {"error": f"Could not convert direct distance_matrix to numpy: {e}"}
        elif distance_matrix_path is not None and distance_matrix_path.exists():
            dist_mat_source_desc = f"hdf5:{distance_matrix_path.name}"
        else:
            return {"error": "distance_matrix_path is required and was not found or provided."}

        try:
            probes_df, filtered_probes_df, triplets_df_full, name_to_idx = self._load_and_prepare_data(item_id_map=item_id_map)
            max_allowable_index = self.max_index_from_map
            if max_allowable_index < 0:
                raise ValueError("Could not determine max index from item_id_map.")
        except Exception as e:
            logger.error("Failed to load/prepare perceptual data: %s", e, exc_info=True)
            return {"error": f"Failed to load/prepare perceptual data: {e}"}

        h5_dset = None
        if dist_mat is None:
            try:
                h5_file = h5py.File(distance_matrix_path, "r")
                if HDF5_DISTANCE_DATASET_NAME not in h5_file:
                    raise ValueError(f"Dataset '{HDF5_DISTANCE_DATASET_NAME}' not found in HDF5 file.")
                h5_dset = h5_file[HDF5_DISTANCE_DATASET_NAME]
                if h5_dset.shape[0] <= max_allowable_index or h5_dset.shape[1] <= max_allowable_index:
                    raise ValueError(
                        "HDF5 matrix shape %s is too small for max index %s found in item_id_map.",
                        h5_dset.shape,
                        max_allowable_index,
                    )
                logger.info(f"Opened distance HDF5 dataset. Shape: {h5_dset.shape}")
            except Exception as e:
                logger.error("Failed to open or validate HDF5 distance matrix file %s: %s", distance_matrix_path, e, exc_info=True)
                if h5_file:
                    h5_file.close()
                return {"error": f"Failed to open/validate HDF5 distance matrix: {e}"}

        triplet_results = {}
        probe_results = {}
        correlation_results = {}

        def get_distance(idx1, idx2):
            if dist_mat is not None:
                return float(dist_mat[idx1, idx2])
            elif h5_dset is not None:
                if not (0 <= idx1 < h5_dset.shape[0] and 0 <= idx2 < h5_dset.shape[1]):
                    logger.warning("Index out of bounds requested (%d, %d) for HDF5 shape %s", idx1, idx2, h5_dset.shape)
                    return np.nan
                return float(h5_dset[idx1, idx2])
            else:
                raise RuntimeError("No distance data source available (direct matrix or HDF5).")

        logger.info("--- Evaluating Triplet Task ---")
        for margin_type_eval in ["all", "high"]:
            if margin_type_eval == "all":
                current_triplets_df = triplets_df_full
            elif margin_type_eval == "high":
                if "Margin_Type" not in triplets_df_full.columns:
                    logger.error("Margin_Type column missing from triplets CSV, cannot filter for 'high' margin.")
                    continue
                current_triplets_df = triplets_df_full[triplets_df_full["Margin_Type"].str.lower() == "high"].copy()

            if current_triplets_df.empty:
                logger.warning("No triplets found for margin type '%s'. Skipping.", margin_type_eval)
                continue
            triplet_acc = 0
            triplet_total = 0
            triplet_model_diffs = []
            triplet_bird_diffs = []
            skipped_names = 0
            skipped_nan = 0

            for row in tqdm(
                current_triplets_df.itertuples(index=False), total=len(current_triplets_df), desc=f"Triplets ({margin_type_eval})", leave=False
            ):
                try:
                    anchor_name = str(getattr(row, "Anchor", ""))
                    pos_name = str(getattr(row, "Positive", ""))
                    neg_name = str(getattr(row, "Negative", ""))

                    if not all(n in name_to_idx for n in [anchor_name, pos_name, neg_name]):
                        skipped_names += 1
                        continue
                    anchor_idx = name_to_idx[anchor_name]
                    pos_idx = name_to_idx[pos_name]
                    neg_idx = name_to_idx[neg_name]

                    dist_pos = get_distance(anchor_idx, pos_idx)
                    dist_neg = get_distance(anchor_idx, neg_idx)

                    if not (np.isfinite(dist_pos) and np.isfinite(dist_neg)):
                        skipped_nan += 1
                        continue

                    model_agrees_with_bird = dist_pos < dist_neg
                    if model_agrees_with_bird:
                        triplet_acc += 1
                    triplet_total += 1
                    triplet_model_diffs.append(dist_pos - dist_neg)
                    triplet_bird_diffs.append(1.0)

                except Exception as e:
                    logger.warning("Err triplet row (%s): %s. E: %s", margin_type_eval, row, e, exc_info=False)
                    continue
            if skipped_names > 0:
                logger.warning("[%s] Skipped %d triplets (names not in map).", margin_type_eval, skipped_names)
            if skipped_nan > 0:
                logger.warning("[%s] Skipped %d triplets (NaN/Inf distances).", margin_type_eval, skipped_nan)

            accuracy = (triplet_acc / triplet_total) if triplet_total > 0 else 0.0
            pvalue = None
            if triplet_total > 0:
                try:
                    pvalue = binomtest(triplet_acc, n=triplet_total, p=0.5, alternative="greater").pvalue
                except ValueError as e:
                    logger.error("Binomial test failed for triplets (%s): %s", margin_type_eval, e)
            triplet_results[f"triplet_{margin_type_eval}_accuracy"] = accuracy
            triplet_results[f"triplet_{margin_type_eval}_pvalue"] = pvalue
            triplet_results[f"triplet_{margin_type_eval}_num_valid"] = triplet_total

            p_str = f"{pvalue:.3e}" if pvalue is not None and np.isfinite(pvalue) else "N/A"
            logger.info("Triplets (%s): Acc=%.4f (%d/%d), p=%s", margin_type_eval, accuracy, triplet_acc, triplet_total, p_str)

            correlation_results.update(
                self._compute_correlations(triplet_model_diffs, triplet_bird_diffs, f"triplet_{margin_type_eval}")
            )

        logger.info("--- Evaluating Probe Task ---")
        probe_corr_data = {"unfiltered": {"model_diffs": [], "bird_choices": []}, "filtered": {"model_diffs": [], "bird_choices": []}}
        for filter_type, df in [("unfiltered", probes_df), ("filtered", filtered_probes_df)]:
            probe_acc = 0
            probe_total = 0
            model_diffs_list = probe_corr_data[filter_type]["model_diffs"]
            bird_choices_list = probe_corr_data[filter_type]["bird_choices"]
            skipped_names = 0
            skipped_nan = 0
            skipped_invalid_decision = 0

            if df.empty:
                logger.warning("No probes found for filter type '%s'. Skipping.", filter_type)
                continue
            for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Probes ({filter_type})", leave=False):
                try:
                    probe_name = str(getattr(row, "sound_id", ""))
                    left_name = str(getattr(row, "left", ""))
                    right_name = str(getattr(row, "right", ""))
                    decision = str(getattr(row, "decision", "")).lower()

                    if decision not in ["left", "right"]:
                        skipped_invalid_decision += 1
                        continue
                    if not all(n in name_to_idx for n in [probe_name, left_name, right_name]):
                        skipped_names += 1
                        continue
                    probe_idx = name_to_idx[probe_name]
                    left_idx = name_to_idx[left_name]
                    right_idx = name_to_idx[right_name]

                    dist_left = get_distance(probe_idx, left_idx)
                    dist_right = get_distance(probe_idx, right_idx)

                    if not (np.isfinite(dist_left) and np.isfinite(dist_right)):
                        skipped_nan += 1
                        continue

                    model_choice_is_left = dist_left < dist_right
                    bird_choice_is_left = decision == "left"
                    if model_choice_is_left == bird_choice_is_left:
                        probe_acc += 1
                    probe_total += 1
                    model_diffs_list.append(dist_left - dist_right)
                    bird_choices_list.append(1.0 if bird_choice_is_left else -1.0)

                except Exception as e:
                    logger.warning("Error processing probe row (%s): %s. E: %s", filter_type, row, e, exc_info=False)
                    continue
            if skipped_names > 0:
                logger.warning("[%s] Skipped %d probes (names not in map).", filter_type, skipped_names)
            if skipped_nan > 0:
                logger.warning("[%s] Skipped %d probes (NaN/Inf dist).", filter_type, skipped_nan)
            if skipped_invalid_decision > 0:
                logger.warning("[%s] Skipped %d probes (invalid decision value).", filter_type, skipped_invalid_decision)

            accuracy = (probe_acc / probe_total) if probe_total > 0 else 0.0
            pvalue = None
            if probe_total > 0:
                try:
                    pvalue = binomtest(probe_acc, n=probe_total, p=0.5, alternative="greater").pvalue
                except ValueError as e:
                    logger.error("Binomial test failed for probes (%s): %s", filter_type, e)
            probe_results[f"probe_accuracy_{filter_type}"] = accuracy
            probe_results[f"probe_pvalue_{filter_type}"] = pvalue
            probe_results[f"probe_num_valid_{filter_type}"] = probe_total

            p_str = f"{pvalue:.3e}" if pvalue is not None and np.isfinite(pvalue) else "N/A"
            logger.info("Probe (%s): Acc=%.4f (%d/%d), p=%s", filter_type, accuracy, probe_acc, probe_total, p_str)

        correlation_results.update(
            self._compute_correlations(probe_corr_data["unfiltered"]["model_diffs"], probe_corr_data["unfiltered"]["bird_choices"], "probe_unfiltered")
        )
        correlation_results.update(
            self._compute_correlations(probe_corr_data["filtered"]["model_diffs"], probe_corr_data["filtered"]["bird_choices"], "probe_filtered")
        )

        if h5_file:
            h5_file.close()
            logger.debug("Closed HDF5 distance matrix file.")
        del dist_mat
        gc.collect()

        final_results = {**triplet_results, **probe_results, **correlation_results}
        for key, value in final_results.items():
            if isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                final_results[key] = float(value) if np.isfinite(value) else None
            elif isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                final_results[key] = int(value)
            elif isinstance(value, list) and len(value) == 2:
                final_results[key] = [float(v) if isinstance(v, (np.number, float, int)) and np.isfinite(v) else None for v in value]
            elif isinstance(value, np.bool_):
                final_results[key] = bool(value)
            elif pd.isna(value):
                final_results[key] = None

        logger.info("PerceptualAlignment evaluation finished.")
        return final_results