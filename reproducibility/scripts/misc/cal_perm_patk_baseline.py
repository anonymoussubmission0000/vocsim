import argparse
import gc
import logging
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml
from datasets import load_dataset
from tqdm import tqdm

# --- Boilerplate to add project root to path ---
try:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except (NameError, IndexError):
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

try:
    from benchmarks.precision import PrecisionAtK
except ImportError:
    print("ERROR: Could not import benchmark class 'PrecisionAtK'. Please ensure 'benchmarks/pak.py' exists and is accessible.")
    sys.exit(1)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
BLIND_TEST_SUBSETS = ["HU3", "HU4", "HW3", "HW4"]
HDF5_DISTANCE_DATASET_NAME = "distance_matrix"
K_VALUES_TO_TEST = [1, 5]


def find_distance_matrix_file(subset_dir: Path, feature_name: str, distance_name: str) -> Path | None:
    """Finds the most recent distance matrix file matching the criteria by parsing filenames."""
    pattern = re.compile(
        rf"distances_{re.escape(distance_name)}.*_{re.escape(feature_name)}_[a-f0-9]+\.h5"
    )
    found_files = [f for f in subset_dir.iterdir() if pattern.match(f.name)] if subset_dir.is_dir() else []
    if found_files:
        return max(found_files, key=lambda p: p.stat().st_mtime)
    logger.warning(f"No distance matrix file found for feature '{feature_name}' and distance '{distance_name}' in {subset_dir}")
    return None


def main():
    """Main function to calculate and report the P@k shuffled baseline with aggregate CIs."""
    parser = argparse.ArgumentParser(
        description="Run a rigorous permutation test for P@k significance, including aggregate CIs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config_file", type=Path, required=True, help="Path to the main VocSim YAML configuration file.")
    parser.add_argument("--features_dir", type=Path, required=True, help="Path to the root features cache directory.")
    parser.add_argument("--feature_name", type=str, default="WhisperEncoderExtractor_mean_row_col_pca_100", help="The feature extractor to evaluate.")
    parser.add_argument("--distance_name", type=str, default="spearman", help="The distance metric used.")
    parser.add_argument("--num_permutations", type=int, default=1000, help="Number of label permutations to create the null distribution.")
    parser.add_argument("--output_csv", type=Path, default="pak_permutation_test_results.csv", help="Path to save detailed per-subset results.")
    args = parser.parse_args()

    logger.info("Loading configuration from: %s", args.config_file)
    with open(args.config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset_cfg = cfg.get("dataset", {})
    subsets_to_run = dataset_cfg.get("subsets_to_run", [])
    dataset_id = dataset_cfg.get("id")

    logger.info("Loading dataset columns 'subset' and 'label'...")
    full_dataset = load_dataset(dataset_id, split=dataset_cfg.get("split", "train")).select_columns(["subset", "label"])

    pak_benchmark = PrecisionAtK(k_values=K_VALUES_TO_TEST)

    all_results = []
    # NEW: Store full distributions for aggregate analysis
    all_distributions = {
        'Public': {f'P@{k}': [] for k in K_VALUES_TO_TEST},
        'Blind': {f'P@{k}': [] for k in K_VALUES_TO_TEST}
    }

    pbar_subsets = tqdm(subsets_to_run, desc="Processing Subsets")
    for subset_key in pbar_subsets:
        pbar_subsets.set_postfix_str(subset_key)
        data_type = 'Blind' if subset_key in BLIND_TEST_SUBSETS else 'Public'

        true_labels = [str(item['label']) for item in full_dataset if item.get('subset') == subset_key and item.get('label') is not None]
        if len(true_labels) < 10:
            logger.warning(f"Skipping subset '{subset_key}': Not enough valid labels ({len(true_labels)}).")
            continue

        subset_features_dir = args.features_dir / subset_key
        dist_mat_path = find_distance_matrix_file(subset_features_dir, args.feature_name, args.distance_name)
        if not dist_mat_path:
            logger.warning(f"Skipping subset '{subset_key}': No distance matrix found.")
            continue
        try:
            with h5py.File(dist_mat_path, 'r') as f:
                dist_mat = f[HDF5_DISTANCE_DATASET_NAME][:]
            if dist_mat.shape[0] != len(true_labels):
                 logger.warning(f"Skipping subset '{subset_key}': Matrix shape {dist_mat.shape} != label count {len(true_labels)}.")
                 continue
        except Exception as e:
            logger.error(f"Failed to load distance matrix for '{subset_key}': {e}")
            continue

        observed_pak_result = pak_benchmark.evaluate(distance_matrix=dist_mat, labels=true_labels)
       
        null_distributions_subset = {k: [] for k in K_VALUES_TO_TEST}
        pbar_perms = tqdm(range(args.num_permutations), desc=f"Permutations for {subset_key}", leave=False)
        for i in pbar_perms:
            rng = np.random.RandomState(seed=i)
            shuffled_labels = rng.permutation(true_labels).tolist()
            perm_result = pak_benchmark.evaluate(distance_matrix=dist_mat, labels=shuffled_labels)
            for k in K_VALUES_TO_TEST:
                score = perm_result.get(f"P@{k}")
                if score is not None:
                    null_distributions_subset[k].append(score)
       
        result_row = {"subset": subset_key}
        for k in K_VALUES_TO_TEST:
            observed_score = observed_pak_result.get(f"P@{k}", np.nan)
            null_dist = null_distributions_subset[k]
           
            if not null_dist:
                logger.error(f"Could not generate null distribution for P@{k} on {subset_key}")
                continue

            null_dist_np = np.array(null_dist)
            all_distributions[data_type][f'P@{k}'].append(null_dist_np) # Store for aggregate CI
           
            result_row.update({
                f"observed_P@{k}": observed_score * 100,
                f"baseline_mean_P@{k}": np.mean(null_dist_np) * 100,
                f"p_value_P@{k}": np.sum(null_dist_np >= observed_score) / len(null_dist_np) if not np.isnan(observed_score) else np.nan
            })
        all_results.append(result_row)
        del dist_mat; gc.collect()

    df = pd.DataFrame(all_results)
    df['type'] = df['subset'].apply(lambda s: 'Blind' if s in BLIND_TEST_SUBSETS else 'Public')
    df.to_csv(args.output_csv, index=False, float_format='%.4f')
    logger.info(f"Detailed per-subset permutation test results saved to: {args.output_csv}")

    # --- Print Aggregated Summary with Aggregate CIs ---
    print("\n" + "="*95)
    print("      EMPIRICAL PERMUTATION TEST SUMMARY FOR PRECISION@K (WITH AGGREGATE 95% CIs)")
    print(f"         (Based on {args.num_permutations} permutations per subset)")
    print("="*95)
    print(f"Feature Extractor: {args.feature_name}")
    print(f"Distance Metric:   {args.distance_name}")
   
    for k in K_VALUES_TO_TEST:
        summary_data = []
        for data_type in ['Public', 'Blind']:
            subset_df = df[df['type'] == data_type]
            if not subset_df.empty:
                observed_mean = subset_df[f'observed_P@{k}'].mean()
               
                perm_arrays = all_distributions[data_type][f'P@{k}']
                if perm_arrays:
                    stacked_perms = np.stack(perm_arrays, axis=0) * 100  
                    mean_perm_scores = np.mean(stacked_perms, axis=0)  
                   
                    baseline_mean = np.mean(mean_perm_scores)
                    ci_95_mean = np.percentile(mean_perm_scores, [2.5, 97.5])
                    p_value_mean = np.sum(mean_perm_scores >= observed_mean) / args.num_permutations
                else:
                    baseline_mean, ci_95_mean, p_value_mean = np.nan, [np.nan, np.nan], np.nan

                summary_data.append({
                    'Type': data_type,
                    f'Observed P@{k} (Mean)': observed_mean,
                    f'Baseline P@{k} (Mean)': baseline_mean,
                    f'Baseline 95% CI (Mean)': f"[{ci_95_mean[0]:.2f}, {ci_95_mean[1]:.2f}]",
                    'Lift over Baseline': observed_mean - baseline_mean,
                    f'p-value (Mean)': f"< 0.001" if p_value_mean < 0.001 else f"{p_value_mean:.3f}"
                })
        
        summary_df = pd.DataFrame(summary_data).set_index('Type')
        print("\n" + "-"*40 + f" P@{k} SUMMARY " + "-"*40)
        print(summary_df.to_string(float_format='%.2f%%'))
    print("="*95)

if __name__ == "__main__":
    main()