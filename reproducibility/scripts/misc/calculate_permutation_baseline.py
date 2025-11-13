# Save this file as: reproducibility/scripts/misc/calculate_permutation_baseline.py

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
    from benchmarks.gsr import GlobalSeparationRate
except ImportError:
    print("ERROR: Could not import benchmark classes. Please run this script from the root of your project.")
    sys.exit(1)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
BLIND_TEST_SUBSETS = ["HU3", "HU4", "HW3", "HW4"]
HDF5_DISTANCE_DATASET_NAME = "distance_matrix"


def find_distance_matrix_file(subset_dir: Path, feature_name: str, distance_name: str) -> Path | None:
    """Finds the most recent distance matrix file matching the criteria by parsing filenames."""
    # This regex is designed to be flexible and match the complex filenames.
    pattern = re.compile(
        rf"distances_{re.escape(distance_name)}.*_{re.escape(feature_name)}_[a-f0-9]+\.h5"
    )
    found_files = [f for f in subset_dir.iterdir() if pattern.match(f.name)] if subset_dir.is_dir() else []
    if found_files:
        return max(found_files, key=lambda p: p.stat().st_mtime)
    logger.warning(f"No distance matrix file found for feature '{feature_name}' and distance '{distance_name}' in {subset_dir}")
    return None


def main():
    """Main function to calculate and report the shuffled baseline."""
    parser = argparse.ArgumentParser(
        description="Run a rigorous permutation test for GSR significance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config_file", type=Path, required=True, help="Path to the main VocSim YAML configuration file.")
    parser.add_argument("--features_dir", type=Path, required=True, help="Path to the root features cache directory.")
    parser.add_argument("--feature_name", type=str, default="WhisperEncoderExtractor_mean_row_col_pca_100", help="The feature extractor to evaluate.")
    parser.add_argument("--distance_name", type=str, default="spearman", help="The distance metric used.")
    parser.add_argument("--num_permutations", type=int, default=1000, help="Number of label permutations to create the null distribution.")
    parser.add_argument("--output_csv", type=Path, default="permutation_test_results.csv", help="Path to save detailed results.")
    args = parser.parse_args()

    logger.info("Loading configuration from: %s", args.config_file)
    with open(args.config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset_cfg = cfg.get("dataset", {})
    subsets_to_run = dataset_cfg.get("subsets_to_run", [])
    dataset_id = dataset_cfg.get("id")

    logger.info("Loading dataset columns 'subset' and 'label' (audio will not be decoded)...")
    full_dataset = load_dataset(dataset_id, split=dataset_cfg.get("split", "train")).select_columns(["subset", "label"])

    gsr_benchmark = GlobalSeparationRate(min_class_size=2)

    all_results = []
    pbar_subsets = tqdm(subsets_to_run, desc="Processing Subsets")
    for subset_key in pbar_subsets:
        pbar_subsets.set_postfix_str(subset_key)

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

        # 1. Calculate the OBSERVED GSR with true labels
        observed_gsr_result = gsr_benchmark.evaluate(distance_matrix=dist_mat, labels=true_labels)
        observed_gsr = observed_gsr_result.get("gsr_score", np.nan)

        # 2. Build the NULL DISTRIBUTION from permutations
        null_distribution = []
        pbar_perms = tqdm(range(args.num_permutations), desc=f"Permutations for {subset_key}", leave=False)
        for i in pbar_perms:
            rng = np.random.RandomState(seed=i)  # Different seed for each shuffle for independence
            shuffled_labels = rng.permutation(true_labels).tolist()
            perm_result = gsr_benchmark.evaluate(distance_matrix=dist_mat, labels=shuffled_labels)
            if perm_result.get("gsr_score") is not None:
                null_distribution.append(perm_result["gsr_score"])
        
        if not null_distribution:
            logger.error(f"Could not generate a null distribution for {subset_key}")
            continue
            
        # 3. Calculate statistics from the null distribution
        null_dist_np = np.array(null_distribution)
        mean_random_gsr = np.mean(null_dist_np)
        ci_95 = np.percentile(null_dist_np, [2.5, 97.5])
        # P-value: the probability of observing a score as high or higher than the true score by chance
        p_value = np.sum(null_dist_np >= observed_gsr) / len(null_dist_np) if not np.isnan(observed_gsr) else np.nan

        all_results.append({
            "subset": subset_key,
            "observed_gsr": observed_gsr * 100,
            "baseline_mean_gsr": mean_random_gsr * 100,
            "baseline_95_ci_low": ci_95[0] * 100,
            "baseline_95_ci_high": ci_95[1] * 100,
            "p_value": p_value
        })

        del dist_mat
        gc.collect()

    if not all_results:
        logger.error("No results were generated.")
        return

    df = pd.DataFrame(all_results)
    df['type'] = df['subset'].apply(lambda s: 'Blind' if s in BLIND_TEST_SUBSETS else 'Public')
    df.to_csv(args.output_csv, index=False, float_format='%.4f')
    logger.info(f"Detailed permutation test results saved to: {args.output_csv}")

    # --- Print Aggregated Summary ---
    summary_data = []
    for data_type in ['Public', 'Blind']:
        subset_df = df[df['type'] == data_type]
        if not subset_df.empty:
            summary_data.append({
                'Type': data_type,
                'Observed GSR (Mean)': subset_df['observed_gsr'].mean(),
                'Baseline GSR (Mean)': subset_df['baseline_mean_gsr'].mean(),
                'Baseline 95% CI (Mean)': f"[{subset_df['baseline_95_ci_low'].mean():.2f}, {subset_df['baseline_95_ci_high'].mean():.2f}]",
                'p < 0.001 (Count)': f"{(subset_df['p_value'] < 0.001).sum()}/{len(subset_df)}"
            })
            
    summary_df = pd.DataFrame(summary_data).set_index('Type')

    print("\n" + "="*80)
    print("PERMUTATION TEST SUMMARY FOR GSR")
    print(f"         (Based on {args.num_permutations} permutations per subset)")
    print("="*80)
    print(f"Feature Extractor: {args.feature_name}")
    print(f"Distance Metric:   {args.distance_name}")
    print("-"*80)
    print(summary_df.to_string(float_format='%.2f%%'))
    print("-"*80)


if __name__ == "__main__":
    main()