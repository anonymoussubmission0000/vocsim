# Save this file as: reproducibility/scripts/misc/calculate_shuffled_baseline.py

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
    # Assumes the script is in reproducibility/scripts/misc/
    project_root = script_path.parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except (NameError, IndexError):
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    print(f"WARNING: Assuming CWD is project root: {project_root}")

# --- Import your project's benchmark classes ---
try:
    from benchmarks.gsr import GlobalSeparationRate
    from benchmarks.precision import PrecisionAtK
except ImportError as e:
    print(f"ERROR: Could not import benchmark classes: {e}")
    print("Please ensure you run this script from the root of your project.")
    sys.exit(1)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
BLIND_TEST_SUBSETS = ["HU3", "HU4", "HW3", "HW4"]
HDF5_DISTANCE_DATASET_NAME = "distance_matrix"


def find_distance_matrix_file(subset_dir: Path, feature_name: str, distance_name: str) -> Path | None:
    """Finds the most recent distance matrix file matching the criteria by parsing filenames."""
    pattern = re.compile(
        rf"distances_{re.escape(distance_name)}.*_{re.escape(feature_name)}_[a-f0-9]+\.h5"
    )
    found_files = [f for f in subset_dir.iterdir() if pattern.match(f.name)] if subset_dir.is_dir() else []
    return max(found_files, key=lambda p: p.stat().st_mtime) if found_files else None


def main():
    """Main function to calculate and report the shuffled baseline."""
    parser = argparse.ArgumentParser(
        description="Calculate GSR and P@k baselines using shuffled labels for a given feature extractor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # ... (rest of the arguments are the same)
    parser.add_argument(
        "--num_shuffles", type=int, default=10,
        help="Number of times to shuffle labels and re-calculate metrics to get a mean and std dev."
    )
    # ... (add back the other arguments)
    parser.add_argument(
        "--config_file", type=Path, required=True,
        help="Path to the main VocSim YAML configuration file (e.g., reproducibility/configs/vocsim_paper.yaml)."
    )
    parser.add_argument(
        "--features_dir", type=Path, required=True,
        help="Path to the root features cache directory containing subset folders with distance matrices."
    )
    parser.add_argument(
        "--feature_name", type=str, default="WhisperEncoderExtractor_mean_row_col_pca_100",
        help="The exact name of the feature extractor to evaluate."
    )
    parser.add_argument(
        "--distance_name", type=str, default="spearman",
        help="The distance metric used to generate the matrices (e.g., 'spearman', 'cosine')."
    )
    parser.add_argument(
        "--output_csv", type=Path, default="shuffled_baseline_results.csv",
        help="Path to save the detailed per-subset results CSV file."
    )
    args = parser.parse_args()

    logger.info("Loading configuration from: %s", args.config_file)
    with open(args.config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset_cfg = cfg.get("dataset", {})
    subsets_to_run = dataset_cfg.get("subsets_to_run", [])
    dataset_id = dataset_cfg.get("id")

    if not subsets_to_run or not dataset_id:
        logger.error("Config must contain 'dataset.subsets_to_run' and 'dataset.id'.")
        return

    logger.info("Loading dataset columns 'subset' and 'label'...")
    full_dataset = load_dataset(dataset_id, split=dataset_cfg.get("split", "train")).select_columns(["subset", "label"])

    gsr_benchmark = GlobalSeparationRate(min_class_size=2)
    pk_benchmark = PrecisionAtK(k_values=[1, 5])

    all_results = []
    # --- NEW: Outer loop for multiple shuffles ---
    pbar_shuffles = tqdm(range(args.num_shuffles), desc="Shuffle Iterations")
    for shuffle_i in pbar_shuffles:
        pbar_subsets = tqdm(subsets_to_run, desc=f"Shuffle {shuffle_i+1}/{args.num_shuffles}", leave=False)
        for subset_key in pbar_subsets:
            pbar_subsets.set_postfix_str(subset_key)

            true_labels = [str(item['label']) for item in full_dataset if item.get('subset') == subset_key and item.get('label') is not None]

            if len(true_labels) < 2:
                continue

            # Use the shuffle iteration as the seed for reproducibility
            rng = np.random.RandomState(seed=shuffle_i)
            shuffled_labels = rng.permutation(true_labels).tolist()

            subset_features_dir = args.features_dir / subset_key
            dist_mat_path = find_distance_matrix_file(subset_features_dir, args.feature_name, args.distance_name)

            if not dist_mat_path:
                continue

            try:
                with h5py.File(dist_mat_path, 'r') as f:
                    dist_mat = f[HDF5_DISTANCE_DATASET_NAME][:]
                if dist_mat.shape[0] != len(true_labels):
                    continue
            except Exception as e:
                logger.error(f"Failed to load distance matrix for '{subset_key}': {e}")
                continue

            gsr_result = gsr_benchmark.evaluate(distance_matrix=dist_mat, labels=shuffled_labels)
            pk_result = pk_benchmark.evaluate(distance_matrix=dist_mat, labels=shuffled_labels)

            all_results.append({
                "shuffle_iteration": shuffle_i,
                "subset": subset_key,
                "gsr_score": gsr_result.get("gsr_score"),
                "P@1": pk_result.get("P@1"),
            })

            del dist_mat
            gc.collect()

    if not all_results:
        logger.error("No results were generated. Please check your paths and configuration.")
        return

    df = pd.DataFrame(all_results)
    df['type'] = df['subset'].apply(lambda s: 'Blind' if s in BLIND_TEST_SUBSETS else 'Public')
    df['gsr_score'] = pd.to_numeric(df['gsr_score'], errors='coerce') * 100
    df['P@1'] = pd.to_numeric(df['P@1'], errors='coerce') * 100

    df.to_csv(args.output_csv, index=False, float_format='%.2f')
    logger.info(f"Detailed results for {args.num_shuffles} shuffles saved to: {args.output_csv}")

    # --- NEW: Aggregation with mean and std ---
    summary = df.groupby('type').agg(
        gsr_mean=('gsr_score', 'mean'),
        gsr_std=('gsr_score', 'std'),
        p1_mean=('P@1', 'mean'),
        p1_std=('P@1', 'std'),
    )

    print("\n" + "="*70)
    print("           EMPIRICAL SHUFFLED BASELINE RESULTS")
    print(f"         (Aggregated over {args.num_shuffles} reproducible shuffles)")
    print("="*70)
    print(f"Feature Extractor: {args.feature_name}")
    print(f"Distance Metric:   {args.distance_name}")
    print("-"*70)
    # Format the summary for printing
    summary_str = summary.to_string(
        formatters={
            'gsr_mean': '{:,.2f}%'.format,
            'gsr_std': '({:,.2f})'.format,
            'p1_mean': '{:,.2f}%'.format,
            'p1_std': '({:,.2f})'.format,
        },
        header=["GSR Mean", "GSR Std", "P@1 Mean", "P@1 Std"]
    )
    print(summary_str)
    print("-"*70)
    print("NOTE: You can now report these mean ± std values in your paper.")

if __name__ == "__main__":
    main()