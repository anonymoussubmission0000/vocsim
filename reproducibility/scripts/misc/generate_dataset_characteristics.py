import datetime
import gc
import logging
import sys
import os
from pathlib import Path
from typing import Optional
import pandas as pd
import argparse
import re
import numpy as np
from collections import Counter
import json
import torch
import librosa


try:
    script_path = Path(__file__).resolve()
    project_root_candidate1 = script_path.parents[1]
    project_root_candidate2 = script_path.parents[3]

    if (project_root_candidate1 / "vocsim" / "runner.py").exists():
        project_root = project_root_candidate1
    elif (project_root_candidate2 / "vocsim" / "runner.py").exists():
        project_root = project_root_candidate2
    elif (Path.cwd() / "vocsim" / "runner.py").exists():
        project_root = Path.cwd()
        print(f"WARNING: Script path resolution heuristic failed. Assuming CWD is project root: {project_root}")
    else:
        project_root = Path.cwd()
        print(f"ERROR: Cannot determine project root relative to script location. runner.py not found. Assuming CWD: {project_root}")
        print("Please ensure this script is run from within the project or adjust path setup if it's moved.")

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"INFO: Added project root to sys.path: {project_root}")
except NameError:
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    print(f"INFO: Assuming CWD project root for interactive session: {project_root}")

try:
    from vocsim.managers.dataset_manager import DatasetManager
    from utils.config_loader import load_config
    from utils.logging_utils import setup_logging
except ImportError as e:
    print(f"ERROR: Could not import project modules: {e}. Ensure an __init__.py exists in 'vocsim' and 'utils' directories and that the project root ({project_root}) is correctly in PYTHONPATH.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BLIND_TEST_SUBSETS = ["HU3", "HU4", "HW3", "HW4"]


class NpEncoder(json.JSONEncoder):
    """Helper class for JSON encoding NumPy types, Path, Tensors etc."""

    def default(self, obj):
        """
        Encodes various object types, including NumPy arrays, scalars, Path objects,
        and PyTorch Tensors, into JSON-serializable formats.

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
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def sanitize_latex_str(text: str) -> str:
    """
    Sanitizes a string for LaTeX output by escaping special characters.

    Args:
        text (str): The input string.

    Returns:
        str: The sanitized string.
    """
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\^{}",
        "\\": r"\textbackslash{}",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def df_to_latex_custom(df: pd.DataFrame, caption: str, label: str, column_format: str = None) -> str:
    """
    Converts a Pandas DataFrame to a LaTeX table string with sanitization.

    Args:
        df (pd.DataFrame): The input DataFrame.
        caption (str): The caption for the LaTeX table.
        label (str): The label for the LaTeX table.
        column_format (str, optional): LaTeX column format string. Defaults to None,
                                        in which case pandas infers it.

    Returns:
        str: The LaTeX table string.
    """
    latex_parts = []
    df_sanitized = df.copy()
    df_sanitized.columns = [sanitize_latex_str(col) for col in df_sanitized.columns]
    if df_sanitized.index.name:
        df_sanitized.index.name = sanitize_latex_str(df_sanitized.index.name)

    for col in df_sanitized.columns:
        df_sanitized[col] = df_sanitized[col].apply(lambda x: sanitize_latex_str(str(x)) if pd.notna(x) else "-")

    if not isinstance(df_sanitized.index, pd.RangeIndex):
        df_sanitized.index = [sanitize_latex_str(str(idx_val)) for idx_val in df_sanitized.index]

    latex_parts.append("\\begin{table}[htp!]")
    latex_parts.append("\\centering")
    latex_parts.append(f"\\caption{{{sanitize_latex_str(caption)}}}")
    latex_parts.append(f"\\label{{{sanitize_latex_str(label)}}}")
    latex_table_body = df_sanitized.to_latex(escape=False, index=True, na_rep="-", column_format=column_format)
    latex_parts.append(latex_table_body)
    latex_parts.append("\\end{table}")
    return "\n".join(latex_parts)


def calculate_snr_energy_based(
    audio_array: np.ndarray,
    sample_rate: int,
    frame_length_ms: float = 30,
    hop_length_ms: float = 15,
    signal_energy_percentile: float = 80,
    noise_energy_percentile: float = 20,
    min_frames_for_estimation: int = 10,
    silence_threshold_rms: float = 1e-4,
) -> float:
    """
    Estimates SNR based on energy percentiles of audio frames.

    Args:
        audio_array (np.ndarray): 1D NumPy array containing the audio waveform.
        sample_rate (int): Sample rate of the audio.
        frame_length_ms (float): Length of each frame in milliseconds.
        hop_length_ms (float): Hop length between frames in milliseconds.
        signal_energy_percentile (float): Percentile used to define signal frames.
        noise_energy_percentile (float): Percentile used to define noise frames.
        min_frames_for_estimation (int): Minimum number of frames needed for estimation.
        silence_threshold_rms (float): RMS threshold to consider audio silent.

    Returns:
        float: Estimated SNR in dB, or np.nan if calculation is not robustly possible.
    """
    local_logger = logging.getLogger(__name__ + ".snr_calc")

    if not isinstance(audio_array, np.ndarray) or audio_array.ndim != 1:
        local_logger.debug("SNR: Input audio must be a 1D NumPy array.")
        return np.nan
    if audio_array.size == 0:
        local_logger.debug("SNR: Input audio array is empty.")
        return np.nan
    if sample_rate <= 0:
        local_logger.debug("SNR: Invalid sample rate.")
        return np.nan

    try:
        audio_float32 = audio_array.astype(np.float32)
        if np.max(np.abs(audio_float32)) < silence_threshold_rms:
            local_logger.debug("SNR: Audio is likely silent based on overall RMS. Returning 0 dB.")
            return 0.0

        frame_length = int(frame_length_ms / 1000 * sample_rate)
        hop_length = int(hop_length_ms / 1000 * sample_rate)

        if frame_length <= 0 or hop_length <= 0 or frame_length > audio_float32.size:
            if audio_float32.size > 0:
                power = np.mean(audio_float32**2)
                return 10 * np.log10(power / 1e-12) if power > 1e-10 else 0.0
            return np.nan

        rms_energy = librosa.feature.rms(y=audio_float32, frame_length=frame_length, hop_length=hop_length)[0]

        if rms_energy.size < min_frames_for_estimation:
            local_logger.debug(f"SNR: Not enough frames ({rms_energy.size}) for robust estimation. Min required: {min_frames_for_estimation}.")
            if rms_energy.size > 0:
                mean_power = np.mean(rms_energy**2)
                return 10 * np.log10(mean_power / (mean_power * 0.01 + 1e-12)) if mean_power > 1e-10 else 0.0
            return np.nan

        energy_values_squared = rms_energy**2

        power_threshold_signal = np.percentile(energy_values_squared, signal_energy_percentile)
        power_threshold_noise = np.percentile(energy_values_squared, noise_energy_percentile)

        if power_threshold_signal <= power_threshold_noise:
            if np.isclose(power_threshold_signal, power_threshold_noise):
                power_threshold_noise = power_threshold_signal * 0.9
                power_threshold_signal = power_threshold_signal * 1.1
                if power_threshold_signal < 1e-10:
                    power_threshold_signal = 1e-10
            else:
                local_logger.debug(
                    "SNR: Signal power threshold (%.2e) <= noise threshold (%.2e)." " Assuming low SNR or constant audio.",
                    power_threshold_signal,
                    power_threshold_noise,
                )
                return 0.0

        signal_powers = energy_values_squared[energy_values_squared >= power_threshold_signal]
        noise_powers = energy_values_squared[energy_values_squared <= power_threshold_noise]

        if len(signal_powers) < (min_frames_for_estimation // 2) or len(noise_powers) < (min_frames_for_estimation // 2):
            local_logger.debug(
                "SNR: Insufficient distinct signal (%d) or noise (%d) frames after thresholding.",
                len(signal_powers),
                len(noise_powers),
            )
            if len(signal_powers) < (min_frames_for_estimation // 2) and len(energy_values_squared) > 0:
                signal_powers = energy_values_squared
            if len(noise_powers) < (min_frames_for_estimation // 2) and len(energy_values_squared) > 0:
                noise_powers = np.array([np.percentile(energy_values_squared, 10)])

        if len(signal_powers) == 0 and len(noise_powers) == 0:
            return np.nan
        if len(signal_powers) == 0:
            return 0.0

        avg_signal_power = np.mean(signal_powers) if len(signal_powers) > 0 else 0.0
        avg_noise_power = np.mean(noise_powers) if len(noise_powers) > 0 else 1e-12

        if avg_noise_power < 1e-12:
            avg_noise_power = 1e-12
        if avg_signal_power < 1e-12:
            return 0.0

        snr_ratio = avg_signal_power / avg_noise_power
        snr_db = 10 * np.log10(snr_ratio)

        return max(0, snr_db)

    except Exception as e:
        local_logger.error("SNR calculation error: %s", e, exc_info=False)
        return np.nan


def calculate_dataset_characteristics(config_path: Path, base_config_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Calculates characteristics for datasets specified in the VocSim configuration.

    Processes each dataset subset listed in the config, calculates metrics
    like number of samples, classes, samples per class, average duration,
    availability (based on blind test status), and estimated SNR.

    Args:
        config_path (Path): Path to the main configuration file.
        base_config_path (Optional[Path]): Path to the base configuration file.

    Returns:
        pd.DataFrame: A DataFrame where the index is the subset ID and columns
                      are the calculated characteristics.
    """
    cfg = load_config(config_path, base_config_path=base_config_path)
    if not cfg:
        logger.error("Failed to load configuration.")
        return pd.DataFrame()

    dataset_manager = DatasetManager(cfg)
    if not dataset_manager.load_full_dataset():
        logger.error("Failed to load the full dataset via DatasetManager.")
        return pd.DataFrame()

    subsets_to_run_config = cfg.get("dataset", {}).get("subsets_to_run")
    if subsets_to_run_config is None:
        top_level_subset = cfg.get("dataset", {}).get("subset")
        subsets_to_run = [top_level_subset] if top_level_subset else ["all"]
        if subsets_to_run == [None]:
            subsets_to_run = ["all"]
    elif isinstance(subsets_to_run_config, str):
        subsets_to_run = [subsets_to_run_config]
    elif isinstance(subsets_to_run_config, list):
        subsets_to_run = subsets_to_run_config
    else:
        subsets_to_run = ["all"]

    all_characteristics_records = []
    default_label_key = cfg.get("dataset", {}).get("default_label_column", "label")
    logger.info("Will use '%s' for class label extraction.", default_label_key)

    for subset_key in subsets_to_run:
        logger.info("Processing subset: %s...", subset_key)
        subset_info = dataset_manager.get_subset_dataset(subset_key)
        if subset_info is None:
            logger.warning("Could not load subset '%s'. Skipping.", subset_key)
            continue

        subset_dataset_obj, _ = subset_info

        num_samples_in_subset = 0
        all_labels_in_subset = []
        all_durations_in_subset = []
        all_snr_values_in_subset = []

        logger.info("Iterating through items in subset '%s' for characteristic calculation...", subset_key)

        for item_idx, item_data in enumerate(subset_dataset_obj):
            num_samples_in_subset += 1
            label_value = item_data.get(default_label_key)
            all_labels_in_subset.append(str(label_value) if label_value is not None else None)

            audio_array_np = None
            sample_rate = None
            current_duration = np.nan
            current_snr = np.nan

            audio_info = item_data.get("audio")
            if audio_info and "array" in audio_info and "sampling_rate" in audio_info:
                audio_array_orig = audio_info["array"]
                sample_rate = audio_info["sampling_rate"]

                if audio_array_orig is not None and sample_rate is not None and sample_rate > 0:
                    if isinstance(audio_array_orig, torch.Tensor):
                        audio_array_np = audio_array_orig.cpu().numpy().astype(np.float32)
                    elif isinstance(audio_array_orig, np.ndarray):
                        audio_array_np = audio_array_orig.astype(np.float32)
                    elif isinstance(audio_array_orig, list):
                        audio_array_np = np.array(audio_array_orig, dtype=np.float32)

                    if audio_array_np is not None:
                        if audio_array_np.ndim > 1:
                            audio_array_np = np.mean(audio_array_np, axis=0) if audio_array_np.shape[0] < audio_array_np.shape[1] else np.mean(audio_array_np, axis=1)

                        if audio_array_np.ndim == 1 and audio_array_np.size > 0:
                            current_duration = len(audio_array_np) / sample_rate
                            current_snr = calculate_snr_energy_based(audio_array_np, sample_rate)

            all_durations_in_subset.append(current_duration)
            all_snr_values_in_subset.append(current_snr)

            if (item_idx + 1) % 1000 == 0:
                logger.info("  Processed %d items in '%s'...", item_idx + 1, subset_key)

        logger.info(f"Finished item iteration for subset '{subset_key}'. Total items processed: {num_samples_in_subset}")

        num_classes = 0
        sam_per_cls_avg_range_str = "N/A"
        if all_labels_in_subset:
            valid_labels = [l for l in all_labels_in_subset if l is not None]
            if valid_labels:
                label_counts = Counter(valid_labels)
                num_classes = len(label_counts)
                if num_classes > 0:
                    counts_per_class_values = list(label_counts.values())
                    avg_sam_per_cls_val = sum(counts_per_class_values) / num_classes
                    min_c = min(counts_per_class_values)
                    max_c = max(counts_per_class_values)
                    sam_per_cls_avg_range_str = f"{avg_sam_per_cls_val:.1f} ({min_c}-{max_c})"

        avg_dur_s_min_max_str = "N/A"
        avg_dur_for_sort = np.nan
        valid_durations = [d for d in all_durations_in_subset if pd.notna(d)]
        if valid_durations:
            avg_dur = np.mean(valid_durations)
            min_dur = np.min(valid_durations)
            max_dur = np.max(valid_durations)
            avg_dur_s_min_max_str = f"{avg_dur:.2f} ({min_dur:.2f}-{max_dur:.2f})"
            avg_dur_for_sort = avg_dur

        snr_db_val_str = "N/A"
        valid_snr_values = [s for s in all_snr_values_in_subset if pd.notna(s) and np.isfinite(s)]
        if valid_snr_values:
            avg_snr_db = np.mean(valid_snr_values)
            snr_db_val_str = f"{avg_snr_db:.0f}"

        avail_char = "X" if subset_key in BLIND_TEST_SUBSETS else "\u2713"

        all_characteristics_records.append({
            "ID": subset_key,
            "N. Samples": num_samples_in_subset,
            "Classes": num_classes,
            "Sam/Cls (avg, range)": sam_per_cls_avg_range_str,
            "Avg. Dur (s) (min-max)": avg_dur_s_min_max_str,
            "Avail": avail_char,
            "SNR (dB)": snr_db_val_str,
            "_sort_avg_dur": avg_dur_for_sort,
        })
        logger.info("Aggregated characteristics for subset: %s", subset_key)
        del subset_dataset_obj, all_labels_in_subset, all_durations_in_subset, all_snr_values_in_subset
        gc.collect()

    df = pd.DataFrame(all_characteristics_records)
    ordered_columns = ["ID", "N. Samples", "Classes", "Sam/Cls (avg, range)", "Avg. Dur (s) (min-max)", "Avail", "SNR (dB)"]
    if "_sort_avg_dur" in df.columns:
        df = df[["ID", "_sort_avg_dur"] + ordered_columns[1:]]

    df = df.set_index("ID")
    return df


def main():
    """
    Main function to calculate and save dataset characteristics to CSV and LaTeX files.
    """
    parser = argparse.ArgumentParser(description="Generate dataset characteristics table from VocSim configuration.")
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the main VocSim YAML configuration file (e.g., reproducibility/configs/vocsim_paper.yaml)",
    )
    parser.add_argument(
        "--base_config_file",
        type=str,
        default=None,
        help="Optional path to a base YAML configuration file (e.g., configs/base.yaml)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_dataset_tables",
        help="Directory to save the output CSV and LaTeX files (relative to project root or absolute).",
    )
    args = parser.parse_args()

    config_p = Path(args.config_file)
    base_config_p = Path(args.base_config_file) if args.base_config_file else None

    output_dir_arg = Path(args.output_dir)
    output_p = output_dir_arg.resolve() if output_dir_arg.is_absolute() else (project_root / output_dir_arg).resolve()
    output_p.mkdir(parents=True, exist_ok=True)

    logger.info("Using main config file: %s", config_p)
    if base_config_p and base_config_p.exists():
        logger.info("Using base config file: %s", base_config_p)
    elif base_config_p:
        logger.warning("Base config file specified but not found: %s", base_config_p)
        base_config_p = None

    characteristics_df = calculate_dataset_characteristics(config_p, base_config_p)

    if characteristics_df.empty:
        logger.error("No characteristics data generated. Exiting.")
        return

    final_display_columns = ["N. Samples", "Classes", "Sam/Cls (avg, range)", "Avg. Dur (s) (min-max)", "Avail", "SNR (dB)"]
    try:
        if "_sort_avg_dur" in characteristics_df.columns:
            characteristics_df_sorted = characteristics_df.sort_values(by="_sort_avg_dur", ascending=True, na_position="last")
            characteristics_df_final = characteristics_df_sorted[final_display_columns]
        else:
            logger.warning("'_sort_avg_dur' column not found. Cannot sort by average duration. Using original order.")
            characteristics_df_final = characteristics_df[final_display_columns]
    except KeyError as e:
        logger.error("KeyError during sorting or column selection: %s. Using unsorted DataFrame with available columns.", e)
        available_cols = [col for col in final_display_columns if col in characteristics_df.columns]
        characteristics_df_final = characteristics_df[available_cols]
    except Exception as e:
        logger.warning("Could not sort by average duration: %s. Using unsorted DataFrame.", e)
        available_cols = [col for col in final_display_columns if col in characteristics_df.columns]
        characteristics_df_final = characteristics_df[available_cols]

    csv_filename = output_p / "dataset_characteristics_calculated.csv"
    try:
        characteristics_df_final.to_csv(csv_filename, index=True, encoding="utf-8")
        logger.info("Dataset characteristics saved to CSV: %s", csv_filename)
    except Exception as e:
        logger.error("Failed to save CSV: %s", e)

    latex_filename = output_p / "dataset_characteristics_calculated.tex"
    column_format_str = "l" + "c" * len(characteristics_df_final.columns)
    latex_caption = "Calculated Characteristics of Datasets Processed by VocSim"
    latex_label = "tab:dataset_characteristics_calculated"

    try:
        latex_string = df_to_latex_custom(characteristics_df_final, latex_caption, latex_label, column_format_str)
        with open(latex_filename, "w", encoding="utf-8") as f:
            f.write(latex_string)
        logger.info("Dataset characteristics saved to LaTeX: %s", latex_filename)
    except Exception as e:
        logger.error("Failed to save LaTeX: %s", e)
        try:
            if "latex_string" not in locals():
                latex_string = characteristics_df_final.to_latex(escape=True, index=True, na_rep="-", column_format=column_format_str)
            print("\n--- LaTeX Fallback Output ---\n" + latex_string + "\n--- End LaTeX Fallback ---")
        except Exception as e_print:
            logger.error("Failed to even print LaTeX to console: %s", e_print)


if __name__ == "__main__":
    main()