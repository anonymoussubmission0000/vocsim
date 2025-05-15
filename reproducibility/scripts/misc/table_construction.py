# -*- coding: utf-8 -*-
import argparse
import json
import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from pandas import DataFrame, MultiIndex
import ast
import io
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import librosa


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURE_CONFIGS: Dict[str, Dict[str, Any]] = {}
FEATURE_SHORT_NAMES: Dict[str, str] = {}
PRETTY_NAMES: Dict[str, str] = {
    "P@1": "P@1",
    "P@5": "P@5",
    "pairwise_f_value": "CS",
    "pccf": "PCCF",
    "weighted_purity": "Purity (W)",
    "num_clusters_found": "Clusters",
    "csr_score": "CSR",
    "accuracy_mean": "Accuracy",
    "top_5_accuracy_mean": "Top-5 Acc.",
    "cosine": "C",
    "euclidean": "E",
    "spearman": "S",
    "all": "Overall",
    "avian_perception": "Avian Perc.",
    "mouse_strain": "Mouse Strain",
    "mouse_identity": "Mouse ID",
    "BS1": "BS1",
    "BS2": "BS2",
    "BS3": "BS3",
    "BS4": "BS4",
    "BS5": "BS5",
    "BC": "BC",
    "ES1": "ES1",
    "HP": "HP",
    "HS1": "HS1",
    "HS2": "HS2",
    "HU1": "HU1",
    "HU2": "HU2",
    "HU3": "HU3",
    "HU4": "HU4",
    "HW1": "HW1",
    "HW2": "HW2",
    "HW3": "HW3",
    "HW4": "HW4",
    "OC1": "OC1",
}

VOCSIM_APPENDIX_S_COLUMN_ORDER = [
    "BC",
    "BS1",
    "BS2",
    "BS3",
    "BS4",
    "BS5",
    "ES1",
    "HP",
    "HS1",
    "HS2",
    "HU1",
    "HU2",
    "HU3",
    "HU4",
    "HW1",
    "HW2",
    "HW3",
    "HW4",
    "OC1",
    "Avg",
    "Avg (Blind)",
]

BLIND_TEST_SUBSETS = ["HU3", "HU4", "HW3", "HW4"]
GOFFINET_FEATURES = OrderedDict(
    [
        ("Spectrogram D=10", "Spectrogram D=10*"),
        ("Spectrogram D=30", "Spectrogram D=30*"),
        ("Spectrogram D=100", "Spectrogram D=100*"),
        ("MUPET D=9", "MUPET D=9*"),
        ("DeepSqueak D=10", "DeepSqueak D=10*"),
        ("Latent D=7", "Latent D=7*"),
        ("Latent D=8", "Latent D=8*"),
    ]
)
GOFFINET_STRAIN_DATA = {
    "k-NN (k=3)": {"Spectrogram D=10": "68.1 (0.2)", "Spectrogram D=30": "76.4 (0.3)", "Spectrogram D=100": "82.3 (0.5)", "MUPET D=9": "86.1 (0.2)", "DeepSqueak D=10": "79.0 (0.3)", "Latent D=7": "89.8 (0.2)"},
    "k-NN (k=10)": {"Spectrogram D=10": "71.0 (0.3)", "Spectrogram D=30": "78.2 (0.1)", "Spectrogram D=100": "82.7 (0.6)", "MUPET D=9": "87.0 (0.1)", "DeepSqueak D=10": "80.7 (0.3)", "Latent D=7": "90.7 (0.4)"},
    "k-NN (k=30)": {"Spectrogram D=10": "72.8 (0.3)", "Spectrogram D=30": "78.5 (0.2)", "Spectrogram D=100": "81.3 (0.5)", "MUPET D=9": "86.8 (0.2)", "DeepSqueak D=10": "81.0 (0.2)", "Latent D=7": "90.3 (0.4)"},
    "RF (depth=10)": {"Spectrogram D=10": "72.8 (0.2)", "Spectrogram D=30": "76.6 (0.2)", "Spectrogram D=100": "79.1 (0.3)", "MUPET D=9": "87.4 (0.5)", "DeepSqueak D=10": "81.2 (0.4)", "Latent D=7": "88.1 (0.5)"},
    "RF (depth=15)": {"Spectrogram D=10": "73.1 (0.3)", "Spectrogram D=30": "78.0 (0.3)", "Spectrogram D=100": "80.5 (0.2)", "MUPET D=9": "87.9 (0.4)", "DeepSqueak D=10": "82.1 (0.3)", "Latent D=7": "89.6 (0.4)"},
    "RF (depth=20)": {"Spectrogram D=10": "73.2 (0.2)", "Spectrogram D=30": "78.3 (0.2)", "Spectrogram D=100": "80.7 (0.3)", "MUPET D=9": "87.9 (0.4)", "DeepSqueak D=10": "81.9 (0.3)", "Latent D=7": "89.6 (0.4)"},
    "MLP (α=0.1)": {"Spectrogram D=10": "72.4 (0.3)", "Spectrogram D=30": "79.1 (0.4)", "Spectrogram D=100": "84.5 (0.3)", "MUPET D=9": "87.8 (0.2)", "DeepSqueak D=10": "82.1 (0.4)", "Latent D=7": "90.1 (0.3)"},
    "MLP (α=0.01)": {"Spectrogram D=10": "72.3 (0.4)", "Spectrogram D=30": "78.6 (0.3)", "Spectrogram D=100": "82.9 (0.4)", "MUPET D=9": "88.1 (0.3)", "DeepSqueak D=10": "82.4 (0.4)", "Latent D=7": "90.0 (0.4)"},
    "MLP (α=0.001)": {"Spectrogram D=10": "72.4 (0.4)", "Spectrogram D=30": "78.5 (0.8)", "Spectrogram D=100": "82.8 (0.1)", "MUPET D=9": "87.9 (0.2)", "DeepSqueak D=10": "81.0 (0.2)", "Latent D=7": "90.4 (0.3)"},
}
GOFFINET_IDENTITY_DATA = {
    "Top-1 accuracy": {
        "MLP (α=0.01)": {"Spectrogram D=10": "9.9 (0.2)", "Spectrogram D=30": "14.9 (0.2)", "Spectrogram D=100": "20.4 (0.4)", "MUPET D=9": "14.7 (0.2)", "Latent D=8": "17.0 (0.3)"},
        "MLP (α=0.001)": {"Spectrogram D=10": "10.8 (0.1)", "Spectrogram D=30": "17.3 (0.4)", "Spectrogram D=100": "25.3 (0.3)", "MUPET D=9": "19.0 (0.3)", "Latent D=8": "22.7 (0.5)"},
        "MLP (α=0.0001)": {"Spectrogram D=10": "10.7 (0.2)", "Spectrogram D=30": "17.3 (0.3)", "Spectrogram D=100": "25.1 (0.3)", "MUPET D=9": "20.6 (0.4)", "Latent D=8": "24.0 (0.2)"},
    },
    "Top-5 accuracy": {
        "MLP (α=0.01)": {"Spectrogram D=10": "36.6 (0.4)", "Spectrogram D=30": "45.1 (0.5)", "Spectrogram D=100": "55.0 (0.3)", "MUPET D=9": "46.5 (0.3)", "Latent D=8": "49.9 (0.4)"},
        "MLP (α=0.001)": {"Spectrogram D=10": "38.6 (0.2)", "Spectrogram D=30": "50.7 (0.6)", "Spectrogram D=100": "62.9 (0.4)", "MUPET D=9": "54.0 (0.2)", "Latent D=8": "59.2 (0.6)"},
        "MLP (α=0.0001)": {"Spectrogram D=10": "38.7 (0.5)", "Spectrogram D=30": "50.8 (0.3)", "Spectrogram D=100": "63.2 (0.4)", "MUPET D=9": "57.3 (0.4)", "Latent D=8": "61.6 (0.4)"},
    },
}
ZANDBERG_RESULTS = {
    "EMB-LUA (Zandberg et al.)": 0.727,
    "Luscinia-U (Zandberg et al.)": 0.698,
    "Luscinia (Zandberg et al.)": 0.66,
    "SAP (Zandberg et al.)": 0.64,
    "Raven (Zandberg et al.)": 0.57,
}
CLASSIFIERS = OrderedDict([
    ("k-NN", OrderedDict([("k=3", {"type_match": "knn", "params_to_match": {"n_neighbors": 3}}), ("k=10", {"type_match": "knn", "params_to_match": {"n_neighbors": 10}}), ("k=30", {"type_match": "knn", "params_to_match": {"n_neighbors": 30}})])),
    ("RF", OrderedDict([("depth=10", {"type_match": "rf", "params_to_match": {"max_depth": 10, "class_weight": "balanced"}}), ("depth=15", {"type_match": "rf", "params_to_match": {"max_depth": 15, "class_weight": "balanced"}}), ("depth=20", {"type_match": "rf", "params_to_match": {"max_depth": 20, "class_weight": "balanced"}})])),
    ("MLP", OrderedDict([("α=0.1", {"type_match": "mlp", "params_to_match": {"alpha": 0.1}}), ("α=0.01", {"type_match": "mlp", "params_to_match": {"alpha": 0.01}}), ("α=0.001", {"type_match": "mlp", "params_to_match": {"alpha": 0.001}})])),
])
MLP_CONFIGS = OrderedDict([("MLP (α=0.01)", {"alpha": 0.01}), ("MLP (α=0.001)", {"alpha": 0.001}), ("MLP (α=0.0001)", {"alpha": 0.0001})])


def parse_value_for_comparison(value: Any) -> float:
    """
    Parses a string value potentially containing LaTeX bolding or parentheses
    into a float for numerical comparison.

    Args:
        value (Any): The input value, typically a string like "12.3 (0.1)".

    Returns:
        float: The parsed numerical value, or np.nan if parsing fails.
    """
    if pd.isna(value) or not isinstance(value, str) or value == "-":
        return np.nan
    cleaned = value.replace("\\textbf{", "").replace("}", "").replace("%", "").strip()
    match = re.match(r"^\s*(-?(?:\d+\.\d+|\d+))", cleaned)
    return float(match.group(1)) if match else np.nan


def bold_string(value: Any) -> str:
    """
    Wraps a string value in LaTeX bold command if it's not already bolded or NaN.

    Args:
        value (Any): The input value.

    Returns:
        str: The bolded LaTeX string, or the original string if NaN or already bolded.
    """
    str_value = str(value)
    if pd.isna(value) or str_value == "-":
        return str_value
    if str_value.startswith("\\textbf{") and str_value.endswith("}"):
        return str_value
    return f"\\textbf{{{str_value}}}"


def bold_best_in_columns(df: DataFrame, columns: List[str], higher_is_better: Dict[str, bool]) -> DataFrame:
    """
    Bolds the best values in specified columns of a DataFrame.

    Args:
        df (DataFrame): The input DataFrame.
        columns (List[str]): List of column names to process.
        higher_is_better (Dict[str, bool]): Dictionary mapping column names to boolean
                                           indicating if higher values are better.

    Returns:
        DataFrame: A new DataFrame with the best values in the specified columns bolded.
    """
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            logger.debug("Column '%s' not found for bolding.", col)
            continue
        is_higher = higher_is_better.get(col, True)
        numeric_values = result[col].apply(parse_value_for_comparison)
        valid_numeric_values = numeric_values.dropna()
        if valid_numeric_values.empty:
            continue
        best_numeric = valid_numeric_values.max() if is_higher else valid_numeric_values.min()
        for idx in result.index:
            original_val = result.loc[idx, col]
            numeric_val = numeric_values.loc[idx]
            if pd.notna(numeric_val) and np.isclose(numeric_val, best_numeric):
                result.loc[idx, col] = bold_string(original_val)
    return result


def bold_overall_best_in_group_df(df: pd.DataFrame, columns_to_consider: List[Any], higher_is_better: bool, N_best: int = 1) -> pd.DataFrame:
    """
    Bolds the top N_best values across specified columns in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_to_consider (List[Any]): List of column identifiers (can be tuples for MultiIndex)
                                         to consider for finding the best values.
        higher_is_better (bool): True if higher values are better, False otherwise.
        N_best (int): Number of top values to bold.

    Returns:
        pd.DataFrame: A new DataFrame with the top values bolded.
    """
    df_out = df.copy()
    if not columns_to_consider:
        return df_out
    all_values_with_loc = []
    for r_idx in df_out.index:
        for c_idx_or_tuple in columns_to_consider:
            if c_idx_or_tuple not in df_out.columns:
                continue
            val_str = df_out.loc[r_idx, c_idx_or_tuple]
            numeric_val = parse_value_for_comparison(val_str)
            if pd.notna(numeric_val):
                all_values_with_loc.append({"val": numeric_val, "r_idx": r_idx, "c_idx": c_idx_or_tuple, "orig_str": val_str})
    if not all_values_with_loc:
        return df_out
    all_values_with_loc.sort(key=lambda x: x["val"], reverse=higher_is_better)
    if N_best <= 0 or not all_values_with_loc:
        return df_out
    actual_N_best_limit = min(N_best, len(all_values_with_loc))
    cutoff_score = all_values_with_loc[actual_N_best_limit - 1]["val"]
    for item in all_values_with_loc:
        is_close_to_cutoff = np.isclose(item["val"], cutoff_score, equal_nan=False)
        should_bold = (item["val"] > cutoff_score or is_close_to_cutoff) if higher_is_better else (item["val"] < cutoff_score or is_close_to_cutoff)
        if should_bold:
            df_out.loc[item["r_idx"], item["c_idx"]] = bold_string(item["orig_str"])
    return df_out


def load_config(config_path: Path) -> Optional[Dict[str, Any]]:
    """
    Loads the main configuration file and updates global feature/pretty name mappings.

    Args:
        config_path (Path): Path to the main configuration file.

    Returns:
        Optional[Dict[str, Any]]: The loaded configuration dictionary, or None if loading fails.
    """
    global FEATURE_CONFIGS, FEATURE_SHORT_NAMES, PRETTY_NAMES
    FEATURE_CONFIGS.clear()
    FEATURE_SHORT_NAMES.clear()
    PRETTY_NAMES = PRETTY_NAMES.copy()
    try:
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        extractors = config.get("feature_extractors", [])
        FEATURE_CONFIGS.update({fc["name"]: fc for fc in extractors if "name" in fc})
        FEATURE_SHORT_NAMES.update({name: cfg.get("short_name", name) for name, cfg in FEATURE_CONFIGS.items()})
        logger.info("Loaded %d feature short names from %s", len(FEATURE_SHORT_NAMES), config_path.name)
        PRETTY_NAMES.update(config.get("table_generator_pretty_names", {}))
        return config
    except Exception as e:
        logger.error("Error loading config %s: %s", config_path, e)
        return None


def get_display_name(name: str, entity_type: str = "feature") -> str:
    """
    Gets the display name for a feature or other entity, using short names and pretty names.

    Args:
        name (str): The raw name of the entity.
        entity_type (str): The type of entity ('feature', 'distance', 'metric', 'subset', 'characteristic').

    Returns:
        str: The display name.
    """
    if entity_type == "feature" and name in FEATURE_SHORT_NAMES:
        return FEATURE_SHORT_NAMES[name]
    return PRETTY_NAMES.get(name, name)


def format_number(value: Any, precision: int = 1, is_percentage: bool = False) -> str:
    """
    Formats a numerical value to a string with specified precision, handling percentages and NaN.

    Args:
        value (Any): The numerical value.
        precision (int): The number of decimal places.
        is_percentage (bool): If True, the input value (assumed 0-1) is multiplied by 100 before formatting.

    Returns:
        str: The formatted string ("-" for NaN/None).
    """
    if pd.isna(value) or value is None:
        return "-"
    if isinstance(value, (int, float, np.number)):
        val_to_format = value * 100 if is_percentage else value
        formatted_str = f"{val_to_format:.{precision}f}"
        return formatted_str
    return str(value)


def find_latest_results_json(directory: Path) -> Optional[Path]:
    """
    Finds the path to the latest results JSON file in a given directory.

    Args:
        directory (Path): The directory to search within.

    Returns:
        Optional[Path]: The path to the latest JSON file, or None if no file is found.
    """
    if not directory.is_dir():
        logger.debug("Directory not found: %s", directory)
        return None
    files = sorted(directory.glob("*_results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if files:
        logger.info("Found results JSON in '%s': %s", directory.name, files[0].name)
        return files[0]
    logger.warning("No '*_results.json' file found in %s", directory)
    return None


def parse_benchmark_params(benchmark_str: str) -> Tuple[str, Dict[str, Any]]:
    """
    Parses a benchmark string (e.g., "MLP(alpha=0.01, max_iter=500)") into type and parameters.

    Args:
        benchmark_str (str): The input benchmark string.

    Returns:
        Tuple[str, Dict[str, Any]]: A tuple containing the benchmark type and a dictionary of parameters.
    """
    match = re.match(r"(\w+)\((.*)\)", benchmark_str)
    if not match:
        return benchmark_str, {}
    clf_type, params_str = match.group(1), match.group(2)
    params = {}
    param_pattern = re.compile(r"(\w+)\s*=\s*('[^']*'|\"[^\"]*\"|\[.*?\]|\(.*?,\s*\)|\(.*?\)|None|True|False|[\w\.-]+(?:e[+-]?\d+)?)")
    for p_match in param_pattern.finditer(params_str):
        key, val_str = p_match.group(1), p_match.group(2).strip()
        try:
            val = ast.literal_eval(val_str)
        except (ValueError, SyntaxError, TypeError):
            if val_str.lower() == "none":
                val = None
            elif val_str.lower() == "true":
                val = True
            elif val_str.lower() == "false":
                val = False
            elif val_str in ("auto", "adam", "relu", "adaptive", "balanced"):
                val = val_str
            else:
                try:
                    val_f = float(val_str)
                    val = int(val_f) if val_f.is_integer() else val_f
                except ValueError:
                    val = val_str
        params[key] = val
    return clf_type, params


def load_results_json(json_path: Path, subset_name: str) -> Optional[DataFrame]:
    """
    Loads and parses benchmark results from a JSON file into a pandas DataFrame.

    Args:
        json_path (Path): Path to the JSON results file.
        subset_name (str): The name of the subset these results belong to.

    Returns:
        Optional[DataFrame]: DataFrame containing parsed results, or None if loading/parsing fails.
    """
    if not json_path.is_file():
        logger.error("JSON file not found: %s", json_path)
        return None
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        records = []
        for feature, feature_data in data.items():
            if not isinstance(feature_data, dict):
                if feature == "error":
                    records.append({"subset": subset_name, "feature": "FeatureProcessingError", "metric_type": "error", "distance": "N/A", "benchmark": "N/A", "error_details": str(feature_data)})
                continue
            if "error" in feature_data:
                records.append({"subset": subset_name, "feature": feature, "metric_type": "error", "distance": "N/A", "benchmark": "FeatureProcessing", "error_details": feature_data["error"]})
                continue
            for metric_type, metric_data in feature_data.items():
                if metric_type == "error":
                    records.append({"subset": subset_name, "feature": feature, "metric_type": metric_type, "distance": "N/A", "benchmark": "N/A", "error_details": str(metric_data)})
                    continue
                if not isinstance(metric_data, dict):
                    records.append({"subset": subset_name, "feature": feature, "metric_type": metric_type, "distance": "N/A", "benchmark": "N/A", "error_details": f"Invalid metric data: {metric_data}"})
                    continue
                base = {"subset": subset_name, "feature": feature, "metric_type": metric_type}
                if metric_type == "distance_based":
                    for distance, benchmarks in metric_data.items():
                        if not isinstance(benchmarks, dict):
                            continue
                        for bench_name, results in benchmarks.items():
                            record = {**base, "distance": distance, "benchmark": bench_name}
                            if bench_name == "ClassificationBenchmark" and isinstance(results, dict):
                                for clf_config, scores in results.items():
                                    records.append({**base, "distance": distance, "benchmark": clf_config, **(scores if isinstance(scores, dict) else {"value": scores})})
                            else:
                                records.append({**record, **(results if isinstance(results, dict) else {"value": results})})
                elif metric_type == "feature_based":
                    for bench_name, results in metric_data.items():
                        record = {**base, "distance": "N/A", "benchmark": bench_name}
                        if bench_name == "ClassificationBenchmark" and isinstance(results, dict):
                            for clf_config, scores in results.items():
                                records.append({**base, "distance": "N/A", "benchmark": clf_config, **(scores if isinstance(scores, dict) else {"value": scores})})
                        else:
                            records.append({**record, **(results if isinstance(results, dict) else {"value": results})})
        if not records:
            logger.warning("No records parsed from %s", json_path.name)
            return DataFrame()
        df = DataFrame(records)
        for col in ["subset", "feature", "metric_type", "distance", "benchmark"]:
            df[col] = df.get(col, pd.NA)
        return df
    except Exception as e:
        logger.error("Error parsing JSON %s: %s", json_path, e, exc_info=True)
        return None


def get_ordered_features(benchmark_only: bool = True) -> List[str]:
    """
    Gets a list of feature names from the loaded config, optionally filtering by 'benchmark_this'.

    Args:
        benchmark_only (bool): If True, return only features with 'benchmark_this: true'.

    Returns:
        List[str]: Ordered list of feature names.
    """
    return [name for name, cfg in FEATURE_CONFIGS.items() if cfg.get("benchmark_this", True) == benchmark_only or not benchmark_only]


def generate_vocsim_main_table(df: DataFrame) -> Optional[DataFrame]:
    """
    Generates the main VocSim results table (overall average performance).

    This table shows average performance metrics (P@1, P@5, etc.) across all
    subsets, primarily using Cosine distance, for each feature.

    Args:
        df (DataFrame): DataFrame containing benchmark results from the 'all' subset.

    Returns:
        Optional[DataFrame]: DataFrame formatted for the main table, or None if input is empty.
    """
    if df.empty:
        return None
    logger.info("Generating VocSim Main Table (Overall Results - Cosine Distance)")
    metrics = OrderedDict([("P@1", ("PrecisionAtK", "P@1")), ("P@5", ("PrecisionAtK", "P@5")), ("CSCF", ("CSCFBenchmark", "pccf")), ("CS", ("FValueBenchmark", "pairwise_f_value")), ("CSR", ("ClassSeparationRatio", "csr_score")), ("Weighted Purity", ("ClusteringPurity", "weighted_purity"))])
    table_data = []
    for feature in df["feature"].unique():
        feature_df = df[df["feature"] == feature]
        row = {"Feature": get_display_name(feature)}
        scores = []
        for metric_name, (bench, col) in metrics.items():
            score = np.nan
            if bench == "ClusteringPurity":
                bench_df_fb = feature_df[(feature_df["benchmark"] == bench) & (feature_df["metric_type"] == "feature_based")]
                if not bench_df_fb.empty and col in bench_df_fb.columns and pd.notna(bench_df_fb[col].iloc[0]):
                    score = bench_df_fb[col].iloc[0]
                else:
                    bench_df_db = feature_df[(feature_df["benchmark"] == bench) & (feature_df["metric_type"] == "distance_based") & (feature_df["distance"].str.lower() == "cosine")]
                    if not bench_df_db.empty and col in bench_df_db.columns and pd.notna(bench_df_db[col].iloc[0]):
                        score = bench_df_db[col].iloc[0]
            else:
                bench_df = feature_df[(feature_df["benchmark"] == bench) & (feature_df["metric_type"] == "distance_based") & (feature_df["distance"].str.lower() == "cosine")]
                if not bench_df.empty and col in bench_df.columns and pd.notna(bench_df[col].iloc[0]):
                    score = bench_df[col].iloc[0]
            row[metric_name] = score
            if pd.notna(score):
                scores.append(score if metric_name in ["P@1", "P@5", "Weighted Purity", "CS"] else (score + 1) / 2 if metric_name == "CSR" else 1 - score)
        row["_sort_score"] = np.mean(scores) if scores else -np.inf
        table_data.append(row)
    if not table_data:
        return None
    result = DataFrame(table_data).sort_values("_sort_score", ascending=False).drop(columns="_sort_score").set_index("Feature")
    ordered_cols = [col for col in metrics if col in result.columns]
    result = result[ordered_cols]
    for col_name_iter in result.columns:
        if col_name_iter in ["P@1", "P@5", "Weighted Purity"]:
            result[col_name_iter] = result[col_name_iter].apply(lambda x: format_number(x, 1, True))
        elif col_name_iter == "CSR":
            result[col_name_iter] = result[col_name_iter].apply(lambda x: format_number(((x + 1) / 2) * 100 if pd.notna(x) else x, 1, False))
        elif col_name_iter in ["CS", "CSCF"]:
            result[col_name_iter] = result[col_name_iter].apply(lambda x: format_number(x, 1, True))
        else:
            result[col_name_iter] = result[col_name_iter].apply(lambda x: format_number(x, 2, False))
    higher_is_better_map = {"P@1": True, "P@5": True, "CSCF": False, "CS": True, "CSR": True, "Weighted Purity": True}
    return bold_best_in_columns(result, ordered_cols, higher_is_better_map)


def generate_full_results_table(
    df: DataFrame,
    metric_name_key: str,
    benchmark_name_prefix: str,
    metric_column_in_json: str,
    is_metric_percentage: bool,
    is_higher_better: bool,
    target_subset_columns: List[str],
) -> Optional[DataFrame]:
    """
    Generates a DataFrame for a specific metric across features, distances, and specified subsets.

    This function is designed to produce data suitable for the strict VocSim appendix longtable format.

    Args:
        df (DataFrame): The full combined DataFrame from loading multiple JSON results.
        metric_name_key (str): The display name of the metric (e.g., "P@1"). Used for logging.
        benchmark_name_prefix (str): The benchmark class name prefix (e.g., "PrecisionAtK").
        metric_column_in_json (str): The exact column name in the parsed DataFrame for this metric (e.g., "P@1", "pccf").
        is_metric_percentage (bool): True if the raw metric value (0-1) needs to be multiplied by 100 for display.
        is_higher_better (bool): True if higher values of this metric indicate better performance.
        target_subset_columns (List[str]): A list of the exact subset keys (e.g., "BC", "BS1")
                                          that should form the data columns of the table, in order.

    Returns:
        Optional[DataFrame]: DataFrame indexed by (Method, Dist) with columns corresponding
                             to `target_subset_columns` + 'Avg' + 'Avg (Blind)',
                             containing pre-formatted string values ready for LaTeX S-columns.
                             Returns None if no data is found.
    """
    if df.empty:
        return None
    logger.info("Generating data for table: %s (across all distances and specified subsets)", metric_name_key)

    rows_data = []
    is_clustering_metric = benchmark_name_prefix == "ClusteringPurity"
    all_features_in_df = df["feature"].unique()
    distances_to_iterate = ["cosine", "euclidean", "spearman"]

    for feature_name_iter in all_features_in_df:
        df_feature_specific = df[df["feature"] == feature_name_iter]

        if is_clustering_metric:
            current_row_values = {"Method": get_display_name(feature_name_iter), "Dist": "-"}
            subset_scores_list = []
            blind_subset_scores_list = []
            has_any_data_for_row = False

            for vocsim_subset_key in target_subset_columns:
                df_vocsim_subset_specific = df_feature_specific[df_feature_specific["subset"] == vocsim_subset_key]
                score_val = np.nan
                if not df_vocsim_subset_specific.empty and metric_column_in_json in df_vocsim_subset_specific.columns:
                    val = df_vocsim_subset_specific[metric_column_in_json].iloc[0]
                    if pd.notna(val):
                        score_val = float(val)
                        has_any_data_for_row = True

                current_row_values[vocsim_subset_key] = score_val
                if pd.notna(score_val):
                    subset_scores_list.append(score_val)
                    if vocsim_subset_key in BLIND_TEST_SUBSETS:
                        blind_subset_scores_list.append(score_val)

            avg_score = np.mean(subset_scores_list) if subset_scores_list else np.nan
            avg_blind_score = np.mean(blind_subset_scores_list) if blind_subset_scores_list else np.nan
            current_row_values["Avg"] = avg_score
            current_row_values["Avg (Blind)"] = avg_blind_score

            sort_val = avg_blind_score if pd.notna(avg_blind_score) else avg_score if pd.notna(avg_score) else (-np.inf if is_higher_better else np.inf)
            current_row_values["_sort_score"] = sort_val * (1 if is_higher_better else -1)

            if has_any_data_for_row:
                rows_data.append(current_row_values)

        for dist_name in distances_to_iterate:
            current_row_values = {"Method": get_display_name(feature_name_iter), "Dist": get_display_name(dist_name, "distance")}
            subset_scores_list = []
            blind_subset_scores_list = []
            has_any_data_for_row = False

            for vocsim_subset_key in target_subset_columns:
                df_metric_specific = df_feature_specific[
                    (df_feature_specific["subset"] == vocsim_subset_key)
                    & (df_feature_specific["metric_type"] == "distance_based")
                    & (df_feature_specific["distance"].str.lower() == dist_name.lower())
                    & (df_feature_specific["benchmark"] == benchmark_name_prefix)
                ]
                score_val = np.nan
                if not df_metric_specific.empty and metric_column_in_json in df_metric_specific.columns:
                    val = df_metric_specific[metric_column_in_json].iloc[0]
                    if pd.notna(val):
                        score_val = float(val)
                        has_any_data_for_row = True

                current_row_values[vocsim_subset_key] = score_val
                if pd.notna(score_val):
                    subset_scores_list.append(score_val)
                    if vocsim_subset_key in BLIND_TEST_SUBSETS:
                        blind_subset_scores_list.append(score_val)

            avg_score = np.mean(subset_scores_list) if subset_scores_list else np.nan
            avg_blind_score = np.mean(blind_subset_scores_list) if blind_subset_scores_list else np.nan
            current_row_values["Avg"] = avg_score
            current_row_values["Avg (Blind)"] = avg_blind_score

            sort_val = avg_blind_score if pd.notna(avg_blind_score) else avg_score if pd.notna(avg_score) else (-np.inf if is_higher_better else np.inf)
            current_row_values["_sort_score"] = sort_val * (1 if is_higher_better else -1)

            if has_any_data_for_row:
                rows_data.append(current_row_values)

    if not rows_data:
        return None

    result_df = DataFrame(rows_data)
    if "_sort_score" not in result_df.columns:
        result_df["_sort_score"] = -np.inf if is_higher_better else np.inf

    result_df = result_df.sort_values(["_sort_score", "Method", "Dist"], ascending=[False, True, True])
    result_df = result_df.drop(columns="_sort_score")
    result_df = result_df.set_index(["Method", "Dist"])

    final_column_order = target_subset_columns + ["Avg", "Avg (Blind)"]
    result_df = result_df.reindex(columns=final_column_order)

    precision_for_format = 1
    if metric_name_key == "CSCF":
        precision_for_format = 2

    numeric_df_for_bolding = result_df.copy()
    if metric_name_key == "CSR":
        numeric_df_for_bolding = numeric_df_for_bolding.applymap(lambda x: ((x + 1) / 2) * 100 if pd.notna(x) else np.nan)
    elif is_metric_percentage:
        numeric_df_for_bolding = numeric_df_for_bolding.applymap(lambda x: x * 100 if pd.notna(x) else np.nan)

    for col_name_iter in result_df.columns:
        if metric_name_key == "CSR":
            result_df[col_name_iter] = result_df[col_name_iter].apply(lambda x: format_number(((x + 1) / 2) * 100 if pd.notna(x) else np.nan, precision_for_format, False))
        else:
            result_df[col_name_iter] = result_df[col_name_iter].apply(lambda x: format_number(x, precision_for_format, is_metric_percentage))

    for col in final_column_order:
        if col not in numeric_df_for_bolding.columns:
            continue

        current_col_numeric_values = numeric_df_for_bolding[col]
        valid_numeric_vals = current_col_numeric_values.dropna()
        if valid_numeric_vals.empty:
            continue

        best_numeric_val = valid_numeric_vals.max() if is_higher_better else valid_numeric_vals.min()

        for idx in result_df.index:
            original_str_val = result_df.loc[idx, col]
            numeric_comp_val = numeric_df_for_bolding.loc[idx, col]

            if pd.notna(numeric_comp_val) and np.isclose(numeric_comp_val, best_numeric_val):
                result_df.loc[idx, col] = bold_string(original_str_val)

    return result_df


def generate_avian_perception_table(df: DataFrame) -> Optional[DataFrame]:
    """
    Generates the Avian Perception table (Triplet Accuracy High).

    This table shows Triplet Accuracy (High Consistency) for each feature-distance
    combination and includes reference values from Zandberg et al. (2024).

    Args:
        df (DataFrame): DataFrame containing benchmark results for the 'avian_perception' subset.

    Returns:
        Optional[DataFrame]: DataFrame formatted for the Avian Perception table, or None if input is empty.
    """
    if df.empty:
        return None
    logger.info("Generating Avian Perception Table (Triplet Acc. High, All Distances)")
    data = []
    features = get_ordered_features() or df["feature"].unique()
    distances = ["cosine", "euclidean", "spearman"]
    for feature_name_iter in features:
        feature_df = df[df["feature"] == feature_name_iter]
        for dist_name_iter in distances:
            dist_df = feature_df[feature_df["distance"].str.lower() == dist_name_iter.lower()]
            if dist_df.empty:
                logger.debug("No data for '%s' with '%s' distance.", feature_name_iter, dist_name_iter)
                continue
            score_val = np.nan
            if "triplet_high_accuracy" in dist_df.columns:
                score_val = dist_df["triplet_high_accuracy"].iloc[0] if not dist_df.empty else np.nan
            else:
                logger.warning("'triplet_high_accuracy' column not found for feature '%s', dist '%s'.", feature_name_iter, dist_name_iter)

            if pd.notna(score_val):
                method_name = f"{get_display_name(feature_name_iter)} ({get_display_name(dist_name_iter, 'distance')})"
                data.append({"Method": method_name, "Triplet Acc. (High)": score_val})
    data.extend([{"Method": method_name_iter, "Triplet Acc. (High)": score_iter} for method_name_iter, score_iter in ZANDBERG_RESULTS.items()])
    if not data:
        return None
    result_df = DataFrame(data).sort_values("Triplet Acc. (High)", ascending=False).set_index("Method")
    result_df["Triplet Acc. (High)"] = result_df["Triplet Acc. (High)"].apply(lambda x: format_number(x, 1, True))
    return bold_best_in_columns(result_df, ["Triplet Acc. (High)"], {"Triplet Acc. (High)": True})


def _compare_params(parsed_val, target_val, atol=1e-6) -> bool:
    """
    Compares parsed parameters from benchmark string against target values.

    Args:
        parsed_val (Any): The value parsed from the benchmark string.
        target_val (Any): The target value from the configuration/definitions.
        atol (float): Absolute tolerance for float comparisons.

    Returns:
        bool: True if the values match (considering float tolerance and types), False otherwise.
    """
    if target_val is None:
        return parsed_val is None
    if parsed_val is None:
        return False
    if isinstance(target_val, float):
        return isinstance(parsed_val, (int, float)) and np.isclose(float(parsed_val), target_val, atol=atol)
    if isinstance(target_val, int):
        if isinstance(parsed_val, float):
            return parsed_val.is_integer() and int(parsed_val) == target_val
        return isinstance(parsed_val, int) and parsed_val == target_val
    if isinstance(target_val, (list, tuple)):
        if not isinstance(parsed_val, (list, tuple)) or len(target_val) != len(parsed_val):
            return False
        try:
            target_comp = [float(x) if isinstance(x, str) and x.replace(".", "", 1).lstrip("-").replace(".", "", 1).isdigit() else x for x in target_val]
            parsed_comp = [float(x) if isinstance(x, str) and x.replace(".", "", 1).lstrip("-").replace(".", "", 1).isdigit() else x for x in parsed_val]
            for p, t in zip(parsed_comp, target_comp):
                if isinstance(p, float) and isinstance(t, float):
                    if not np.isclose(p, t, atol=atol):
                        return False
                elif type(p) != type(t):
                    if isinstance(p, (int, float)) and isinstance(t, (int, float)):
                        if not np.isclose(float(p), float(t), atol=atol):
                            return False
                    else:
                        return False
                elif p != t:
                    return False
            return True
        except (ValueError, TypeError):
            return list(parsed_val) == list(target_val)
    return parsed_val == target_val


def format_float(x, precision=1, is_percentage=False):
    """
    Formats a float number to a string with a specific precision, handling NaN and percentages.

    Args:
        x: The float number.
        precision (int): Number of decimal places.
        is_percentage (bool): If True, multiplies by 100.

    Returns:
        str: The formatted string.
    """
    if pd.isna(x) or x is None:
        return "-"
    if isinstance(x, (int, float, np.number)):
        val = x * 100 if is_percentage else x
        return f"{val:.{precision}f}"
    return str(x)


def get_pretty_name(name: str, entity_type: str = "feature") -> str:
    """
    Gets the pretty display name for an entity type.

    Args:
        name (str): The raw name.
        entity_type (str): The entity type ('feature', 'distance', 'metric', 'subset').

    Returns:
        str: The pretty name.
    """
    if entity_type == "feature" and name in FEATURE_SHORT_NAMES:
        return FEATURE_SHORT_NAMES[name]
    return PRETTY_NAMES.get(name, name)


def generate_mouse_strain_table(
    df_strain_data: pd.DataFrame,
    your_target_feature_full_names: List[str],
    goffinet_target_feature_map: OrderedDict,
    classifier_column_config_map: OrderedDict,
    metric_name: str = "accuracy_mean",
    std_dev_name: str = "accuracy_std",
) -> Optional[pd.DataFrame]:
    """
    Generates the Mouse Strain classification accuracy table, combining own results with Goffinet et al. (2021).

    Args:
        df_strain_data (pd.DataFrame): DataFrame containing benchmark results for the 'mouse_strain' subset.
        your_target_feature_full_names (List[str]): List of full feature names from your results to include as rows.
        goffinet_target_feature_map (OrderedDict): Mapping from Goffinet et al. feature names to display names.
        classifier_column_config_map (OrderedDict): Configuration for classifier columns (groups and specific configs).
        metric_name (str): The name of the mean metric column (e.g., "accuracy_mean").
        std_dev_name (str): The name of the std dev metric column (e.g., "accuracy_std").

    Returns:
        Optional[pd.DataFrame]: DataFrame formatted for the Mouse Strain table, or None if no data is found.
    """
    logger.info("Generating Mouse Strain table with FEATURES AS ROWS, classifiers as columns.")
    goffinet_strain_raw = GOFFINET_STRAIN_DATA
    all_row_display_names = [get_pretty_name(f_full_name, "feature") for f_full_name in your_target_feature_full_names] if your_target_feature_full_names else []
    if goffinet_target_feature_map:
        for goffinet_display_name in goffinet_target_feature_map.values():
            if goffinet_display_name not in all_row_display_names:
                all_row_display_names.append(goffinet_display_name)
    if not all_row_display_names:
        logger.warning("No features defined for rows in strain table.")
        return None
    column_tuples = [(clf_group_name, specific_config_disp_name) for clf_group_name, specific_configs_map in classifier_column_config_map.items() for specific_config_disp_name in specific_configs_map.keys()]
    if not column_tuples:
        logger.warning("No classifier columns defined for strain table.")
        return None
    result_df = pd.DataFrame(index=all_row_display_names, columns=pd.MultiIndex.from_tuples(column_tuples))
    result_df.index.name = "Method"

    if df_strain_data is not None and not df_strain_data.empty and your_target_feature_full_names:
        for your_f_full_name in your_target_feature_full_names:
            your_f_display_name = get_pretty_name(your_f_full_name, "feature")
            if your_f_display_name not in result_df.index:
                continue
            df_feature_specific = df_strain_data[df_strain_data["feature"] == your_f_full_name]
            if df_feature_specific.empty:
                continue
            for clf_group_name, specific_configs_map in classifier_column_config_map.items():
                for specific_config_disp_name, clf_match_details in specific_configs_map.items():
                    target_clf_type, target_clf_params_to_match = clf_match_details["type_match"], clf_match_details["params_to_match"]
                    is_mlp_alpha_only_match = target_clf_type == "mlp" and "alpha" in target_clf_params_to_match and len(target_clf_params_to_match) == 1
                    candidate_runs_for_cell = []
                    for _, r_series in df_feature_specific.iterrows():
                        bench_str = r_series.get("benchmark", "")
                        parsed_type, parsed_params = parse_benchmark_params(bench_str)
                        if parsed_type == target_clf_type:
                            param_match = True
                            for p_key, p_val_target in target_clf_params_to_match.items():
                                if is_mlp_alpha_only_match and p_key == "hidden_layer_sizes":
                                    pass
                                elif not _compare_params(parsed_params.get(p_key), p_val_target):
                                    param_match = False
                                    break
                            if param_match:
                                candidate_runs_for_cell.append(r_series)
                    best_run_for_cell = None
                    if candidate_runs_for_cell:
                        if is_mlp_alpha_only_match and len(candidate_runs_for_cell) > 0:
                            best_score = -np.inf
                            for run in candidate_runs_for_cell:
                                score = run.get(metric_name, -np.inf)
                                if pd.notna(score) and score > best_score:
                                    best_score = score
                                    best_run_for_cell = run
                            if best_run_for_cell is None:
                                best_run_for_cell = candidate_runs_for_cell[0]
                        elif candidate_runs_for_cell:
                            best_run_for_cell = candidate_runs_for_cell[0]
                    if best_run_for_cell is not None:
                        mean_val_prop, std_val_prop = best_run_for_cell.get(metric_name, np.nan), best_run_for_cell.get(std_dev_name, np.nan)
                        mean_display = mean_val_prop if pd.notna(mean_val_prop) else np.nan
                        std_display = std_val_prop if pd.notna(std_val_prop) else np.nan
                        formatted_val = f"{format_float(mean_display, 1, False)} ({format_float(std_display, 1, False)})" if pd.notna(mean_display) and pd.notna(std_display) else ("-" if pd.isna(mean_display) else format_float(mean_display, 1, False))
                        result_df.loc[your_f_display_name, (clf_group_name, specific_config_disp_name)] = formatted_val
                    else:
                        result_df.loc[your_f_display_name, (clf_group_name, specific_config_disp_name)] = "-"
    if goffinet_target_feature_map:
        for goffinet_paper_feat_key, goffinet_display_name_row in goffinet_target_feature_map.items():
            if goffinet_display_name_row not in result_df.index:
                continue
            for clf_group_name, specific_configs_map in classifier_column_config_map.items():
                for specific_config_disp_name in specific_configs_map.keys():
                    goffinet_classifier_key = f"{clf_group_name} ({specific_config_disp_name})"
                    score_str = goffinet_strain_raw.get(goffinet_classifier_key, {}).get(goffinet_paper_feat_key, "-")
                    result_df.loc[goffinet_display_name_row, (clf_group_name, specific_config_disp_name)] = score_str
    result_df = result_df.fillna("-")
    all_data_cols_for_bouding = result_df.columns.tolist()
    if all_data_cols_for_bouding:
        result_df = bold_overall_best_in_group_df(result_df, all_data_cols_for_bouding, higher_is_better=True, N_best=1)
    return result_df


def generate_mouse_identity_table(df_mouse_identity_full: pd.DataFrame, target_feature_full_names_ours: List[str], mlp_alpha_configs_map: OrderedDict, goffinet_feature_display_map: OrderedDict) -> Optional[pd.DataFrame]:
    """
    Generates the Mouse Identity classification accuracy table, combining own results with Goffinet et al. (2021).

    Args:
        df_mouse_identity_full (pd.DataFrame): DataFrame containing benchmark results for the 'mouse_identity' subset.
        target_feature_full_names_ours (List[str]): List of full feature names from your results to include as rows.
        mlp_alpha_configs_map (OrderedDict): Configuration for MLP alpha columns.
        goffinet_feature_display_map (OrderedDict): Mapping from Goffinet et al. feature names to display names.

    Returns:
        Optional[pd.DataFrame]: DataFrame formatted for the Mouse Identity table, or None if no data is found.
    """
    if (df_mouse_identity_full is None or df_mouse_identity_full.empty) and not goffinet_feature_display_map:
        logger.warning("No own data and no Goffinet features for mouse identity table.")
        return None
    logger.info("Generating Mouse Identity Classification Table (Features as Rows, including Goffinet).")
    goffinet_identity_raw = GOFFINET_IDENTITY_DATA
    all_feature_rows_display_names = [get_pretty_name(f_full_name, "feature") for f_full_name in target_feature_full_names_ours] if target_feature_full_names_ours else []
    if goffinet_feature_display_map:
        for goffinet_feat_display_starred in goffinet_feature_display_map.values():
            if goffinet_feat_display_starred not in all_feature_rows_display_names:
                all_feature_rows_display_names.append(goffinet_feat_display_starred)
    if not all_feature_rows_display_names:
        logger.warning("No features (neither own nor Goffinet) to display for mouse identity table.")
        return None
    metric_types_display = ["Top-1 accuracy", "Top-5 accuracy"]
    column_tuples = [(mlp_disp_name, metric_type_disp) for mlp_disp_name in mlp_alpha_configs_map.keys() for metric_type_disp in metric_types_display]
    combined_df = pd.DataFrame(index=all_feature_rows_display_names, columns=pd.MultiIndex.from_tuples(column_tuples))
    combined_df.index.name = "Feature Set"

    if df_mouse_identity_full is not None and not df_mouse_identity_full.empty and target_feature_full_names_ours:
        for f_full_name_ours in target_feature_full_names_ours:
            f_display_name_ours = get_pretty_name(f_full_name_ours, "feature")
            if f_display_name_ours not in combined_df.index:
                continue
            df_feature_specific = df_mouse_identity_full[df_mouse_identity_full["feature"] == f_full_name_ours]
            if df_feature_specific.empty:
                continue
            for mlp_disp_name_col_group, mlp_params_to_match in mlp_alpha_configs_map.items():
                candidate_rows = []
                for _, r_series in df_feature_specific.iterrows():
                    bench_str = r_series.get("benchmark", "")
                    parsed_type, parsed_params = parse_benchmark_params(bench_str)
                    if parsed_type == "mlp":
                        param_match = all(_compare_params(parsed_params.get(p_key), p_val_target) for p_key, p_val_target in mlp_params_to_match.items())
                        if param_match:
                            candidate_rows.append(r_series)
                best_mlp_run_for_alpha = None
                if candidate_rows:
                    best_score_top1 = -np.inf
                    for cand_row in candidate_rows:
                        score_prop = cand_row.get("accuracy_mean", -np.inf)
                        if pd.notna(score_prop) and score_prop > best_score_top1:
                            best_score_top1 = score_prop
                            best_mlp_run_for_alpha = cand_row
                    if best_mlp_run_for_alpha is None and candidate_rows:
                        best_mlp_run_for_alpha = candidate_rows[0]
                if best_mlp_run_for_alpha is not None:
                    top1_mean_disp = best_mlp_run_for_alpha.get("accuracy_mean", np.nan)
                    top1_std_disp = best_mlp_run_for_alpha.get("accuracy_std", np.nan)
                    top5_mean_disp = best_mlp_run_for_alpha.get("top_5_accuracy_mean", np.nan)
                    top5_std_disp = best_mlp_run_for_alpha.get("top_5_accuracy_std", np.nan)
                    val_top1 = f"{format_float(top1_mean_disp, 1, False)} ({format_float(top1_std_disp, 1, False)})" if pd.notna(top1_mean_disp) and pd.notna(top1_std_disp) else ("-" if pd.isna(top1_mean_disp) else format_float(top1_mean_disp, 1, False))
                    val_top5 = f"{format_float(top5_mean_disp, 1, False)} ({format_float(top5_std_disp, 1, False)})" if pd.notna(top5_mean_disp) and pd.notna(top5_std_disp) else ("-" if pd.isna(top5_mean_disp) else format_float(top5_mean_disp, 1, False))
                    combined_df.loc[f_display_name_ours, (mlp_disp_name_col_group, "Top-1 accuracy")] = val_top1
                    combined_df.loc[f_display_name_ours, (mlp_disp_name_col_group, "Top-5 accuracy")] = val_top5
                else:
                    combined_df.loc[f_display_name_ours, (mlp_disp_name_col_group, "Top-1 accuracy")] = "-"
                    combined_df.loc[f_display_name_ours, (mlp_disp_name_col_group, "Top-5 accuracy")] = "-"
    if goffinet_feature_display_map:
        for goffinet_feat_paper_name_key, goffinet_feat_display_starred in goffinet_feature_display_map.items():
            if goffinet_feat_display_starred not in combined_df.index:
                continue
            for mlp_alpha_disp_name_col_group in mlp_alpha_configs_map.keys():
                score_str_top1 = goffinet_identity_raw["Top-1 accuracy"].get(mlp_alpha_disp_name_col_group, {}).get(goffinet_feat_paper_name_key, "-")
                combined_df.loc[goffinet_feat_display_starred, (mlp_alpha_disp_name_col_group, "Top-1 accuracy")] = score_str_top1
                score_str_top5 = goffinet_identity_raw["Top-5 accuracy"].get(mlp_alpha_disp_name_col_group, {}).get(goffinet_feat_paper_name_key, "-")
                combined_df.loc[goffinet_feat_display_starred, (mlp_alpha_disp_name_col_group, "Top-5 accuracy")] = score_str_top5
    combined_df = combined_df.fillna("-")
    top1_cols_to_consider = [(mlp_disp, "Top-1 accuracy") for mlp_disp in mlp_alpha_configs_map.keys() if (mlp_disp, "Top-1 accuracy") in combined_df.columns]
    if top1_cols_to_consider:
        combined_df = bold_overall_best_in_group_df(combined_df, top1_cols_to_consider, higher_is_better=True, N_best=1)
    top5_cols_to_consider = [(mlp_disp, "Top-5 accuracy") for mlp_disp in mlp_alpha_configs_map.keys() if (mlp_disp, "Top-5 accuracy") in combined_df.columns]
    if top5_cols_to_consider:
        combined_df = bold_overall_best_in_group_df(combined_df, top5_cols_to_consider, higher_is_better=True, N_best=1)
    return combined_df


def sanitize(text: Any) -> str:
    """
    Sanitizes text for LaTeX output, avoiding changes to likely LaTeX commands.

    Args:
        text (Any): Input text.

    Returns:
        str: Sanitized string.
    """
    if pd.isna(text):
        return "-"
    s_text = str(text)
    if any(cmd in s_text for cmd in ["\\textbf{", "\\makecell{", "\\multicolumn{", "\\textit{", "\\emph{", "\\texttt{"]):
        return s_text
    s_text = s_text.replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
    s_text = s_text.replace("#", r"\#").replace("_", r"\_").replace("{", r"\{")
    s_text = s_text.replace("}", r"\}").replace("~", r"\textasciitilde{}")
    s_text = s_text.replace("^", r"\^{}").replace("*", r"$^*$")
    s_text = s_text.replace("<", r"\textless{}").replace(">", r"\textgreater{}")
    if "\\" in s_text and not any(cmd in s_text for cmd in ["\\makecell", "\\textbf", "\\textit", "\\emph", "\\texttt", "\\&", "\\%", "\\$", "\\#", "\\_", "\\{", "\\}", "\\~", "\\^", "\\*", "\\textbackslash"]):
        s_text = s_text.replace("\\", r"\textbackslash{}")
    return s_text


def generate_vocsim_appendix_longtable_latex(
    df_data: pd.DataFrame, caption_text: str, table_label: str, output_file: Path
):
    """
    Generates and writes a VocSim appendix longtable using the STRICT format.

    Args:
        df_data (pd.DataFrame): DataFrame indexed by ('Method', 'Dist') with columns
                                matching VOCSIM_APPENDIX_S_COLUMN_ORDER, containing
                                pre-formatted string values.
        caption_text (str): Caption for the table.
        table_label (str): LaTeX label for the table.
        output_file (Path): Path to save the .tex file.
    """
    if df_data.empty:
        logger.warning("No data for VocSim Appendix table '%s'. Skipping file write.", caption_text)
        return

    num_data_columns_in_df = len(df_data.columns)
    expected_num_s_columns = 21

    if num_data_columns_in_df != expected_num_s_columns:
        logger.error(
            "FATAL: VocSim appendix table '%s' generator expects exactly %d data columns (as defined in VOCSIM_APPENDIX_S_COLUMN_ORDER),"
            " but found %d in the provided DataFrame. Columns found: %s. Expected: %s. Halting generation for this table.",
            caption_text,
            expected_num_s_columns,
            num_data_columns_in_df,
            list(df_data.columns),
            VOCSIM_APPENDIX_S_COLUMN_ORDER,
        )
        logger.error("Please ensure `generate_full_results_table` produces these columns in the correct order.")
        return

    latex_lines = []
    latex_lines.append("% Add to your LaTeX preamble: \\usepackage{longtable, booktabs, siunitx, makecell}")
    latex_lines.append("% WARNING: This table uses S[table-format=2.1]. Values like 100.0 or -10.0 will cause siunitx errors.")
    latex_lines.append("\\scriptsize")
    latex_lines.append("\\setlength{\\tabcolsep}{2pt}")

    s_caption = sanitize(caption_text)
    s_label = sanitize(table_label)

    latex_lines.append(f"\\begin{{longtable}}{{l l *{{{expected_num_s_columns}}}{{S[table-format=2.1]}}}}")
    latex_lines.append(f"\\caption{{{s_caption}}}\\label{{{s_label}}}\\\\")
    latex_lines.append("\\toprule")

    header_line_parts = ["Method", "Dist"]
    for s_col_name in VOCSIM_APPENDIX_S_COLUMN_ORDER:
        display_col_name = get_display_name(s_col_name, "subset")
        if display_col_name == "Avg (Blind)":
            header_line_parts.append("\\multicolumn{1}{c}{\\makecell{Avg\\\\(Blind)}}")
        else:
            header_line_parts.append(f"\\multicolumn{{1}}{{c}}{{{sanitize(display_col_name)}}}")

    header_full_line = " & ".join(header_line_parts) + " \\\\"

    latex_lines.append(header_full_line)
    latex_lines.append("\\midrule")
    latex_lines.append("\\endfirsthead")
    latex_lines.append("")
    latex_lines.append(f"\\caption[]{{(Continued) {s_caption}}}\\\\")
    latex_lines.append("\\toprule")
    latex_lines.append(header_full_line)
    latex_lines.append("\\midrule")
    latex_lines.append("\\endhead")
    latex_lines.append("")

    total_table_cols_inc_index = 2 + expected_num_s_columns
    latex_lines.append("\\midrule")
    latex_lines.append(f"\\multicolumn{{{total_table_cols_inc_index}}}{{r}}{{\\textit{{Continued on next page}}}}\\\\")
    latex_lines.append("\\endfoot")
    latex_lines.append("")
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\endlastfoot")
    latex_lines.append("")

    for (method_val_idx, dist_val_idx), row_series_data in df_data.iterrows():
        row_str_parts = [sanitize(str(method_val_idx)), sanitize(str(dist_val_idx))]
        for cell_val_str in row_series_data:
            row_str_parts.append(str(cell_val_str))
        latex_lines.append(" & ".join(row_str_parts) + " \\\\")

    latex_lines.append("\\end{longtable}")

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            f.write("\n".join(latex_lines))
        logger.info("VocSim Appendix LaTeX table '%s' saved to %s", s_caption, output_file)
    except Exception as e:
        logger.error("Error saving VocSim Appendix LaTeX table %s: %s", output_file, e)


def output_latex_table(
    df: DataFrame,
    caption: str,
    label: str,
    output_file: Path,
    column_format: Optional[str] = None,
    notes: Optional[List[str]] = None,
    is_longtable: bool = False,
) -> None:
    """
    Generates and writes a LaTeX table (standard or longtable) from a DataFrame.

    Args:
        df (DataFrame): The input DataFrame.
        caption (str): The caption for the table.
        label (str): The LaTeX label for the table.
        output_file (Path): Path to save the .tex file.
        column_format (Optional[str]): LaTeX column format string. Defaults to pandas inference.
        notes (Optional[List[str]]): List of strings to include as table notes.
        is_longtable (bool): If True, generates a longtable environment. Otherwise, a standard table.
    """
    if df.empty:
        logger.warning("No data for LaTeX table '%s'.", caption)
        return

    df_copy = df.copy()

    if df_copy.index.name or isinstance(df_copy.index, MultiIndex):
        sanitized_index_names = [sanitize(name) for name in df_copy.index.names]
        if isinstance(df_copy.index, MultiIndex):
            sanitized_index_tuples = [tuple(sanitize(level) for level in idx_tuple) for idx_tuple in df_copy.index.to_list()]
            df_copy.index = MultiIndex.from_tuples(sanitized_index_tuples, names=sanitized_index_names)
        else:
            sanitized_index_values = [sanitize(idx_val) for idx_val in df_copy.index.to_list()]
            df_copy.index = pd.Index(sanitized_index_values, name=sanitized_index_names[0] if sanitized_index_names else None)

    if isinstance(df_copy.columns, MultiIndex):
        sanitized_column_names = [sanitize(name) for name in df_copy.columns.names]
        sanitized_column_tuples = [tuple(sanitize(level) for level in col_tuple) for col_tuple in df_copy.columns.to_list()]
        df_copy.columns = MultiIndex.from_tuples(sanitized_column_tuples, names=sanitized_column_names)
    else:
        df_copy.columns = pd.Index([sanitize(col_name) for col_name in df_copy.columns.to_list()])

    for col in df_copy.columns:
        df_copy[col] = df_copy[col].apply(lambda x: sanitize(x) if not (isinstance(x, str) and x.startswith("\\")) else x)

    latex = []
    index_levels = df_copy.index.nlevels if (df_copy.index.name or isinstance(df_copy.index, MultiIndex)) else 0
    total_cols = len(df_copy.columns) + index_levels

    s_caption = sanitize(caption)
    s_label = sanitize(label)

    if is_longtable:
        col_fmt = column_format or ("l" * index_levels + "S[table-align-text-post=false]" * len(df_copy.columns))

        latex.append(f"% Longtable: {s_caption}")
        latex.append(f"\\begin{{longtable}}{{{col_fmt}}}")
        latex.append(f"\\caption{{{s_caption}}}\\label{{{s_label}}}\\\\")
        latex.append("\\toprule")

        header_parts = []
        if index_levels > 0:
            header_parts.extend(df_copy.index.names)

        if isinstance(df_copy.columns, MultiIndex):
            for i in range(df_copy.columns.nlevels):
                level_headers = []
                if index_levels > 0 and i == 0:
                    level_headers.extend([""] * index_levels)

                current_level_names = df_copy.columns.get_level_values(i)
                if i == 0:
                    spans = []
                    if len(current_level_names) > 0:
                        curr_name, curr_count = current_level_names[0], 1
                        for name_val in current_level_names[1:]:
                            if name_val == curr_name:
                                curr_count += 1
                            else:
                                spans.append((curr_name, curr_count))
                                curr_name, curr_count = name_val, 1
                        spans.append((curr_name, curr_count))
                    level_headers.extend([f"\\multicolumn{{{count}}}{{c}}{{{name}}}" for name, count in spans])
                else:
                    level_headers.extend(list(current_level_names))

                if i == 0 and index_levels > 0:
                    final_header_row_parts = list(df_copy.index.names) + level_headers[index_levels:]
                elif i > 0 and index_levels > 0:
                    final_header_row_parts = [""] * index_levels + level_headers
                else:
                    final_header_row_parts = level_headers

                latex.append(" & ".join(final_header_row_parts) + "\\\\")
        else:
            header_parts.extend(list(df_copy.columns))
            latex.append(" & ".join(header_parts) + "\\\\")

        latex.append("\\midrule")
        latex.append("\\endfirsthead")

        latex.append(f"\\caption[]{{(Continued) {s_caption}}}\\\\")
        latex.append("\\toprule")
        if isinstance(df_copy.columns, MultiIndex):
            for i in range(df_copy.columns.nlevels):
                level_headers = []
                if index_levels > 0 and i == 0:
                    level_headers.extend([""] * index_levels)
                current_level_names = df_copy.columns.get_level_values(i)
                if i == 0:
                    spans = []
                    if len(current_level_names) > 0:
                        curr_name, curr_count = current_level_names[0], 1
                        for name_val in current_level_names[1:]:
                            if name_val == curr_name:
                                curr_count += 1
                            else:
                                spans.append((curr_name, curr_count))
                                curr_name, curr_count = name_val, 1
                        spans.append((curr_name, curr_count))
                    level_headers.extend([f"\\multicolumn{{{count}}}{{c}}{{{name}}}" for name, count in spans])
                else:
                    level_headers.extend(list(current_level_names))
                if i == 0 and index_levels > 0:
                    final_header_row_parts = list(df_copy.index.names) + level_headers[index_levels:]
                elif i > 0 and index_levels > 0:
                    final_header_row_parts = [""] * index_levels + level_headers
                else:
                    final_header_row_parts = level_headers
                latex.append(" & ".join(final_header_row_parts) + "\\\\")
        else:
            repeated_header_parts = []
            if index_levels > 0:
                repeated_header_parts.extend(df_copy.index.names)
            repeated_header_parts.extend(list(df_copy.columns))
            latex.append(" & ".join(repeated_header_parts) + "\\\\")
        latex.append("\\midrule")
        latex.append("\\endhead")

        latex.append("\\midrule")
        latex.append(f"\\multicolumn{{{total_cols}}}{{r}}{{\\textit{{Continued on next page}}}}\\\\")
        latex.append("\\endfoot")

        latex.append("\\bottomrule")
        if notes:
            sanitized_notes = []
            for note_line in notes:
                if "\\cite{" in note_line:
                    parts = note_line.split("\\cite{")
                    processed_parts = [sanitize(parts[0])]
                    for part in parts[1:]:
                        cite_key_and_rest = part.split("}", 1)
                        processed_parts.append("\\cite{" + cite_key_and_rest[0] + "}")
                        if len(cite_key_and_rest) > 1:
                            processed_parts.append(sanitize(cite_key_and_rest[1]))
                    sanitized_notes.append("".join(processed_parts))
                else:
                    sanitized_notes.append(sanitize(note_line))

            notes_str = ('\\\\\n').join(sanitized_notes)
            latex.append(f"\\multicolumn{{{total_cols}}}{{p{{\\dimexpr\\linewidth-2\\tabcolsep\\relax}}}}{{\\footnotesize {notes_str}}}\\\\")
        latex.append("\\endlastfoot")

        for idx, row_series in df_copy.iterrows():
            row_data = []
            if isinstance(idx, tuple):
                row_data.extend([str(i) for i in idx])
            elif index_levels:
                row_data.append(str(idx))

            row_data.extend([str(val) for val in row_series])
            latex.append(" & ".join(row_data) + "\\\\")
        latex.append("\\end{longtable}")

    else:
        latex.append("% Standard table, not longtable")
        latex.append("\\begin{table}[htp!]\\centering")
        latex.append(f"\\caption{{{s_caption}}}\\label{{{s_label}}}")
        col_fmt_standard = column_format or ("l" * index_levels + "r" * len(df_copy.columns))

        latex_data_str = df_copy.to_latex(
            escape=False,
            na_rep="-",
            column_format=col_fmt_standard,
            index=bool(index_levels),
            header=True,
            multirow=True,
            multicolumn_format="c",
        )

        if "S[" in col_fmt_standard:
            logger.warning("Table '%s': Using S columns with pandas.to_latex. Headers might need manual adjustment or \\multicolumn.", s_caption)
            if not isinstance(df_copy.columns, MultiIndex):
                lines = latex_data_str.splitlines()
                header_idx = -1
                for i, line_txt in enumerate(lines):
                    if "\\midrule" in line_txt and header_idx == -1:
                        header_idx = i - 1
                        if header_idx >= 0 and "&" in lines[header_idx]:
                            header_content = lines[header_idx].split("&")
                            new_header_parts = []
                            col_specs = col_fmt_standard.replace(" ", "")

                            current_col_spec_idx = 0
                            if index_levels > 0:
                                new_header_parts.extend(header_content[:index_levels])
                                current_col_spec_idx += index_levels
                                header_content = header_content[index_levels:]

                            for i_h, head_item in enumerate(header_content):
                                spec_part = ""
                                if "S[" in col_specs[current_col_spec_idx:]:
                                    end_s = col_specs.find("]", current_col_spec_idx)
                                    if end_s != -1:
                                        spec_part = col_specs[current_col_spec_idx : end_s + 1]
                                        current_col_spec_idx = end_s + 1
                                else:
                                    spec_part = col_specs[current_col_spec_idx]
                                    current_col_spec_idx += 1

                                if "S[" in spec_part:
                                    new_header_parts.append(f"\\multicolumn{{1}}{{c}}{{{head_item.strip()}}}")
                                else:
                                    new_header_parts.append(head_item)
                            lines[header_idx] = " & ".join(new_header_parts)
                            latex_data_str = "\n".join(lines)
                            break

        latex.append(latex_data_str)
        if notes:
            sanitized_notes = []
            for note_line in notes:
                if "\\cite{" in note_line:
                    parts = note_line.split("\\cite{")
                    processed_parts = [sanitize(parts[0])]
                    for part in parts[1:]:
                        cite_key_and_rest = part.split("}", 1)
                        processed_parts.append("\\cite{" + cite_key_and_rest[0] + "}")
                        if len(cite_key_and_rest) > 1:
                            processed_parts.append(sanitize(cite_key_and_rest[1]))
                    sanitized_notes.append("".join(processed_parts))
                else:
                    sanitized_notes.append(sanitize(note_line))
            notes_str = ('\\\\\n').join(sanitized_notes)
            latex.append("\\smallskip\n\\begin{minipage}{\\textwidth}\\footnotesize")
            latex.append(notes_str)
            latex.append("\\end{minipage}")
        latex.append("\\end{table}")

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            f.write("\n".join(latex))
        logger.info("Generic LaTeX table '%s' saved to %s", s_caption, output_file)
    except Exception as e:
        logger.error("Error saving generic LaTeX table %s: %s", output_file, e)


def main() -> None:
    """
    Main function to generate LaTeX tables for different benchmarks based on configuration and results JSONs.
    """
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from VocSim JSON results.")
    parser.add_argument("--paper_configs", type=str, nargs="+", required=True, help="Paths to YAML configuration files.")
    parser.add_argument("--default_output_tables_dir_name", type=str, default="paper_tables_script_generated", help="Default subdirectory for tables.")
    args = parser.parse_args()

    for config_path_str in args.paper_configs:
        config_file = Path(config_path_str)
        if not config_file.is_file():
            logger.error("Config file not found: %s. Skipping.", config_file)
            continue
        logger.info("\n===== Processing Config: %s =====", config_file.name)
        cfg = load_config(config_file)
        if not cfg:
            continue

        project_root = Path(cfg.get("project_root", ".")).resolve()
        results_dir_base = Path(cfg.get("results_dir", project_root / "results")).resolve()
        output_dir = Path(cfg.get("output_tables_dir", results_dir_base / args.default_output_tables_dir_name)).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Outputting tables for %s to: %s", config_file.name, output_dir)

        vocsim_data_subset_keys = [key for key in VOCSIM_APPENDIX_S_COLUMN_ORDER if key not in ["Avg", "Avg (Blind)"]]

        config_subsets_to_load_from_json = cfg.get("dataset", {}).get("subsets_to_run", ["all"])
        if not isinstance(config_subsets_to_load_from_json, list):
            config_subsets_to_load_from_json = [config_subsets_to_load_from_json]

        dfs_loaded = []
        for subset_key_to_load in config_subsets_to_load_from_json:
            json_dir_path = results_dir_base / subset_key_to_load
            json_file_path = find_latest_results_json(json_dir_path)
            if json_file_path:
                df_loaded_single = load_results_json(json_file_path, subset_key_to_load)
                if df_loaded_single is not None and not df_loaded_single.empty:
                    dfs_loaded.append(df_loaded_single)
            else:
                logger.warning("No JSON found for subset key '%s' in %s", subset_key_to_load, json_dir_path)

        if not dfs_loaded:
            logger.warning("No data loaded for %s from any specified subset. Skipping table generation.", config_file.name)
            continue
        combined_df_all_loaded = pd.concat(dfs_loaded, ignore_index=True)
        config_name_stem = config_file.stem.lower()

        if "vocsim_paper" in config_name_stem:
            df_for_vocsim_main_table = combined_df_all_loaded[combined_df_all_loaded["subset"] == "all"]
            if not df_for_vocsim_main_table.empty:
                table_vocsim_main = generate_vocsim_main_table(df_for_vocsim_main_table)
                if table_vocsim_main is not None:
                    output_latex_table(
                        table_vocsim_main,
                        "Performance Comparison on VocSim (Overall Average, Cosine Distance Preferred).",
                        "tab:main-results-comparison",
                        output_dir / "table_main_vocsim_results.tex",
                    )
            else:
                logger.warning("No 'all' subset data in loaded DFs for VocSim main table generation.")

            metrics_for_vocsim_appendix = OrderedDict([
                ("P@1", ("PrecisionAtK", "P@1", True, True)),
                ("P@5", ("PrecisionAtK", "P@5", True, True)),
                ("CSCF", ("CSCFBenchmark", "pccf", False, False)),
                ("CS", ("FValueBenchmark", "pairwise_f_value", True, True)),
                ("CSR", ("ClassSeparationRatio", "csr_score", True, False)),
                ("Weighted Purity", ("ClusteringPurity", "weighted_purity", True, True)),
            ])

            for metric_disp_name_iter, (bench_prefix_iter, json_col_iter, higher_bool_iter, is_perc_bool_iter) in metrics_for_vocsim_appendix.items():
                logger.info("Preparing data for VocSim Appendix table: %s", metric_disp_name_iter)

                table_data_df_for_appendix = generate_full_results_table(
                    combined_df_all_loaded, metric_disp_name_iter, bench_prefix_iter, json_col_iter, is_perc_bool_iter, higher_bool_iter, vocsim_data_subset_keys
                )

                if table_data_df_for_appendix is not None and not table_data_df_for_appendix.empty:
                    try:
                        table_data_df_for_appendix = table_data_df_for_appendix.reindex(columns=VOCSIM_APPENDIX_S_COLUMN_ORDER)
                    except Exception as e:
                        logger.error(
                            "Failed to reindex columns for %s table: %s. Columns present: %s. Expected: %s",
                            metric_disp_name_iter,
                            e,
                            table_data_df_for_appendix.columns,
                            VOCSIM_APPENDIX_S_COLUMN_ORDER,
                        )
                        continue

                    if table_data_df_for_appendix.isnull().all().all():
                        logger.warning("Data for %s resulted in all NaN columns after reordering. Skipping table generation.", metric_disp_name_iter)
                        continue

                    suffix_str_app = "($\\uparrow$ better)" if higher_bool_iter else "($\\downarrow$ better)"
                    table_title_app = f"{get_display_name(metric_disp_name_iter, 'metric')} Results Across Subsets and Distances {suffix_str_app}"
                    table_label_str_app = f"tab:appendix_{metric_disp_name_iter.lower().replace('@', '').replace(' ', '_').replace('(','').replace(')','')}_all_dists"
                    output_file_path_app = output_dir / f"table_appendix_{metric_disp_name_iter.lower().replace('@', '').replace(' ', '_').replace('(','').replace(')','')}_all_dists.tex"

                    logger.info("Ensuring generate_vocsim_appendix_longtable_latex is called for metric: %s", metric_disp_name_iter)
                    generate_vocsim_appendix_longtable_latex(table_data_df_for_appendix, table_title_app, table_label_str_app, output_file_path_app)
                else:
                    logger.warning("No data DataFrame generated by `generate_full_results_table` for metric: %s. Cannot create appendix table.", metric_disp_name_iter)

        elif "avian_paper" in config_name_stem:
            df_for_avian_table = combined_df_all_loaded[combined_df_all_loaded["subset"] == "avian_perception"]
            if df_for_avian_table.empty and "all" in combined_df_all_loaded["subset"].unique():
                df_for_avian_table = combined_df_all_loaded[combined_df_all_loaded["subset"] == "all"]

            if not df_for_avian_table.empty:
                table_avian = generate_avian_perception_table(df_for_avian_table)
                if table_avian is not None:
                    notes_avian = [
                        sanitize("\\textit{Comparison based on Triplet Accuracy (High), >70\\% consistency (Zandberg et al., 2024).}"),
                        sanitize("\\textit{Methods with * are reference values from \\cite{zandberg2024bird}.}"),
                    ]

                    logger.info("Generating Avian Perception table as a LONGTABLE.")
                    output_latex_table(
                        table_avian,
                        "Avian Perception Alignment: Triplet Accuracy (High Consistency)",
                        "tab:avian-perception-triplet-high",
                        output_dir / "table_avian_perception_triplet_high.tex",
                        column_format="lS[table-format=2.1]",
                        notes=notes_avian,
                        is_longtable=True,
                    )
            else:
                logger.warning("No 'avian_perception' or 'all' subset data found for Avian paper.")

        elif "mouse_strain_paper" in config_name_stem:
            df_for_strain_table = combined_df_all_loaded[combined_df_all_loaded["subset"] == "mouse_strain"]
            if df_for_strain_table.empty:
                logger.warning("No 'mouse_strain' data found for Mouse Strain paper.")
            else:
                desired_order_strain = ["EF", "ET", "ETF", "EMTF", "EF (D=30)", "EF (D=100)", "ET (D=30)", "ET (D=100)", "ETF (D=30)", "ETF (D=100)", "EMTF (D=30)", "EMTF (D=100)"]
                short_to_full_map_strain = {cfg.get("short_name", n): n for n, cfg in FEATURE_CONFIGS.items()}
                unique_features_strain = sorted(df_for_strain_table["feature"].unique())
                features_for_strain_table = [short_to_full_map_strain.get(s, s) for s in desired_order_strain if short_to_full_map_strain.get(s, s) in unique_features_strain]
                features_for_strain_table.extend([f for f in unique_features_strain if f not in features_for_strain_table])
                table_strain = generate_mouse_strain_table(df_for_strain_table, features_for_strain_table, GOFFINET_FEATURES, CLASSIFIERS)
                if table_strain is not None:
                    col_count_strain = len(table_strain.columns)
                    col_fmt_strain = "l" + "c" * col_count_strain
                    notes_strain = ["Results with (*) are from Goffinet et al. (2021). Values are Top-1 accuracy (\\%) with std."]
                    output_latex_table(
                        table_strain,
                        "Predicting mouse strain. Classification accuracy (Top-1, \\%) and std over 5 splits.",
                        "tab:mouse-strain-features-rows",
                        output_dir / "table_mouse_strain_features_rows.tex",
                        col_fmt_strain,
                        notes_strain,
                        is_longtable=True,
                    )

        elif "mouse_identity_paper" in config_name_stem:
            df_for_identity_table = combined_df_all_loaded[combined_df_all_loaded["subset"] == "mouse_identity"]
            if df_for_identity_table.empty:
                logger.warning("No 'mouse_identity' data found for Mouse Identity paper.")
            else:
                features_for_identity_table = sorted(df_for_identity_table["feature"].unique())
                table_identity = generate_mouse_identity_table(df_for_identity_table, features_for_identity_table, MLP_CONFIGS, GOFFINET_FEATURES)
                if table_identity is not None:
                    col_fmt_identity = "l" + ("cc" * len(MLP_CONFIGS))
                    notes_identity = ["Results with (*) are from Goffinet et al. (2021). Values are accuracy (\\%) with std."]
                    output_latex_table(
                        table_identity,
                        "Predicting mouse identity. Classification accuracy (\\%) and std over 5 splits.",
                        "tab:mouse-identity-combined",
                        output_dir / "table_mouse_identity_combined.tex",
                        col_fmt_identity,
                        notes_identity,
                        is_longtable=False,
                    )
        else:
            logger.warning("No specific table generation logic defined for config stem: %s", config_name_stem)

    logger.info("--- All configs processed ---")


if __name__ == "__main__":
    main()