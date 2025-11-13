# In: reproducibility/scripts/paper/table_generators.py

# -*- coding: utf-8 -*-
"""
Functions for generating the specific pandas DataFrames for each paper table.
This includes the main VocSim summary, detailed appendix tables, application-specific
tables (Avian, Mouse), and the metric correlation matrix.
"""
import logging
from collections import OrderedDict
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from .data_loader import ConfigManager, parse_benchmark_params
from .latex_utils import bold_best_in_columns, bold_overall_best_in_group_df, bold_string, format_number
from .table_configs import (BLIND_TEST_SUBSETS, GOFFINET_IDENTITY_DATA,
                            GOFFINET_STRAIN_DATA, METRICS_FOR_CORRELATION,
                            VOCSIM_APPENDIX_S_COLUMN_ORDER, ZANDBERG_RESULTS)

logger = logging.getLogger(__name__)

def _compare_params(parsed_val: Any, target_val: Any, atol: float = 1e-6) -> bool:
    """Helper to robustly compare parsed parameters against target values."""
    if target_val is None: return parsed_val is None
    if parsed_val is None: return False
    if isinstance(target_val, float):
        return isinstance(parsed_val, (int, float)) and np.isclose(float(parsed_val), target_val, atol=atol)
    if isinstance(target_val, int):
        if isinstance(parsed_val, float):
            return parsed_val.is_integer() and int(parsed_val) == target_val
        return isinstance(parsed_val, int) and parsed_val == target_val
    if isinstance(target_val, (list, tuple)):
        if not isinstance(parsed_val, (list, tuple)) or len(target_val) != len(parsed_val):
            return False
        return all(_compare_params(p, t, atol) for p, t in zip(parsed_val, target_val))
    return parsed_val == target_val


def generate_metric_correlation_table(df: DataFrame) -> Optional[DataFrame]:
    """Generates a Spearman correlation matrix table for key performance metrics."""
    if df.empty: return None
    logger.info("Generating Metric Correlation Table...")

    df['method'] = df['feature'] + ' | ' + df['distance']
    public_df = df[~df['subset'].isin(BLIND_TEST_SUBSETS)].copy()

    agg_data = []
    for method in public_df['method'].unique():
        method_df = public_df[public_df['method'] == method]
        method_scores = {'Method': method}
        for metric_name, config in METRICS_FOR_CORRELATION.items():
            score_df = method_df[method_df['benchmark'] == config['benchmark']]
            if not score_df.empty and config['column'] in score_df.columns:
                raw_score = score_df[config['column']].mean()
                if pd.notna(raw_score):
                    method_scores[metric_name] = config['transform'](raw_score)
                else:
                    method_scores[metric_name] = np.nan
            else:
                method_scores[metric_name] = np.nan
        agg_data.append(method_scores)

    scores_df = pd.DataFrame(agg_data).set_index('Method').dropna()
    if scores_df.empty:
        logger.warning("No methods with complete scores found. Cannot generate correlation table.")
        return None

    correlation_matrix = scores_df.corr(method='spearman')
    formatted_matrix = correlation_matrix.applymap(lambda x: f"{x:.2f}")
    formatted_matrix.index.name = "Metric"
    return formatted_matrix


# In reproducibility/scripts/paper/table_generators.py

def generate_vocsim_main_table(df: DataFrame, config_manager: ConfigManager) -> Optional[DataFrame]:
    """Generates the main VocSim results table."""
    if df.empty: return None
    logger.info("Generating VocSim Main Table (Overall Results - Cosine Distance)")
    metrics = OrderedDict([
        ("GSR", ("GlobalSeparationRate", "gsr_score")),
        ("P@1", ("PrecisionAtK", "P@1")), 
        ("P@5", ("PrecisionAtK", "P@5")), 
        ("CSR", ("ClassSeparationRatio", "csr_score")), 
        ("CS", ("FValueBenchmark", "pairwise_f_value")), 
        ("CSCF", ("CSCFBenchmark", "pccf")), 
        ("Weighted Purity", ("ClusteringPurity", "weighted_purity"))
    ])
    
    table_data = []
    for feature in df["feature"].unique():
        feature_df = df[df["feature"] == feature]
        row = {"Feature": config_manager.get_display_name(feature)}
        scores_for_sorting = [] # This list will hold "higher is better" scores for ranking
        for metric_name, (bench, col) in metrics.items():
            score = np.nan
            bench_df = feature_df[(feature_df["benchmark"] == bench) & (feature_df["metric_type"] == "distance_based") & (feature_df["distance"].str.lower() == "cosine")]
            if not bench_df.empty and col in bench_df.columns and pd.notna(bench_df[col].iloc[0]): score = bench_df[col].iloc[0]
            elif bench == "ClusteringPurity": # Special case for feature-based metric
                bench_df_fb = feature_df[(feature_df["benchmark"] == bench) & (feature_df["metric_type"] == "feature_based")]
                if not bench_df_fb.empty and col in bench_df_fb.columns and pd.notna(bench_df_fb[col].iloc[0]): score = bench_df_fb[col].iloc[0]
            
            row[metric_name] = score 
            
            if pd.notna(score):
                if metric_name in ["GSR", "P@1", "P@5", "Weighted Purity", "CSR"]:
                    scores_for_sorting.append(score) # Already 0-1, higher is better
                elif metric_name in ["CS", "CSCF"]:
                    scores_for_sorting.append(1 - score) # Transform 0-1, lower is better -> higher is better
        
        row["_sort_score"] = np.mean(scores_for_sorting) if scores_for_sorting else -np.inf
        table_data.append(row)

    if not table_data: return None
    result = pd.DataFrame(table_data).sort_values("_sort_score", ascending=False).drop(columns="_sort_score").set_index("Feature")
    ordered_cols = [col for col in metrics if col in result.columns]
    result = result[ordered_cols]
    
    for col_name in result.columns:
        if col_name in ["GSR", "P@1", "P@5", "Weighted Purity", "CSR"]:
            result[col_name] = result[col_name].apply(lambda x: format_number(x, 1, True))
        elif col_name in ["CS", "CSCF"]:
            result[col_name] = result[col_name].apply(lambda x: format_number(1 - x if pd.notna(x) else np.nan, 1, True))

    # All metrics are now "higher is better" for bolding
    return bold_best_in_columns(result, ordered_cols, {k: True for k in result.columns})

# In: reproducibility/scripts/paper/table_generators.py

def generate_full_results_table(
    df: DataFrame,
    metric_name_key: str,
    benchmark_name: str,
    metric_col: str,
    is_percent: bool,
    is_higher_better: bool,
    subsets: List[str],
    config_manager: "ConfigManager",
) -> Optional[DataFrame]:
    """
    Generates a DataFrame for a specific metric across features, distances, and subsets.
    This function handles both distance-based and feature-based metrics correctly.
    """
    if df.empty:
        return None
    logger.info(f"Generating Appendix Table for: {metric_name_key}")

    rows_data = []
    features = config_manager.get_ordered_features()
    distances = ["cosine", "euclidean", "spearman"]
    
    is_feature_based_metric = not df[df["benchmark"] == benchmark_name]["distance"].dropna().any()

    for feature_name in features:
        df_feature = df[df["feature"] == feature_name]
        if df_feature.empty:
            continue

        if is_feature_based_metric:
            row = {
                "Method": config_manager.get_display_name(feature_name),
                "Dist": "-",
            }
            # --- START FIX ---
            public_scores, blind_scores, has_data = [], [], False
            for subset in subsets:
                df_metric = df_feature[
                    (df_feature["subset"] == subset) &
                    (df_feature["metric_type"] == "feature_based") &
                    (df_feature["benchmark"] == benchmark_name)
                ]
                score = np.nan
                if not df_metric.empty and metric_col in df_metric.columns:
                    val = df_metric[metric_col].iloc[0]
                    if pd.notna(val):
                        score = float(val)
                        has_data = True
                row[subset] = score
                if pd.notna(score):
                    if subset in BLIND_TEST_SUBSETS:
                        blind_scores.append(score)
                    else:
                        public_scores.append(score)
            
            if has_data:
                row["Avg"] = np.mean(public_scores) if public_scores else np.nan
                row["Avg (Blind)"] = np.mean(blind_scores) if blind_scores else np.nan
                sort_val = row["Avg (Blind)"] if pd.notna(row["Avg (Blind)"]) else row["Avg"] if pd.notna(row["Avg"]) else (-np.inf if is_higher_better else np.inf)
                row["_sort_score"] = sort_val * (1 if is_higher_better else -1)
                rows_data.append(row)
            # --- END FIX ---

        else:
            for dist_name in distances:
                row = {
                    "Method": config_manager.get_display_name(feature_name),
                    "Dist": config_manager.get_display_name(dist_name, "distance"),
                }
                # --- START FIX ---
                public_scores, blind_scores, has_data = [], [], False
                for subset in subsets:
                    df_metric = df_feature[
                        (df_feature["subset"] == subset) &
                        (df_feature["metric_type"] == "distance_based") &
                        (df_feature["distance"].str.lower() == dist_name.lower()) &
                        (df_feature["benchmark"] == benchmark_name)
                    ]
                    score = np.nan
                    if not df_metric.empty and metric_col in df_metric.columns:
                        val = df_metric[metric_col].iloc[0]
                        if pd.notna(val):
                            score = float(val)
                            has_data = True
                    row[subset] = score
                    if pd.notna(score):
                        if subset in BLIND_TEST_SUBSETS:
                            blind_scores.append(score)
                        else:
                            public_scores.append(score)
                
                if has_data:
                    row["Avg"] = np.mean(public_scores) if public_scores else np.nan
                    row["Avg (Blind)"] = np.mean(blind_scores) if blind_scores else np.nan
                    sort_val = row["Avg (Blind)"] if pd.notna(row["Avg (Blind)"]) else row["Avg"] if pd.notna(row["Avg"]) else (-np.inf if is_higher_better else np.inf)
                    row["_sort_score"] = sort_val * (1 if is_higher_better else -1)
                    rows_data.append(row)
                # --- END FIX ---

    if not rows_data:
        return None

    result_df = pd.DataFrame(rows_data).sort_values("_sort_score", ascending=False).drop(columns="_sort_score").set_index(["Method", "Dist"])
    result_df = result_df.reindex(columns=VOCSIM_APPENDIX_S_COLUMN_ORDER)

    numeric_df = result_df.copy()
    if is_percent:
        numeric_df = numeric_df.applymap(lambda x: x * 100 if pd.notna(x) else np.nan)

    for col in result_df.columns:
        result_df[col] = result_df[col].apply(lambda x: format_number(x, 1, is_percent))

    for col in VOCSIM_APPENDIX_S_COLUMN_ORDER:
        if col not in numeric_df.columns:
            continue
        valid_vals = numeric_df[col].dropna()
        if valid_vals.empty:
            continue
        best_val = valid_vals.max() if is_higher_better else valid_vals.min()
        for idx in result_df.index:
            if pd.notna(numeric_df.loc[idx, col]) and np.isclose(numeric_df.loc[idx, col], best_val):
                result_df.loc[idx, col] = bold_string(result_df.loc[idx, col])

    return result_df
    
def generate_avian_perception_table(df: DataFrame, config_manager: ConfigManager) -> Optional[DataFrame]:
    if df.empty: return None
    logger.info("Generating Avian Perception Table (Triplet Acc. High)")
    data = []
    features = config_manager.get_ordered_features()
    distances = ["cosine", "euclidean", "spearman"]
    for feature_name in features:
        feature_df = df[df["feature"] == feature_name]
        for dist_name in distances:
            dist_df = feature_df[feature_df["distance"].str.lower() == dist_name.lower()]
            if dist_df.empty: continue
            score = dist_df["triplet_high_accuracy"].iloc[0] if "triplet_high_accuracy" in dist_df.columns else np.nan
            if pd.notna(score):
                method_name = f"{config_manager.get_display_name(feature_name)} ({config_manager.get_display_name(dist_name, 'distance')})"
                data.append({"Method": method_name, "Triplet Acc. (High)": score})
    data.extend([{"Method": k, "Triplet Acc. (High)": v} for k, v in ZANDBERG_RESULTS.items()])
    if not data: return None
    result_df = pd.DataFrame(data).sort_values("Triplet Acc. (High)", ascending=False).set_index("Method")
    result_df["Triplet Acc. (High)"] = result_df["Triplet Acc. (High)"].apply(lambda x: format_number(x, 1, True))
    return bold_best_in_columns(result_df, ["Triplet Acc. (High)"], {"Triplet Acc. (High)": True})


def generate_mouse_strain_table(df_strain: pd.DataFrame, features: List[str], goffinet_features: Dict, classifiers: Dict, config_manager: ConfigManager, metric_name="accuracy_mean", std_dev_name="accuracy_std") -> Optional[DataFrame]:
    logger.info("Generating Mouse Strain table with FEATURES AS ROWS, classifiers as columns.")
    all_rows = [config_manager.get_display_name(f) for f in features] + list(goffinet_features.values())
    all_rows = sorted(list(set(all_rows)))
    
    column_tuples = [(g, s) for g, s_map in classifiers.items() for s in s_map.keys()]
    result_df = pd.DataFrame(index=all_rows, columns=pd.MultiIndex.from_tuples(column_tuples))
    result_df.index.name = "Method"

    # Populate with our results
    for f_name in features:
        f_display = config_manager.get_display_name(f_name)
        df_feature = df_strain[df_strain["feature"] == f_name]
        if df_feature.empty: continue
        for clf_group, specific_configs in classifiers.items():
            for specific_name, match_details in specific_configs.items():
                target_type, target_params = match_details["type_match"], match_details["params_to_match"]
                
                best_run = None
                best_score = -np.inf
                for _, row in df_feature.iterrows():
                    parsed_type, parsed_params = parse_benchmark_params(row.get("benchmark", ""))
                    if parsed_type == target_type:
                        is_match = all(_compare_params(parsed_params.get(k), v) for k, v in target_params.items())
                        if is_match:
                            score = row.get(metric_name, -np.inf)
                            if pd.notna(score) and score > best_score:
                                best_score = score
                                best_run = row
                
                if best_run is not None:
                    mean_val = best_run.get(metric_name, np.nan)
                    std_val = best_run.get(std_dev_name, np.nan)
                    val_str = f"{format_number(mean_val, 1)} ({format_number(std_val, 1)})" if pd.notna(mean_val) and pd.notna(std_val) else format_number(mean_val, 1)
                    result_df.loc[f_display, (clf_group, specific_name)] = val_str

    # Populate with Goffinet's results
    for goffinet_key, goffinet_display in goffinet_features.items():
        for clf_group, specific_configs in classifiers.items():
            for specific_name in specific_configs.keys():
                goffinet_clf_key = f"{clf_group} ({specific_name})"
                score_str = GOFFINET_STRAIN_DATA.get(goffinet_clf_key, {}).get(goffinet_key, "-")
                result_df.loc[goffinet_display, (clf_group, specific_name)] = score_str
                
    result_df = result_df.fillna("-")
    return bold_overall_best_in_group_df(result_df, result_df.columns.tolist(), higher_is_better=True, n_best=1)


def generate_mouse_identity_table(df_identity: pd.DataFrame, features: List[str], mlp_configs: Dict, goffinet_features: Dict, config_manager: ConfigManager) -> Optional[DataFrame]:
    logger.info("Generating Mouse Identity Classification Table.")
    all_rows = [config_manager.get_display_name(f) for f in features] + list(goffinet_features.values())
    all_rows = sorted(list(set(all_rows)))

    metric_types = ["Top-1 accuracy", "Top-5 accuracy"]
    column_tuples = [(mlp_disp, metric_disp) for mlp_disp in mlp_configs.keys() for metric_disp in metric_types]
    result_df = pd.DataFrame(index=all_rows, columns=pd.MultiIndex.from_tuples(column_tuples))
    result_df.index.name = "Feature Set"

    # Populate with our results
    for f_name in features:
        f_display = config_manager.get_display_name(f_name)
        df_feature = df_identity[df_identity["feature"] == f_name]
        if df_feature.empty: continue
        for mlp_disp, mlp_params in mlp_configs.items():
            best_run = None
            best_score = -np.inf
            for _, row in df_feature.iterrows():
                parsed_type, parsed_params = parse_benchmark_params(row.get("benchmark", ""))
                if parsed_type == "mlp":
                    is_match = all(_compare_params(parsed_params.get(k), v) for k, v in mlp_params.items())
                    if is_match:
                        score = row.get("accuracy_mean", -np.inf)
                        if pd.notna(score) and score > best_score:
                            best_score = score
                            best_run = row
            if best_run is not None:
                top1_mean = best_run.get("accuracy_mean", np.nan)
                top1_std = best_run.get("accuracy_std", np.nan)
                top5_mean = best_run.get("top_5_accuracy_mean", np.nan)
                top5_std = best_run.get("top_5_accuracy_std", np.nan)
                val_top1 = f"{format_number(top1_mean, 1)} ({format_number(top1_std, 1)})" if pd.notna(top1_mean) and pd.notna(top1_std) else format_number(top1_mean, 1)
                val_top5 = f"{format_number(top5_mean, 1)} ({format_number(top5_std, 1)})" if pd.notna(top5_mean) and pd.notna(top5_std) else format_number(top5_mean, 1)
                result_df.loc[f_display, (mlp_disp, "Top-1 accuracy")] = val_top1
                result_df.loc[f_display, (mlp_disp, "Top-5 accuracy")] = val_top5
    
    # Populate with Goffinet's results
    for goffinet_key, goffinet_display in goffinet_features.items():
        for mlp_disp in mlp_configs.keys():
            score_top1 = GOFFINET_IDENTITY_DATA["Top-1 accuracy"].get(mlp_disp, {}).get(goffinet_key, "-")
            score_top5 = GOFFINET_IDENTITY_DATA["Top-5 accuracy"].get(mlp_disp, {}).get(goffinet_key, "-")
            result_df.loc[goffinet_display, (mlp_disp, "Top-1 accuracy")] = score_top1
            result_df.loc[goffinet_display, (mlp_disp, "Top-5 accuracy")] = score_top5

    result_df = result_df.fillna("-")
    top1_cols = [(mlp_disp, "Top-1 accuracy") for mlp_disp in mlp_configs.keys() if (mlp_disp, "Top-1 accuracy") in result_df.columns]
    top5_cols = [(mlp_disp, "Top-5 accuracy") for mlp_disp in mlp_configs.keys() if (mlp_disp, "Top-5 accuracy") in result_df.columns]
    result_df = bold_overall_best_in_group_df(result_df, top1_cols, higher_is_better=True, n_best=1)
    result_df = bold_overall_best_in_group_df(result_df, top5_cols, higher_is_better=True, n_best=1)
    return result_df



def generate_averaged_correlation_table(df: DataFrame, config_manager: ConfigManager, methods_to_include: Optional[List[str]] = None) -> Optional[DataFrame]:
    """
    Generates a Spearman correlation matrix of the STANDARDIZED scores.
    This version correctly identifies Silhouette as a distance-based metric.
    """
    # CORRECTED: Silhouette is now correctly marked as 'distance_based'
    METRICS_FOR_CORRELATION_FORMAT = OrderedDict([
        ('GSR', {'benchmark': 'GlobalSeparationRate', 'column': 'gsr_score', 'transform': lambda x: x, 'metric_type': 'distance_based'}),
        ('Silhouette', {'benchmark': 'SilhouetteBenchmark', 'column': 'silhouette_score', 'transform': lambda x: (x + 1) / 2, 'metric_type': 'distance_based'}),
        ('P@1', {'benchmark': 'PrecisionAtK', 'column': 'P@1', 'transform': lambda x: x, 'metric_type': 'distance_based'}),
        ('P@5', {'benchmark': 'PrecisionAtK', 'column': 'P@5', 'transform': lambda x: x, 'metric_type': 'distance_based'}),
        ('CSR', {'benchmark': 'ClassSeparationRatio', 'column': 'csr_score', 'transform': lambda x: x, 'metric_type': 'distance_based'}),
        ('CS', {'benchmark': 'FValueBenchmark', 'column': 'pairwise_f_value', 'transform': lambda x: 1 - x, 'metric_type': 'distance_based'}),
        ('CSCF', {'benchmark': 'CSCFBenchmark', 'column': 'pccf', 'transform': lambda x: 1 - x, 'metric_type': 'distance_based'}),
    ])

    if df.empty:
        return None
    logger.info("Generating averaged metric correlation table on standardized [0, 1] scores...")

    public_subsets = [s for s in df['subset'].unique() if s not in BLIND_TEST_SUBSETS]
    if not public_subsets:
        logger.warning("No public subsets found for correlation.")
        return None
    
    public_df = df[df['subset'].isin(public_subsets)].copy()

    if methods_to_include:
        public_df = public_df[public_df['feature'].isin(methods_to_include)]

    if public_df.empty:
        logger.warning("No data remains after filtering.")
        return None

    available_benchmarks = public_df['benchmark'].unique()
    metrics_to_correlate = OrderedDict()
    for name, config in METRICS_FOR_CORRELATION_FORMAT.items():
        if config['benchmark'] in available_benchmarks:
            metrics_to_correlate[name] = config
        else:
            logger.warning(f"Metric '{name}' excluded from correlation (benchmark not found in results).")

    if len(metrics_to_correlate) < 2:
        logger.error("Fewer than 2 metrics with available data found.")
        return None
    
    logger.info(f"Metrics available for correlation: {list(metrics_to_correlate.keys())}")
    
    agg_data = []
    for feature_name in sorted(public_df['feature'].unique()):
        feature_df = public_df[public_df['feature'] == feature_name]
        feature_avg_scores = {'Feature': feature_name}
        
        for metric_name, config in metrics_to_correlate.items():
            metric_type = config['metric_type']
            
            # This logic is now simplified and correct for all cases
            if metric_type == 'distance_based':
                score_df = feature_df[
                    (feature_df['benchmark'] == config['benchmark']) &
                    (feature_df['metric_type'] == 'distance_based') &
                    (feature_df['distance'].str.lower() == 'cosine')
                ]
            else: # feature_based (though none are left in this list, this is good practice)
                score_df = feature_df[
                    (feature_df['benchmark'] == config['benchmark']) &
                    (feature_df['metric_type'] == 'feature_based')
                ]

            if not score_df.empty and config['column'] in score_df.columns:
                avg_raw_score = score_df[config['column']].mean()
                if pd.notna(avg_raw_score):
                    feature_avg_scores[metric_name] = config['transform'](avg_raw_score)
                else:
                    feature_avg_scores[metric_name] = np.nan
            else:
                feature_avg_scores[metric_name] = np.nan
        
        agg_data.append(feature_avg_scores)

    if not agg_data:
        return None

    scores_df = pd.DataFrame(agg_data).set_index('Feature')
    scores_df.dropna(inplace=True)

    if len(scores_df) < 2:
        logger.warning(f"Need at least 2 complete feature extractors for correlation, found {len(scores_df)}.")
        return None
    
    logger.info(f"Computing final correlation matrix on {len(scores_df)} feature extractors.")

    correlation_matrix = scores_df.corr(method='spearman')
    
    ordered_metrics = list(metrics_to_correlate.keys())
    correlation_matrix = correlation_matrix.reindex(index=ordered_metrics, columns=ordered_metrics)
    
    formatted_matrix = correlation_matrix.apply(lambda s: s.map('{:.2f}'.format))
    formatted_matrix.index.name = None
    
    formatted_matrix.columns = [config_manager.get_display_name(c, 'metric') for c in formatted_matrix.columns]
    formatted_matrix.index = [config_manager.get_display_name(i, 'metric') for i in formatted_matrix.index]

    return formatted_matrix