# In: reproducibility/scripts/paper/generate_paper_tables.py

# -*- coding: utf-8 -*-
"""
Main driver script to generate all LaTeX tables for the paper from VocSim JSON results.
This script orchestrates the loading of configurations and data, calls the specific
table generators for each benchmark (VocSim, Avian, Mouse, Correlation), and saves
the final .tex files.
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np


from .data_loader import (ConfigManager, find_latest_results_json,
                          load_results_json)
from .latex_utils import output_latex_table, generate_vocsim_appendix_longtable_latex
from .table_configs import (CLASSIFIERS, GOFFINET_FEATURES, MLP_CONFIGS,
                            VOCSIM_APPENDIX_S_COLUMN_ORDER, BLIND_TEST_SUBSETS, METRICS_FOR_CORRELATION)
from .table_generators import (generate_avian_perception_table,
                               generate_averaged_correlation_table,
                               generate_mouse_identity_table,
                               generate_mouse_strain_table,
                               generate_vocsim_main_table,
                               generate_full_results_table)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main function to generate LaTeX tables for different benchmarks based on
    configuration and results JSONs.
    """
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from VocSim JSON results.")
    parser.add_argument("--paper_configs", type=str, nargs="+", required=True, help="Paths to YAML configuration files.")
    parser.add_argument("--default_output_tables_dir_name", type=str, default="paper_tables_script_generated", help="Default subdirectory for tables.")
    args = parser.parse_args()

    for config_path_str in args.paper_configs:
        config_file = Path(config_path_str)
        if not config_file.is_file():
            logger.error(f"Config file not found: {config_file}. Skipping.")
            continue

        logger.info(f"\n===== Processing Config: {config_file.name} =====")
        config_manager = ConfigManager(config_file)
        cfg = config_manager.config
        if not cfg:
            continue

        project_root = Path(cfg.get("project_root", ".")).resolve()
        results_dir_base = Path(cfg.get("results_dir", project_root / "results")).resolve()
        output_dir = Path(cfg.get("output_tables_dir", results_dir_base / args.default_output_tables_dir_name)).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Outputting tables for {config_file.name} to: {output_dir}")

        subsets_to_load = cfg.get("dataset", {}).get("subsets_to_run", ["all"])
        if not isinstance(subsets_to_load, list):
            subsets_to_load = [subsets_to_load]

        dfs_loaded = []
        for subset_key in subsets_to_load:
            json_dir = results_dir_base / subset_key
            json_file = find_latest_results_json(json_dir)
            if json_file:
                df_single = load_results_json(json_file, subset_key)
                if df_single is not None and not df_single.empty:
                    dfs_loaded.append(df_single)
            else:
                logger.warning(f"No JSON found for subset '{subset_key}' in {json_dir}")

        if not dfs_loaded:
            logger.warning(f"No data loaded for {config_file.name}. Skipping table generation.")
            continue

        combined_df = pd.concat(dfs_loaded, ignore_index=True)
        config_name_stem = config_file.stem.lower()

        # --- Generate Tables Based on Config File Name ---
        if "vocsim_paper" in config_name_stem:
            df_for_main = combined_df[combined_df["subset"] == "all"]
            if not df_for_main.empty:
                logger.info("--- Generating VocSim Main Table ---")
                table_main = generate_vocsim_main_table(df_for_main, config_manager)
                if table_main is not None:
                    output_latex_table(
                        table_main,
                        "Performance Comparison on VocSim (Overall Average, Cosine Distance Preferred).",
                        "tab:main-results-comparison",
                        output_dir / "table_main_vocsim_results.tex",
                    )

            logger.info("--- Generating Metric Correlation Tables ---")
            
            # 1. Generate correlation table for ALL methods
            logger.info("Calculating correlation for ALL methods...")
            correlation_table_all = generate_averaged_correlation_table(combined_df, config_manager)
            if correlation_table_all is not None:
                output_latex_table(
                    correlation_table_all,
                    caption="Average Spearman Rank Correlation ($\\rho$) of Key Metrics (All Methods)",
                    label="tab:metric-correlations-all",
                    output_file=output_dir / "table_metric_correlations_all.tex",
                    column_format="l" + "c" * len(correlation_table_all.columns)
                )

            # 2. Identify the "best performing" methods to generate a second table
            logger.info("Identifying top 20% of methods for focused correlation analysis...")
            public_df = combined_df[~combined_df['subset'].isin(BLIND_TEST_SUBSETS)].copy()
            public_df['method'] = public_df['feature'] + ' | ' + public_df['distance']
            
            method_scores = {}
            for method in public_df['method'].unique():
                method_df = public_df[public_df['method'] == method]
                scores = []
                for metric_name, config in METRICS_FOR_CORRELATION.items():
                    score_df = method_df[method_df['benchmark'] == config['benchmark']]
                    if not score_df.empty and config['column'] in score_df.columns:
                        # Average the transformed score across public subsets
                        avg_score = score_df[config['column']].apply(config['transform']).mean()
                        if pd.notna(avg_score):
                            scores.append(avg_score)
                if scores:
                    method_scores[method] = np.mean(scores)
            
            if method_scores:
                sorted_methods = sorted(method_scores.items(), key=lambda item: item[1], reverse=True)
                top_20_percent_count = int(len(sorted_methods) * 0.2)
                best_methods = [method for method, score in sorted_methods[:top_20_percent_count]]
                logger.info(f"Identified {len(best_methods)} best-performing methods.")

                # 3. Generate correlation table for ONLY the best methods
                logger.info("Calculating correlation for BEST PERFORMING methods...")
                correlation_table_best = generate_averaged_correlation_table(combined_df, config_manager, methods_to_include=best_methods)
                if correlation_table_best is not None:
                    output_latex_table(
                        correlation_table_best,
                        caption="Average Spearman Rank Correlation ($\\rho$) of Key Metrics (Top 20\\% of Methods)",
                        label="tab:metric-correlations-best",
                        output_file=output_dir / "table_metric_correlations_best.tex",
                        column_format="l" + "c" * len(correlation_table_best.columns)
                    )
            else:
                logger.warning("Could not determine best performing methods.")


            logger.info("--- Generating VocSim Appendix Tables ---")
            metrics_for_appendix = config_manager.METRICS_FOR_APPENDIX
            vocsim_subsets = [k for k in VOCSIM_APPENDIX_S_COLUMN_ORDER if k not in ["Avg", "Avg (Blind)"]]
            for metric_name, (bench, col, higher_is_better, is_percent) in metrics_for_appendix.items():
                table_appendix = generate_full_results_table(
                    combined_df, metric_name, bench, col, is_percent, higher_is_better, vocsim_subsets, config_manager
                )
                if table_appendix is not None:
                    suffix = "($\\uparrow$ better)" if higher_is_better else "($\\downarrow$ better)"
                    caption = f"{config_manager.get_display_name(metric_name, 'metric')} Results Across Subsets and Distances {suffix}"
                    label = f"tab:appendix_{metric_name.lower().replace('@', '')}"
                    generate_vocsim_appendix_longtable_latex(table_appendix, caption, label, output_dir / f"table_appendix_{metric_name.lower().replace('@','')}.tex")

        elif "avian_paper" in config_name_stem:
            df_for_avian = combined_df[combined_df["subset"] == "avian_perception"]
            if not df_for_avian.empty:
                table_avian = generate_avian_perception_table(df_for_avian, config_manager)
                if table_avian is not None:
                    notes = ["\\textit{Comparison based on Triplet Accuracy (High), >70\\% consistency (Zandberg et al., 2024).}", "\\textit{Methods with * are reference values from \\cite{zandberg2024bird}.}"]
                    output_latex_table(
                        table_avian,
                        "Avian Perception Alignment: Triplet Accuracy (High Consistency)",
                        "tab:avian-perception-triplet-high",
                        output_dir / "table_avian_perception_triplet_high.tex",
                        column_format="lS[table-format=2.1]",
                        notes=notes,
                        is_longtable=True
                    )

        elif "mouse_strain_paper" in config_name_stem:
            df_for_strain = combined_df[combined_df["subset"] == "mouse_strain"]
            if not df_for_strain.empty:
                features_for_table = config_manager.get_ordered_features()
                table_strain = generate_mouse_strain_table(df_for_strain, features_for_table, GOFFINET_FEATURES, CLASSIFIERS, config_manager)
                if table_strain is not None:
                    notes = ["Results with (*) are from Goffinet et al. (2021). Values are Top-1 accuracy (\\%) with std."]
                    output_latex_table(
                        table_strain,
                        "Predicting mouse strain. Classification accuracy (Top-1, \\%) and std over 5 splits.",
                        "tab:mouse-strain-features-rows",
                        output_dir / "table_mouse_strain_features_rows.tex",
                        notes=notes,
                        is_longtable=True
                    )

        elif "mouse_identity_paper" in config_name_stem:
            df_for_identity = combined_df[combined_df["subset"] == "mouse_identity"]
            if not df_for_identity.empty:
                features_for_table = config_manager.get_ordered_features()
                table_identity = generate_mouse_identity_table(df_for_identity, features_for_table, MLP_CONFIGS, GOFFINET_FEATURES, config_manager)
                if table_identity is not None:
                    notes = ["Results with (*) are from Goffinet et al. (2021). Values are accuracy (\\%) with std."]
                    output_latex_table(
                        table_identity,
                        "Predicting mouse identity. Classification accuracy (\\%) and std over 5 splits.",
                        "tab:mouse-identity-combined",
                        output_dir / "table_mouse_identity_combined.tex",
                        notes=notes,
                        is_longtable=False
                    )

    logger.info("--- All configs processed ---")


if __name__ == "__main__":
    main()