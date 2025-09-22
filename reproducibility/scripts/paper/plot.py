# File: plot.py
import argparse
import io
import json
import logging
from pathlib import Path
import yaml
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration Constants ---
BLIND_TEST_SUBSETS = ["HU3", "HU4", "HW3", "HW4"]
METRICS_CONFIG = {
    'P@1': {'benchmark': 'PrecisionAtK', 'column': 'P@1', 'transform': lambda x: x},
    'P@5': {'benchmark': 'PrecisionAtK', 'column': 'P@5', 'transform': lambda x: x},
    'GSR': {'benchmark': 'GlobalSeparationRate', 'column': 'gsr_score', 'transform': lambda x: x},
}

# --- Data Loading and Parsing Functions ---

def load_config(config_path: Path) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def find_latest_results_json(directory: Path) -> Path | None:
    """Finds the most recent '*_results.json' file in a directory."""
    if not directory.is_dir():
        return None
    files = sorted(directory.glob("*_results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def load_and_parse_results_json(json_path: Path, subset_name: str) -> pd.DataFrame:
    """Loads a VocSim results JSON and flattens it into a DataFrame."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    records = []
    for feature, feature_data in data.items():
        if not isinstance(feature_data, dict): continue
        for metric_type, metric_data in feature_data.items():
            if not isinstance(metric_data, dict): continue
            base_record = {'subset': subset_name, 'feature': feature, 'metric_type': metric_type}
            if metric_type == 'distance_based':
                for distance, benchmarks in metric_data.items():
                    for bench_name, results in benchmarks.items():
                        if isinstance(results, dict):
                           record = {**base_record, 'distance': distance, 'benchmark': bench_name, **results}
                           records.append(record)
            elif metric_type == 'feature_based':
                 for bench_name, results in metric_data.items():
                    if isinstance(results, dict):
                        record = {**base_record, 'distance': 'N/A', 'benchmark': bench_name, **results}
                        records.append(record)
    return pd.DataFrame(records)

def load_subset_characteristics() -> pd.DataFrame:
    """Loads the VocSim subset characteristics from a string into a DataFrame."""
    table1_data = """
"ID","N. Samples","Classes","Sam/Cls (avg, range)","Avg. Dur (s) (min-max)","Avail","SNR (dB)"
BS3,9988,46,"217.1 (2-1374)","0.07 (0.03-0.20)",Y,20
BS1,473,6,"78.8 (78-79)","0.08 (0.03-0.17)",Y,20
HP,10687,68,"157.2 (8-176)","0.09 (0.03-0.49)",Y,18
BS2,10001,36,"277.8 (6-1209)","0.13 (0.03-0.26)",Y,15
BS4,7035,64,"109.9 (9-129)","0.13 (0.03-0.83)",Y,31
BC,3321,28,"118.6 (6-605)","0.33 (0.03-2.99)",Y,15
HW2,8827,1324,"6.7 (5-7)","0.39 (0.08-1.00)",Y,19
HW1,11532,754,"15.3 (5-100)","0.40 (0.07-1.10)",Y,20
HU2,17041,1366,"12.5 (5-130)","0.64 (0.03-0.49)",Y,22
BS5,8244,30,"274.8 (42-333)","0.70 (0.03-4.99)",Y,26
OC1,441,21,"21.0 (9-32)","0.87 (0.28-5.32)",Y,61
HS1,1670,14,"119.3 (5-713)","0.88 (0.04-2.98)",Y,26
HW4,3497,215,"16.3 (5-46)","0.97 (0.46-2.00)",X,15
HU4,1001,80,"12.5 (5-44)","0.99 (0.47-1.98)",X,15
HW3,4540,368,"12.3 (5-24)","1.01 (0.32-2.00)",X,20
HU3,1703,183,"9.3 (5-24)","1.40 (0.40-3.00)",X,20
HS2,8918,236,"37.8 (9-93)","2.88 (0.04-6.00)",Y,28
HU1,14463,1245,"11.6 (5-50)","3.21 (1.24-6.10)",Y,34
ES1,2000,50,"40.0 (40-40)","5.00 (5.00-5.00)",Y,35
"""
    df_info = pd.read_csv(io.StringIO(table1_data.strip()), sep=',', engine='python', header=0)
    df_info.columns = df_info.columns.str.strip()
    return df_info.set_index('ID')


def generate_trends_figure_with_gsr(
    df_full: pd.DataFrame,
    df_subsets_info: pd.DataFrame,
    output_path: Path
):
    """
    Generates and saves 3 versions of the 6-panel trends figure,
    using the top 5%, 50%, and 100% of methods based on performance.
    Fonts are enlarged and tick labels are bolded for clarity.
    """
    logger.info("Generating generalization trends figures for top 5%, 50%, and 100% of methods...")

    # --- 1. Data Preparation and Ranking Score Calculation ---
    df_public = df_full[~df_full['subset'].isin(BLIND_TEST_SUBSETS)].copy()
    df_public['method'] = df_public['feature'] + ' | ' + df_public['distance']

    agg_data_ranking = []
    for method in df_public['method'].unique():
        method_df = df_public[df_public['method'] == method]
        scores = []
        for metric_name, config in METRICS_CONFIG.items():
            score_df = method_df[method_df['benchmark'] == config['benchmark']]
            if not score_df.empty and config['column'] in score_df.columns:
                avg_score = score_df[config['column']].apply(config['transform']).mean()
                if pd.notna(avg_score):
                    scores.append(avg_score)
        if scores:
            agg_data_ranking.append({'method': method, 'avg_rank_score': np.mean(scores)})

    if not agg_data_ranking:
        logger.error("No method scores calculated for ranking. Aborting figure generation.")
        return

    df_ranking = pd.DataFrame(agg_data_ranking)

    # --- 2. Loop to generate a plot for each percentile ---
    for percentile in [5, 50, 100]:
        logger.info(f"--- Generating plot for top {percentile}% of methods ---")

        quantile_val = 1.0 - (percentile / 100.0)
        score_threshold = df_ranking['avg_rank_score'].quantile(quantile_val)
        top_methods = df_ranking[df_ranking['avg_rank_score'] >= score_threshold]['method'].tolist()

        if not top_methods:
            logger.warning(f"No methods found for top {percentile}%. Skipping plot.")
            continue
        logger.info(f"Identified {len(top_methods)} methods for top {percentile}% plot.")

        df_top_raw = df_public[df_public['method'].isin(top_methods)].copy()

        plot_records = []
        for (subset, method), group in df_top_raw.groupby(['subset', 'method']):
            record = {'subset': subset, 'method': method}
            for metric_name, config in METRICS_CONFIG.items():
                score_series = group[group['benchmark'] == config['benchmark']][config['column']]
                if not score_series.empty and pd.notna(score_series.iloc[0]):
                    record[metric_name] = score_series.iloc[0] * 100
            plot_records.append(record)

        df_plot = pd.DataFrame(plot_records)
        scatter_data = df_plot.merge(df_subsets_info, left_on='subset', right_index=True)

        # --- 3. Plotting with Enhanced Fonts ---
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(3, 2, figsize=(14, 18))
        axes = axes.flatten()

        colors = {'P@5': '#003566', 'P@1': '#0077b6', 'GSR': '#ffc300'}
        x_vars = {
            '# Samples (Log)': 'N. Samples',
            '# Classes (Log)': 'Classes',
            'Avg. Samples / Class (Log)': 'Sam/Cls (avg, range)',
            'Avg. Duration (s) (Log)': 'Avg. Dur (s) (min-max)',
            'SNR (dB)': 'SNR (dB)'
        }
        subplot_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']

        for i, (xlabel, col) in enumerate(x_vars.items()):
            ax = axes[i]
            temp_df = scatter_data.copy()
            temp_df['x_numeric'] = pd.to_numeric(temp_df[col].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
            
            is_log = '(Log)' in xlabel
            if is_log:
                temp_df = temp_df[temp_df['x_numeric'] > 0]
            
            if temp_df.empty or temp_df['x_numeric'].nunique() < 2:
                ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center')
                continue
            
            for metric in colors.keys():
                ax.scatter(temp_df['x_numeric'], temp_df[metric], color=colors[metric], alpha=0.15, s=20, zorder=1)
                
                n_bins = 8
                plot_df = temp_df[['x_numeric', metric]].dropna()
                if plot_df.empty: continue
                
                if is_log:
                    plot_df = plot_df[plot_df['x_numeric'] > 0]
                    if plot_df.empty: continue
                    log_bins = np.logspace(np.log10(plot_df['x_numeric'].min()), np.log10(plot_df['x_numeric'].max()), n_bins + 1)
                    plot_df['x_bin'] = pd.cut(plot_df['x_numeric'], bins=log_bins, include_lowest=True)
                else:
                    plot_df['x_bin'] = pd.cut(plot_df['x_numeric'], bins=n_bins, include_lowest=True)

                agg = plot_df.groupby('x_bin').agg(
                    x_mean=('x_numeric', 'mean'),
                    y_mean=(metric, 'mean'),
                    y_std=(metric, 'std')
                ).dropna()

                ax.plot(agg['x_mean'], agg['y_mean'], color=colors[metric], zorder=2, label=metric, linewidth=2.5)
                ax.fill_between(agg['x_mean'], agg['y_mean'] - agg['y_std'], agg['y_mean'] + agg['y_std'], color=colors[metric], alpha=0.2, zorder=1)

            if is_log:
                ax.set_xscale('log')
                
            # CHANGED: Increased font size for axis labels
            ax.set_xlabel(xlabel, fontsize=18, fontweight='bold')
            if i % 2 == 0:
                ax.set_ylabel('Performance (%)', fontsize=18, fontweight='bold')
            else:
                ax.tick_params(axis='y', labelleft=False)

            ax.set_ylim(0, 105)
            
            # CHANGED: Increased font size for subplot labels
            ax.text(-0.05, 1.02, subplot_labels[i], transform=ax.transAxes, 
                    fontsize=22, fontweight='bold', va='bottom', ha='left')

        # CHANGED: Increased font size for legend
        axes[0].legend(fontsize=16)

        # Subplot f: Box plot
        ax_f = axes[5]
        def get_category(subset_id):
            cat = ''.join(filter(str.isalpha, subset_id))
            return cat if cat else 'Other'
            
        df_plot['category'] = df_plot['subset'].apply(get_category)
        df_melted = df_plot.melt(id_vars=['category'], value_vars=list(METRICS_CONFIG.keys()), var_name='Metric', value_name='Score')
        
        sns.boxplot(x='category', y='Score', hue='Metric', data=df_melted, ax=ax_f,
                    palette=list(colors.values()), order=['ES','BS','BC','OC','HP','HS','HW','HU'])
        
        # CHANGED: Increased font size for axis labels
        ax_f.set_xlabel('Sound Category', fontsize=18, fontweight='bold')
        ax_f.set_ylabel('')
        ax_f.tick_params(axis='y', labelleft=False)
        # CHANGED: Increased font size for legend
        ax_f.legend(title=None, loc='lower left', fontsize=16)
        ax_f.set_ylim(0, 105)
        
        # CHANGED: Increased font size for subplot labels
        ax_f.text(-0.05, 1.02, subplot_labels[5], transform=ax_f.transAxes, 
                  fontsize=22, fontweight='bold', va='bottom', ha='left')
        
        # --- NEW: Loop through all axes to make tick labels bigger and bold ---
        for ax in axes:
            ax.tick_params(axis='both', which='major', labelsize=16)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')

        fig.subplots_adjust(wspace=0.1, hspace=0.3)

        # --- 4. Saving the figure ---
        base_name = output_path.stem
        suffix = output_path.suffix
        new_filename = f"{base_name}_top_{percentile}_percent{suffix}"
        new_output_path = output_path.parent / new_filename
        
        new_output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(new_output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Trends figure saved to: {new_output_path}")
        plt.close(fig)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate generalization trends figure from a VocSim configuration file.")
    parser.add_argument("config_file", type=Path, help="Path to the VocSim YAML configuration file (e.g., reproducibility/configs/vocsim_paper.yaml).")
    parser.add_argument("--output_file", type=Path, default="figure_generalization_trends_gsr.pdf", help="Path to save the output figure.")
    args = parser.parse_args()

    if not args.config_file.is_file():
        logger.error(f"Configuration file not found: {args.config_file}")
        exit(1)

    cfg = load_config(args.config_file)
    project_root = Path(cfg.get("project_root", ".")).resolve()
    results_dir_base = Path(cfg.get("results_dir", project_root / "results")).resolve()
    subsets_to_run = cfg.get("dataset", {}).get("subsets_to_run", ["all"])
    
    dfs_loaded = []
    for subset_key in subsets_to_run:
        json_dir = results_dir_base / subset_key
        json_file = find_latest_results_json(json_dir)
        if json_file:
            logger.info(f"Loading results for subset '{subset_key}' from {json_file.name}")
            df_single = load_and_parse_results_json(json_file, subset_key)
            if df_single is not None and not df_single.empty:
                dfs_loaded.append(df_single)
        else:
            logger.warning(f"No results JSON found for subset '{subset_key}' in {json_dir}")

    if not dfs_loaded:
        logger.error("No valid results data could be loaded. Aborting figure generation.")
        exit(1)

    combined_df = pd.concat(dfs_loaded, ignore_index=True)
    logger.info(f"Successfully loaded and combined results for {len(combined_df['subset'].unique())} subsets.")
    
    df_subset_characteristics = load_subset_characteristics()
    
    generate_trends_figure_with_gsr(
        df_full=combined_df,
        df_subsets_info=df_subset_characteristics,
        output_path=args.output_file
    )