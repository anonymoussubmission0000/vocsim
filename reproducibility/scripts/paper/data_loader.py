# -*- coding: utf-8 -*-
"""
Handles loading of configurations and parsing of benchmark result files.
"""
import ast
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import OrderedDict

import pandas as pd
import yaml
from pandas import DataFrame

from .table_configs import PRETTY_NAMES

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages loading and accessing configuration for table generation."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.feature_configs: Dict[str, Any] = {}
        self.feature_short_names: Dict[str, str] = {}
        self.pretty_names = PRETTY_NAMES.copy()
        
        # This will be populated from the config file.
        self.METRICS_FOR_APPENDIX: OrderedDict[str, Tuple] = OrderedDict()

        self._load_config()

    def _load_config(self):
        """Loads the YAML config and populates instance attributes."""
        try:
            with self.config_path.open("r") as f:
                self.config = yaml.safe_load(f) or {}
            
            extractors = self.config.get("feature_extractors", [])
            self.feature_configs = {fc["name"]: fc for fc in extractors if "name" in fc}
            self.feature_short_names = {name: cfg.get("short_name", name) for name, cfg in self.feature_configs.items()}
            self.pretty_names.update(self.config.get("table_generator_pretty_names", {}))
            
            # Define metrics for appendix based on the main config file
            self.METRICS_FOR_APPENDIX = OrderedDict([
                ("P@1", ("PrecisionAtK", "P@1", True, True)),
                ("P@5", ("PrecisionAtK", "P@5", True, True)),
                ("GSR", ("GlobalSeparationRate", "gsr_score", True, True)),
                ("Sil", ("SilhouetteBenchmark", "silhouette_score", True, True)),
                ("CSR", ("ClassSeparationRatio", "csr_score", True, True)),
                ("CS", ("FValueBenchmark", "pairwise_f_value", False, True)),
                ("CSCF", ("CSCFBenchmark", "pccf", False, True)), 
                ("Weighted Purity", ("ClusteringPurity", "weighted_purity", True, True)),
            ])

            logger.info(f"Loaded {len(self.feature_short_names)} feature configs from {self.config_path.name}")
        except Exception as e:
            logger.error(f"Error loading config {self.config_path}: {e}")
            self.config = {}

    def get_display_name(self, name: str, entity_type: str = "feature") -> str:
        """Gets the display name for a feature or other entity."""
        if entity_type == "feature" and name in self.feature_short_names:
            return self.feature_short_names[name]
        return self.pretty_names.get(name, name)
    
    def get_ordered_features(self, benchmark_only: bool = True) -> List[str]:
        """Gets a list of feature names from the config."""
        return [
            name for name, cfg in self.feature_configs.items()
            if not benchmark_only or cfg.get("benchmark_this", True)
        ]


def find_latest_results_json(directory: Path) -> Optional[Path]:
    """Finds the path to the latest results JSON file in a directory."""
    if not directory.is_dir():
        logger.debug(f"Directory not found: {directory}")
        return None
    files = sorted(directory.glob("*_results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if files:
        logger.info(f"Found results JSON in '{directory.name}': {files[0].name}")
        return files[0]
    logger.warning(f"No '*_results.json' file found in {directory}")
    return None

def parse_benchmark_params(benchmark_str: str) -> Tuple[str, Dict[str, Any]]:
    """Parses a benchmark string like 'MLP(alpha=0.01)' into type and params."""
    match = re.match(r"(\w+)\((.*)\)", benchmark_str)
    if not match:
        return benchmark_str, {}
    clf_type, params_str = match.groups()
    params = {}
    # A robust regex to capture different parameter value types
    param_pattern = re.compile(r"(\w+)\s*=\s*('[^']*'|\"[^\"]*\"|\[.*?\]|\(.*?,\s*\)|\(.*?\)|None|True|False|[\w\.-]+(?:e[+-]?\d+)?)")
    for p_match in param_pattern.finditer(params_str):
        key, val_str = p_match.groups()
        try:
            val = ast.literal_eval(val_str.strip())
        except (ValueError, SyntaxError, TypeError):
            val = val_str.strip()
    return clf_type, params

def load_results_json(json_path: Path, subset_name: str) -> Optional[DataFrame]:
    """Loads and parses benchmark results from a JSON file into a pandas DataFrame."""
    if not json_path.is_file(): return None
    try:
        with json_path.open("r", encoding="utf-8") as f: data = json.load(f)
        records = []
        for feature, feature_data in data.items():
            if not isinstance(feature_data, dict): continue
            for metric_type, metric_data in feature_data.items():
                if not isinstance(metric_data, dict): continue
                base = {"subset": subset_name, "feature": feature, "metric_type": metric_type}
                if metric_type == "distance_based":
                    for distance, benchmarks in metric_data.items():
                        for bench_name, results in benchmarks.items():
                            if bench_name == "ClassificationBenchmark" and isinstance(results, dict):
                                for clf_config, scores in results.items():
                                    records.append({**base, "distance": distance, "benchmark": clf_config, **(scores if isinstance(scores, dict) else {"value": scores})})
                            else:
                                records.append({**base, "distance": distance, "benchmark": bench_name, **(results if isinstance(results, dict) else {"value": results})})
                elif metric_type == "feature_based":
                    for bench_name, results in metric_data.items():
                        if bench_name == "ClassificationBenchmark" and isinstance(results, dict):
                            for clf_config, scores in results.items():
                                records.append({**base, "distance": "N/A", "benchmark": clf_config, **(scores if isinstance(scores, dict) else {"value": scores})})
                        else:
                            records.append({**base, "distance": "N/A", "benchmark": bench_name, **(results if isinstance(results, dict) else {"value": results})})
        return DataFrame(records) if records else DataFrame()
    except Exception as e:
        logger.error(f"Error parsing JSON {json_path}: {e}"); return None