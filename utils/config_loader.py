import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Union, Optional, List
import os

logger = logging.getLogger(__name__)


def _deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """
    Recursively updates a dictionary with values from another dictionary.
    List values in update_dict completely replace list values in base_dict.

    Args:
        base_dict (Dict): The dictionary to update.
        update_dict (Dict): The dictionary containing updates.

    Returns:
        Dict: The updated dictionary.
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def load_config(config_path: Union[str, Path], base_config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Loads the main configuration file and optionally merges it with a base configuration.

    Args:
        config_path (Union[str, Path]): Path to the main YAML configuration file.
        base_config_path (Optional[Union[str, Path]]): Path to a base YAML configuration file.
                                                       Values in the main config override base config values.

    Returns:
        Dict[str, Any]: The loaded (and potentially merged) configuration dictionary with paths resolved.

    Raises:
        FileNotFoundError: If the specified configuration file(s) do not exist.
        yaml.YAMLError: If there is an error parsing the YAML file(s).
    """
    main_path = Path(config_path)
    if not main_path.exists():
        raise FileNotFoundError(f"Main configuration file not found: {main_path}")

    final_config = {}

    if base_config_path:
        base_path = Path(base_config_path)
        if base_path.exists():
            try:
                with open(base_path, "r", encoding="utf-8") as f:
                    base_loaded = yaml.safe_load(f)
                    if base_loaded:
                        final_config = base_loaded
                logger.debug("Loaded base configuration from %s", base_path)
            except yaml.YAMLError as e:
                logger.error("Error parsing base YAML file %s: %s", base_path, e)
                raise
            except Exception as e:
                logger.error("Error reading base config file %s: %s", base_path, e)
                raise
        else:
            logger.warning("Base configuration file specified but not found: %s", base_path)

    try:
        with open(main_path, "r", encoding="utf-8") as f:
            main_config = yaml.safe_load(f)
            if main_config is None:
                main_config = {}
        logger.debug("Loaded main configuration from %s", main_path)

        final_config = _deep_update(final_config, main_config)

    except yaml.YAMLError as e:
        logger.error("Error parsing main YAML file %s: %s", main_path, e)
        raise
    except Exception as e:
        logger.error("Error reading main config file %s: %s", main_path, e)
        raise

    project_root_str = final_config.get("project_root", ".")
    project_root = (main_path.parent / project_root_str).resolve()
    final_config["project_root"] = str(project_root)

    path_keys = ["results_dir", "features_dir", "models_dir", "log_dir", "data_dir", "probe_csv_path", "triplet_csv_path", "model_path"]

    def resolve_path_values(cfg_part):
        if isinstance(cfg_part, dict):
            resolved_dict = {}
            for k, v in cfg_part.items():
                if any(path_key in k for path_key in path_keys) and isinstance(v, str):
                    p = Path(v)
                    if p.is_absolute():
                        resolved_path = p.resolve()
                    else:
                        resolved_path = (project_root / p).resolve()
                    resolved_dict[k] = str(resolved_path)
                else:
                    resolved_dict[k] = resolve_path_values(v)
            return resolved_dict
        elif isinstance(cfg_part, list):
            return [resolve_path_values(elem) for elem in cfg_part]
        else:
            return cfg_part

    final_config = resolve_path_values(final_config)
    logger.debug("Resolved relative paths in configuration.")

    return final_config