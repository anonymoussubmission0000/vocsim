import logging
from pathlib import Path
import sys
import argparse
import os
from typing import List


try:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    expected_vocsim_dir = project_root / "vocsim"
    if not expected_vocsim_dir.is_dir() and not (project_root / "vocsim" / "runner.py").exists():
        project_root = Path.cwd()
        print(f"WARN: Assuming CWD project root: {project_root}")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"INFO: Added project root: {project_root}")
except NameError:
    project_root = Path.cwd()
    sys.path.insert(0, str(project_root))
    print(f"INFO: Assuming CWD project root for interactive session: {project_root}")

from vocsim.runner import PipelineRunner
from utils.config_loader import load_config
from utils.logging_utils import setup_logging

CONFIG_NAME = "avian_paper.yaml"
BASE_CONFIG_NAME = "base.yaml"
CONFIG_DIR = project_root / "reproducibility" / "configs"
BASE_CONFIG_DIR = project_root / "configs"


def main(steps_to_run: List[str]):
    """
    Main function to run the Avian Perception pipeline stages.

    Args:
        steps_to_run (List[str]): A list of pipeline stage names to execute.
    """
    config_path = CONFIG_DIR / CONFIG_NAME
    base_config_path = BASE_CONFIG_DIR / BASE_CONFIG_NAME
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)
    cfg = load_config(config_path, base_config_path=base_config_path if base_config_path.exists() else None)
    log_config = cfg.get("logging", {})
    log_config.setdefault("log_dir", project_root / "logs")
    log_config.setdefault("log_file", "avian_perception_run.log")
    setup_logging(log_config)
    logger = logging.getLogger(__name__)
    logger.info(f"Loaded config from {config_path}" + (f" and merged with {base_config_path}" if base_config_path.exists() else ""))
    logger.info(f"Executing pipeline steps: {steps_to_run}")
    if Path.cwd() != project_root:
        logger.warning("Running from '%s', changing CWD to project root '%s'.", Path.cwd(), project_root)
        os.chdir(project_root)
    try:
        runner = PipelineRunner(cfg)
        runner.run(steps=steps_to_run)
        logger.info("Avian Perception script finished successfully.")
    except Exception as e:
        logger.error("Error in Avian Perception run: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Avian Perception Pipeline Stages.")
    parser.add_argument("--steps", nargs="+", default=["features", "distances", "benchmarks"], choices=["train", "features", "distances", "benchmarks", "all"], help="Pipeline stages to run. 'all' runs train -> features -> distances -> benchmarks.")
    args = parser.parse_args()
    if "all" in args.steps:
        selected_steps = ["train", "features", "distances", "benchmarks"]
    else:
        ordered_steps = []
        step_order = ["train", "features", "distances", "benchmarks"]
        for step in step_order:
            if step in args.steps:
                ordered_steps.append(step)
        selected_steps = ordered_steps
    main(selected_steps)