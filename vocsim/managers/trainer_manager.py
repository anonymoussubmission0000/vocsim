import functools
import gc
import importlib
import logging
import sys
import time
from pathlib import Path
from typing import (Any, Callable, Dict, List, Optional, Tuple, Type, Union)

import torch
from torch.utils.data import DataLoader, IterableDataset

from trainers.base import Trainer
from utils.logging_utils import setup_logging
from utils.torch_utils import get_device

from reproducibility.models.autoencoder import Autoencoder, AudioConfig
from reproducibility.trainers.autoencoder import ae_collate_fn
from reproducibility.models.vae import VariationalAutoencoder
from reproducibility.trainers.vae import vae_collate_fn as top_level_vae_collate_fn
from vocsim.managers.dataset_manager import DatasetManager


logger = logging.getLogger(__name__)


class TrainerManager:
    """Handles trainer and model instantiation, dataloader creation, and training execution."""

    def __init__(self, config: Dict[str, Any], models_dir: Path, device: torch.device):
        """
        Initializes the TrainerManager.

        Args:
            config (Dict[str, Any]): The configuration dictionary.
            models_dir (Path): The base directory for saving trained models.
            device (torch.device): The device to use for training.
        """
        self.cfg = config
        self.models_dir = models_dir
        self.device = device
        self.training_jobs = config.get("train", [])
        self.trained_model_paths: Dict[str, Path] = {}

    def _get_class_from_module(self, module_name: str, class_name: str) -> Type:
        """
        Dynamically imports a class from a module path.

        Args:
            module_name (str): The dotted module path.
            class_name (str): The name of the class within the module.

        Returns:
            Type: The imported class object.

        Raises:
            ImportError: If the module or class cannot be found/imported.
        """
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except Exception as e:
            logger.error("Failed import: %s from %s.", class_name, module_name, exc_info=True)
            raise ImportError(f"Failed import {class_name} from {module_name}.") from e

    def _instantiate_model(self, model_config: Dict[str, Any]) -> torch.nn.Module:
        """
        Instantiates the torch.nn.Module based on model configuration.

        Args:
            model_config (Dict[str, Any]): Configuration for the model.

        Returns:
            torch.nn.Module: An instance of the model.

        Raises:
            ValueError: If model config is invalid or instantiation fails.
        """
        model_name = model_config.get("name")
        model_module_path = model_config.get("module")
        model_params = model_config.get("params", {})
        if not model_name or not model_module_path:
            raise ValueError("Model config requires 'name' and 'module'.")
        logger.info("Instantiating model: %s", model_name)
        ModelClass = self._get_class_from_module(model_module_path, model_name)
        instance = None
        if ModelClass is Autoencoder:
            if AudioConfig is None:
                raise ImportError("AudioConfig not available for Autoencoder.")
            ae_audio_cfg_dict = model_params.get("audio_config")
            ae_dims = model_params.get("dimensions")
            ae_bneck = model_params.get("bottleneck_dim")
            if not all([ae_audio_cfg_dict, ae_dims, ae_bneck is not None]):
                raise ValueError("Missing 'audio_config', 'dimensions', or 'bottleneck_dim' for Autoencoder model params.")
            try:
                ae_audio_cfg = AudioConfig(**ae_audio_cfg_dict)
                instance = Autoencoder(config=ae_audio_cfg, max_spec_width=ae_dims["max_spec_width"], bottleneck_dim=ae_bneck)
            except Exception as ae_init_err:
                logger.error("Error initializing Autoencoder: %s", ae_init_err, exc_info=True)
                raise
        elif ModelClass is VariationalAutoencoder:
            vae_z_dim = model_params.get("z_dim")
            vae_precision = model_params.get("model_precision", 10.0)
            vae_lr = model_params.get("learning_rate", 1e-3)
            if vae_z_dim is None:
                raise ValueError("Missing 'z_dim' for VariationalAutoencoder model params.")
            try:
                instance = VariationalAutoencoder(z_dim=vae_z_dim, model_precision=vae_precision, device_name=self.device.type, lr=vae_lr)
            except Exception as vae_init_err:
                logger.error("Error initializing VariationalAutoencoder: %s", vae_init_err, exc_info=True)
                raise
        else:
            try:
                instance = ModelClass(**model_params)
            except Exception as gen_init_err:
                logger.error("Error initializing model %s: %s", model_name, gen_init_err, exc_info=True)
                raise
        if instance is None:
            raise RuntimeError(f"Model instantiation failed for {model_name}")
        logger.info("Model '%s' instantiated successfully.", model_name)
        return instance

    def _get_trainer(self, trainer_config: Dict[str, Any], model: torch.nn.Module, training_scope: str) -> Trainer:
        """
        Instantiates a Trainer, adjusting the output directory based on the training scope.

        Args:
            trainer_config (Dict[str, Any]): Configuration for the trainer.
            model (torch.nn.Module): The model instance to train.
            training_scope (str): The scope of training (e.g., dataset subset name).

        Returns:
            Trainer: An instance of the Trainer.

        Raises:
            ValueError: If trainer config is invalid.
            ImportError: If trainer class cannot be instantiated.
        """
        trainer_name = trainer_config.get("name")
        trainer_module_path = trainer_config.get("module")
        trainer_params = trainer_config.get("params", {})
        if not trainer_name or not trainer_module_path:
            raise ValueError("Trainer config requires 'name' and 'module'.")
        logger.info("Instantiating trainer: %s (Scope: %s)", trainer_name, training_scope)
        TrainerClass = self._get_class_from_module(trainer_module_path, trainer_name)
        scoped_trainer_name = f"{trainer_name}_{training_scope}"
        trainer_output_dir = self.models_dir / scoped_trainer_name
        instance = TrainerClass(model=model, config=trainer_params, output_dir=trainer_output_dir, device=self.device.type)
        logger.info("Trainer '%s' instantiated. Output: %s", scoped_trainer_name, trainer_output_dir)
        return instance

    def _get_collate_fn(self, trainer: Trainer) -> Optional[Callable]:
        """
        Determines the appropriate collate_fn based on the trainer type.

        Args:
            trainer (Trainer): The trainer instance.

        Returns:
            Optional[Callable]: The collate function, or None if a default collate should be used or an error occurs.
        """
        collate_fn = None
        trainer_class_name = trainer.__class__.__name__
        if trainer_class_name == "PaperVAETrainer":
            logger.info("Using VAE-specific collate function.")
            if top_level_vae_collate_fn is None:
                logger.error("VAE collate function not imported.")
                return None
            try:
                if not all(hasattr(trainer, attr) for attr in ["target_sr", "n_fft", "hop_length", "win_length", "window_fn_str", "spec_height", "spec_width", "window_samples", "hop_samples"]):
                    logger.error("Trainer '%s' missing frontend parameters for collate_fn.", trainer_class_name)
                    return None
                collate_fn = functools.partial(top_level_vae_collate_fn, target_sr=trainer.target_sr, n_fft=trainer.n_fft, hop_length=trainer.hop_length, win_length=trainer.win_length, window_fn_str=trainer.window_fn_str, spec_height=trainer.spec_height, spec_width=trainer.spec_width, window_samples=trainer.window_samples, hop_samples=trainer.hop_samples)
            except AttributeError as e:
                logger.error("VAE trainer missing param for collate: %s", e, exc_info=True)
                return None
        elif trainer_class_name == "PaperAutoencoderTrainer":
            logger.info("Using AE-specific collate function (padding).")
            if ae_collate_fn is None:
                logger.error("AE collate function not imported.")
                return None
            collate_fn = ae_collate_fn
        else:
            logger.debug("Using default DataLoader collate for %s.", trainer_class_name)
        return collate_fn

    def run_training_job(self, job_config: Dict[str, Any], dataset: Any, training_scope: str):
        """
        Instantiates model/trainer, creates dataloaders, runs training for a single job.

        Args:
            job_config (Dict[str, Any]): Configuration for the training job.
            dataset (Any): The dataset object for training.
            training_scope (str): The scope of training (e.g., dataset subset name).
        """
        trainer_config = job_config.get("trainer")
        model_config = job_config.get("model")
        if not trainer_config:
            logger.error("Missing 'trainer' section. Skip.")
            return
        if not model_config:
            model_config = trainer_config.get("model")
            logger.warning("Using nested 'model' config. Prefer top-level.")
        if not model_config:
            logger.error("Missing 'model' section. Skip.")
            return

        base_trainer_name = trainer_config.get("name", "unknown_trainer")
        scoped_trainer_name = f"{base_trainer_name}_{training_scope}"
        logger.info("--- Starting Training Job: %s ---", scoped_trainer_name)
        start_time = time.time()
        model, trainer, train_loader, val_loader = None, None, None, None
        try:
            model = self._instantiate_model(model_config)
            trainer = self._get_trainer(trainer_config, model, training_scope)
            batch_size = job_config.get("batch_size", 64)
            num_workers = 0 if sys.platform == "win32" else job_config.get("num_workers", 0)
            pin_memory = job_config.get("pin_memory", True) and self.device.type == "cuda"
            train_dataset = dataset
            val_dataset = None
            collate_fn = self._get_collate_fn(trainer)
            if collate_fn is None and trainer.__class__.__name__ in ["PaperVAETrainer", "PaperAutoencoderTrainer"]:
                logger.error("Missing collate for %s. Abort.", trainer.__class__.__name__)
                return
            is_iterable = isinstance(train_dataset, IterableDataset)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=not is_iterable, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn, drop_last=True)
            val_loader = None
            logger.info("DataLoaders created (Batch: %d, Workers: %d).", batch_size, num_workers)
            trainer.train(train_loader, val_loader)
            model_save_path = trainer.output_dir / "final_model.pt"
            if model_save_path.exists():
                self.trained_model_paths[scoped_trainer_name] = model_save_path
                logger.info("Stored path for '%s': %s", scoped_trainer_name, model_save_path)
            else:
                logger.warning("Final model not found for '%s' in %s", scoped_trainer_name, trainer.output_dir)
        except (ImportError, ValueError, FileNotFoundError) as job_err:
            logger.error("Config/Init error job '%s': %s", scoped_trainer_name, job_err, exc_info=True)
        except Exception as e:
            logger.error("Runtime error job '%s': %s", scoped_trainer_name, e, exc_info=True)
        finally:
            del model, trainer, train_loader, val_loader
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            elapsed = time.time() - start_time
            logger.info("Job '%s' finished in %.2fs.", scoped_trainer_name, elapsed)

    def run_all_training_jobs(self, dataset_manager: DatasetManager):
        """
        Iterates through all training jobs configured and executes them.

        Args:
            dataset_manager (DatasetManager): The dataset manager instance.
        """
        if not self.training_jobs:
            logger.info("No training jobs defined.")
            return
        logger.info("Found %d training job(s).", len(self.training_jobs))
        for i, job_config in enumerate(self.training_jobs):
            trainer_name_cfg = job_config.get("trainer", {}).get("name", f"Job_{i+1}")
            training_scope = job_config.get("train_on_subset", "all")
            logger.info("--- Processing Training Job %d/%d (%s, Scope: %s) ---", i + 1, len(self.training_jobs), trainer_name_cfg, training_scope)
            target_dataset_obj = None
            if training_scope == "all":
                target_dataset_obj = dataset_manager.full_dataset_obj
                if target_dataset_obj is None:
                    logger.error("Full dataset not loaded. Skip job %s.", trainer_name_cfg)
                    continue
                logger.info("Job targets full dataset.")
            else:
                subset_info = dataset_manager.get_subset_dataset(training_scope)
                if subset_info:
                    target_dataset_obj, _ = subset_info
                    logger.info("Job targets subset: '%s'", training_scope)
                else:
                    logger.error("Subset '%s' not found/loaded. Skip job %s.", training_scope, trainer_name_cfg)
                    continue
            self.run_training_job(job_config, target_dataset_obj, training_scope)
        logger.info("--- All Configured Training Jobs Processed ---")

    def get_trained_model_paths(self) -> Dict[str, Path]:
        """
        Returns a dictionary mapping SCOPED trainer names to their final saved model paths.

        Returns:
            Dict[str, Path]: Mapping of scoped trainer names to file paths.
        """
        return self.trained_model_paths