import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import os

from features.base import FeatureExtractor
from reproducibility.models.autoencoder import Autoencoder, AudioConfig

logger = logging.getLogger(__name__)


class PaperAutoencoderExtractor(FeatureExtractor):
    """
    Loads and uses the specific Autoencoder model trained for the paper (scoped).
    """

    def _initialize(
        self,
        model_scope: str,
        base_models_dir: Union[str, Path],
        audio_config: Dict[str, Any],
        dimensions: Dict[str, Any],
        bottleneck_dim: int,
        checkpoint_key: str = "model_state_dict",
        **kwargs,
    ):
        """
        Load the AE model trained for a specific scope (e.g., 'all' or subset key).

        Args:
            model_scope (str): The scope identifier for the model (e.g., dataset subset name).
            base_models_dir (Union[str, Path]): Base directory where trainer checkpoints are stored.
            audio_config (Dict[str, Any]): Dictionary containing audio configuration details (sr, n_mels, fmin, fmax).
            dimensions (Dict[str, Any]): Dictionary containing model dimension details (nfft, max_spec_width).
            bottleneck_dim (int): The dimension of the bottleneck layer in the autoencoder.
            checkpoint_key (str): The key in the checkpoint file containing the model state dictionary.
            **kwargs: Additional keyword arguments (ignored).
        """
        if not model_scope:
            raise ValueError("model_scope must be provided ('all' or subset key)")
        self.model_scope = model_scope
        self.base_models_dir = Path(base_models_dir)
        self.audio_config = audio_config
        self.dimensions = dimensions
        self.bottleneck_dim = bottleneck_dim

        logger.info("Initializing PaperAutoencoderExtractor for scope: '%s'", self.model_scope)
        trainer_base_name = "PaperAutoencoderTrainer"
        scoped_trainer_name = f"{trainer_base_name}_{self.model_scope}"
        checkpoint_dir = self.base_models_dir / scoped_trainer_name / "checkpoints"
        model_path = checkpoint_dir / "final_model.pt"

        if not model_path.exists():
            logger.warning("'final_model.pt' not found in %s. Looking for latest checkpoint...", checkpoint_dir)
            available_checkpoints = sorted(checkpoint_dir.glob("*.pt"), key=os.path.getmtime, reverse=True)
            if available_checkpoints:
                model_path = available_checkpoints[0]
                logger.info("Using latest checkpoint: %s", model_path.name)
            else:
                raise FileNotFoundError(f"No AE model checkpoint found for scope '{self.model_scope}' in {checkpoint_dir}")

        try:
            ae_audio_cfg = AudioConfig(
                sr=self.audio_config["sr"],
                n_mels=self.audio_config["n_mels"],
                nfft=self.dimensions["nfft"],
                fmin=self.audio_config.get("fmin", 0.0),
                fmax=self.audio_config.get("fmax", self.audio_config["sr"] // 2),
            )
            max_spec_width = self.dimensions["max_spec_width"]
        except KeyError as e:
            raise ValueError(f"Missing key in audio_config/dimensions for AE init: {e}") from e
        except Exception as e:
            raise ValueError(f"Error creating AudioConfig: {e}") from e

        try:
            self.model = Autoencoder(config=ae_audio_cfg, max_spec_width=max_spec_width, bottleneck_dim=self.bottleneck_dim)
        except Exception as e:
            logger.error("Failed instantiate Autoencoder: %s", e, exc_info=True)
            raise

        try:
            logger.info("Loading AE model checkpoint: %s", model_path)
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get(checkpoint_key)
            if state_dict is None:
                dummy_keys = self.model.state_dict().keys()
                if all(k in checkpoint for k in dummy_keys):
                    state_dict = checkpoint
                else:
                    raise KeyError(f"Key '{checkpoint_key}' not found. Keys: {list(checkpoint.keys())}")

            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning("Missing keys loading AE state_dict: %s", missing)
            if unexpected:
                logger.warning("Unexpected keys loading AE state_dict: %s", unexpected)

            self.model.to(self.device)
            self.model.eval()
            logger.info("Paper AE model (Scope: %s) loaded successfully.", self.model_scope)
        except Exception as e:
            logger.error("Failed to load AE state_dict from %s: %s", model_path, e, exc_info=True)
            raise

    @torch.no_grad()
    def extract(self, audio_data: Union[np.ndarray, torch.Tensor], sample_rate: int, **kwargs: Any) -> torch.Tensor:
        """
        Extract features using the loaded paper-specific Autoencoder's encoder.

        Args:
            audio_data (Union[np.ndarray, torch.Tensor]): Input audio waveform (numpy array or torch tensor).
            sample_rate (int): Sample rate of the audio data.
            **kwargs (Any): Additional keyword arguments (ignored).

        Returns:
            torch.Tensor: The extracted pooled bottleneck features ([bottleneck_dim]) on CPU,
                          or an empty tensor if extraction fails.
        """
        try:
            if isinstance(audio_data, np.ndarray):
                audio_tensor = torch.from_numpy(audio_data.astype(np.float32))
            else:
                audio_tensor = audio_data.float()

            if audio_tensor.ndim > 1 and audio_tensor.shape[0] > 1:
                audio_tensor = torch.mean(audio_tensor, dim=0)
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            audio_tensor = audio_tensor.to(self.device)

            if sample_rate != self.model.config.sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.model.config.sr).to(self.device)
                audio_tensor = resampler(audio_tensor)

            max_val = torch.max(torch.abs(audio_tensor))
            if max_val > 1e-6:
                audio_tensor = audio_tensor / max_val

            _, encoded_features = self.model(audio_tensor)

            pooled_features = torch.mean(encoded_features, dim=[2, 3])

            return pooled_features.squeeze(0).cpu()

        except Exception as e:
            logger.error("Paper AE feature extraction failed: %s", e, exc_info=True)
            return torch.empty(0, device="cpu")