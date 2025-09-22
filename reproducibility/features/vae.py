import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.distributions import LowRankMultivariateNormal

from features.base import FeatureExtractor
from reproducibility.models.vae import VariationalAutoencoder, preprocess_vae_input

logger = logging.getLogger(__name__)


class PaperVAEExtractor(FeatureExtractor):
    """
    Loads and uses the specific VAE model trained for the paper (scoped).
    """

    def _initialize(
        self,
        model_scope: str,
        base_models_dir: Union[str, Path],
        checkpoint_key: str = "model_state_dict",
        extract_mode: str = "mean",
        target_sr: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        window_fn_str: str = "hann",
        spec_height: int = 128,
        spec_width: int = 128,
        window_overlap: float = 0.5,
        **kwargs,
    ):
        """
        Load the VAE model trained for a specific scope (e.g., 'all' or subset key).

        Args:
            model_scope (str): The scope identifier for the model (e.g., dataset subset name).
            base_models_dir (Union[str, Path]): Base directory where trainer checkpoints are stored.
            checkpoint_key (str): The key in the checkpoint file containing the model state dictionary.
            extract_mode (str): Determines whether to extract the 'mean' of the latent
                distribution or a 'sample'. Defaults to 'mean'.
            target_sr (int): Target sample rate for VAE frontend processing.
            n_fft (int): FFT size for VAE frontend spectrogram calculation.
            hop_length (int): Hop length for VAE frontend spectrogram calculation.
            win_length (int): Window length for VAE frontend spectrogram calculation.
            window_fn_str (str): Window function name for VAE frontend (e.g., 'hann').
            spec_height (int): Target Mel spectrogram height (number of Mel bins) for VAE.
            spec_width (int): Target spectrogram width (number of time frames) for VAE.
            window_overlap (float): Overlap between spectrogram windows for VAE frontend.
            **kwargs: Additional keyword arguments (ignored).
        """
        if not model_scope:
            raise ValueError("model_scope must be provided ('all' or subset key)")
        self.model_scope = model_scope
        self.base_models_dir = Path(base_models_dir)

        logger.info("Initializing PaperVAEExtractor for scope: '%s'", self.model_scope)

        self.extract_mode = extract_mode
        if self.extract_mode not in ["mean", "sample"]:
            raise ValueError("VAE extract_mode must be 'mean' or 'sample'")

        self.target_sr = target_sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length else n_fft
        self.window_fn_str = window_fn_str
        self.spec_height = spec_height
        self.spec_width = spec_width
        self.window_overlap = window_overlap
        self.window_samples = (self.spec_width - 1) * self.hop_length
        self.hop_samples = int(self.window_samples * (1 - self.window_overlap))

        trainer_base_name = "PaperVAETrainer"
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
                raise FileNotFoundError(f"No VAE model checkpoint found for scope '{self.model_scope}' in {checkpoint_dir}")

        try:
            logger.info("Loading VAE model checkpoint: %s", model_path)
            checkpoint = torch.load(model_path, map_location="cpu")
            state_dict = checkpoint.get(checkpoint_key)
            if state_dict is None:
                dummy_model_state = VariationalAutoencoder(z_dim=32).state_dict().keys()
                if all(k in checkpoint for k in dummy_model_state):
                    state_dict = checkpoint
                else:
                    raise KeyError(f"Key '{checkpoint_key}' not found. Keys: {list(checkpoint.keys())}")

            z_dim = checkpoint.get("z_dim")
            if z_dim is None:
                z_dim_keys = ["fc41.weight", "fc5.bias"]
                for k in z_dim_keys:
                    if k in state_dict:
                        z_dim = state_dict[k].shape[0]
                        break
                if z_dim is None:
                    raise ValueError("Could not determine z_dim from VAE checkpoint.")
            logger.info(f"Inferred z_dim={z_dim} from checkpoint.")

            self.model = VariationalAutoencoder(z_dim=z_dim, device_name=self.device.type)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning("Missing keys loading VAE state_dict: %s", missing)
            if unexpected:
                logger.warning("Unexpected keys loading VAE state_dict: %s", unexpected)

            self.model.eval()
            logger.info("Paper VAE model (Scope: %s) loaded successfully.", self.model_scope)
        except Exception as e:
            logger.error("Failed to load/initialize VAE model from %s: %s", model_path, e, exc_info=True)
            raise

    @torch.no_grad()
    def extract(self, audio_data: Union[np.ndarray, torch.Tensor], sample_rate: int, **kwargs: Any) -> torch.Tensor:
        """
        Extracts features (latent representation) using the loaded VAE encoder.

        Audio is first converted to spectrogram chunks using the VAE's
        frontend parameters, then passed through the VAE encoder. Latent
        vectors are averaged over the chunks.

        Args:
            audio_data (Union[np.ndarray, torch.Tensor]): Input audio waveform.
            sample_rate (int): Sample rate of the audio data.
            **kwargs (Any): Additional keyword arguments (ignored).

        Returns:
            torch.Tensor: The extracted average latent vector ([z_dim]) on CPU,
                          or an empty tensor if extraction fails.
        """
        try:
            spec_chunks = preprocess_vae_input(
                audio_tensor=audio_data,
                sample_rate=sample_rate,
                target_sr=self.target_sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window_fn_str=self.window_fn_str,
                spec_height=self.spec_height,
                spec_width=self.spec_width,
                window_samples=self.window_samples,
                hop_samples=self.hop_samples,
                device=self.device,
            )
            if not spec_chunks:
                return torch.empty(0, device="cpu")

            all_latents = []
            batch_size = 128
            for i in range(0, len(spec_chunks), batch_size):
                spec_batch = torch.cat([c.to(self.device) for c in spec_chunks[i : i + batch_size]], dim=0)
                mu, u, d_diag = self.model.encode(spec_batch)
                if self.extract_mode == "mean":
                    latent_batch = mu
                else:
                    latent_batch = LowRankMultivariateNormal(mu, u, d_diag).rsample()
                all_latents.append(latent_batch.cpu())

            if not all_latents:
                return torch.empty(0, device="cpu")
            all_latents_tensor = torch.cat(all_latents, dim=0)
            avg_latent = all_latents_tensor.mean(dim=0)
            return avg_latent

        except Exception as e:
            logger.error("Paper VAE feature extraction failed: %s", e, exc_info=True)
            return torch.empty(0, device="cpu")