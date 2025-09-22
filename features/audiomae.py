# -*- coding: utf-8 -*-
"""AudioMAE feature extractor (hance-ai/audiomae)."""

import logging
from typing import Union, Any
from features.base import FeatureExtractor
import numpy as np
import torch
from transformers import AutoModel
import librosa
import soundfile as sf
import tempfile
import os

logger = logging.getLogger(__name__)


class AudioMAEExtractor(FeatureExtractor):
    """
    Feature extractor using the AudioMAE model from hance-ai on Hugging Face.
    Reshapes output to (768, 512).
    """

    def _initialize(self, model_id: str = "hance-ai/audiomae", trust_remote_code: bool = True, **kwargs):
        """
        Load the pre-trained AudioMAE model.

        Args:
            model_id (str): The Hugging Face model identifier.
            trust_remote_code (bool): Whether to trust remote code for loading.
            **kwargs: Additional arguments passed to FeatureExtractor init.
        """
        logger.info("Loading AudioMAE model '%s'...", model_id)
        try:
            self.model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code).to(self.device)
            self.model.eval()
            logger.info("AudioMAE model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load AudioMAE model '%s': %s", model_id, e, exc_info=True)
            raise

    @torch.no_grad()
    def extract(self, audio_data: Union[np.ndarray, torch.Tensor, str], sample_rate: int, **kwargs: Any) -> torch.Tensor:
        """
        Extracts features using AudioMAE and reshapes them.

        Args:
            audio_data: Input audio as numpy array, torch tensor, or file path.
            sample_rate: Sample rate of the audio (if array/tensor).
            **kwargs: Additional arguments (ignored).

        Returns:
            A torch.Tensor of shape (768, 512) or an empty tensor on failure.
        """
        try:
            tmp_file_to_delete = None
            if isinstance(audio_data, (np.ndarray, torch.Tensor)):
                if isinstance(audio_data, torch.Tensor):
                    audio_data = audio_data.cpu().numpy()

                if audio_data.ndim > 1:
                    if audio_data.shape[0] > audio_data.shape[1]:
                        audio_data = np.mean(audio_data, axis=0)
                    else:
                        audio_data = np.mean(audio_data, axis=1)

                target_sr = 16000
                if sample_rate != target_sr:
                    audio_data = librosa.resample(audio_data.astype(np.float32), orig_sr=sample_rate, target_sr=target_sr)
                    sample_rate = target_sr

                audio_data = audio_data.astype(np.float32)
                max_val = np.max(np.abs(audio_data))
                if max_val > 1e-6:
                    audio_data = audio_data / max_val
                else:
                    logger.warning("Audio data appears to be silent or near-silent.")

                if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
                    logger.error("Audio data contains NaN or infinite values after processing.")
                    raise ValueError("Audio data contains NaN or infinite values.")

                custom_temp_dir = "d:/code/vocsim/temp"
                os.makedirs(custom_temp_dir, exist_ok=True)
                with tempfile.NamedTemporaryFile(suffix=".wav", dir=custom_temp_dir, delete=False) as tmpfile:
                    sf.write(tmpfile.name, audio_data, sample_rate)
                    audio_input_path = tmpfile.name
                    tmp_file_to_delete = tmpfile.name

            elif isinstance(audio_data, str):
                audio_input_path = audio_data
            else:
                raise TypeError(f"Unsupported audio_data type: {type(audio_data)}")

            features = self.model(audio_input_path)
            features = features.to(self.device)

            original_shape = features.shape

            target_shape = (768, 512)

            if features.numel() == target_shape[0] * target_shape[1]:
                reshaped_features = features.reshape(target_shape)
            else:
                logger.warning(
                    "Unexpected feature shape %s from AudioMAE model."
                    " Total elements %d do not match expected %d for target shape %s. Returning empty tensor.",
                    original_shape,
                    features.numel(),
                    target_shape[0] * target_shape[1],
                    target_shape,
                )
                return torch.empty(0, device=self.device).cpu()

            return reshaped_features.cpu()

        except Exception as e:
            logger.error("AudioMAE feature extraction failed: %s", e, exc_info=True)
            return torch.empty(0).cpu()

        finally:
            if tmp_file_to_delete and os.path.exists(tmp_file_to_delete):
                try:
                    os.unlink(tmp_file_to_delete)
                except OSError as e:
                    logger.error("Error deleting temporary file %s: %s", tmp_file_to_delete, e)
                    pass