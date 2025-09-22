import logging
from typing import Any, Union
import numpy as np
import torch
import librosa
from transformers import ClapModel, ClapProcessor
from features.base import FeatureExtractor

logger = logging.getLogger(__name__)


class CLAPExtractor(FeatureExtractor):
    """
    Feature extractor using the CLAP model from LAION on Hugging Face.
    """

    def _initialize(self, model_id: str = "laion/larger_clap_general", **kwargs):
        """
        Load the pre-trained CLAP model and processor.

        Args:
            model_id (str): The Hugging Face model identifier.
            **kwargs: Additional arguments (ignored).
        """
        logger.info("Loading CLAP model and processor '%s'...", model_id)
        try:
            self.model = ClapModel.from_pretrained(model_id).to(self.device)
            self.processor = ClapProcessor.from_pretrained(model_id)
            self.target_sr = self.processor.feature_extractor.sampling_rate
            self.model.eval()
            logger.info("CLAP model and processor loaded successfully.")
        except Exception as e:
            logger.error("Failed to load CLAP model '%s': %s", model_id, e, exc_info=True)
            raise

    @torch.no_grad()
    def extract(self, audio_data: Union[np.ndarray, torch.Tensor], sample_rate: int, **kwargs: Any) -> torch.Tensor:
        """
        Extract audio features using the CLAP model.

        Args:
            audio_data (Union[np.ndarray, torch.Tensor]): Input audio waveform
                (numpy array or torch tensor). Expected to be MONO.
            sample_rate (int): Sample rate of the audio data.

        Returns:
            torch.Tensor: The extracted audio embedding tensor ([1, embedding_dim])
                          on CPU.
        """
        try:
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.cpu().numpy()

            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=0)
            if sample_rate != self.target_sr:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.target_sr)
                sample_rate = self.target_sr

            inputs = self.processor(audios=audio_data, sampling_rate=sample_rate, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            audio_features = self.model.get_audio_features(**inputs)

            return audio_features.cpu()
        except Exception as e:
            logger.error("CLAP feature extraction failed: %s", e, exc_info=True)
            return torch.empty(0)