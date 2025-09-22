import logging
from typing import Union, Any
import numpy as np
import torch
import torchaudio
from transformers import EncodecModel, AutoProcessor

from features.base import FeatureExtractor

logger = logging.getLogger(__name__)


class EncodecExtractor(FeatureExtractor):
    """
    Feature extractor using the EnCodec model from Facebook/Meta on Hugging Face.
    """

    def _initialize(self, model_id: str = "facebook/encodec_24khz", bandwidth: float = 6.0, **kwargs):
        """
        Load the pre-trained Encodec model and processor.

        Args:
            model_id (str): The Hugging Face model identifier (e.g., "facebook/encodec_24khz").
            bandwidth (float): Target bandwidth for the model (e.g., 1.5, 3.0, 6.0, 12.0, 24.0).
                               This influences the returned codebook size/indices.
            **kwargs: Additional arguments (ignored).
        """
        logger.info("Loading Encodec model and processor '%s'...", model_id)
        try:
            self.model = EncodecModel.from_pretrained(model_id).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.target_sr = self.processor.sampling_rate
            self.bandwidth = bandwidth
            self.model.eval()
            logger.info(f"Encodec model loaded successfully (Target SR: {self.target_sr} Hz, BW: {self.bandwidth} kbps).")
        except Exception as e:
            logger.error("Failed to load Encodec model '%s': %s", model_id, e, exc_info=True)
            raise

    @torch.no_grad()
    def extract(self, audio_data: Union[np.ndarray, torch.Tensor], sample_rate: int, **kwargs: Any) -> torch.Tensor:
        """
        Extract features (discrete codes) using the Encodec model's encoder.

        Args:
            audio_data (Union[np.ndarray, torch.Tensor]): Input audio waveform
                (numpy array or torch tensor). Assumed MONO.
            sample_rate (int): Sample rate of the audio data.
            **kwargs (Any): Additional keyword arguments (ignored).

        Returns:
            torch.Tensor: The extracted discrete codes tensor ([1, num_codebooks, num_frames])
                        on CPU.
        """
        try:
            if isinstance(audio_data, np.ndarray):
                audio_data = torch.from_numpy(audio_data).float()

            if audio_data.ndim == 1:
                audio_data = audio_data.unsqueeze(0)
            elif audio_data.ndim == 2 and audio_data.shape[0] > 1:
                audio_data = torch.mean(audio_data, dim=0, keepdim=True)
            elif audio_data.ndim != 2 or audio_data.shape[0] != 1:
                raise ValueError(f"Expected mono audio with 1 channel, got shape {audio_data.shape}")

            audio_data = audio_data.to(self.device)

            if sample_rate != self.target_sr:
                logger.debug("Resampling audio from %d Hz to %d Hz.", sample_rate, self.target_sr)
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sr).to(self.device)
                audio_data = resampler(audio_data)

            audio_np = audio_data.cpu().numpy()
            audio_np = audio_np.squeeze(0)

            inputs = self.processor(raw_audio=audio_np, sampling_rate=self.target_sr, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            encoder_outputs = self.model.encode(inputs["input_values"], inputs["padding_mask"])
            codes = encoder_outputs.audio_codes

            if codes.shape[0] == 1:
                codes = codes.squeeze(0)

            return codes.cpu()
        except Exception as e:
            logger.error("Encodec feature extraction failed: %s", e, exc_info=True)
            raise