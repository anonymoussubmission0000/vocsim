import logging
from typing import Any, Union, Optional
import torch
import torch.nn.functional as F
import numpy as np
import dac
from audiotools import AudioSignal
from features.base import FeatureExtractor

logger = logging.getLogger(__name__)


class DACExtractor(FeatureExtractor):
    """
    Feature extractor using the Descript Audio Codec (DAC).
    This extractor can return the continuous latent 'z', discrete 'codes',
    or the 'latents' from the RVQ bottleneck. It can ensure a fixed-length
    output by padding or truncating if `max_len` is specified.
    """

    def _initialize(
        self,
        model_type: str = "44khz",
        output_type: str = "latents",
        max_len: Optional[int] = None,
        **kwargs,
    ):
        """
        Load the pre-trained DAC model.

        Args:
            model_type (str): The model variant to load ('44khz', '24khz', '16khz').
            output_type (str): The type of feature to return. One of:
                               - 'z': The continuous latent vector before quantization.
                               - 'codes': The discrete integer codes from the RVQ.
                               - 'latents': The continuous vectors from the RVQ bottleneck.
                               - 'all': Concatenation of z, codes (flattened), and latents.
            max_len (Optional[int]): If specified, the number of time steps for the output feature.
                                     Shorter sequences will be padded, longer ones will be truncated.
                                     If None, features are returned with their natural length.
            **kwargs: Additional arguments (ignored).
        """
        logger.info("Loading Descript Audio Codec (DAC) model '%s'...", model_type)
        try:
            model_path = dac.utils.download(model_type=model_type)
            self.model = dac.DAC.load(model_path).to(self.device)
            self.model.eval()

            self.output_type = output_type.lower()
            self.max_len = max_len

            valid_outputs = ["z", "codes", "latents", "all"]
            if self.output_type not in valid_outputs:
                raise ValueError(f"output_type must be one of {valid_outputs}")

            log_msg = f"DAC model '{model_type}' loaded. Output type: '{self.output_type}'."
            if self.max_len is not None:
                log_msg += f" Output length fixed to {self.max_len}."
            else:
                log_msg += " Output length is variable."
            logger.info(log_msg)

        except Exception as e:
            logger.error("Failed to load DAC model '%s': %s", model_type, e, exc_info=True)
            raise

    @torch.no_grad()
    def extract(self, audio_data: Union[np.ndarray, torch.Tensor], sample_rate: int, **kwargs: Any) -> torch.Tensor:
        """
        Extracts features using the DAC model.

        Args:
            audio_data (Union[np.ndarray, torch.Tensor]): Input audio waveform.
            sample_rate (int): Sample rate of the audio data.

        Returns:
            torch.Tensor: The extracted features as a tensor on the CPU. The time dimension
                          will be padded/truncated only if `max_len` was set during initialization.
        """
        try:
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.cpu().numpy()

            signal = AudioSignal(audio_data, sample_rate)
            signal.to(self.model.device)
            x = self.model.preprocess(signal.audio_data, signal.sample_rate)
            z, codes, latents, _, _ = self.model.encode(x)
            z = z.squeeze(0)
            codes = codes.squeeze(0)
            latents = latents.squeeze(0)

            features = None
            if self.output_type == "z":
                features = z.float()
            elif self.output_type == "codes":
                features = codes.float()
            elif self.output_type == "latents":
                features = latents.float()
            elif self.output_type == "all":
                features = torch.cat([z, codes.float(), latents], dim=0)

            if features is None:
                raise ValueError(f"Feature extraction failed for output_type '{self.output_type}'")

            if self.max_len is not None:
                current_len = features.shape[-1]
                if current_len < self.max_len:
                    pad_amount = self.max_len - current_len
                    features = F.pad(features, (0, pad_amount), "constant", 0)
                elif current_len > self.max_len:
                    features = features[..., : self.max_len]

            return features.cpu()

        except Exception as e:
            logger.error("DAC feature extraction failed: %s", e, exc_info=True)
            return torch.empty(0)