import logging
from typing import Any, Union, List
import numpy as np
import torch
from transformers import AutoFeatureExtractor, WavLMModel

from features.base import FeatureExtractor

logger = logging.getLogger(__name__)


class WavLMExtractor(FeatureExtractor):
    """
    Feature extractor using WavLM models from Hugging Face (e.g., microsoft/wavlm-large).
    """

    def _initialize(self, model_id: str = "microsoft/wavlm-large", output_hidden_states: bool = False, **kwargs):
        """
        Load the pre-trained WavLM model and its feature extractor.

        Args:
            model_id (str): The Hugging Face model identifier (e.g., "microsoft/wavlm-large").
            output_hidden_states (bool): If True, return all hidden states instead of just the last one.
            **kwargs: Additional arguments passed to the base class (e.g., device).
        """
        logger.info("Loading WavLM model and feature extractor '%s'...", model_id)
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            self.model = WavLMModel.from_pretrained(model_id).to(self.device)
            self.target_sr = self.feature_extractor.sampling_rate
            self.output_hidden_states = output_hidden_states
            self.model.eval()
            logger.info("WavLM model and feature extractor loaded successfully.")
        except Exception as e:
            logger.error("Failed to load WavLM model/feature extractor '%s': %s", model_id, e, exc_info=True)
            raise

    @torch.no_grad()
    def extract(self, audio_data: Union[np.ndarray, torch.Tensor], sample_rate: int, **kwargs: Any) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Extract features using the WavLM model.

        Args:
            audio_data (Union[np.ndarray, torch.Tensor]): Input audio waveform
                (numpy array or torch tensor). Expected to be MONO.
            sample_rate (int): Sample rate of the audio data.
            **kwargs (Any): Additional keyword arguments (ignored).

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: The extracted features.
                - If output_hidden_states is False: Returns the last hidden state tensor
                  ([1, sequence_length, hidden_size]).
                - If output_hidden_states is True: Returns a list of tensors for all hidden states.
                Returns tensor(s) on CPU.
        """
        try:
            if isinstance(audio_data, np.ndarray):
                audio_data = audio_data.astype(np.float32)
            elif isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.float().cpu().numpy()
            else:
                logger.error("Unsupported audio_data type: %s", type(audio_data))
                return torch.empty(0, device="cpu")

            if audio_data.ndim > 1:
                if audio_data.shape[0] > 1 and audio_data.shape[1] > 1:
                    logger.warning("Input audio has more than one channel. Averaging channels to mono.")
                    channel_axis = 0 if audio_data.shape[0] < audio_data.shape[1] else 1
                    audio_data = np.mean(audio_data, axis=channel_axis)
                elif audio_data.shape[0] == 1 or audio_data.shape[1] == 1:
                    audio_data = audio_data.flatten()
            audio_data = audio_data.squeeze()

            inputs = self.feature_extractor(audio_data, sampling_rate=sample_rate, return_tensors="pt", padding=True)

            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs, output_hidden_states=self.output_hidden_states)

            if self.output_hidden_states:
                features = [h.detach().cpu() for h in outputs.hidden_states]
            else:
                features = outputs.last_hidden_state.detach().cpu()

            return features
        except Exception as e:
            logger.error("WavLM feature extraction failed: %s", e, exc_info=True)
            return torch.empty(0, device="cpu")