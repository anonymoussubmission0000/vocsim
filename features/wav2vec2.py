import logging
from typing import List, Union, Any, Optional
import numpy as np
import torch
import torchaudio.transforms as T
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from features.base import FeatureExtractor

logger = logging.getLogger(__name__)


class Wav2Vec2Extractor(FeatureExtractor):
    """
    Feature extractor using Wav2Vec2 models from Hugging Face (e.g., facebook/wav2vec2-base-960h).
    """

    def _initialize(self, model_id: str = "facebook/wav2vec2-base-960h", output_hidden_states: bool = False, **kwargs):
        """
        Load the pre-trained Wav2Vec2 model and processor.

        Args:
            model_id (str): The Hugging Face model identifier.
            output_hidden_states (bool): If True, return all hidden states instead of just the last one.
            **kwargs: Additional arguments (ignored).
        """
        logger.info("Loading Wav2Vec2 model and processor '%s'...", model_id)
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_id)
            self.model = Wav2Vec2Model.from_pretrained(model_id).to(self.device)
            self.target_sr = self.processor.feature_extractor.sampling_rate
            self.output_hidden_states = output_hidden_states
            self.model.eval()
            logger.info("Wav2Vec2 model and processor loaded successfully.")
        except Exception as e:
            logger.error("Failed to load Wav2Vec2 model '%s': %s", model_id, e, exc_info=True)
            raise

    @torch.no_grad()
    def extract(self, audio_data: Union[np.ndarray, torch.Tensor], sample_rate: int, **kwargs: Any) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Extract features using the Wav2Vec2 model.

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
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.cpu().numpy()

            if audio_data.ndim > 1:
                logger.warning("Wav2Vec2 expects mono audio, input has %d dims. Taking mean.", audio_data.ndim)
                audio_data = np.mean(audio_data, axis=0)

            inputs = self.processor(audio_data, sampling_rate=sample_rate, return_tensors="pt", padding=True)

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs, output_hidden_states=self.output_hidden_states)

            if self.output_hidden_states:
                features = [h.cpu() for h in outputs.hidden_states]
            else:
                features = outputs.last_hidden_state.cpu()

            return features
        except Exception as e:
            logger.error("Wav2Vec2 feature extraction failed: %s", e, exc_info=True)
            return torch.empty(0)