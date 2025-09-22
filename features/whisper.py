import logging
from typing import Union, Any, List
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperModel
import librosa

from features.base import FeatureExtractor

logger = logging.getLogger(__name__)


class WhisperEncoderExtractor(FeatureExtractor):
    """
    Feature extractor using the encoder part of Whisper models from Hugging Face.
    """

    def _initialize(
        self,
        model_id: str = "openai/whisper-large-v3",
        output_hidden_states: bool = False,
        **kwargs,
    ):
        """
        Load the pre-trained Whisper model and processor.

        Args:
            model_id: The Hugging Face model identifier.
            output_hidden_states: If True, return all hidden states.
        """
        logger.info(f"Loading Whisper model and processor '{model_id}'...")
        try:
            self.processor = WhisperProcessor.from_pretrained(model_id)
            self.model = WhisperModel.from_pretrained(model_id).to(self.device)
            self.target_sr = self.processor.feature_extractor.sampling_rate
            self.output_hidden_states = output_hidden_states
            self.model.eval()
            logger.info(f"Whisper model and processor loaded successfully on device {self.device}.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{model_id}': {e}", exc_info=True)
            raise

    @torch.no_grad()
    def extract(self, audio_data: Union[np.ndarray, torch.Tensor], sample_rate: int, **kwargs: Any) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Extract features using the Whisper encoder. Handles resampling to 16kHz.

        Args:
            audio_data: Input audio waveform (numpy array or torch tensor). Mono expected.
            sample_rate: Sample rate of the audio data.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Features from the encoder. Last hidden state ([1, sequence_length, hidden_size])
            or list of all hidden states if output_hidden_states is True. Tensor(s) on CPU.
        """
        try:
            if isinstance(audio_data, torch.Tensor):
                audio_np = audio_data.float().cpu().numpy()
            elif isinstance(audio_data, np.ndarray):
                audio_np = audio_data.astype(np.float32)
            else:
                raise TypeError("audio_data must be numpy array or torch tensor")

            if audio_np.ndim > 1:
                logger.warning(f"Whisper expects mono audio, input has {audio_np.ndim} dims. Taking mean.")
                channel_axis = 0 if audio_np.shape[0] < audio_np.shape[1] else 1
                audio_np = np.mean(audio_np, axis=channel_axis)

            if sample_rate != self.target_sr:
                logger.debug(f"Resampling audio from {sample_rate} Hz to {self.target_sr} Hz...")
                try:
                    audio_np = librosa.resample(y=audio_np, orig_sr=sample_rate, target_sr=self.target_sr)
                    current_sample_rate = self.target_sr
                    logger.debug("Resampling successful.")
                except Exception as resample_err:
                    logger.error(f"Librosa resampling failed: {resample_err}", exc_info=True)
                    return torch.empty(0, device="cpu")
            else:
                current_sample_rate = sample_rate

            inputs = self.processor(audio_np, sampling_rate=current_sample_rate, return_tensors="pt")
            input_features = inputs.input_features.to(self.device)

            encoder_outputs = self.model.encoder(input_features, output_hidden_states=self.output_hidden_states)

            if self.output_hidden_states:
                features = [h.cpu() for h in encoder_outputs.hidden_states]
            else:
                features = encoder_outputs.last_hidden_state.cpu()
            return features

        except Exception as e:
            logger.error(f"Whisper Encoder feature extraction failed: {e}", exc_info=True)
            return torch.empty(0, device="cpu")