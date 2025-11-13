import logging
from typing import Any, Union
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from transformers import AutoModel
from features.base import FeatureExtractor

logger = logging.getLogger(__name__)


class EATExtractor(FeatureExtractor):
    """
    Feature extractor using the EAT (Efficient Audio Transformer) model.

    EAT is a self-supervised audio model designed for both effectiveness and efficiency.
    Paper: "EAT: Self-Supervised Pre-Training with Efficient Audio Transformer" (IJCAI 2024)
    GitHub: https://github.com/cwx-worst-one/EAT
    """

    def _initialize(self, model_id: str = "worstchan/EAT-large_epoch20_finetune_AS2M", **kwargs):
        """
        Load the pre-trained EAT model.

        Args:
            model_id (str): The Hugging Face model identifier for EAT.
            **kwargs: Additional arguments (ignored).
        """
        logger.info("Loading EAT model '%s'...", model_id)
        try:
            self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(self.device)
            self.target_sr = 16000  # EAT uses 16kHz sample rate

            # Mel-spectrogram parameters (from EAT documentation)
            self.n_mels = 128
            self.n_fft = 400  # 25ms window at 16kHz
            self.hop_length = 160  # 10ms shift at 16kHz
            self.fmin = 0
            self.fmax = 8000  # Nyquist frequency at 16kHz

            # Normalization parameters (from EAT model card)
            self.norm_mean = -4.268
            self.norm_std = 4.569

            self.model.eval()
            logger.info("EAT model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load EAT model '%s': %s", model_id, e, exc_info=True)
            raise

    def _compute_mel_spectrogram(self, waveform: np.ndarray) -> torch.Tensor:
        """
        Compute mel-spectrogram from waveform.

        Args:
            waveform (np.ndarray): Audio waveform at 16kHz

        Returns:
            torch.Tensor: Mel-spectrogram with shape [1, 1, time, freq]
        """
        # Ensure minimum audio length for mel-spectrogram computation
        # EAT uses 16x16 kernel, so we need at least 16 time frames
        min_samples = self.n_fft + (15 * self.hop_length)  # 16 frames minimum

        if len(waveform) < min_samples:
            # Pad audio if too short
            waveform = np.pad(waveform, (0, min_samples - len(waveform)), mode='constant')

        # Compute mel-spectrogram using librosa
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.target_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0,
        )

        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize
        mel_spec_normalized = (mel_spec_db - self.norm_mean) / self.norm_std

        # Convert to tensor and reshape to [1, 1, time, freq]
        # librosa returns [freq, time], we need [time, freq]
        mel_tensor = torch.from_numpy(mel_spec_normalized.T).float()  # [time, freq]
        mel_tensor = mel_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, time, freq]

        # Ensure time dimension is a multiple of 16 (pad if necessary)
        time_frames = mel_tensor.shape[2]
        if time_frames % 16 != 0:
            pad_frames = 16 - (time_frames % 16)
            mel_tensor = F.pad(mel_tensor, (0, 0, 0, pad_frames), mode='constant', value=0)

        return mel_tensor

    @torch.no_grad()
    def extract(self, audio_data: Union[np.ndarray, torch.Tensor], sample_rate: int, **kwargs: Any) -> torch.Tensor:
        """
        Extract audio features using the EAT model.

        Args:
            audio_data (Union[np.ndarray, torch.Tensor]): Input audio waveform
                (numpy array or torch tensor). Expected to be MONO.
            sample_rate (int): Sample rate of the audio data.
            **kwargs: Additional arguments:
                - return_type (str): 'utterance' (CLS token), 'frame' (all frames), or 'all' (default: 'utterance')

        Returns:
            torch.Tensor: The extracted audio features on CPU.
                - If return_type='utterance': shape [1, embedding_dim]
                - If return_type='frame': shape [1, num_frames, embedding_dim]
                - If return_type='all': shape [1, num_frames+1, embedding_dim] (includes CLS)
        """
        return_type = kwargs.get("return_type", "utterance")

        try:
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.cpu().numpy()

            # Ensure mono
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=0)

            # Resample if necessary
            if sample_rate != self.target_sr:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.target_sr)
                sample_rate = self.target_sr

            # Convert to mel-spectrogram
            mel_tensor = self._compute_mel_spectrogram(audio_data).to(self.device)

            # Extract features with EAT model
            outputs = self.model(mel_tensor)

            # Handle different output formats
            # EAT typically returns a dictionary or tuple with hidden states
            if isinstance(outputs, dict):
                if "last_hidden_state" in outputs:
                    features = outputs["last_hidden_state"]
                elif "hidden_states" in outputs:
                    features = outputs["hidden_states"][-1]
                else:
                    # Try to get the main output
                    features = outputs.get("logits", outputs.get("embeddings", outputs))
            elif isinstance(outputs, tuple):
                features = outputs[0]
            else:
                features = outputs

            # Return based on return_type
            if return_type == "utterance":
                # Return CLS token (first token)
                if features.dim() == 3:  # [batch, seq_len, dim]
                    features = features[:, 0, :]  # [batch, dim]
            elif return_type == "frame":
                # Return all frames except CLS token
                if features.dim() == 3:
                    features = features[:, 1:, :]  # [batch, seq_len-1, dim]
            # else return_type == "all": return everything as is

            return features.cpu()

        except Exception as e:
            logger.error("EAT feature extraction failed: %s", e, exc_info=True)
            return torch.empty(0)
