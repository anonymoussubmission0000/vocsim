"""Mel Spectrogram feature extractor."""

import logging
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T

from features.base import FeatureExtractor

logger = logging.getLogger(__name__)


class MelExtractor(FeatureExtractor):
    """
    Computes Mel spectrograms using torchaudio.
    """

    def _initialize(
        self,
        sr: int = 16000,
        n_fft: int = 1024,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        power: float = 2.0,
        normalized: bool = False,
        log_scale: bool = True,
        **kwargs,
    ):
        """
        Initialize the Mel spectrogram extractor.

        Args:
            sr (int): Target sample rate.
            n_fft (int): Size of the FFT window.
            hop_length (Optional[int]): The number of frames between successive STFT windows. Defaults to n_fft // 4.
            win_length (Optional[int]): The window length. Defaults to n_fft.
            n_mels (int): Number of Mel filter banks.
            f_min (float): Minimum frequency.
            f_max (Optional[float]): Maximum frequency. Defaults to sr / 2.0.
            power (float): Exponent for the magnitude spectrogram (2.0 for power spectrogram).
            normalized (bool): Whether to normalize Mel banks.
            log_scale (bool): Whether to convert to log scale using AmplitudeToDB.
            **kwargs: Additional keyword arguments (ignored).
        """
        self.target_sr = sr
        self.log_scale = log_scale
        self.n_fft = n_fft

        _hop_length = hop_length if hop_length is not None else n_fft // 4
        _win_length = win_length if win_length is not None else n_fft
        _f_max = f_max if f_max is not None else sr / 2.0

        self._win_length = _win_length
        self._hop_length = _hop_length

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            win_length=_win_length,
            hop_length=_hop_length,
            f_min=f_min,
            f_max=_f_max,
            n_mels=n_mels,
            power=power,
            normalized=normalized,
            center=True,
        ).to(self.device)

        if self.log_scale:
            _stype = "power" if power == 2.0 else "magnitude"
            self.log_scaler = T.AmplitudeToDB(stype=_stype, top_db=80.0).to(
                self.device
            )

        logger.info(
            f"MelExtractor initialized (SR: {sr}, N_FFT: {n_fft}, Hop: {_hop_length}, Mels: {n_mels})"
        )

    def extract(
        self, audio_data: Union[np.ndarray, torch.Tensor], sample_rate: int, **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute the Mel spectrogram.

        Handles resampling, mono conversion, and padding for short inputs.

        Args:
            audio_data (Union[np.ndarray, torch.Tensor]): Input audio waveform
                (numpy array or torch tensor). Can be mono or stereo.
            sample_rate (int): Sample rate of the audio data.
            **kwargs (Any): Additional keyword arguments (ignored).

        Returns:
            torch.Tensor: The computed Mel spectrogram tensor ([n_mels, num_frames]) on CPU,
                          or an empty tensor if extraction fails.
        """
        try:
            if isinstance(audio_data, np.ndarray):
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                audio_tensor = torch.from_numpy(audio_data)
            elif isinstance(audio_data, torch.Tensor):
                if audio_data.dtype != torch.float32:
                    audio_tensor = audio_data.float()
                else:
                    audio_tensor = audio_data
            else:
                raise TypeError("Input audio_data must be ndarray or tensor")

            audio_tensor = audio_tensor.to(self.device)

            if sample_rate != self.target_sr:
                resampler = T.Resample(
                    orig_freq=sample_rate, new_freq=self.target_sr
                ).to(self.device)
                audio_tensor = resampler(audio_tensor)

            if audio_tensor.ndim > 1:
                if audio_tensor.shape[0] > 1 and audio_tensor.shape[1] > 1:
                    audio_tensor = torch.mean(audio_tensor, dim=0)
                elif audio_tensor.shape[0] == 1:
                    audio_tensor = audio_tensor.squeeze(0)
                elif audio_tensor.shape[1] == 1:
                    audio_tensor = audio_tensor.squeeze(1)
                elif audio_tensor.ndim > 2:
                    raise ValueError(f"Unsupported audio dimensionality {audio_tensor.ndim}. Expected 1D or 2D [C, T].")

            min_len = self._win_length
            current_len = audio_tensor.shape[-1] if audio_tensor.ndim > 0 else 0

            if current_len == 0:
                logger.warning("Input audio is empty after potential resampling/mono.")
                return torch.empty(0, self.mel_transform.n_mels).transpose(0, 1).cpu()

            if current_len < min_len:
                pad_amount = min_len - current_len
                audio_tensor = F.pad(
                    audio_tensor,
                    (pad_amount // 2, pad_amount - pad_amount // 2),
                    mode="constant",
                    value=0,
                )
                logger.debug("Padded short audio to length %d", audio_tensor.shape[-1])

            audio_tensor_input = audio_tensor.unsqueeze(0).unsqueeze(0)

            mel_spec = self.mel_transform(audio_tensor_input)

            if self.log_scale:
                mel_spec = torch.clamp(mel_spec, min=1e-10)
                mel_spec = self.log_scaler(mel_spec)
                if not torch.isfinite(mel_spec).all():
                    logger.warning("Non-finite values found in MelSpec AFTER log scaling. Replacing.")
                    mel_spec = torch.nan_to_num(
                        mel_spec, nan=0.0, posinf=torch.max(mel_spec[torch.isfinite(mel_spec)]), neginf=-80.0
                    )

            return mel_spec.squeeze(0).squeeze(0).cpu()

        except Exception as e:
            input_shape_str = str(getattr(audio_tensor, "shape", "N/A"))
            logger.error("Mel extraction failed for initial input shape %s: %s", input_shape_str, e, exc_info=True)
            return torch.empty(0).cpu()