import logging
from typing import Optional

import numpy as np
import torch
from transformers.audio_utils import mel_filter_bank

logger = logging.getLogger(__name__)

CUSTOM_N_MELS = 128


class WhisperSegFrontend:
    """
    Calculates the 128-bin log-Mel spectrogram required by WhisperSeg
    for a single, pre-processed audio chunk.
    """

    def __init__(
        self,
        sr: int,
        n_fft: int,
        hop_length: int,
        min_frequency: float = 0.0,
        max_frequency: Optional[float] = None,
    ):
        """
        Initializes the WhisperSegFrontend.

        Args:
            sr (int): The sampling rate of the audio.
            n_fft (int): The FFT size for the STFT.
            hop_length (int): The hop length for the STFT.
            min_frequency (float): The minimum frequency for the Mel filterbank.
            max_frequency (Optional[float]): The maximum frequency for the Mel filterbank.
        """
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_freq_bins = 1 + self.n_fft // 2
        self.num_mel_bins = CUSTOM_N_MELS

        _max_frequency = max_frequency if max_frequency is not None else float(sr // 2)

        try:
            mel_filters_np = mel_filter_bank(
                num_frequency_bins=self.num_freq_bins,
                num_mel_filters=self.num_mel_bins,
                min_frequency=min_frequency,
                max_frequency=_max_frequency,
                sampling_rate=sr,
                norm="slaney",
                mel_scale="htk",
            )
            mel_filters_np_transposed = mel_filters_np.T
            self.mel_filters_torch = torch.from_numpy(mel_filters_np_transposed).float()

            self.window = torch.hann_window(self.n_fft)
            logger.debug(f"Created Mel filterbank tensor with shape: {self.mel_filters_torch.shape}")

        except Exception as e:
            logger.error(f"Failed to create Mel filterbank: {e}", exc_info=True)
            raise

    def __call__(self, audio_chunk_padded: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculates the log-mel spectrogram for a single audio chunk.

        Args:
            audio_chunk_padded (np.ndarray): A 1D numpy array containing the audio chunk,
                                             zero-padded to be a multiple of hop_length
                                             plus n_fft at the end.

        Returns:
            Optional[np.ndarray]: The log-mel spectrogram as a numpy array
                                  of shape [num_mel_bins, num_frames], or None if an error occurs.
        """
        if not isinstance(audio_chunk_padded, np.ndarray):
            logger.error("Input audio_chunk_padded must be a numpy array.")
            return None
        if audio_chunk_padded.ndim != 1:
            logger.error(f"Input audio_chunk_padded must be 1D, got shape {audio_chunk_padded.shape}")
            return None
        if audio_chunk_padded.dtype != np.float32:
            audio_chunk_padded = audio_chunk_padded.astype(np.float32)

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            waveform_tensor = torch.from_numpy(audio_chunk_padded).to(device)
            window_d = self.window.to(device)
            mel_filters_tensor_d = self.mel_filters_torch.to(device)

            stft = torch.stft(
                waveform_tensor,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=window_d,
                center=True,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
            if stft.shape[0] != self.num_freq_bins:
                logger.error(f"STFT output freq bins {stft.shape[0]} != expected {self.num_freq_bins}")
                return None

            magnitudes_power = stft.abs() ** 2

            if mel_filters_tensor_d.shape[1] != magnitudes_power.shape[0]:
                logger.error(
                    f"Cannot multiply Mel filter ({mel_filters_tensor_d.shape}) and STFT"
                    f" ({magnitudes_power.shape}): Freq bins mismatch."
                )
                return None

            mel_spec = torch.matmul(mel_filters_tensor_d, magnitudes_power)

            log_spec = torch.log10(torch.clamp(mel_spec, min=1e-10))
            if log_spec.numel() > 0:
                log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
            else:
                return np.zeros((self.num_mel_bins, 0), dtype=np.float32)
            log_spec = (log_spec + 4.0) / 4.0

            return log_spec.cpu().numpy()

        except Exception as spec_err:
            logger.error(f"Error computing spectrogram: {spec_err}", exc_info=True)
            return None