import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from utils.torch_utils import check_tensor

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Configuration for audio processing within the Autoencoder."""

    sr: int = 16000
    n_mels: int = 128
    nfft: int = 2048
    fmin: float = 100.0
    fmax: float = 8000.0
    sample_dur: Optional[float] = None


class STFT(nn.Module):
    """STFT front-end for the autoencoder"""

    def __init__(self, nfft: int, hop_length: int, max_spec_width: int):
        """
        Initializes the STFT frontend.

        Args:
            nfft (int): FFT size.
            hop_length (int): Hop length.
            max_spec_width (int): Target spectrogram width (number of time frames).
        """
        super().__init__()
        self.nfft = nfft
        self.hop_length = hop_length
        self.max_spec_width = ((max_spec_width + 31) // 32) * 32
        self.register_buffer("window", torch.hann_window(nfft))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs STFT on the input audio.

        Args:
            x (torch.Tensor): Input audio tensor of shape [B, Time].

        Returns:
            torch.Tensor: Magnitude spectrogram tensor of shape [B, 1, Freq, Time].
        """
        target_length = (self.max_spec_width - 1) * self.hop_length

        if x.size(1) < target_length:
            padding = torch.zeros(x.size(0), target_length - x.size(1), device=x.device)
            x = torch.cat([x, padding], dim=1)
        elif x.size(1) > target_length:
            x = x[:, :target_length]

        z = torch.stft(
            x,
            n_fft=self.nfft,
            hop_length=self.hop_length,
            win_length=self.nfft,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        mag = torch.abs(z)

        if mag.size(-1) < self.max_spec_width:
            pad_size = self.max_spec_width - mag.size(-1)
            mag = torch.nn.functional.pad(mag, (0, pad_size))
        elif mag.size(-1) > self.max_spec_width:
            mag = mag[..., : self.max_spec_width]

        return mag.unsqueeze(1)


class MelFilter(nn.Module):
    """Mel filterbank for the autoencoder"""

    def __init__(self, sr: int, nfft: int, n_mels: int, fmin: float, fmax: float):
        """
        Initializes the Mel filterbank.

        Args:
            sr (int): Sample rate.
            nfft (int): FFT size.
            n_mels (int): Number of Mel bins.
            fmin (float): Minimum frequency.
            fmax (float): Maximum frequency.
        """
        super().__init__()
        self.n_mels = n_mels
        self.mel_scale = T.MelScale(
            n_mels=n_mels,
            sample_rate=sr,
            f_min=fmin,
            f_max=fmax,
            n_stft=nfft // 2 + 1,
            norm="slaney",
            mel_scale="slaney",
        )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Applies the Mel filterbank and log scaling to the magnitude spectrogram.

        Args:
            spec (torch.Tensor): Magnitude spectrogram tensor of shape [B, 1, Freq, Time].

        Returns:
            torch.Tensor: Log-Mel spectrogram tensor of shape [B, 1, Mels, Time].
        """
        spec = spec.squeeze(1)

        power_spec = spec**2
        mel_power_spec = self.mel_scale(power_spec)

        log_mel_spec = torch.log1p(torch.clamp(mel_power_spec, min=1e-9))

        if not check_tensor(log_mel_spec, "LogMelSpec in MelFilter"):
            logger.error("NaN/Inf detected after MelScale+log1p in MelFilter! Returning zeros.")
            return torch.zeros_like(log_mel_spec).unsqueeze(1)

        return log_mel_spec.unsqueeze(1)


class Autoencoder(nn.Module):
    """Autoencoder model for audio features (Paper Specific Structure)."""

    def __init__(
        self,
        config: AudioConfig,
        max_spec_width: int,
        bottleneck_dim: int = 256,
    ):
        """
        Initializes the Autoencoder model.

        Args:
            config (AudioConfig): Audio processing configuration.
            max_spec_width (int): Target spectrogram width.
            bottleneck_dim (int): Dimension of the bottleneck layer.
        """
        super().__init__()
        self.config = config
        hop_length = max(config.nfft // 4, 1)
        self.target_width = ((max_spec_width + 31) // 32) * 32

        self.frontend = nn.Sequential(
            STFT(config.nfft, hop_length, max_spec_width),
            MelFilter(config.sr, config.nfft, config.n_mels, config.fmin, config.fmax),
            nn.InstanceNorm2d(1, eps=1e-4),
        )

        self.encoder = nn.Sequential(
            self._make_conv_block(1, 32),
            self._make_conv_block(32, 64),
            self._make_conv_block(64, 128),
            self._make_conv_block(128, 256),
            nn.Conv2d(256, bottleneck_dim, 3, stride=2, padding=1),
        )

        self.decoder = nn.Sequential(
            self._make_deconv_block(bottleneck_dim, 256),
            self._make_deconv_block(256, 128),
            self._make_deconv_block(128, 64),
            self._make_deconv_block(64, 32),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input audio tensor of shape [B, Time].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the decoded spectrogram
                                               ([B, 1, Mels, Time]) and the encoded features
                                               ([B, bottleneck_dim, H, W]).
        """
        mel_spec = self.frontend(x)
        if not check_tensor(mel_spec, "Frontend Output"):
            logger.error("NaN/Inf detected in frontend output! Returning zeros.")
            est_h = mel_spec.shape[2] // 32
            est_w = mel_spec.shape[3] // 32
            bottleneck_channels = self.encoder[-1].out_channels
            dummy_encoded_shape = (
                x.shape[0],
                bottleneck_channels,
                est_h,
                est_w,
            )
            return torch.zeros_like(mel_spec), torch.zeros(dummy_encoded_shape, device=mel_spec.device)

        encoded = self.encoder(mel_spec)
        if not check_tensor(encoded, "Encoder Output"):
            logger.error("NaN/Inf detected in encoder output! Returning zeros.")
            return torch.zeros_like(mel_spec), torch.zeros_like(encoded)

        decoded = self.decoder(encoded)
        if not check_tensor(decoded, "Decoder Output"):
            logger.error("NaN/Inf detected in decoder output! Returning zeros.")
            return torch.zeros_like(decoded), encoded

        if decoded.shape != mel_spec.shape:
            decoded = torch.nn.functional.interpolate(
                decoded,
                size=mel_spec.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        return decoded, encoded

    @staticmethod
    def _make_conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """Creates a convolutional block for the encoder."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    @staticmethod
    def _make_deconv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """Creates a deconvolutional block for the decoder."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )