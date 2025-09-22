import logging
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.distributions import LowRankMultivariateNormal

logger = logging.getLogger(__name__)

X_SHAPE = (128, 128)
X_DIM = np.prod(X_SHAPE)


def preprocess_vae_input(
    audio_tensor: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    target_sr: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window_fn_str: str,
    spec_height: int,
    spec_width: int,
    window_samples: int,
    hop_samples: int,
    device: torch.device,
) -> List[torch.Tensor]:
    """
    Takes raw audio, resamples, chunks, computes spectrograms, and resizes.

    Args:
        audio_tensor (Union[np.ndarray, torch.Tensor]): Raw audio waveform.
        sample_rate (int): Original sample rate of the audio.
        target_sr (int): Target sample rate for resampling.
        n_fft (int): FFT size for spectrogram calculation.
        hop_length (int): Hop length for spectrogram calculation.
        win_length (int): Window length for spectrogram calculation.
        window_fn_str (str): Name of the window function ('hann', 'hamming').
        spec_height (int): Target spectrogram height (number of Mel bins).
        spec_width (int): Target spectrogram width (number of time frames).
        window_samples (int): Number of audio samples corresponding to the desired spectrogram window width.
        hop_samples (int): Number of audio samples corresponding to the desired hop between spectrogram windows.
        device (torch.device): Device to perform computation on.

    Returns:
        List[torch.Tensor]: A list of preprocessed spectrogram chunks for the VAE.
                            Shape of each chunk: [1, H, W]
    """
    if not isinstance(audio_tensor, torch.Tensor):
        if not isinstance(audio_tensor, np.ndarray):
            raise TypeError(f"Expected numpy array or torch tensor, got {type(audio_tensor)}")
        if audio_tensor.dtype != np.float32:
            audio_tensor = audio_tensor.astype(np.float32)
        audio_tensor = torch.from_numpy(audio_tensor)

    if audio_tensor.ndim > 1 and audio_tensor.shape[0] > 1:
        audio_tensor = torch.mean(audio_tensor, dim=0)
    if audio_tensor.ndim == 2 and audio_tensor.shape[0] == 1:
        audio_tensor = audio_tensor.squeeze(0)

    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr).to(device)
        audio_tensor = resampler(audio_tensor.to(device))
    else:
        audio_tensor = audio_tensor.to(device)

    if len(audio_tensor) < window_samples:
        padding = window_samples - len(audio_tensor)
        audio_tensor = F.pad(audio_tensor, (0, padding))
        num_chunks = 1
        starts = [0]
    else:
        starts = list(range(0, len(audio_tensor) - window_samples + 1, hop_samples))
        if not starts or (starts[-1] + window_samples < len(audio_tensor)):
            starts.append(len(audio_tensor) - window_samples)
        num_chunks = len(starts)

    if num_chunks == 0:
        return []

    window_fn_map = {"hann": torch.hann_window, "hamming": torch.hamming_window}
    window_fn = window_fn_map.get(window_fn_str, torch.hann_window)
    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window_fn=window_fn,
        power=None,
        center=True,
        pad_mode="reflect",
    ).to(device)

    processed_chunks = []
    for start_idx in starts:
        end_idx = start_idx + window_samples
        audio_chunk = audio_tensor[start_idx:end_idx]

        if audio_chunk.numel() == 0:
            logger.warning("Encountered empty audio chunk during VAE preprocessing, skipping.")
            continue

        spec_complex = spec_transform(audio_chunk)
        spec = spec_complex.abs()
        spec = torch.log1p(spec)

        spec = spec.unsqueeze(0).unsqueeze(0)
        spec_resized = F.interpolate(spec, size=(spec_height, spec_width), mode="bilinear", align_corners=False)
        processed_chunks.append(spec_resized.squeeze(1))

    return processed_chunks


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder class for single-channel spectrograms (Paper Specific)."""

    def __init__(
        self,
        z_dim: int = 32,
        model_precision: float = 10.0,
        device_name: str = "cpu",
        lr: float = 1e-3,
    ):
        """
        Initializes the Variational Autoencoder model.

        Args:
            z_dim (int): The dimension of the latent space.
            model_precision (float): Precision parameter for the reconstruction likelihood term.
            device_name (str): Name of the device to move the model to.
            lr (float): Learning rate (stored for context, not used in forward pass).
        """
        super().__init__()
        self.z_dim = z_dim
        self.model_precision = model_precision
        self.device = torch.device(device_name)
        self.lr = lr

        self.conv1 = nn.Conv2d(1, 8, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 2, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, 2, padding=1)
        self.conv5 = nn.Conv2d(16, 24, 3, 1, padding=1)
        self.conv6 = nn.Conv2d(24, 24, 3, 2, padding=1)
        self.conv7 = nn.Conv2d(24, 32, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(8)
        self.bn4 = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(16)
        self.bn6 = nn.BatchNorm2d(24)
        self.bn7 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(32 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc31 = nn.Linear(256, 64)
        self.fc32 = nn.Linear(256, 64)
        self.fc33 = nn.Linear(256, 64)
        self.fc41 = nn.Linear(64, self.z_dim)
        self.fc42 = nn.Linear(64, self.z_dim)
        self.fc43 = nn.Linear(64, self.z_dim)

        self.fc5 = nn.Linear(self.z_dim, 64)
        self.fc6 = nn.Linear(64, 256)
        self.fc7 = nn.Linear(256, 1024)
        self.fc8 = nn.Linear(1024, 8192)
        self.convt1 = nn.ConvTranspose2d(32, 24, 3, 1, padding=1)
        self.convt2 = nn.ConvTranspose2d(24, 24, 3, 2, padding=1, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(24, 16, 3, 1, padding=1)
        self.convt4 = nn.ConvTranspose2d(16, 16, 3, 2, padding=1, output_padding=1)
        self.convt5 = nn.ConvTranspose2d(16, 8, 3, 1, padding=1)
        self.convt6 = nn.ConvTranspose2d(8, 8, 3, 2, padding=1, output_padding=1)
        self.convt7 = nn.ConvTranspose2d(8, 1, 3, 1, padding=1)
        self.bn8 = nn.BatchNorm2d(32)
        self.bn9 = nn.BatchNorm2d(24)
        self.bn10 = nn.BatchNorm2d(24)
        self.bn11 = nn.BatchNorm2d(16)
        self.bn12 = nn.BatchNorm2d(16)
        self.bn13 = nn.BatchNorm2d(8)
        self.bn14 = nn.BatchNorm2d(8)

        self.to(self.device)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes the input spectrogram into latent distribution parameters (mu, u, d_diag).

        Args:
            x (torch.Tensor): Input spectrogram tensor of shape [B, H, W] or [B, 1, H, W].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing:
                - mu (torch.Tensor): Mean vector of the latent distribution [B, z_dim].
                - u (torch.Tensor): Factor `u` for the covariance matrix [B, z_dim, 1].
                - d_diag (torch.Tensor): Diagonal factor `d` for the covariance matrix [B, z_dim].
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() != 4 or x.shape[1] != 1:
            raise ValueError(f"Input tensor must be [B, H, W] or [B, 1, H, W], got {x.shape}")

        x = F.relu(self.conv1(self.bn1(x)))
        x = F.relu(self.conv2(self.bn2(x)))
        x = F.relu(self.conv3(self.bn3(x)))
        x = F.relu(self.conv4(self.bn4(x)))
        x = F.relu(self.conv5(self.bn5(x)))
        x = F.relu(self.conv6(self.bn6(x)))
        x = F.relu(self.conv7(self.bn7(x)))

        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = self.fc41(F.relu(self.fc31(x)))
        u = self.fc42(F.relu(self.fc32(x))).unsqueeze(-1)
        d_diag = torch.exp(self.fc43(F.relu(self.fc33(x))))

        return mu, u, d_diag

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes a latent vector z back into a flattened spectrogram representation.

        Args:
            z (torch.Tensor): Latent vector tensor of shape [B, z_dim].

        Returns:
            torch.Tensor: Flattened reconstructed spectrogram tensor of shape [B, H*W].
        """
        z = F.relu(self.fc5(z))
        z = F.relu(self.fc6(z))
        z = F.relu(self.fc7(z))
        z = F.relu(self.fc8(z))
        z = z.view(-1, 32, 16, 16)
        z = F.relu(self.convt1(self.bn8(z)))
        z = F.relu(self.convt2(self.bn9(z)))
        z = F.relu(self.convt3(self.bn10(z)))
        z = F.relu(self.convt4(self.bn11(z)))
        z = F.relu(self.convt5(self.bn12(z)))
        z = F.relu(self.convt6(self.bn13(z)))
        z = self.convt7(self.bn14(z))

        return z.view(-1, X_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass and calculates the VAE loss (negative ELBO).

        Args:
            x (torch.Tensor): Input spectrogram tensor of shape [B, 1, H, W] or [B, H, W].

        Returns:
            torch.Tensor: The negative ELBO loss for the batch.
        """
        mu, u, d_diag = self.encode(x)
        latent_dist = LowRankMultivariateNormal(mu, u, d_diag)
        z = latent_dist.rsample()
        x_rec_flat = self.decode(z)
        x_flat = x.view(x.shape[0], -1)

        log_pxz = -0.5 * X_DIM * torch.log(
            torch.tensor(2.0 * np.pi / self.model_precision, device=self.device)
        ) - 0.5 * self.model_precision * torch.sum(torch.pow(x_flat - x_rec_flat, 2), dim=1)
        log_prior_term = -0.5 * torch.sum(torch.pow(z, 2), dim=1) - 0.5 * self.z_dim * np.log(2 * np.pi)
        entropy_term = latent_dist.entropy()
        elbo_per_sample = log_pxz + log_prior_term + entropy_term
        return -torch.mean(elbo_per_sample)