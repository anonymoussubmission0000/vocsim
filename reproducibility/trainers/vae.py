import functools
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from reproducibility.models.vae import VariationalAutoencoder, preprocess_vae_input
from trainers.base import Trainer


logger = logging.getLogger(__name__)

X_DIM = 128 * 128


def vae_collate_fn(
    batch: List[Dict[str, Any]],
    target_sr: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window_fn_str: str,
    spec_height: int,
    spec_width: int,
    window_samples: int,
    hop_samples: int,
) -> Tuple[Optional[torch.Tensor], List[Optional[Any]]]:
    """
    Collate function for VAE training.

    Args:
        batch (List[Dict[str, Any]]): A list of dataset items (dictionaries).
        target_sr (int): Target sample rate for resampling.
        n_fft (int): FFT size for spectrogram calculation.
        hop_length (int): Hop length for spectrogram calculation.
        win_length (int): Window length for spectrogram calculation.
        window_fn_str (str): Name of the window function ('hann', 'hamming').
        spec_height (int): Target spectrogram height (number of Mel bins).
        spec_width (int): Target spectrogram width (number of time frames).
        window_samples (int): Number of audio samples corresponding to the desired spectrogram window width.
        hop_samples (int): Number of audio samples corresponding to the desired hop between spectrogram windows.

    Returns:
        Tuple[Optional[torch.Tensor], List[Optional[Any]]]: A tuple containing:
            - Padded spectrogram chunks tensor [batch_size, spec_height, spec_width], or empty tensor.
            - List of metadata/IDs corresponding to original items.
    """
    all_processed_chunks = []
    metadata_list = []
    preprocess_device = torch.device("cpu")

    for item in batch:
        audio_info = item.get("audio")
        metadata = item.get("metadata_or_id", None)
        if not audio_info or "array" not in audio_info or "sampling_rate" not in audio_info:
            metadata_list.append(metadata)
            continue
        audio_data = audio_info["array"]
        sample_rate = audio_info["sampling_rate"]
        if audio_data is None or sample_rate is None:
            metadata_list.append(metadata)
            continue

        try:
            processed_chunks = preprocess_vae_input(
                audio_tensor=audio_data,
                sample_rate=sample_rate,
                target_sr=target_sr,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window_fn_str=window_fn_str,
                spec_height=spec_height,
                spec_width=spec_width,
                window_samples=window_samples,
                hop_samples=hop_samples,
                device=preprocess_device,
            )
            all_processed_chunks.extend(processed_chunks)
            metadata_list.extend([metadata] * len(processed_chunks))
        except Exception as e:
            metadata_list.append(metadata)
            continue

    if not all_processed_chunks:
        return (
            torch.empty((0, spec_height, spec_width), device=preprocess_device),
            [],
        )

    try:
        stacked_chunks = torch.cat(all_processed_chunks, dim=0)
        return stacked_chunks, metadata_list
    except Exception as e:
        return (
            torch.empty((0, spec_height, spec_width), device=preprocess_device),
            [],
        )


class PaperVAETrainer(Trainer):
    """
    Trains the specific Variational Autoencoder architecture used in the paper.
    """

    def _initialize_components(self):
        """
        Initialize optimizer, loss (ELBO), etc. for the paper's VAE.
        """
        logger.info("Initializing components for PaperVAETrainer...")

        self.vae_params = self.config.get("vae_frontend_params", {})
        self.target_sr = self.vae_params.get("target_sr", 16000)
        self.n_fft = self.vae_params.get("n_fft", 512)
        self.hop_length = self.vae_params.get("hop_length", 256)
        self.win_length = self.vae_params.get("win_length", self.n_fft)
        self.window_fn_str = self.vae_params.get("window_fn_str", "hann")
        self.spec_height = self.vae_params.get("spec_height", 128)
        self.spec_width = self.vae_params.get("spec_width", 128)
        window_overlap = self.vae_params.get("window_overlap", 0.5)
        self.window_samples = (self.spec_width - 1) * self.hop_length
        self.hop_samples = int(self.window_samples * (1 - window_overlap))

        lr = self.config.get("learning_rate", 1e-3)
        self.test_freq = self.config.get("test_frequency", 25)
        self.save_freq = self.config.get("save_frequency_epochs", 10)
        self.mixed_precision = self.config.get("mixed_precision", False)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        logger.debug("Optimizer: Adam (lr=%f)", lr)
        self.criterion = None
        logger.debug("Criterion: Negative ELBO (calculated in VAE forward)")

        if self.mixed_precision and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
            logger.debug("Mixed precision enabled with GradScaler.")
        else:
            self.scaler = None
            if self.mixed_precision and self.device.type == "cpu":
                logger.warning("Mixed precision requested but device is CPU. Disabling.")
                self.mixed_precision = False

        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.loss_history = {"train": {}, "val": {}}
        self.writer = SummaryWriter(log_dir=str(self.logs_dir))

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """
        Execute the training loop for the paper's VAE.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (Optional[DataLoader]): DataLoader for the validation dataset.
        """
        num_epochs = self.config.get("num_epochs", 50)

        logger.info("Starting VAE training for %d epochs...", num_epochs)
        if hasattr(train_loader, "batch_size"):
            logger.info("Batch size: %d", train_loader.batch_size)

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            start_time = time.time()

            self.model.train()
            total_train_loss = 0.0
            total_train_samples = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for batch_idx, batch in enumerate(pbar):
                data, _ = batch
                if data is None or data.numel() == 0:
                    continue

                data = data.to(self.device)

                batch_num_samples = data.size(0)
                if batch_num_samples == 0:
                    continue
                total_train_samples += batch_num_samples

                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    loss = self.model(data)

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                total_train_loss += loss.item() * batch_num_samples
                if total_train_samples > 0:
                    pbar.set_postfix({"avg_chunk_loss": f"{total_train_loss / total_train_samples:.4f}"})

            avg_train_loss = total_train_loss / total_train_samples if total_train_samples > 0 else float("inf")
            self.loss_history["train"][epoch] = avg_train_loss

            epoch_time = time.time() - start_time
            logger.info(
                "Epoch %d Train Summary | Time: %.2fs | Avg Loss (Neg ELBO per chunk): %.4f",
                epoch + 1,
                epoch_time,
                avg_train_loss,
            )
            self.writer.add_scalar("train/loss_per_chunk", avg_train_loss, epoch)
            self.writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]["lr"], epoch)

            if val_loader and (epoch + 1) % self.test_freq == 0:
                self.model.eval()
                total_val_loss = 0.0
                total_val_samples = 0
                with torch.no_grad():
                    pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
                    for batch in pbar_val:
                        data, _ = batch
                        if data is None or data.numel() == 0:
                            continue
                        data = data.to(self.device)
                        batch_num_samples = data.size(0)
                        if batch_num_samples == 0:
                            continue
                        total_val_samples += batch_num_samples

                        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                            loss = self.model(data)

                        total_val_loss += loss.item() * batch_num_samples
                        if total_val_samples > 0:
                            pbar_val.set_postfix({"avg_chunk_loss": f"{total_val_loss / total_val_samples:.4f}"})

                avg_val_loss = total_val_loss / total_val_samples if total_val_samples > 0 else float("inf")
                self.loss_history["val"][epoch] = avg_val_loss
                logger.info("Epoch %d Val Summary | Avg Loss (Neg ELBO per chunk): %.4f", epoch + 1, avg_val_loss)
                self.writer.add_scalar("val/loss_per_chunk", avg_val_loss, epoch)

                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    logger.info("New best validation loss: %.4f", self.best_val_loss)
                    self.save_model(epoch=epoch + 1, is_best=True)

            if (epoch + 1) % self.save_freq == 0 and epoch > 0:
                self.save_model(epoch=epoch + 1, is_best=False)

        logger.info("VAE Training finished.")
        self.save_model(epoch=self.current_epoch + 1, is_best=False, final=True)
        self.writer.close()

    def save_model(self, epoch: Optional[int] = None, is_best: bool = False, final: bool = False):
        """
        Saves the VAE model checkpoint.

        Args:
            epoch (Optional[int]): The current epoch number.
            is_best (bool): True if this is the best model so far.
            final (bool): True if this is the final model after training completes.
        """
        filename = "final_model.pt" if final else f"checkpoint_epoch_{epoch}.pt"
        checkpoint_data = {
            "z_dim": self.model.z_dim,
            "model_precision": self.model.model_precision,
            "lr": self.model.lr,
            "vae_frontend_params": {
                "target_sr": self.target_sr,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "win_length": self.win_length,
                "window_fn_str": self.window_fn_str,
                "spec_height": self.spec_height,
                "spec_width": self.spec_width,
                "window_overlap": self.config.get("vae_frontend_params", {}).get("window_overlap", 0.5),
            },
        }
        self._save_checkpoint(filename, epoch, is_best, extra_data=checkpoint_data)

    def _save_checkpoint(
        self,
        filename: str,
        epoch: Optional[int],
        is_best: bool,
        extra_data: Optional[Dict] = None,
    ):
        """
        Helper method to save a checkpoint dictionary including VAE params.

        Args:
            filename (str): The name of the checkpoint file.
            epoch (Optional[int]): The current epoch number.
            is_best (bool): True if this is the best model checkpoint.
            extra_data (Optional[Dict]): Additional data to include in the checkpoint.
        """
        if self.optimizer is None:
            logger.warning("Optimizer not initialized, cannot save its state.")

        base_checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": None,
            "config": self.config,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "best_val_loss": self.best_val_loss,
            "loss_history": self.loss_history,
        }
        if extra_data:
            base_checkpoint.update(extra_data)

        filepath = self.checkpoints_dir / filename
        torch.save(base_checkpoint, filepath)
        logger.info("Checkpoint saved to %s", filepath)

        if is_best:
            best_filepath = self.output_dir / "final_model.pt"
            torch.save(base_checkpoint, best_filepath)
            logger.info("Best model checkpoint saved to %s", best_filepath)