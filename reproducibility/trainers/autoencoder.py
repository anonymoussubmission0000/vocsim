# -*- coding: utf-8 -*-
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from reproducibility.models.autoencoder import Autoencoder
from trainers.base import Trainer

try:
    from utils.torch_utils import check_tensor
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("utils.torch_utils.check_tensor not found. NaN checks disabled.")

    def check_tensor(tensor, name):
        return True


logger = logging.getLogger(__name__)


def ae_collate_fn(batch: List[Dict[str, Any]]) -> Tuple[Optional[torch.Tensor], List[Optional[Any]]]:
    """
    Collate function for the Autoencoder DataLoader.

    Pads audio tensors to the maximum length in the batch and collects metadata.
    Handles potential errors or missing data in items.

    Args:
        batch (List[Dict[str, Any]]): A list of dataset items (dictionaries).

    Returns:
        Tuple[Optional[torch.Tensor], List[Optional[Any]]]: A tuple containing:
            - Padded audio tensor [batch_size, max_len], or empty tensor if no valid audio.
            - List of metadata/IDs corresponding to items in the batch.
    """
    audio_list = []
    metadata_list = []
    max_len = 0
    for item in batch:
        audio_info = item.get("audio")
        meta = item.get("metadata_or_id", None)
        if not audio_info or "array" not in audio_info or audio_info["array"] is None:
            metadata_list.append(meta)
            continue
        audio_data = audio_info["array"]
        if not isinstance(audio_data, torch.Tensor):
            if isinstance(audio_data, np.ndarray):
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                audio_data = torch.from_numpy(audio_data)
            elif isinstance(audio_data, list):
                audio_data = torch.tensor(audio_data, dtype=torch.float32)
            else:
                metadata_list.append(meta)
                continue
        elif audio_data.dtype != torch.float32:
            audio_data = audio_data.float()
        if audio_data.ndim > 1:
            if audio_data.shape[0] > 1 and audio_data.shape[1] > 1:
                audio_data = torch.mean(audio_data, dim=0)
            elif audio_data.shape[1] == 1:
                audio_data = audio_data.squeeze(1)
            elif audio_data.shape[0] == 1:
                audio_data = audio_data.squeeze(0)
        if audio_data.ndim != 1:
            logger.warning("Audio bad dims %d. Skip.", audio_data.ndim)
            metadata_list.append(meta)
            continue
        audio_list.append(audio_data)
        metadata_list.append(meta)
        if len(audio_data) > max_len:
            max_len = len(audio_data)
    if not audio_list:
        return torch.empty((0, 0), dtype=torch.float32), []
    try:
        padded_audio = pad_sequence([t.cpu() for t in audio_list], batch_first=True, padding_value=0.0)
        return padded_audio, metadata_list
    except Exception as e:
        logger.error("Pad failed: %s", e, exc_info=True)
        return torch.empty((0, 0), dtype=torch.float32), []


class PaperAutoencoderTrainer(Trainer):
    """
    Trains the specific Autoencoder architecture used in the paper.
    """

    def _initialize_components(self):
        """
        Initializes optimizer, scheduler, criterion, and TensorBoard writer.
        """
        logger.info("Initializing components for PaperAutoencoderTrainer...")
        lr = self.config.get("learning_rate", 0.0003)
        weight_decay = self.config.get("weight_decay", 0.01)
        betas = tuple(self.config.get("betas", (0.9, 0.999)))
        scheduler_mode = self.config.get("scheduler_mode", "min")
        scheduler_factor = self.config.get("scheduler_factor", 0.5)
        scheduler_patience = self.config.get("scheduler_patience", 5)
        self.mixed_precision = self.config.get("mixed_precision", True)
        self.grad_accum_steps = self.config.get("gradient_accumulation_steps", 1)
        self.reg_weight = self.config.get("regularization_weight", 0.01)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        logger.debug("Optimizer: AdamW (lr=%f, wd=%f, betas=%s)", lr, weight_decay, betas)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode=scheduler_mode, factor=scheduler_factor, patience=scheduler_patience, verbose=True
        )
        logger.debug(
            "Scheduler: ReduceLROnPlateau (mode=%s, factor=%f, patience=%d)",
            scheduler_mode,
            scheduler_factor,
            scheduler_patience,
        )
        self.criterion = nn.L1Loss()
        logger.debug("Criterion: L1Loss")
        if self.mixed_precision and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
            logger.debug("Mixed precision enabled with GradScaler.")
        else:
            self.scaler = None
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.early_stopping_counter = 0
        self.writer = SummaryWriter(log_dir=str(self.logs_dir))

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """
        Execute the training loop for the paper's Autoencoder.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (Optional[DataLoader]): DataLoader for the validation dataset.
        """
        num_epochs = self.config.get("num_epochs", 50)
        early_stopping_patience = self.config.get("early_stopping_patience", 10)

        logger.info("Starting AE training for %d epochs...", num_epochs)
        if hasattr(train_loader, "batch_size"):
            logger.info("Batch size: %d, Grad Accum: %d", train_loader.batch_size, self.grad_accum_steps)

        nan_batches_skipped_train = 0

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            start_time = time.time()

            self.model.train()
            total_train_loss = 0.0
            total_recon_loss = 0.0
            total_reg_loss = 0.0
            processed_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for i, batch in enumerate(pbar):
                if batch is None:
                    continue
                try:
                    audio, _ = batch
                    if audio is None or audio.numel() == 0:
                        continue
                    audio = audio.to(self.device)
                    if not check_tensor(audio, f"Input Batch {i}"):
                        logger.warning("Skipping batch %d due to NaN/Inf in input audio.", i)
                        nan_batches_skipped_train += 1
                        continue
                except Exception as e:
                    logger.error("Error processing batch %d: %s. Skipping.", i, e, exc_info=True)
                    continue

                try:
                    with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                        reconstructed, encoded = self.model(audio)
                        with torch.no_grad():
                            target_mel_spec = self.model.frontend(audio)

                        valid_target = check_tensor(target_mel_spec, f"Target Batch {i}")
                        valid_recon = check_tensor(reconstructed, f"Recon Batch {i}")
                        valid_encoded = check_tensor(encoded, f"Encoded Batch {i}")

                        if not (valid_target and valid_recon and valid_encoded):
                            logger.warning("Skipping batch %d due to NaN/Inf in model outputs.", i)
                            nan_batches_skipped_train += 1
                            if (i + 1) % self.grad_accum_steps != 0:
                                self.optimizer.zero_grad(set_to_none=True)
                            continue

                        recon_loss = self.criterion(reconstructed, target_mel_spec)
                        reg_loss = self.reg_weight * torch.mean(torch.abs(encoded))
                        loss = recon_loss + reg_loss
                        loss = loss / self.grad_accum_steps

                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if (i + 1) % self.grad_accum_steps == 0 or (i + 1) == len(train_loader):
                        if self.scaler:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

                    total_train_loss += loss.item() * self.grad_accum_steps
                    total_recon_loss += recon_loss.item()
                    total_reg_loss += reg_loss.item()
                    processed_batches += 1
                    if processed_batches > 0:
                        pbar.set_postfix(
                            {
                                "loss": f"{total_train_loss / processed_batches:.4f}",
                                "recon": f"{total_recon_loss / processed_batches:.4f}",
                                "reg": f"{total_reg_loss / processed_batches:.4f}",
                                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                            }
                        )

                except Exception as model_err:
                    logger.error("Error during training step %d: %s. Skipping batch.", i, model_err, exc_info=True)
                    if (i + 1) % self.grad_accum_steps != 0:
                        try:
                            self.optimizer.zero_grad(set_to_none=True)
                        except Exception:
                            pass
                    continue

            if nan_batches_skipped_train > 0:
                logger.warning("Skipped %d batches in TRAIN epoch %d due to NaN/Inf.", nan_batches_skipped_train, epoch + 1)
                nan_batches_skipped_train = 0

            avg_train_loss = total_train_loss / processed_batches if processed_batches > 0 else float("nan")
            avg_recon_loss = total_recon_loss / processed_batches if processed_batches > 0 else float("nan")
            avg_reg_loss = total_reg_loss / processed_batches if processed_batches > 0 else float("nan")
            epoch_time = time.time() - start_time

            logger.info(
                "Epoch %d Train Summary | Time: %.2fs | Avg Loss: %.4f | Recon Loss: %.4f | Reg Loss: %.4f",
                epoch + 1,
                epoch_time,
                avg_train_loss,
                avg_recon_loss,
                avg_reg_loss,
            )
            self.writer.add_scalar("train/loss", avg_train_loss, epoch)
            self.writer.add_scalar("train/recon_loss", avg_recon_loss, epoch)
            self.writer.add_scalar("train/reg_loss", avg_reg_loss, epoch)
            self.writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]["lr"], epoch)

            avg_val_loss = float("inf")
            nan_batches_skipped_val = 0
            if val_loader:
                self.model.eval()
                total_val_loss = 0.0
                processed_val_batches = 0
                with torch.no_grad():
                    pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
                    for batch in pbar_val:
                        if batch is None:
                            continue
                        try:
                            audio, _ = batch
                            if audio is None or audio.numel() == 0:
                                continue
                            audio = audio.to(self.device)
                            if not check_tensor(audio, "Val Input"):
                                logger.warning("NaN/Inf in val input. Skipping batch.")
                                nan_batches_skipped_val += 1
                                continue

                            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                                reconstructed, encoded = self.model(audio)
                                target_mel_spec = self.model.frontend(audio)

                                if not check_tensor(target_mel_spec, "Val Target") or not check_tensor(reconstructed, "Val Recon") or not check_tensor(encoded, "Val Encoded"):
                                    logger.warning("NaN/Inf in val outputs. Skipping batch.")
                                    nan_batches_skipped_val += 1
                                    continue

                                recon_loss = self.criterion(reconstructed, target_mel_spec)
                                reg_loss = self.reg_weight * torch.mean(torch.abs(encoded))
                                loss = recon_loss + reg_loss
                            total_val_loss += loss.item()
                            processed_val_batches += 1
                            if processed_val_batches > 0:
                                pbar_val.set_postfix({"loss": f"{total_val_loss / processed_val_batches:.4f}"})
                        except Exception as val_err:
                            logger.error("Error during validation batch: %s. Skipping.", val_err, exc_info=True)
                            continue

                if nan_batches_skipped_val > 0:
                    logger.warning("Skipped %d batches in VAL epoch %d due to NaN/Inf.", nan_batches_skipped_val, epoch + 1)

                avg_val_loss = total_val_loss / processed_val_batches if processed_val_batches > 0 else float("nan")
                logger.info("Epoch %d Val Summary | Avg Loss: %.4f", epoch + 1, avg_val_loss)
                self.writer.add_scalar("val/loss", avg_val_loss, epoch)
            else:
                avg_val_loss = avg_train_loss

            if not np.isnan(avg_val_loss) and not np.isinf(avg_val_loss):
                self.scheduler.step(avg_val_loss)
                is_best = avg_val_loss < self.best_loss
                if is_best:
                    self.best_loss = avg_val_loss
                    self.save_model(epoch=epoch + 1, is_best=True)
                    self.early_stopping_counter = 0
                    logger.info("New best model at epoch %d, loss %.4f", epoch + 1, self.best_loss)
                else:
                    self.early_stopping_counter += 1
            else:
                logger.warning("Skipping scheduler/best model check due to non-finite val loss (%f)", avg_val_loss)
                self.early_stopping_counter += 1

            save_freq = self.config.get("save_frequency_epochs", 10)
            if save_freq > 0 and (epoch + 1) % save_freq == 0:
                self.save_model(epoch=epoch + 1, is_best=False)

            if early_stopping_patience > 0 and self.early_stopping_counter >= early_stopping_patience:
                logger.info("Early stopping after %d epochs.", early_stopping_patience)
                break

        logger.info("AE Training finished.")
        self.save_model(epoch=self.current_epoch + 1, is_best=False, final=True)
        self.writer.close()

    def save_model(self, epoch: Optional[int] = None, is_best: bool = False, final: bool = False):
        """
        Saves the model checkpoint.

        Args:
            epoch (Optional[int]): The current epoch number.
            is_best (bool): True if this is the best model so far.
            final (bool): True if this is the final model after training completes.
        """
        if final:
            filename = "final_model.pt"
            super()._save_checkpoint(filename=filename, epoch=epoch, is_best=True)
            return
        elif is_best:
            filename = f"checkpoint_epoch_{epoch}.pt"
            super()._save_checkpoint(filename=filename, epoch=epoch, is_best=True)
            return
        else:
            filename = f"checkpoint_epoch_{epoch}.pt"
            super()._save_checkpoint(filename=filename, epoch=epoch, is_best=False)