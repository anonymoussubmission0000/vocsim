import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Union
from pathlib import Path
import torch
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


class Trainer(ABC):
    """
    Abstract Base Class for all model trainers.
    """

    def __init__(self, model: torch.nn.Module, config: Dict[str, Any], output_dir: Union[str, Path], device: str = "cpu"):
        """
        Initialize the base trainer.

        Args:
            model (torch.nn.Module): The model to be trained.
            config (Dict[str, Any]): Training configuration parameters (e.g., lr, epochs, batch_size).
            output_dir (Union[str, Path]): Directory to save models and logs.
            device (str): The device to run training on ('cpu' or 'cuda').
        """
        self.model = model.to(torch.device(device))
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self._initialize_components()
        logger.info(f"{self.__class__.__name__} initialized. Output Dir: {self.output_dir}, Device: {self.device}")

    def _initialize_components(self):
        """
        Initialize components like optimizer, scheduler, loss function, etc.
        """
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.criterion: Optional[torch.nn.Module] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None

        logger.warning("Base Trainer _initialize_components called. Subclasses should override this.")

    @abstractmethod
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """
        Execute the training loop.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (Optional[DataLoader]): DataLoader for the validation dataset.
        """
        pass

    @abstractmethod
    def save_model(self, epoch: Optional[int] = None, is_best: bool = False):
        """
        Save the current state of the model.

        Args:
            epoch (Optional[int]): The epoch number (used for checkpoint naming).
            is_best (bool): Flag indicating if this is the best model found so far.
        """
        pass

    def _save_checkpoint(self, filename: str, epoch: Optional[int], is_best: bool):
        """
        Helper method to save a checkpoint dictionary.

        Args:
            filename (str): The name of the checkpoint file.
            epoch (Optional[int]): The current epoch number.
            is_best (bool): True if this is the best model checkpoint.
        """
        if self.optimizer is None:
            logger.warning("Optimizer not initialized, cannot save its state.")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "config": self.config,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
        }
        filepath = self.checkpoints_dir / filename
        torch.save(checkpoint, filepath)
        logger.info("Checkpoint saved to %s", filepath)

        if is_best:
            best_filepath = self.output_dir / "final_model.pt"
            torch.save(checkpoint, best_filepath)
            logger.info("Best model checkpoint saved to %s", best_filepath)