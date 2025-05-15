"""PyTorch specific helpers."""

import logging
import torch

logger = logging.getLogger(__name__)

_DEVICE = None


def get_device(force_cpu: bool = False) -> torch.device:
    """
    Gets the recommended device (CUDA if available, else CPU).

    Args:
        force_cpu (bool): If True, forces the use of CPU even if CUDA is available.

    Returns:
        torch.device: The selected torch device object.
    """
    global _DEVICE
    if _DEVICE is None:
        if force_cpu:
            _DEVICE = torch.device("cpu")
            logger.info("Device forced to CPU.")
        elif torch.cuda.is_available():
            _DEVICE = torch.device("cuda")
            try:
                props = torch.cuda.get_device_properties(0)
                logger.info("CUDA available. Using GPU: %s (CUDA Compute Capability %d.%d)", props.name, props.major, props.minor)
            except Exception as e:
                logger.warning("CUDA available, but failed to get device properties: %s", e)
        else:
            _DEVICE = torch.device("cpu")
            logger.info("CUDA not available. Using CPU.")
    return _DEVICE


def check_tensor(tensor: torch.Tensor, name: str = "Tensor") -> bool:
    """
    Checks a tensor for NaN or Inf values.

    Args:
        tensor (torch.Tensor): The tensor to check.
        name (str): An optional name for the tensor for logging purposes.

    Returns:
        bool: True if the tensor is valid (no NaN/Inf), False otherwise.
    """
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan:
        logger.error("%s contains NaN values!", name)
    if has_inf:
        logger.error("%s contains Inf values!", name)

    return not (has_nan or has_inf)