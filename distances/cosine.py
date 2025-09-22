import logging
from typing import Any, Union, Optional
from distances.base import DistanceCalculator
import torch
import numpy as np
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

logger = logging.getLogger(__name__)


class CosineDistance(DistanceCalculator):
    """
    Calculates pairwise Cosine distances using torchmetrics.
    """

    def _initialize(self, use_torchmetrics: bool = True, zero_diagonal: bool = True, **kwargs):
        """
        Initialize CosineDistance calculator.

        Args:
            use_torchmetrics (bool): Whether to use torchmetrics for calculation (currently required).
            zero_diagonal (bool): Whether to set the diagonal of the distance matrix to zero
                when computing distances between the same set of features (X vs X).
            **kwargs: Additional keyword arguments.
        """
        if not use_torchmetrics or torch is None:
            raise NotImplementedError("CosineDistance currently requires torchmetrics and PyTorch.")
        self.zero_diagonal = zero_diagonal
        logger.debug(f"CosineDistance initialized (torchmetrics, zero_diag={zero_diagonal})")

    def compute_pairwise(self, features_X: Any, features_Y: Optional[Any] = None) -> Union[torch.Tensor, np.ndarray]:
        """
        Compute the pairwise Cosine distance matrix.

        Args:
            features_X (Any): A 2D tensor or array [n_samples_X, feature_dim].
            features_Y (Optional[Any]): An optional second 2D tensor or array [n_samples_Y, feature_dim].
                                       If provided, computes distances between X and Y.
                                       If None, computes distances between X and X.

        Returns:
            torch.Tensor: Pairwise distance matrix [n_samples_X, n_samples_Y (or X)], scaled [0, 1].
                         Always returns a torch.Tensor on the device used for computation.
        """
        if not isinstance(features_X, (torch.Tensor, np.ndarray)):
            raise TypeError(f"Input features_X must be Tensor/ndarray, got {type(features_X)}")
        if features_X.ndim != 2:
            raise ValueError(f"Input features_X must be 2D, got {features_X.shape}")

        if isinstance(features_X, np.ndarray):
            features_tensor_X = torch.from_numpy(features_X).float()
        else:
            features_tensor_X = features_X.float()

        features_tensor_Y = None
        if features_Y is not None:
            if not isinstance(features_Y, (torch.Tensor, np.ndarray)):
                raise TypeError(f"Input features_Y must be Tensor/ndarray, got {type(features_Y)}")
            if features_Y.ndim != 2:
                raise ValueError(f"Input features_Y must be 2D, got {features_Y.shape}")
            if features_X.shape[1] != features_Y.shape[1]:
                raise ValueError(f"Feature dimensions mismatch: X={features_X.shape[1]}, Y={features_Y.shape[1]}")

            if isinstance(features_Y, np.ndarray):
                features_tensor_Y = torch.from_numpy(features_Y).float()
            else:
                features_tensor_Y = features_Y.float()
            features_tensor_Y = features_tensor_Y.to(features_tensor_X.device)
            target_shape = (features_tensor_X.shape[0], features_tensor_Y.shape[0])
            compute_mode = "X vs Y"
            zero_diag_effective = False
        else:
            target_shape = (features_tensor_X.shape[0], features_tensor_X.shape[0])
            compute_mode = "X vs X"
            zero_diag_effective = self.zero_diagonal

        logger.debug(f"Computing Cosine Distance ({compute_mode}) for shapes {features_tensor_X.shape} vs {features_tensor_Y.shape if features_tensor_Y is not None else 'self'}")

        try:
            similarity_matrix = pairwise_cosine_similarity(features_tensor_X, features_tensor_Y)

            distance_matrix = (1.0 - similarity_matrix) / 2.0

            if zero_diag_effective:
                torch.diagonal(distance_matrix).fill_(0)

            distance_matrix.clamp_(min=0.0, max=1.0)

            if distance_matrix.shape != target_shape:
                logger.warning(f"Output shape mismatch: {distance_matrix.shape} != {target_shape}. Check torchmetrics version/behavior.")

            return distance_matrix
        except Exception as e:
            logger.error("Torchmetrics Cosine distance failed: %s", e, exc_info=True)
            raise