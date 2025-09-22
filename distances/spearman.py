import logging
from typing import Any, Union, Optional
from distances.base import DistanceCalculator
import torch
import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


class SpearmanDistance(DistanceCalculator):
    """
    Calculates pairwise Spearman Rank Correlation distances.
    """

    def _initialize(self, use_gpu_if_available: bool = True, **kwargs):
        """
        Initialize SpearmanDistance calculator.

        Args:
            use_gpu_if_available (bool): If True and CUDA is available, use the GPU implementation.
            **kwargs: Additional arguments (ignored).
        """
        self.use_gpu = use_gpu_if_available and torch.cuda.is_available()
        if self.use_gpu:
            logger.debug("SpearmanDistance will use GPU implementation.")
        else:
            logger.debug("SpearmanDistance will use CPU (SciPy) implementation.")

    def _compute_ranks_gpu(self, features: torch.Tensor) -> torch.Tensor:
        """
        Computes ranks along the feature dimension (dim=1) on GPU.

        Args:
            features (torch.Tensor): A 2D tensor [N, D].

        Returns:
            torch.Tensor: A 2D tensor of ranks [N, D].
        """
        ranks = features.argsort(dim=1).argsort(dim=1).float()
        return ranks

    def _fast_spearman_gpu(self, features_X: torch.Tensor, features_Y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Custom PyTorch implementation for Spearman correlation distance on GPU.

        Args:
            features_X (torch.Tensor): A 2D tensor [n_samples_X, feature_dim] on a GPU device.
            features_Y (Optional[torch.Tensor]): An optional second 2D tensor [n_samples_Y, feature_dim]
                                                 on the same GPU device.

        Returns:
            torch.Tensor: Pairwise distance matrix [n_samples_X, n_samples_Y (or X)] on the GPU.
        """
        n_samples_X, n_features = features_X.shape
        if n_samples_X < 1:
            if features_Y is None:
                return torch.zeros((0, 0), device=features_X.device)
            else:
                return torch.zeros((0, features_Y.shape[0]), device=features_X.device)

        ranks_X = self._compute_ranks_gpu(features_X)

        ranks_X_centered = ranks_X - ranks_X.mean(dim=1, keepdim=True)

        std_devs_X = ranks_X_centered.std(dim=1, unbiased=True)
        std_devs_X[std_devs_X == 0] = 1.0

        if features_Y is None:
            n_samples_Y = n_samples_X
            ranks_Y_centered = ranks_X_centered
            std_devs_Y = std_devs_X
            target_shape = (n_samples_X, n_samples_X)
        else:
            n_samples_Y = features_Y.shape[0]
            if features_Y.shape[1] != n_features:
                raise ValueError(f"Feature dimension mismatch: X={n_features}, Y={features_Y.shape[1]}")
            ranks_Y = self._compute_ranks_gpu(features_Y)
            ranks_Y_centered = ranks_Y - ranks_Y.mean(dim=1, keepdim=True)
            std_devs_Y = ranks_Y_centered.std(dim=1, unbiased=True)
            std_devs_Y[std_devs_Y == 0] = 1.0
            target_shape = (n_samples_X, n_samples_Y)

        numerator = torch.matmul(ranks_X_centered, ranks_Y_centered.t())

        denominator = torch.outer(std_devs_X, std_devs_Y) * (n_features - 1)

        rho = torch.zeros_like(numerator)
        valid_denom_mask = denominator != 0
        rho[valid_denom_mask] = numerator[valid_denom_mask] / denominator[valid_denom_mask]

        rho = torch.clamp(rho, -1.0, 1.0)

        distance_matrix = (1.0 - rho) / 2.0

        if features_Y is None:
            torch.diagonal(distance_matrix).fill_(0.0)

        if distance_matrix.shape != target_shape:
            logger.warning(f"Spearman GPU shape mismatch: {distance_matrix.shape} != {target_shape}")

        return distance_matrix

    def _scipy_spearman_cpu(self, features_X: np.ndarray, features_Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes Spearman correlation distance using SciPy on CPU.

        Args:
            features_X (np.ndarray): A 2D numpy array [n_samples_X, feature_dim].
            features_Y (Optional[np.ndarray]): An optional second 2D numpy array [n_samples_Y, feature_dim].

        Returns:
            np.ndarray: Pairwise distance matrix [n_samples_X, n_samples_Y (or X)] as a numpy array.
        """
        logger.debug(
            f"Using SciPy Spearman on CPU for shapes {features_X.shape} vs {features_Y.shape if features_Y is not None else 'self'}"
        )

        if features_X.ndim != 2:
            raise ValueError("SciPy Spearman requires 2D input X.")
        n_samples_X = features_X.shape[0]

        try:
            if features_Y is None:
                rho_matrix, _ = spearmanr(features_X, axis=1)
                if np.isscalar(rho_matrix):
                    rho_matrix = np.array([[1.0]]) if n_samples_X == 1 else np.eye(n_samples_X)
            else:
                raise NotImplementedError(
                    "CPU-based Spearman distance computation currently only supports X vs X (calculating the full matrix). Use GPU for X vs Y."
                )

            rho_matrix = np.nan_to_num(rho_matrix, nan=0.0)

            distance_matrix = (1.0 - rho_matrix) / 2.0
            if features_Y is None:
                np.fill_diagonal(distance_matrix, 0.0)

            return distance_matrix.astype(np.float32)

        except ValueError as e:
            logger.error("SciPy spearmanr failed. Check input data (NaNs/Infs?). Error: %s", e, exc_info=True)
            n_samples_Y = features_Y.shape[0] if features_Y is not None else n_samples_X
            dist_mat = np.full((n_samples_X, n_samples_Y), 0.5, dtype=np.float32)
            if features_Y is None:
                np.fill_diagonal(dist_mat, 0.0)
            return dist_mat
        except Exception as e:
            logger.error("Spearman CPU distance calculation failed: %s", e, exc_info=True)
            raise

    def compute_pairwise(self, features_X: Any, features_Y: Optional[Any] = None) -> Union[torch.Tensor, np.ndarray]:
        """
        Compute the pairwise Spearman distance matrix. Uses GPU if enabled and possible.

        Args:
            features_X (Any): A 2D tensor or array [n_samples_X, feature_dim].
            features_Y (Optional[Any]): An optional second 2D tensor or array [n_samples_Y, feature_dim].
                                       If provided, computes distances between X and Y (GPU required).
                                       If None, computes distances between X and X.

        Returns:
            Union[torch.Tensor, np.ndarray]: Pairwise distance matrix [n_samples_X, n_samples_Y (or X)].
                                             Returns torch.Tensor if computed on GPU, np.ndarray if on CPU.
        """
        use_gpu_runtime = self.use_gpu and isinstance(features_X, torch.Tensor) and features_X.is_cuda
        if features_Y is not None:
            use_gpu_runtime = use_gpu_runtime and isinstance(features_Y, torch.Tensor) and features_Y.is_cuda
            if not use_gpu_runtime:
                logger.warning("X vs Y Spearman requested but GPU conditions not met. Trying CPU (may fail/be slow).")
                if isinstance(features_X, torch.Tensor):
                    features_X_np = features_X.cpu().numpy()
                else:
                    features_X_np = np.asarray(features_X)
                features_Y_np = None
                if features_Y is not None:
                    if isinstance(features_Y, torch.Tensor):
                        features_Y_np = features_Y.cpu().numpy()
                    else:
                        features_Y_np = np.asarray(features_Y)
                return self._scipy_spearman_cpu(features_X_np, features_Y_np)

        if use_gpu_runtime:
            logger.debug("Using GPU Spearman computation.")
            try:
                return self._fast_spearman_gpu(features_X, features_Y)
            except Exception as e:
                logger.error("GPU Spearman failed: %s. Falling back to CPU.", e, exc_info=True)
                features_X_np = features_X.cpu().numpy()
                features_Y_np = features_Y.cpu().numpy() if features_Y is not None else None
                return self._scipy_spearman_cpu(features_X_np, features_Y_np)
        else:
            logger.debug("Using CPU Spearman computation.")
            if isinstance(features_X, torch.Tensor):
                features_X_np = features_X.cpu().numpy()
            else:
                features_X_np = np.asarray(features_X)
            features_Y_np = None
            if features_Y is not None:
                if isinstance(features_Y, torch.Tensor):
                    features_Y_np = features_Y.cpu().numpy()
                else:
                    features_Y_np = np.asarray(features_Y)
            return self._scipy_spearman_cpu(features_X_np, features_Y_np)