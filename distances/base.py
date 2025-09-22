"""Abstract Base Class for distance calculators."""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class DistanceCalculator(ABC):
    """
    Abstract Base Class for calculating pairwise distances between feature vectors.

    Subclasses must implement the `compute_pairwise` method.
    """

    def __init__(self, **kwargs):
        """
        Initialize the base distance calculator.

        Args:
            **kwargs: Additional keyword arguments for subclasses.
        """
        self._initialize(**kwargs)

    def _initialize(self, **kwargs):
        """Optional initialization for subclasses."""
        pass

    @abstractmethod
    def compute_pairwise(self, features: Any) -> Any:
        """
        Compute the pairwise distance matrix for a set of features.

        Args:
            features (Any): A collection of features, typically a 2D tensor or array
                           where rows represent samples and columns represent feature dimensions
                           (e.g., torch.Tensor or np.ndarray of shape [n_samples, feature_dim]).

        Returns:
            Any: A square matrix (e.g., torch.Tensor or np.ndarray of shape
                 [n_samples, n_samples]) containing the pairwise distances.
                 The exact format depends on the implementation.
        """
        pass

    def __call__(self, features: Any) -> Any:
        """Allows calling the instance like a function."""
        return self.compute_pairwise(features)