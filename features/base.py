from abc import ABC, abstractmethod
from typing import Any, Union, Dict
import numpy as np
import torch


class FeatureExtractor(ABC):
    """Abstract Base Class for all feature extractors."""

    def __init__(self, device: str = "cpu", **kwargs):
        """
        Initializes the FeatureExtractor base class.

        Args:
            device (str): The device (e.g., 'cpu', 'cuda') to use for feature extraction.
            **kwargs: Additional keyword arguments passed to the _initialize method.
        """
        self.device = torch.device(device)
        self._initialize(**kwargs)

    def _initialize(self, **kwargs):
        """
        Optional initialization hook for subclasses.

        Subclasses can override this method to perform specific setup tasks
        after the instance is created but before extraction begins.

        Args:
            **kwargs: Keyword arguments passed during initialization.
        """
        pass

    @abstractmethod
    def extract(self, audio_data: Union[np.ndarray, torch.Tensor], sample_rate: int, **kwargs: Any) -> Any:
        """
        Extract features from audio data.

        Args:
            audio_data (Union[np.ndarray, torch.Tensor]): Input audio waveform as a numpy array or torch tensor.
            sample_rate (int): Sample rate of the audio data in Hz.
            **kwargs (Any): Additional keyword arguments specific to the extractor implementation.

        Returns:
            Any: The extracted features. The format and type depend on the specific
                 subclass implementation.
        """
        pass