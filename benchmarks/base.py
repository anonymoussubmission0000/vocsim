import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pathlib import Path 

logger = logging.getLogger(__name__)

class Benchmark(ABC):
    """Abstract Base Class for all benchmark evaluations."""

    def __init__(self, **kwargs):
        self._initialize(**kwargs)

    def _initialize(self, **kwargs):
        """Optional initialization for subclasses to process parameters."""
        pass

    @abstractmethod
    def evaluate(self, *, 
                 distance_matrix: Optional[Any] = None,
                 features: Optional[Any] = None, 
                 distance_matrix_path: Optional[Path] = None,
                 feature_hdf5_path: Optional[Path] = None, 
                 dataset: Optional[Any] = None,
                 labels: Optional[List[Any]] = None,
                 item_id_map: Optional[Dict[str, Dict]] = None,
                 feature_config: Optional[Dict[str, Any]] = None, 
                 **kwargs) -> Dict[str, Any]: 
        """
        Run the benchmark evaluation.

        Accepts either data directly (e.g., distance_matrix) OR paths
        to HDF5 files (feature_hdf5_path).
        Implementations should handle loading data from paths if direct data is None
        or implement chunked processing if feasible (like ClassificationBenchmark).
        """
        pass

    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Allows calling the instance like a function."""
        return self.evaluate(**kwargs)