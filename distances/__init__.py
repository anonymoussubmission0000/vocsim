"""
VocSim Benchmark: Distance Computation Modules.
"""

from features.base import FeatureExtractor
from .cosine import CosineDistance
from .euclidean import EuclideanDistance
from .spearman import SpearmanDistance

__all__ = [
    "DistanceCalculator",
    "CosineDistance",
    "EuclideanDistance",
    "SpearmanDistance",
]