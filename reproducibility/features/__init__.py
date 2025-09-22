"""
VocSim Benchmark: Paper Reproducibility Feature Extractors.
"""

from .autoencoder import PaperAutoencoderExtractor
from .vae import PaperVAEExtractor

__all__ = [
    "PaperAutoencoderExtractor",
    "PaperVAEExtractor",
]