# -*- coding: utf-8 -*-
"""
VocSim Benchmark: Paper Reproducibility Model Trainers.

This package contains trainer implementations specific to the
models trained in the original paper (e.g., AE, VAE).
"""

from .autoencoder import PaperAutoencoderTrainer
from .vae import PaperVAETrainer

__all__ = [
    "PaperAutoencoderTrainer",
    "PaperVAETrainer",
]