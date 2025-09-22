"""
VocSim Benchmark: Core API and Orchestration Module.
This package contains the central runner and logic for executing
benchmarks within the VocSim framework.
"""
__version__ = '0.0.1'

from .runner import PipelineRunner

__all__ = [
    "PipelineRunner",
]