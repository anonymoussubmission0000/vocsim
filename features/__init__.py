"""
VocSim Benchmark: Feature Extraction Modules.
"""

from features.base import FeatureExtractor
from .audiomae import AudioMAEExtractor
from .clap import CLAPExtractor
from .encodec import EncodecExtractor
from .mel import MelExtractor
from .wav2vec2 import Wav2Vec2Extractor
from .wavlm import WavLMExtractor
from .whisper import WhisperEncoderExtractor
# Import from sub-package
from .whisperseg.extractor import WhisperSegExtractor

__all__ = [
    "FeatureExtractor",
    "AudioMAEExtractor",
    "CLAPExtractor",
    "EncodecExtractor",
    "MelExtractor",
    "Wav2Vec2Extractor",
    "WavLMExtractor",
    "WhisperEncoderExtractor",
    "WhisperSegExtractor",
]