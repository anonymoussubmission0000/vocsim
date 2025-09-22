# In reproducibility/scripts/misc/calculate_efficiency.py

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
import logging
import warnings

# --- Boilerplate to add project root to path ---
try:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[3] # Assumes misc/scripts/reproducibility/
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    print(f"INFO: Added project root to path: {project_root}")
except NameError:
    project_root = Path.cwd()
    sys.path.insert(0, str(project_root))
    print(f"INFO: Assuming CWD for interactive session: {project_root}")

warnings.filterwarnings("ignore", category=UserWarning)
from thop import profile, clever_format

# --- Import your project's feature extractors and models ---
from features.base import FeatureExtractor
from features.whisper import WhisperEncoderExtractor
from features.wavlm import WavLMExtractor
from features.clap import CLAPExtractor
from features.encodec import EncodecExtractor
from features.audiomae import AudioMAEExtractor
from features.wav2vec2 import Wav2Vec2Extractor
from reproducibility.features.vae import PaperVAEExtractor
from reproducibility.features.autoencoder import PaperAutoencoderExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TABLE_HEADER = f"{'Model Pipeline':<30} | {'Parameters (M)':<16} | {'MACs (G)':<10} | {'Peak Memory (GB)':<18}"
TABLE_SEPARATOR = "-" * (len(TABLE_HEADER) + 2)

class FeatureExtractorProfiler(nn.Module):
    """A wrapper to make a FeatureExtractor's `extract` method profilable by thop."""
    def __init__(self, extractor: FeatureExtractor):
        super().__init__()
        # Important: The extractor's internal model must be a property of this wrapper
        # for thop to find its parameters. We'll search for it.
        self.extractor = extractor
        if hasattr(self.extractor, 'model'):
             self.model = self.extractor.model
        if hasattr(self.extractor, 'feature_extractor'):
             # For models like WavLM
             self.feature_extractor = self.extractor.feature_extractor

    def forward(self, audio_data, sample_rate):
        # The forward pass simply calls the extract method
        return self.extractor.extract(audio_data, sample_rate)

def profile_pipeline(model_name: str, extractor: FeatureExtractor, input_args: tuple):
    """
    Profiles a full feature extraction pipeline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Wrap the extractor to make it profilable
    profiler_wrapper = FeatureExtractorProfiler(extractor).to(device).eval()
    
    # --- Profile MACs and Parameters with thop ---
    try:
        # We pass the raw audio waveform, thop traces PyTorch ops inside .extract()
        macs, params = profile(profiler_wrapper, inputs=input_args, verbose=False)
        params_m = params / 1e6
        macs_g = macs / 1e9
    except Exception:
        # Fallback for complex models (like AudioMAE) where thop might fail
        # but we can still count parameters.
        logging.warning(f"Could not profile MACs for {model_name} with thop. Reporting params only.")
        params = sum(p.numel() for p in profiler_wrapper.parameters())
        params_m = params / 1e6
        macs_g = -1.0

    # --- Measure Peak Memory Usage ---
    peak_mem_gb = -1.0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        try:
            # Run the forward pass to measure activation memory
            _ = profiler_wrapper(*input_args)
            peak_mem = torch.cuda.max_memory_allocated(device)
            peak_mem_gb = peak_mem / 1e9
        except Exception as e:
            logging.error(f"Failed forward pass for memory profiling on {model_name}: {e}")

    # --- Print results ---
    print(f"| {model_name:<28} | {f'{params_m:.2f}':<16} | {f'{macs_g:.2f}' if macs_g >= 0 else 'N/A':<10} | {f'{peak_mem_gb:.3f}' if peak_mem_gb >= 0 else 'N/A':<18} |")


def main():
    """
    Main function to orchestrate the profiling of all model pipelines.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Profiling full feature extraction pipelines on device: {device}")

    # --- Define a standard input audio clip ---
    # 30-second, 16kHz mono audio. This will be the input to the entire pipeline.
    sample_rate = 16000
    duration_sec = 30
    input_waveform = torch.randn(sample_rate * duration_sec).to(device)
    input_args = (input_waveform, sample_rate) # Arguments for our profiler wrapper

    print("\n" + TABLE_SEPARATOR)
    print(f"| {'Model Pipeline':<28} | {'Parameters (M)':<16} | {'MACs (G)':<10} | {'Peak Memory (GB)':<18} |")
    print(TABLE_SEPARATOR)

    # === Pretrained Models from Hugging Face ===

    # 1. Whisper Encoder Pipeline
    try:
        extractor = WhisperEncoderExtractor(model_id="openai/whisper-large-v3", device=device.type)
        profile_pipeline("Whisper-L-v3", extractor, input_args)
        del extractor
    except Exception as e:
        logging.error(f"Failed to profile Whisper pipeline: {e}")

    # 2. WavLM Pipeline
    try:
        extractor = WavLMExtractor(model_id="microsoft/wavlm-large", device=device.type)
        profile_pipeline("WavLM-Large", extractor, input_args)
        del extractor
    except Exception as e:
        logging.error(f"Failed to profile WavLM pipeline: {e}")

    # 3. Wav2Vec2 Pipeline
    try:
        extractor = Wav2Vec2Extractor(model_id="facebook/wav2vec2-base-960h", device=device.type)
        profile_pipeline("Wav2Vec2-Base", extractor, input_args)
        del extractor
    except Exception as e:
        logging.error(f"Failed to profile Wav2Vec2 pipeline: {e}")

    # 4. CLAP Pipeline
    try:
        extractor = CLAPExtractor(model_id="laion/larger_clap_general", device=device.type)
        profile_pipeline("CLAP", extractor, input_args)
        del extractor
    except Exception as e:
        logging.error(f"Failed to profile CLAP pipeline: {e}")

    # 5. AudioMAE Pipeline
    # NOTE: AudioMAE's `extract` writes to a file, making it untraceable by thop.
    # We will get an N/A for MACs, but this is the correct behavior.
    try:
        extractor = AudioMAEExtractor(model_id="hance-ai/audiomae", trust_remote_code=True, device=device.type)
        profile_pipeline("AudioMAE", extractor, input_args)
        del extractor
    except Exception as e:
        logging.error(f"Failed to profile AudioMAE pipeline: {e}")

    # 6. Encodec Pipeline
    try:
        extractor = EncodecExtractor(model_id="facebook/encodec_24khz", device=device.type)
        profile_pipeline("Encodec", extractor, input_args)
        del extractor
    except Exception as e:
        logging.error(f"Failed to profile Encodec pipeline: {e}")

    print(TABLE_SEPARATOR)
    print(f"| {'Special Cases / Custom':<28} | {'':<16} | {'':<10} | {'':<18} |")
    print(TABLE_SEPARATOR)

    # 8. Paper VAE Pipeline
    try:
        # Note: You must have a trained model checkpoint for this to work.
        # Update path to your actual model directory.
        models_dir = project_root / "d:/data2/vocsim/paper_models_scoped" 
        extractor = PaperVAEExtractor(model_scope="all", base_models_dir=models_dir, device=device.type)
        profile_pipeline("Paper VAE", extractor, input_args)
        del extractor
    except Exception as e:
        logging.error(f"Failed to profile Paper VAE pipeline: {e}. Check model path and checkpoints.")

    # 9. Paper Autoencoder Pipeline
    try:
        # Update path to your actual model directory.
        models_dir = project_root / "d:/data2/vocsim/paper_models_scoped"
        ae_audio_config = {'sr': 16000, 'n_mels': 128, 'nfft': 1024, 'fmin': 0.0}
        ae_dimensions = {'nfft': 1024, 'max_spec_width': 512}
        extractor = PaperAutoencoderExtractor(model_scope="all", base_models_dir=models_dir, 
                                              audio_config=ae_audio_config, dimensions=ae_dimensions,
                                              bottleneck_dim=256, device=device.type)
        profile_pipeline("Paper Autoencoder", extractor, input_args)
        del extractor
    except Exception as e:
        logging.error(f"Failed to profile Paper Autoencoder pipeline: {e}. Check model path and checkpoints.")

    # 10. Mel Spectrogram (Baseline)
    print(f"| {'Mel Spectrogram':<28} | {'0.00':<16} | {'0.00':<10} | {'Minimal':<18} |")
    print(f"| {'  (Note: Parameter-free transformation, not a neural model.)':<90} |")

    print(TABLE_SEPARATOR)
    if device.type == 'cuda':
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

