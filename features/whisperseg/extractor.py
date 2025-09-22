import logging
import json
from pathlib import Path
from typing import Union, Any, Optional, List, Dict, Tuple

import ctranslate2
import numpy as np
import torch

from features.base import FeatureExtractor
from features.whisperseg.audio_frontend import WhisperSegFrontend
from features.whisperseg.model_loader import load_whisperseg_model

logger = logging.getLogger(__name__)

DEFAULT_TOTAL_SPEC_COLUMNS = 1500


class WhisperSegExtractor(FeatureExtractor):
    """
    Feature extractor using the WhisperSeg model (CTranslate2 format).
    """

    def _initialize(
        self,
        model_path: str,
        device_index: Union[int, List[int]] = 0,
        compute_type: Optional[str] = None,
        batch_size: int = 8,
        output_type: str = "embedding",
        default_spec_time_step: float = 0.02,
        default_min_frequency: float = 0.0,
        default_num_trials: int = 1,
        **kwargs,
    ):
        """
        Load the CTranslate2 model and tokenizer. Stores default parameters.

        Args:
            model_path (str): Path to the CTranslate2 model directory.
            device_index (Union[int, List[int]]): Device index(es) for CTranslate2.
            compute_type (Optional[str]): Compute type for CTranslate2 (e.g., 'float16', 'int8').
            batch_size (int): Batch size for the CTranslate2 encoder.
            output_type (str): Type of output features ('embedding' or 'spectrogram').
            default_spec_time_step (float): Default time step for spectrogram frames.
            default_min_frequency (float): Default minimum frequency for the Mel filterbank.
            default_num_trials (int): Default number of padding/chunking trials.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            None
        """
        logger.info(f"Initializing WhisperSegExtractor (Original Loop Mode) with model: {model_path}")
        self.model, self.tokenizer = load_whisperseg_model(
            model_path, device=self.device.type, device_index=device_index, compute_type=compute_type
        )
        self.batch_size = batch_size
        self.output_type = output_type.lower()
        if self.output_type not in ["embedding", "spectrogram"]:
            raise ValueError("output_type must be 'embedding' or 'spectrogram'")

        self.default_spec_time_step = default_spec_time_step
        self.default_min_frequency = default_min_frequency
        self.default_num_trials = default_num_trials

        config_path = Path(model_path) / "config.json"
        model_cfg = {}
        if config_path.is_file():
            try:
                with open(config_path, "r") as f:
                    model_cfg = json.load(f)
            except Exception as e:
                logger.warning("Could not read config.json: %s", e)
        self.total_spec_columns = model_cfg.get("total_spec_columns", DEFAULT_TOTAL_SPEC_COLUMNS)

        self._frontend_cache: Dict[Tuple[int, float, float], WhisperSegFrontend] = {}

        logger.debug(
            "WhisperSeg Extractor (Original Loop) Initialized: Output='%s',"
            " TotalSpecCols=%s, Defaults(step=%s, fmin=%s, trials=%s)"
            " CT2Device=%s:%s, CT2Compute=%s",
            self.output_type,
            self.total_spec_columns,
            self.default_spec_time_step,
            self.default_min_frequency,
            self.default_num_trials,
            self.device.type,
            device_index,
            self.model.compute_type,
        )

    def _get_frontend_calculator(self, sr: int, spec_time_step: float, min_frequency: float) -> WhisperSegFrontend:
        """
        Gets or creates a frontend calculator instance.

        Args:
            sr (int): Sample rate.
            spec_time_step (float): Spectrogram time step.
            min_frequency (float): Minimum frequency for Mel filterbank.

        Returns:
            WhisperSegFrontend: Frontend calculator instance.
        """
        key = (sr, spec_time_step, min_frequency)
        if key not in self._frontend_cache:
            logger.debug("Creating frontend calculator for SR=%d, Step=%f, Fmin=%f", sr, spec_time_step, min_frequency)
            if sr <= 32000:
                n_fft = 512
            elif sr <= 80000:
                n_fft = 1024
            elif sr <= 150000:
                n_fft = 2048
            elif sr <= 300000:
                n_fft = 4096
            else:
                n_fft = 8192
            hop_length = int(round(spec_time_step * sr))

            self._frontend_cache[key] = WhisperSegFrontend(
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                min_frequency=min_frequency,
                max_frequency=None,
            )
        return self._frontend_cache[key]

    @torch.no_grad()
    def extract(
        self,
        audio_data: Union[np.ndarray, torch.Tensor],
        sample_rate: int,
        spec_time_step: Optional[float] = None,
        min_frequency: Optional[float] = None,
        num_trials: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Extract features using the original WhisperSeg padding/chunking loop.

        Args:
            audio_data (Union[np.ndarray, torch.Tensor]): Input audio waveform.
            sample_rate (int): Sample rate of the audio.
            spec_time_step (Optional[float]): Spectrogram time step (overrides default).
            min_frequency (Optional[float]): Minimum frequency (overrides default).
            num_trials (Optional[int]): Number of padding/chunking trials (overrides default).
            **kwargs (Any): Additional keyword arguments.

        Returns:
            torch.Tensor: Extracted features (embeddings or spectrogram chunks).
        """
        _spec_time_step = spec_time_step if spec_time_step is not None else self.default_spec_time_step
        _min_frequency = min_frequency if min_frequency is not None else self.default_min_frequency
        _num_trials = num_trials if num_trials is not None else self.default_num_trials

        try:
            if isinstance(audio_data, torch.Tensor):
                audio_np = audio_data.cpu().numpy()
            elif isinstance(audio_data, np.ndarray):
                audio_np = audio_data
            else:
                raise TypeError("Input must be numpy array or torch tensor.")

            if audio_np.dtype != np.float32:
                audio_np = audio_np.astype(np.float32)
            if audio_np.ndim > 1:
                min_dim = np.argmin(audio_np.shape)
                if audio_np.shape[min_dim] > 1:
                    audio_np = np.mean(audio_np, axis=min_dim)
                else:
                    audio_np = audio_np.squeeze()
            if audio_np.ndim != 1:
                raise ValueError(f"Audio not mono. Shape: {audio_np.shape}")

            frontend_calculator = self._get_frontend_calculator(sample_rate, _spec_time_step, _min_frequency)
            sr = sample_rate

            clip_duration = self.total_spec_columns * _spec_time_step
            max_num_padding_samples = int(np.ceil(clip_duration * sr))
            audio_left_pad = np.zeros(max_num_padding_samples, dtype=np.float32)
            audio_clip_length = int(np.ceil(clip_duration * sr))

            all_valid_feature_chunks = []

            for trial_id in range(_num_trials):
                padding_time = 0.0
                if _num_trials > 1:
                    padding_time = np.round(clip_duration * trial_id / _num_trials / _spec_time_step) * _spec_time_step
                num_padding_samples = int(np.round(padding_time * sr))

                current_left_pad = audio_left_pad[len(audio_left_pad) - num_padding_samples :]
                audio_padded = np.concatenate([current_left_pad, audio_np], axis=0)

                for pos in range(0, max(1, len(audio_padded)), audio_clip_length):
                    audio_clip = audio_padded[pos : pos + audio_clip_length]
                    if len(audio_clip) < audio_clip_length:
                        padding_needed = audio_clip_length - len(audio_clip)
                        audio_clip_padded = np.pad(audio_clip, (0, padding_needed), mode="constant", constant_values=0.0)
                    else:
                        audio_clip_padded = audio_clip

                    input_features = frontend_calculator(audio_clip_padded)

                    if input_features is None:
                        logger.warning("Frontend calculator returned None for a chunk. Skipping.")
                        continue

                    current_width = input_features.shape[1]
                    if current_width > self.total_spec_columns:
                        input_features = input_features[:, : self.total_spec_columns]
                    elif current_width < self.total_spec_columns:
                        pad_cols = self.total_spec_columns - current_width
                        pad_val = input_features.min() if input_features.size > 0 else 0.0
                        input_features = np.pad(input_features, ((0, 0), (0, pad_cols)), mode="constant", constant_values=pad_val)

                    if input_features.shape != (frontend_calculator.num_mel_bins, self.total_spec_columns):
                        logger.warning("Final feature chunk shape %s unexpected. Skipping.", input_features.shape)
                        continue

                    all_valid_feature_chunks.append(input_features.astype(np.float32))

            if not all_valid_feature_chunks:
                logger.warning("No valid feature chunks generated after processing audio.")
                return torch.empty(0, dtype=torch.float32, device="cpu")

            if self.output_type == "spectrogram":
                try:
                    spectrograms_tensor = torch.from_numpy(np.stack(all_valid_feature_chunks)).float()
                    logger.debug("Returning %d WS spectrogram chunks, stacked shape: %s", len(all_valid_feature_chunks), spectrograms_tensor.shape)
                    return spectrograms_tensor.cpu()
                except ValueError as e:
                    logger.error("Could not stack spec chunks: %s. Returning empty.", e)
                    return torch.empty(0, dtype=torch.float32, device="cpu")

            elif self.output_type == "embedding":
                all_encoder_outputs_list = []
                num_total_chunks = len(all_valid_feature_chunks)
                logger.debug("Processing %d feature chunks through CTranslate2 encoder...", num_total_chunks)

                for i in range(0, num_total_chunks, self.batch_size):
                    batch_chunks_list = all_valid_feature_chunks[i : i + self.batch_size]
                    try:
                        batch_features_np = np.stack(batch_chunks_list)
                    except ValueError as stack_err:
                        shapes = [f.shape for f in batch_chunks_list]
                        logger.error("Inconsistent shapes in batch %d: %s. Skip. Err: %s", i // self.batch_size, shapes, stack_err)
                        continue

                    features_view = ctranslate2.StorageView.from_array(batch_features_np)
                    encoder_output_sv = self.model.encode(features_view, to_cpu=True)
                    all_encoder_outputs_list.append(torch.tensor(np.array(encoder_output_sv).tolist(), dtype=torch.float32))

                if not all_encoder_outputs_list:
                    logger.warning("No encoder outputs generated.")
                    return torch.empty(0, dtype=torch.float32, device="cpu")

                final_features = torch.cat(all_encoder_outputs_list, dim=0)
                first_dim_size = torch.prod(torch.tensor(final_features.shape[:-1]))
                last_dim_size = final_features.shape[-1]
                final_features = final_features.view(first_dim_size, last_dim_size)
                return final_features.cpu()

        except Exception as e:
            logger.error("WhisperSeg feature extraction failed: %s", e, exc_info=True)
            return torch.empty(0, dtype=torch.float32, device="cpu")