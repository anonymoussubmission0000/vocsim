"""Helper to load WhisperSeg CTranslate2 model."""

import logging
import os
import json
from pathlib import Path
from typing import List, Tuple, Union, Optional

import ctranslate2
import tokenizers
from transformers import WhisperTokenizer

logger = logging.getLogger(__name__)

VALID_COMPUTE_TYPES = ["float16", "int8_float16", "int8", "float32"]


def load_whisperseg_model(
    model_path: str,
    device: str = "cuda",
    device_index: Union[int, List[int]] = 0,
    compute_type: Optional[str] = None,
) -> Tuple[ctranslate2.models.Whisper, Union[WhisperTokenizer, tokenizers.Tokenizer]]:
    """
    Loads the WhisperSeg CTranslate2 model and associated tokenizer.

    Handles potential differences in tokenizer file locations based on conversion.

    Args:
        model_path (str): Path to the converted CTranslate2 WhisperSeg model directory.
        device (str): Device to load the model onto ('cuda' or 'cpu').
        device_index (Union[int, List[int]]): Index or list of indices for CUDA devices.
        compute_type (Optional[str]): CTranslate2 computation type (e.g., 'float16', 'int8').
                                       If None, defaults based on device (float16/GPU, float32/CPU).

    Returns:
        Tuple[ctranslate2.models.Whisper, Union[WhisperTokenizer, tokenizers.Tokenizer]]: Loaded model and tokenizer.

    Raises:
        FileNotFoundError: If the model path or required files don't exist.
        ValueError: If an invalid compute_type is provided.
        Exception: For other CTranslate2 or Tokenizer loading errors.
    """
    model_path_obj = Path(model_path)
    if not model_path_obj.is_dir():
        raise FileNotFoundError(f"CTranslate2 model directory not found: {model_path}")
    logger.info(f"Loading WhisperSeg CTranslate2 model from: {model_path_obj}")

    if compute_type is None:
        _compute_type = "float16" if device == "cuda" else "float32"
    elif compute_type in VALID_COMPUTE_TYPES:
        _compute_type = compute_type
    else:
        raise ValueError(f"Invalid compute_type '{compute_type}'. Must be one of {VALID_COMPUTE_TYPES}")

    try:
        if isinstance(device_index, list):
            if len(device_index) > 1:
                logger.warning(
                    "Multiple device indices (%s) provided for CTranslate2 Whisper. Loading model onto first device (%s) only.",
                    device_index,
                    device_index[0],
                )
                _device_index = device_index[0]
            elif len(device_index) == 1:
                _device_index = device_index[0]
            else:
                _device_index = 0
                logger.warning("Empty device_index list provided, defaulting to device 0.")
        else:
            _device_index = device_index

        logger.info(f"Using compute_type: {_compute_type}, device: {device}, device_index: {_device_index}")

        model = ctranslate2.models.Whisper(str(model_path_obj), device=device, device_index=_device_index, compute_type=_compute_type)
        logger.info("CTranslate2 model loaded successfully.")
    except Exception as e:
        logger.error("Failed to load CTranslate2 model from %s: %s", model_path_obj, e, exc_info=True)
        raise

    tokenizer = None
    hf_token_path = model_path_obj / "hf_model"
    tok_json_path = model_path_obj / "tokenizer.json"

    try:
        if hf_token_path.is_dir() and (hf_token_path / "tokenizer_config.json").exists():
            tokenizer = WhisperTokenizer.from_pretrained(hf_token_path)
            logger.info(f"Loaded WhisperTokenizer from HF subdirectory: {hf_token_path}")
        elif tok_json_path.is_file():
            tokenizer = tokenizers.Tokenizer.from_file(str(tok_json_path))
            logger.info(f"Loaded base Tokenizer from: {tok_json_path}")
        else:
            config_path = model_path_obj / "config.json"
            original_model_id = None
            if config_path.is_file():
                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)
                        original_model_id = config.get("_name_or_path") or config.get("model_type")
                        if original_model_id and "/" not in original_model_id:
                            original_model_id = None
                except Exception as cfg_err:
                    logger.warning("Error reading config.json: %s", cfg_err)

            if original_model_id:
                logger.warning("Tokenizer files missing. Attempting load from original ID: %s", original_model_id)
                try:
                    tokenizer = WhisperTokenizer.from_pretrained(original_model_id)
                except Exception as fallback_err:
                    logger.error("Failed tokenizer fallback %s: %s", original_model_id, fallback_err)
                    raise FileNotFoundError(f"Could not find/load tokenizer in {model_path_obj} or from fallback.")
            else:
                raise FileNotFoundError(f"Could not find tokenizer files in {model_path_obj}.")

        logger.info("Tokenizer loaded successfully.")
        return model, tokenizer

    except Exception as e:
        logger.error("Failed to load tokenizer for %s: %s", model_path_obj, e, exc_info=True)
        raise