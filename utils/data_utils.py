import logging
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union
import scipy.io as sio
import torchaudio
import pandas as pd
from datasets import Dataset as HFDataset
from datasets import Audio, Value, ClassLabel, Features
import numpy as np
from tqdm.auto import tqdm
import torch

logger = logging.getLogger(__name__)


def _get_unique_filename(dest_dir: Path, filename: str) -> str:
    """
    Generates a unique filename if a conflict occurs in the destination directory.

    Args:
        dest_dir (Path): The directory where the file will be saved.
        filename (str): The desired filename.

    Returns:
        str: A unique filename.
    """
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while (dest_dir / new_filename).exists():
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return new_filename


def _flatten_mouse_identity_raw_structure(raw_data_dir: Path, target_dir: Path) -> None:
    """
    Flattens the directory structure from the original MUPET mouse identity data.

    Moves all .mat and .wav files from subdirectories into the target_dir.
    This is often necessary before processing MAT files if they reference WAV files
    assuming a flat structure.

    Args:
        raw_data_dir (Path): Path to the directory containing the raw MUPET data.
        target_dir (Path): Path to the directory where files will be moved (flattened structure).
    """
    logger.info("Flattening directory structure from %s to %s...", raw_data_dir, target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    moved_count = 0
    skipped_count = 0

    for dirpath, _, filenames in os.walk(raw_data_dir):
        current_dir = Path(dirpath)
        if current_dir == raw_data_dir or current_dir == target_dir:
            continue

        for filename in filenames:
            if filename.endswith((".mat", ".wav")):
                src_path = current_dir / filename
                dest_filename = _get_unique_filename(target_dir, filename)
                dest_path = target_dir / dest_filename
                try:
                    shutil.move(str(src_path), str(dest_path))
                    logger.debug("Moved: %s -> %s", src_path, dest_path)
                    moved_count += 1
                except Exception as e:
                    logger.error("Error moving %s: %s", src_path, e)
                    skipped_count += 1
            else:
                skipped_count += 1
                logger.debug("Skipping non-mat/wav file: %s", current_dir / filename)

    logger.info("Flattening complete. Moved %d files, skipped %d.", moved_count, skipped_count)


def _process_single_mat_file(mat_path: Path, audio_base_dir: Path, output_audio_dir: Path) -> List[Dict[str, Any]]:
    """
    Processes a single MUPET .mat file to extract syllable audio segments.

    Args:
        mat_path (Path): Path to the .mat file.
        audio_base_dir (Path): Base directory where corresponding .wav files are located.
        output_audio_dir (Path): Directory to save the extracted syllable .wav files.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a syllable,
                              containing metadata like file path, label (identity), speaker (identity), etc.
    """
    syllable_records = []
    try:
        mat_data = sio.loadmat(mat_path)
        syllable_data = mat_data.get("syllable_data")
        syllable_stats = mat_data.get("syllable_stats")
        filestats = mat_data.get("filestats")

        if syllable_data is None or syllable_stats is None or filestats is None:
            logger.warning("Skipping %s: Missing required data fields (syllable_data, syllable_stats, or filestats).", mat_path.name)
            return []
        if syllable_data.shape[1] == 0 or syllable_stats.shape[1] == 0:
            logger.warning("Skipping %s: No syllables found in data/stats.", mat_path.name)
            return []
        if syllable_data.shape[1] != syllable_stats.shape[1]:
            logger.warning(
                "Skipping %s: Mismatch between syllable_data (%d) and syllable_stats (%d) counts.",
                mat_path.name,
                syllable_data.shape[1],
                syllable_stats.shape[1],
            )
            return []

        try:
            wav_filename = syllable_data[0, 0][0]
            wav_path = audio_base_dir / wav_filename
            if not wav_path.exists():
                potential_paths = list(audio_base_dir.glob(f"**/{wav_filename}"))
                if not potential_paths:
                    logger.error("WAV file '%s' referenced in %s not found in %s.", wav_filename, mat_path.name, audio_base_dir)
                    return []
                wav_path = potential_paths[0]
                logger.debug("Found WAV file at %s", wav_path)

            fs = filestats[0, 0]["fs"][0, 0]
        except (IndexError, KeyError, TypeError) as e:
            logger.error("Error extracting metadata from %s: %s. Check MAT file structure.", mat_path.name, e, exc_info=True)
            return []

        try:
            audio, wav_fs = torchaudio.load(wav_path)
            if wav_fs != fs:
                logger.warning("Sample rate mismatch for %s: MAT=%d, WAV=%d. Using WAV's sample rate.", wav_path.name, fs, wav_fs)
                fs = wav_fs
            audio = audio[0].numpy()
            audio = audio.astype(np.float32)
            max_abs = np.max(np.abs(audio))
            if max_abs > 1e-6:
                audio = audio / max_abs
        except Exception as e:
            logger.error("Error loading or processing WAV file %s: %s", wav_path, e, exc_info=True)
            return []

        try:
            onsets_ms = syllable_stats[3, :]
            durations_ms = syllable_stats[1, :]
        except IndexError:
            logger.error("Error accessing onset/duration rows in syllable_stats for %s. Check MAT file structure.", mat_path.name)
            return []

        onsets_samples = (onsets_ms * fs / 1000).astype(int)
        durations_samples = (durations_ms * fs / 1000).astype(int)
        offsets_samples = onsets_samples + durations_samples

        identity = mat_path.stem.split("_")[0]
        file_id = mat_path.stem

        identity_output_dir = output_audio_dir / identity
        identity_output_dir.mkdir(parents=True, exist_ok=True)

        num_syllables = len(onsets_samples)
        for i in range(num_syllables):
            onset, offset = onsets_samples[i], offsets_samples[i]

            if onset < 0 or offset > len(audio) or onset >= offset:
                logger.warning("Skipping syllable %d in %s: Invalid boundaries [%d, %d] for audio length %d.", i + 1, mat_path.name, onset, offset, len(audio))
                continue

            syllable = audio[onset:offset]

            syllable_filename = f"{identity}_{file_id}_syllable{i+1}.wav"
            output_path = identity_output_dir / syllable_filename
            try:
                syllable_tensor = torch.from_numpy(syllable).unsqueeze(0)
                torchaudio.save(output_path, syllable_tensor, fs)

                syllable_records.append({
                    "audio": str(output_path),
                    "label": identity,
                    "speaker": identity,
                    "source_mat": str(mat_path),
                    "source_wav": str(wav_path),
                    "syllable_index": i + 1,
                    "onset_ms": onsets_ms[i],
                    "duration_ms": durations_ms[i],
                    "sampling_rate": fs,
                    "subset": "mouse_identity",
                })
            except Exception as e:
                logger.error("Error saving syllable %d from %s to %s: %s", i + 1, mat_path.name, output_path, e, exc_info=True)

    except Exception as e:
        logger.error("Critical error processing MAT file %s: %s", mat_path, e, exc_info=True)

    return syllable_records


def convert_mouse_identity_data(
    raw_data_dir: Union[str, Path], output_dir: Union[str, Path], flatten_first: bool = True
) -> Optional[str]:
    """
    Converts the raw MUPET mouse identity dataset (.mat files) into a structured
    format with individual syllable audio files and metadata.

    Args:
        raw_data_dir (Union[str, Path]): Path to the directory containing the raw
                                         MUPET data (potentially nested).
        output_dir (Union[str, Path]): Path to the directory where the processed data
                                       (syllable audio files, metadata JSON/HF dataset) will be saved.
        flatten_first (bool): If True, first flatten the raw_data_dir structure by moving
                              all .mat/.wav files to a single directory (`output_dir`/raw_flat).

    Returns:
        str: Path to the saved Hugging Face dataset directory, or None if conversion failed.
    """
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_audio_dir = output_dir / "audio_syllables"
    processed_audio_dir.mkdir(parents=True, exist_ok=True)
    hf_dataset_dir = output_dir / "hf_dataset"

    if flatten_first:
        flat_dir = output_dir / "raw_flat"
        _flatten_mouse_identity_raw_structure(raw_data_dir, flat_dir)
        mat_search_dir = flat_dir
        wav_base_dir = flat_dir
    else:
        logger.warning("Not flattening directory structure. MAT file processing might fail if WAV paths are relative.")
        mat_search_dir = raw_data_dir
        wav_base_dir = raw_data_dir

    mat_files = list(mat_search_dir.glob("**/*.mat"))
    if not mat_files:
        logger.error("No .mat files found in %s. Cannot proceed.", mat_search_dir)
        return None
    logger.info("Found %d .mat files to process.", len(mat_files))

    all_syllable_data = []
    for mat_file in tqdm(mat_files, desc="Processing MAT files"):
        syllable_records = _process_single_mat_file(mat_file, wav_base_dir, processed_audio_dir)
        all_syllable_data.extend(syllable_records)

    if not all_syllable_data:
        logger.error("No syllables were successfully extracted from any MAT file.")
        return None

    logger.info("Successfully extracted %d syllables.", len(all_syllable_data))

    try:
        features = Features({
            "audio": Audio(sampling_rate=all_syllable_data[0]["sampling_rate"]),
            "label": Value(dtype="string"),
            "speaker": Value(dtype="string"),
            "source_mat": Value(dtype="string"),
            "source_wav": Value(dtype="string"),
            "syllable_index": Value(dtype="int32"),
            "onset_ms": Value(dtype="float32"),
            "duration_ms": Value(dtype="float32"),
            "sampling_rate": Value(dtype="int32"),
            "subset": Value(dtype="string"),
        })

        def generator():
            for record in all_syllable_data:
                audio_path = record["audio"]
                record["audio"] = audio_path
                yield record

        hf_dataset = HFDataset.from_generator(generator, features=features)

        hf_dataset.save_to_disk(str(hf_dataset_dir))
        logger.info("Successfully created and saved Hugging Face dataset to %s", hf_dataset_dir)

        if flatten_first and flat_dir.exists():
            logger.info("Intermediate flat directory left at %s. Remove manually if desired.", flat_dir)

        return str(hf_dataset_dir)

    except Exception as e:
        logger.error("Failed to create or save Hugging Face dataset: %s", e, exc_info=True)
        return None