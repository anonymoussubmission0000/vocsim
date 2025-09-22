# -*- coding: utf-8 -*-
"""
Script to generate images of VAE/AE inputs and reconstructions, and an example P@1 table.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List

import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import gc
from tqdm import tqdm
import torch
import torchaudio
import torch.nn.functional as F
import pandas as pd
import argparse
from torch.distributions import LowRankMultivariateNormal

# --- Project Path Setup ---
try:
    script_path = Path(__file__).resolve()
    project_root_candidate1 = script_path.parents[1]
    project_root_candidate2 = script_path.parents[3]

    if (project_root_candidate1 / "vocsim" / "runner.py").exists():
        project_root = project_root_candidate1
    elif (project_root_candidate2 / "vocsim" / "runner.py").exists():
        project_root = project_root_candidate2
    elif (Path.cwd() / "vocsim" / "runner.py").exists():
        project_root = Path.cwd()
        print(f"WARNING: Script path resolution heuristic failed. Assuming CWD is project root: {project_root}")
    else:
        project_root = Path.cwd()
        print(f"ERROR: Cannot determine project root relative to script location. runner.py not found. Assuming CWD: {project_root}")
        print("Please ensure this script is run from within the project or adjust path setup if it's moved.")

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"INFO: Added project root to sys.path: {project_root}")
except NameError:
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    print(f"INFO: Assuming CWD project root for interactive session: {project_root}")

try:
    from vocsim.managers.dataset_manager import DatasetManager
    from utils.config_loader import load_config
    from utils.logging_utils import setup_logging
    from utils.torch_utils import get_device

    from reproducibility.models.vae import VariationalAutoencoder, preprocess_vae_input
    from reproducibility.models.autoencoder import Autoencoder, AudioConfig
except ImportError as e:
    print(f"ERROR: Could not import project modules: {e}. Ensure an __init__.py exists in 'vocsim' and 'utils' directories and that the project root ({project_root}) is correctly in PYTHONPATH.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG_NAME = "vocsim_paper.yaml"
BASE_CONFIG_NAME = "base.yaml"
CONFIG_DIR = project_root / "reproducibility" / "configs"
BASE_CONFIG_DIR = project_root / "configs"

OUTPUT_DIR = project_root / "reproducibility_outputs" / "vae_ae_images"
MODEL_SCOPE = "all"


def find_first_item_for_subset(dataset_manager: DatasetManager, subset_key: str) -> Optional[Dict[str, Any]]:
    """
    Finds the first item in a dataset subset that has valid audio data.

    Args:
        dataset_manager (DatasetManager): The dataset manager instance.
        subset_key (str): The key of the subset to search.

    Returns:
        Optional[Dict[str, Any]]: The first valid item found, or None if none is found.
    """
    logger.info("Searching for first item in subset '%s'...", subset_key)
    if dataset_manager.full_dataset_obj is None:
        logger.error("Full dataset not loaded in DatasetManager.")
        return None
    try:
        for item in dataset_manager.full_dataset_obj:
            if item.get("subset") == subset_key:
                if item.get("audio") and item["audio"].get("array") is not None:
                    logger.info("Found item for subset '%s'.", subset_key)
                    return item
                else:
                    logger.warning("Found item for subset '%s' but audio data is missing/invalid.", subset_key)
    except Exception as e:
        logger.error("Error searching for subset item '%s': %s", subset_key, e, exc_info=True)

    logger.warning("Could not find any valid items for subset '%s'.", subset_key)
    return None


def load_paper_model(model_type: str, config: Dict[str, Any], device: torch.device) -> Optional[torch.nn.Module]:
    """
    Loads either the VAE or AE model trained on the 'all' scope.

    Args:
        model_type (str): Type of model to load ('vae' or 'ae').
        config (Dict[str, Any]): The loaded configuration dictionary.
        device (torch.device): The device to load the model onto.

    Returns:
        Optional[torch.nn.Module]: The loaded model instance, or None if loading fails.
    """
    model_class = None
    model_params = None
    trainer_base_name = ""
    checkpoint_key = "model_state_dict"
    definition_key = ""

    models_base_dir = Path(config.get("models_dir", project_root / "models")).resolve()

    if model_type.lower() == "vae":
        model_class = VariationalAutoencoder
        trainer_base_name = "PaperVAETrainer"
        definition_key = "vae_model_def"

    elif model_type.lower() == "ae":
        model_class = Autoencoder
        trainer_base_name = "PaperAutoencoderTrainer"
        definition_key = "ae_model_def"

    else:
        logger.error("Unknown model_type: %s", model_type)
        return None

    logger.debug("Searching for %s params...", model_type.upper())
    definitions = config.get("definitions", {})
    if definition_key in definitions:
        model_params = definitions[definition_key].get("params")
        if model_params:
            logger.info("Found %s params in config['definitions']['%s'].", model_type.upper(), definition_key)
        else:
            logger.warning("Found '%s' in definitions, but 'params' key is missing or empty.", definition_key)

    if model_params is None:
        logger.debug("'%s' params not found in definitions. Searching 'train' block...", definition_key)
        train_jobs = config.get("train", [])
        for job in train_jobs:
            if job.get("trainer", {}).get("name") == trainer_base_name:
                model_cfg_in_train = job.get("model", {})
                model_params = model_cfg_in_train.get("params")
                if model_params:
                    logger.info("Found %s params in config['train'] block for '%s'.", model_type.upper(), trainer_base_name)
                    break
        if model_params is None:
            logger.error("Could not find %s model 'params' in definitions or train blocks.", model_type.upper())
            return None

    instance_params = {}
    try:
        if model_type.lower() == "vae":
            instance_params["z_dim"] = int(model_params["z_dim"])
            instance_params["model_precision"] = model_params.get("model_precision", 10.0)
            instance_params["device_name"] = device.type
        elif model_type.lower() == "ae":
            audio_config_dict = model_params["audio_config"]
            dims_dict = model_params.get("dimensions", {})
            instance_params["config"] = AudioConfig(**audio_config_dict)
            instance_params["max_spec_width"] = dims_dict.get("max_spec_width", 256)
            instance_params["bottleneck_dim"] = int(model_params["bottleneck_dim"])
    except KeyError as e:
        logger.error("Missing required key '%s' in found %s 'params'. Params found: %s", e, model_type.upper(), model_params)
        return None
    except Exception as e:
        logger.error(f"Error processing {model_type.upper()} params for instantiation: {e}. Params: {model_params}")
        return None

    scoped_trainer_name = f"{trainer_base_name}_{MODEL_SCOPE}"
    checkpoint_dir = models_base_dir / scoped_trainer_name / "checkpoints"
    model_path = checkpoint_dir / "final_model.pt"

    if not model_path.exists():
        logger.warning("'final_model.pt' not found in %s. Looking for latest checkpoint...", checkpoint_dir)
        available_checkpoints = sorted(checkpoint_dir.glob("*.pt"), key=os.path.getmtime, reverse=True)
        if available_checkpoints:
            model_path = available_checkpoints[0]
            logger.info("Using latest checkpoint: %s", model_path.name)
        else:
            logger.error("No %s model checkpoint found for scope '%s' in %s", model_type.upper(), MODEL_SCOPE, checkpoint_dir)
            return None
    try:
        logger.info("Instantiating %s model...", model_type.upper())
        model = model_class(**instance_params)

        logger.info("Loading %s checkpoint: %s", model_type.upper(), model_path)
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get(checkpoint_key)
        if state_dict is None:
            dummy_keys = model.state_dict().keys()
            if all(k in checkpoint for k in dummy_keys):
                state_dict = checkpoint
            else:
                raise KeyError(f"Checkpoint key '{checkpoint_key}' not found and checkpoint is not a state_dict. Keys: {list(checkpoint.keys())}")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys loading %s state_dict: %s", model_type.upper(), missing)
        if unexpected:
            logger.warning("Unexpected keys loading %s state_dict: %s", model_type.upper(), unexpected)

        model.to(device)
        model.eval()
        logger.info("%s model (Scope: %s) loaded successfully to %s.", model_type.upper(), MODEL_SCOPE, device)
        return model
    except Exception as e:
        logger.error("Failed to load/initialize %s model from %s: %s", model_type.upper(), model_path, e, exc_info=True)
        return None


def plot_spectrogram(
    spec_data: np.ndarray,
    title: str,
    output_path: Path,
    sr: int,
    hop_length: int,
    is_mel: bool = True,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    n_fft: Optional[int] = None,
):
    """
    Plots and saves a spectrogram.

    Args:
        spec_data (np.ndarray): 2D numpy array containing spectrogram data.
        title (str): Title for the plot.
        output_path (Path): Path to save the plot file.
        sr (int): Sample rate used for the spectrogram.
        hop_length (int): Hop length used for the spectrogram.
        is_mel (bool): True if it's a Mel spectrogram, False otherwise.
        fmin (Optional[float]): Minimum frequency for plotting (for non-Mel).
        fmax (Optional[float]): Maximum frequency for plotting (for non-Mel).
        n_fft (Optional[int]): FFT size used for the spectrogram (for non-Mel).
    """
    if spec_data is None or spec_data.size == 0:
        logger.warning("Cannot plot empty spectrogram for %s", title)
        return
    plt.figure(figsize=(10, 4))
    y_axis = "mel" if is_mel else "linear"
    plot_fmin = fmin if fmin is not None else 0
    plot_fmax = fmax if fmax is not None else sr / 2.0
    try:
        img = librosa.display.specshow(
            spec_data, sr=sr, hop_length=hop_length, x_axis="time", y_axis=y_axis, fmin=plot_fmin, fmax=plot_fmax, cmap="magma"
        )
        plt.colorbar(img, format="%+2.0f dB" if is_mel and np.any(spec_data < -10) else "%.2f")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, format="svg", bbox_inches="tight")
        logger.info("Saved plot: %s", output_path.name)
    except Exception as e:
        logger.error("Failed to plot/save spectrogram %s: %s", output_path, e, exc_info=True)
    finally:
        plt.close()


def plot_latent_vector(latent_vec: np.ndarray, title: str, output_path: Path):
    """
    Plots and saves a 1D latent vector as an image.

    Args:
        latent_vec (np.ndarray): 1D numpy array containing the latent vector.
        title (str): Title for the plot.
        output_path (Path): Path to save the plot file.
    """
    if latent_vec is None or latent_vec.size == 0:
        logger.warning("Cannot plot empty latent vector for %s", title)
        return
    plt.figure(figsize=(10, 1))
    try:
        img = plt.imshow(latent_vec.reshape(1, -1), aspect="auto", cmap="viridis")
        plt.colorbar(img)
        plt.title(title)
        plt.xlabel("Latent Dimension")
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(output_path, format="svg", bbox_inches="tight")
        logger.info("Saved plot: %s", output_path.name)
    except Exception as e:
        logger.error("Failed to plot/save latent vector %s: %s", output_path, e, exc_info=True)
    finally:
        plt.close()


def sanitize_latex_str(text: str) -> str:
    """
    Sanitizes a string for LaTeX output by escaping special characters.

    Args:
        text (str): The input string.

    Returns:
        str: The sanitized string.
    """
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\^{}",
        "\\": r"\textbackslash{}",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def generate_p1_latex_table(
    data_rows: list[dict],
    table_label: str = "tab:p1_results",
    caption_text: str = "P@1 Results Across Subsets and Distances ($\\uparrow$ better)",
    table_format_str: str = "2.1",
) -> str:
    """
    Generates LaTeX code for a P@1 results longtable.

    Args:
        data_rows: A list of dictionaries, where each dictionary represents a row.
                   Example: {"Method": "M1", "Dist": "D1", "BC": 10.1, ...}
        table_label: The LaTeX label for the table.
        caption_text: The caption for the table.
        table_format_str: The siunitx table-format string (e.g., "2.1", "3.1").
                          Ensure this matches your data range (e.g., "3.1" if 100.0 is possible).
    Returns:
        A string containing the LaTeX code for the table.
    """
    num_s_columns = 21
    total_columns = 2 + num_s_columns

    s_column_headers_config = [
        {"key": "BC", "display": "BC"},
        {"key": "BS1", "display": "BS1"},
        {"key": "BS2", "display": "BS2"},
        {"key": "BS3", "display": "BS3"},
        {"key": "BS4", "display": "BS4"},
        {"key": "BS5", "display": "BS5"},
        {"key": "ES1", "display": "ES1"},
        {"key": "HP", "display": "HP"},
        {"key": "HS1", "display": "HS1"},
        {"key": "HS2", "display": "HS2"},
        {"key": "HU1", "display": "HU1"},
        {"key": "HU2", "display": "HU2"},
        {"key": "HU3", "display": "HU3"},
        {"key": "HU4", "display": "HU4"},
        {"key": "HW1", "display": "HW1"},
        {"key": "HW2", "display": "HW2"},
        {"key": "HW3", "display": "HW3"},
        {"key": "HW4", "display": "HW4"},
        {"key": "OC1", "display": "OC1"},
        {"key": "Avg", "display": "Avg"},
        {"key": "Avg (Blind)", "display": "\\makecell{Avg\\\\(Blind)}"},
    ]
    s_column_keys = [item["key"] for item in s_column_headers_config]

    latex_lines = []
    latex_lines.append("% Add to your LaTeX preamble: \\usepackage{longtable, booktabs, siunitx, makecell}")
    latex_lines.append("\\scriptsize")
    latex_lines.append("\\setlength{\\tabcolsep}{2pt}")
    latex_lines.append(f"% For S columns, table-format={table_format_str} is used. Ensure this matches your data (e.g., use \"3.1\" if scores like 100.0 can occur).")
    latex_lines.append(f"\\begin{{longtable}}{{l l *{{{num_s_columns}}}{{S[table-format={table_format_str}]}}}}")

    latex_lines.append(f"\\caption{{{caption_text}}}\\label{{{table_label}}}\\\\")
    latex_lines.append("\\toprule")

    header_line_parts = ["Method", "Dist"]
    for item in s_column_headers_config:
        header_line_parts.append(f"\\multicolumn{{1}}{{c}}{{{item['display']}}}")
    header_line = " & ".join(header_line_parts) + " \\\\"

    latex_lines.append(header_line)
    latex_lines.append("\\midrule")
    latex_lines.append("\\endfirsthead")
    latex_lines.append("")
    latex_lines.append(f"\\caption[]{{(Continued) {caption_text}}}\\\\")
    latex_lines.append("\\toprule")
    latex_lines.append(header_line)
    latex_lines.append("\\midrule")
    latex_lines.append("\\endhead")
    latex_lines.append("")

    latex_lines.append("\\midrule")
    latex_lines.append(f"\\multicolumn{{{total_columns}}}{{r}}{{\\textit{{Continued on next page}}}}\\\\")
    latex_lines.append("\\endfoot")
    latex_lines.append("")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\endlastfoot")
    latex_lines.append("")

    for row_dict in data_rows:
        row_str_parts = [str(row_dict.get("Method", "")), str(row_dict.get("Dist", ""))]
        for key in s_column_keys:
            val = row_dict.get(key)
            if isinstance(val, (int, float)):
                row_str_parts.append(str(val))
            elif val is None or str(val).strip() == "":
                row_str_parts.append("{}")
            else:
                row_str_parts.append(f"\\multicolumn{{1}}{{c}}{{{str(val)}}}")
        latex_lines.append(" & ".join(row_str_parts) + " \\\\")

    latex_lines.append("\\end{longtable}")
    return "\n".join(latex_lines)


def main():
    """
    Main function to generate VAE/AE input/reconstruction images and a P@1 LaTeX table example.
    """
    log_dir = OUTPUT_DIR.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging({"level": "INFO", "log_file": log_dir / "vae_image_script.log"})
    logger.info("--- Starting Latent Reconstruction Script ---")

    config_path = CONFIG_DIR / CONFIG_NAME
    base_config_path = BASE_CONFIG_DIR / BASE_CONFIG_NAME
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)
    cfg = load_config(config_path, base_config_path=base_config_path if base_config_path.exists() else None)
    logger.info("Loaded config from %s", config_path.name)

    device = get_device(cfg.get("force_cpu", False))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Using device: %s. Output directory: %s", device, OUTPUT_DIR)

    dataset_manager = DatasetManager(cfg)
    if not dataset_manager.load_full_dataset():
        logger.error("Failed to load dataset.")
        sys.exit(1)

    item_bs5 = find_first_item_for_subset(dataset_manager, "BS5")
    item_hw2 = find_first_item_for_subset(dataset_manager, "HW2")
    if not item_bs5 or not item_hw2:
        logger.error("Could not find required example items from BS5 and/or hw2.")
        sys.exit(1)
    items_to_process = {"BS5": item_bs5, "HW2": item_hw2}

    vae_model = load_paper_model("vae", cfg, device)
    ae_model = load_paper_model("ae", cfg, device)
    if not vae_model or not ae_model:
        logger.error("Failed to load VAE and/or AE models.")
        sys.exit(1)

    vae_model_params_from_cfg = cfg.get("definitions", {}).get("vae_model_def", {}).get("params", {})
    vae_trainer_params_from_cfg = cfg.get("definitions", {}).get("vae_trainer_def", {}).get("params", {})
    vae_frontend_params = vae_trainer_params_from_cfg.get("vae_frontend_params", {})

    if not vae_frontend_params:
        logger.error("VAE frontend params (vae_frontend_params) not found in config definitions (vae_trainer_def).")
        sys.exit(1)

    target_sr_vae = vae_frontend_params.get("target_sr", 16000)
    n_fft_vae = vae_frontend_params.get("n_fft", 512)
    hop_length_vae = vae_frontend_params.get("hop_length", 128)
    win_length_vae = vae_frontend_params.get("win_length", n_fft_vae)
    window_fn_str_vae = vae_frontend_params.get("window_fn_str", "hann_window")
    spec_height_vae = vae_frontend_params.get("spec_height", 128)
    spec_width_vae = vae_frontend_params.get("spec_width", 128)
    window_overlap_vae = vae_frontend_params.get("window_overlap", 0.75)

    vae_window_samples = (spec_width_vae - 1) * hop_length_vae
    vae_hop_samples = int(vae_window_samples * (1 - window_overlap_vae))

    ae_audio_config: AudioConfig = ae_model.config
    ae_n_fft = ae_audio_config.nfft
    ae_hop_length = ae_audio_config.hop_length
    ae_n_mels = ae_audio_config.n_mels
    ae_fmin = ae_audio_config.fmin
    ae_fmax = ae_audio_config.fmax
    ae_sr = ae_audio_config.sr

    for model_type, model_instance in [("VAE", vae_model), ("AE", ae_model)]:
        logger.info("\n--- Processing Model: %s ---", model_type)
        if model_instance is None:
            logger.warning("Model instance for %s is None. Skipping.", model_type)
            continue

        for subset_key, item_data in items_to_process.items():
            logger.info("-- Processing Example from Subset: %s --", subset_key)
            audio_array = item_data["audio"]["array"]
            sample_rate = item_data["audio"]["sampling_rate"]
            original_name = item_data.get("original_name", f"item_{subset_key}")

            original_mel_spec = None
            reconstructed_mel_spec = None
            latent_representation = None

            try:
                with torch.no_grad():
                    if model_type == "VAE":
                        vae_input_chunks = preprocess_vae_input(
                            audio_tensor=torch.from_numpy(audio_array.astype(np.float32)),
                            sample_rate=sample_rate,
                            device=device,
                            target_sr=target_sr_vae,
                            n_fft=n_fft_vae,
                            hop_length=hop_length_vae,
                            win_length=win_length_vae,
                            window_fn_str=window_fn_str_vae,
                            spec_height=spec_height_vae,
                            spec_width=spec_width_vae,
                            window_samples=vae_window_samples,
                            hop_samples=vae_hop_samples,
                        )
                        if not vae_input_chunks:
                            raise ValueError("VAE preprocessing yielded no chunks.")
                        vae_input_tensor = vae_input_chunks[0].to(device)
                        original_mel_spec = vae_input_tensor.squeeze(0).cpu().numpy()

                        mu, u, d_diag = model_instance.encode(vae_input_tensor.unsqueeze(0))
                        z = mu
                        reconstructed_output = model_instance.decode(z)
                        reconstructed_mel_spec = reconstructed_output.view(spec_height_vae, spec_width_vae).cpu().numpy()
                        latent_representation = mu.squeeze(0).cpu().numpy()

                        plot_sr = target_sr_vae
                        plot_hop_length = hop_length_vae
                        plot_n_fft = n_fft_vae
                        plot_fmin = None
                        plot_fmax = None

                    elif model_type == "AE":
                        audio_tensor = torch.from_numpy(audio_array.astype(np.float32)).unsqueeze(0).to(device)
                        if sample_rate != ae_sr:
                            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=ae_sr).to(device)
                            audio_tensor = resampler(audio_tensor)

                        max_val = torch.max(torch.abs(audio_tensor))
                        if max_val > 1e-6:
                            audio_tensor = audio_tensor / max_val

                        original_mel_spec_tensor = model_instance.frontend(audio_tensor)
                        original_mel_spec = original_mel_spec_tensor.squeeze(0).squeeze(0).cpu().numpy()

                        reconstructed_output, encoded_bottleneck = model_instance(audio_tensor)
                        reconstructed_mel_spec = reconstructed_output.squeeze(0).squeeze(0).cpu().numpy()

                        plot_sr = ae_sr
                        plot_hop_length = ae_hop_length
                        plot_n_fft = ae_n_fft
                        plot_fmin = ae_fmin
                        plot_fmax = ae_fmax

            except Exception as e:
                logger.error("Error processing %s for %s ('%s'): %s", model_type, subset_key, original_name, e, exc_info=True)
                continue

            safe_original_name = Path(original_name).stem.replace(" ", "_").replace("/", "_")
            base_filename = f"{model_type}_{subset_key}_{safe_original_name}"

            plot_spectrogram(
                original_mel_spec,
                f"Original Mel ({model_type} Input) - {subset_key}",
                OUTPUT_DIR / f"{base_filename}_original_mel.svg",
                sr=plot_sr,
                hop_length=plot_hop_length,
                is_mel=True,
                fmin=plot_fmin,
                fmax=plot_fmax,
                n_fft=plot_n_fft,
            )

            plot_spectrogram(
                reconstructed_mel_spec,
                f"Reconstructed Mel - {model_type} - {subset_key}",
                OUTPUT_DIR / f"{base_filename}_reconstructed_mel.svg",
                sr=plot_sr,
                hop_length=plot_hop_length,
                is_mel=True,
                fmin=plot_fmin,
                fmax=plot_fmax,
                n_fft=plot_n_fft,
            )

            if latent_representation is not None:
                plot_latent_vector(
                    latent_representation, f"{model_type} Latent/Bottleneck - {subset_key}", OUTPUT_DIR / f"{base_filename}_latent.svg"
                )

            del original_mel_spec, reconstructed_mel_spec, latent_representation
            if "audio_tensor" in locals():
                del audio_tensor
            if "vae_input_tensor" in locals():
                del vae_input_tensor
            if "vae_input_chunks" in locals():
                del vae_input_chunks
            if "original_mel_spec_tensor" in locals():
                del original_mel_spec_tensor
            if "reconstructed_output" in locals():
                del reconstructed_output
            if "encoded_bottleneck" in locals():
                del encoded_bottleneck
            if "mu" in locals():
                del mu
            if "u" in locals():
                del u
            if "d_diag" in locals():
                del d_diag
            if "z" in locals():
                del z
            gc.collect()

        del model_instance
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("--- VAE/AE Image Generation Finished ---")

    logger.info("--- Generating P@1 LaTeX Table Example ---")
    example_p1_data = [
        {"Method": "MethodA", "Dist": "DistX", "BC": 10.1, "BS1": 12.3, "BS2": 13.4, "BS3": 14.5, "BS4": 15.6, "BS5": 16.7, "ES1": 17.8, "HP": 18.9, "HS1": 19.0, "HS2": 20.1, "HU1": 21.2, "HU2": 22.3, "HU3": 23.4, "HU4": 24.5, "HW1": 25.6, "HW2": 26.7, "HW3": 27.8, "HW4": 28.9, "OC1": 29.0, "Avg": 30.1, "Avg (Blind)": 31.2},
        {"Method": "MethodB", "Dist": "DistY", "BC": 11.1, "BS1": None, "BS2": "E", "BS3": 15.5, "BS4": 16.6, "BS5": 17.7, "ES1": 18.8, "HP": 19.9, "HS1": 20.0, "HS2": 21.1, "HU1": 22.2, "HU2": 23.3, "HU3": 24.4, "HU4": 25.5, "HW1": 26.6, "HW2": 27.7, "HW3": 28.8, "HW4": 29.9, "OC1": 30.0, "Avg": 31.1, "Avg (Blind)": 100.0},
        {"Method": "MethodC", "Dist": "DistZ", "BC": 5.1, "BS1": 3.3, "BS2": 4.4, "BS3": "-", "BS4": 6.6, "BS5": 7.7, "ES1": 8.8, "HP": 9.9, "HS1": 0.0, "HS2": 1.1, "HU1": 2.2, "HU2": 3.3, "HU3": 4.4, "HU4": 5.5, "HW1": 6.6, "HW2": 7.7, "HW3": 8.8, "HW4": 9.9, "OC1": 10.0, "Avg": 11.1, "Avg (Blind)": 12.2},
    ]

    logger.info("Attempting to generate table with table-format='2.1'. This will fail in LaTeX if scores like 100.0 exist with this format.")
    latex_table_2_1 = generate_p1_latex_table(
        example_p1_data,
        table_label="tab:p1_results_example_2_1",
        caption_text="Example P@1 Results (2.1 format)",
        table_format_str="2.1",
    )
    output_path_2_1 = OUTPUT_DIR / "example_p1_table_format_2_1.tex"
    try:
        with open(output_path_2_1, "w", encoding="utf-8") as f:
            f.write(latex_table_2_1)
        logger.info("Generated P@1 LaTeX table (2.1 format) to: %s", output_path_2_1)
    except Exception as e:
        logger.error("Failed to write 2.1 format table: %s", e)

    print("\n--- Example LaTeX P@1 Table (format: 2.1) ---")
    print(latex_table_2_1)

    logger.info("Generating table with table-format='3.1'. This is safer for scores up to 100.0.")
    latex_table_3_1 = generate_p1_latex_table(
        example_p1_data,
        table_label="tab:p1_results_example_3_1",
        caption_text="Example P@1 Results (3.1 format)",
        table_format_str="3.1",
    )
    output_path_3_1 = OUTPUT_DIR / "example_p1_table_format_3_1.tex"
    try:
        with open(output_path_3_1, "w", encoding="utf-8") as f:
            f.write(latex_table_3_1)
        logger.info("Generated P@1 LaTeX table (3.1 format) to: %s", output_path_3_1)
    except Exception as e:
        logger.error("Failed to write 3.1 format table: %s", e)

    print("\n--- Example LaTeX P@1 Table (format: 3.1) ---")
    print(latex_table_3_1)

    logger.info("--- Script Finished (including table generation example) ---")


if __name__ == "__main__":
    main()