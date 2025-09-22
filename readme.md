# VocSim: A Training-Free Benchmark for Content Identity in Single-Source Audio Embeddings

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-VocSim-blue)](https://huggingface.co/datasets/vocsim/public-benchmark)
[![Paper](https://img.shields.io/badge/Paper-arXiv%20(TBD)-red)](https://example.com/link-to-your-paper)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-%F0%9F%A4%97%20-green)](https://huggingface.co/spaces/vocsim/vocsim)

**VocSim** is a framework for evaluating neural embeddings in **zero-shot audio similarity** tasks. It provides a robust, extensible platform with a diverse benchmark dataset, advanced evaluation metrics, and a modular pipeline to compare audio feature representations—from traditional Mel spectrograms to pretrained models and custom self-supervised approaches. All official datasets are hosted on the [Hugging Face Hub](https://huggingface.co/vocsim).

## Installation

### Prerequisites

- Python 3.8+
- Git
- An environment manager: `uv` (fastest), `Conda`/`Mamba`, or `virtualenv`.
- CUDA-enabled GPU (strongly recommended for performance; see *Appendix A.8* in the paper for compute details).

### Steps

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/vocsim/vocsim.git
    cd vocsim
    ```

2.  **Set Up Environment**:

    Choose one of the following methods. We recommend `uv` for the fastest Python package installation.

    **Using `uv`**:
    ```bash
    # Create and activate a virtual environment
    uv venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    # The command below is for CUDA 12.1. Find the correct one for your system at https://pytorch.org/get-started/locally/
    uv pip install torch==2.3.1+cu121 torchaudio==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121

    # Install the remaining packages from requirements.txt
    uv pip install -r requirements.txt
    ```

    **Using `virtualenv` + `pip`**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate

    # Note: Ensure you install a version of PyTorch compatible with your
    # system's CUDA version for GPU acceleration before installing other packages.
    pip install torch==2.3.1+cu121 torchaudio==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    ```

34.  **Download Datasets**:
    -   **VocSim Core Dataset**: [vocsim/public-benchmark](https://huggingface.co/datasets/vocsim/public-benchmark)
    -   **Application Datasets**:
        -   Avian Perception: [vocsim/avian-perception-benchmark](https://huggingface.co/datasets/vocsim/avian-perception-benchmark)
        -   Mouse Strain: [vocsim/mouse-strain-classification-benchmark](https://huggingface.co/datasets/vocsim/mouse-strain-classification-benchmark)
        -   Mouse Identity: [vocsim/mouse-identity-classification-benchmark](https://huggingface.co/datasets/vocsim/mouse-identity-classification-benchmark)

4.  **Configure Paths**:
    -   Edit `configs/base.yaml` to set `project_root`, `data_dir`, `results_dir`, `features_dir`, and `models_dir`.
    -   For paper reproduction, copy `reproducibility/configs/vocsim_paper.yaml` to `configs/run_config.yaml` and update paths (e.g., `results_dir`, dataset paths).

## Usage

VocSim’s pipeline is orchestrated by `vocsim.runner.PipelineRunner`, with stages for dataset management, training, feature extraction, distance computation, and benchmarking. Run experiments via scripts in `reproducibility/scripts/`.

### Example: Reproduce Paper Benchmarks
```bash
python reproducibility/scripts/vocsim.py --steps all
```
This executes training (if configured), feature extraction, distance computation, and benchmarking per `reproducibility/configs/vocsim_paper.yaml`.

### Command-Line Options
- `--steps`: Select pipeline stages (`train`, `features`, `distances`, `benchmarks`, `all`).
- Cached outputs are reused if a step is skipped.

### Outputs
- **Logs**: Stored in `logging:log_dir` (e.g., `logs/`).
- **Cached Data**: Features, distances, and PCA models in `features_dir` (e.g., `features_cache/`).
- **Models**: Saved in `models_dir` (e.g., `models/`).
- **Results**: Benchmark outputs in `results_dir` (e.g., `results/`), with JSON/CSV summaries.

## Extending VocSim

VocSim’s modular design supports custom components.

### Add a Feature Extractor
1. Create a class in `features/` inheriting from `features.base.FeatureExtractor`.
2. Implement `_initialize` and `extract` methods.
3. Update `configs/features.yaml`:
   ```yaml
   - name: MyExtractor
     module: features.my_extractor
     class: MyExtractorClass
     params:
       model_path: "path/to/model.pt"
   ```
4. Add to `features/__init__.py`.

### Add a Distance Metric
1. Create a class in `distances/` inheriting from `distances.base.DistanceCalculator`.
2. Implement `_initialize` and `compute_pairwise`.
3. Update `configs/distances.yaml`.
4. Add to `distances/__init__.py`.

### Add a Benchmark
1. Create a class in `benchmarks/` inheriting from `benchmarks.base.Benchmark`.
2. Implement `_initialize` and `evaluate`.
3. Update `configs/benchmarks.yaml`.
4. Add to `benchmarks/__init__.py`.

### Add a Dataset
1. Prepare data in Hugging Face format with `audio` and `label` columns.
2. Update `configs/datasets.yaml`:
   ```yaml
   my_dataset:
     id: "path/to/hf_dataset"
     subset: null
     split: train
   ```
3. Specify in run config’s `dataset` section.

## Reproducing the Paper

The `reproducibility/` directory includes:
- **Configs**: `vocsim_paper.yaml`, `avian_paper.yaml`, `mouse_strain_paper.yaml`, `mouse_identity_paper.yaml`.
- **Scripts**: `vocsim.py`, `avian_perception.py`, `mouse_strain.py`, `mouse_identity.py`.
- **Models/Trainers**: Custom AE/VAE implementations in `reproducibility/models/` and `reproducibility/trainers/`.

**Steps**:
1. Update paths in config files.
2. Run:
   ```bash
   python reproducibility/scripts/vocsim.py --steps all
   ```

## Utilities
- `reproducibility/scripts/misc/`:
  - `table_construction.py`: Generate LaTeX tables.
  - `plots_visualization.py`: Create performance plots.
  - `feature_extraction.py`: Inspect features for a single sample.
  - `latent_reconstruction.py`: Visualize AE/VAE reconstructions.
  - `generate_dataset_characteristics.py`: Compute dataset statistics.

## Citation

## License
MIT License (see `LICENSE`). Datasets and pretrained models follow their respective licenses (see *Appendix A.1.1* in the paper).

## Acknowledgements
Thanks to the dataset creators and pretrained model developers whose work enables VocSim.

---
*VocSim is under active development. Check [GitHub Issues](https://github.com/vocsim/vocsim/issues) for updates or to report bugs.*