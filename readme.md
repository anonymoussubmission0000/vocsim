# VocSim: Evaluating Embedding Generalization for Zero-Shot Audio Similarity
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-VocSim-blue)](https://huggingface.co/datasets/anonymous-submission000/vocsim)
[![Paper](https://img.shields.io/badge/Paper-arXiv%20(TBD)-red)](https://example.com/link-to-your-paper)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-PwC%20(TBD)-green)](https://example.com/link-to-leaderboard)

**VocSim** is a framework for evaluating neural embeddings in **zero-shot audio similarity** tasks. It provides a robust, extensible platform with a diverse benchmark dataset, advanced evaluation metrics, and a modular pipeline to compare audio feature representations—from traditional Mel spectrograms to pretrained models and custom self-supervised approaches.

## Installation

### Prerequisites

- Python 3.8+
- Conda (recommended) or virtualenv
- Git
- CUDA-enabled GPU (strongly recommended for performance; see *Appendix A.8* in the paper for compute details)

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/anonymoussubmission0000/vocsim.git
   cd vocsim
   ```

2. **Set Up Environment**:
   Using Conda:
   ```bash
   conda env create -f environment.yml
   conda activate vocsim_env
   ```
   Or with virtualenv:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   *Note*: Ensure PyTorch is compatible with your CUDA version for GPU acceleration.

3. **Install VocSim**:
   ```bash
   pip install .
   ```

4. **Download Datasets**:
   - **VocSim Core Dataset**: [Hugging Face](https://huggingface.co/datasets/anonymous-submission000/vocsim).
   - **Application Datasets**:
     - Avian Perception: [Hugging Face](https://huggingface.co/datasets/anonymous-submission000/vocsim-applications-avian-perception)
     - Mouse Strain: [Hugging Face](https://huggingface.co/datasets/anonymous-submission000/vocsim-applications-mouse-strain) 
     - Mouse Identity: [Hugging Face](https://huggingface.co/datasets/anonymous-submission000/vocsim-applications-mouse-identity)

5. **Configure Paths**:
   - Edit `configs/base.yaml` to set `project_root`, `data_dir`, `results_dir`, `features_dir`, and `models_dir`.
   - For paper reproduction, copy `reproducibility/configs/vocsim_paper.yaml` to `configs/run_config.yaml` and update paths (e.g., `results_dir`, dataset paths).

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
*VocSim is under active development. Check [GitHub Issues](https://github.com/anonymoussubmission0000/vocsim/issues) for updates or to report bugs.*