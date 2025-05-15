Okay, let's integrate the details from the provided Python code files into the README, making it more specific and accurate to the implementation.

```markdown
# VocSim: Evaluating Embedding Generalization for Zero-Shot Audio Similarity
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-VocSim-blue)](https://huggingface.co/datasets/anonymous-submission000/vocsim)
[![Paper](https://img.shields.io/badge/Paper-arXiv%20(TBD)-red)](https://example.com/link-to-your-paper)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-PwC%20(TBD)-green)](https://example.com/link-to-leaderboard)

**VocSim** is a framework for evaluating neural embeddings in **zero-shot audio similarity** tasks. It provides a robust, extensible platform with a diverse benchmark dataset, advanced evaluation metrics, and a modular pipeline to compare audio feature representations—from traditional Mel spectrograms to pretrained models and custom self-supervised approaches. It is designed to facilitate reproducible evaluation of audio embeddings.

## Installation

### Prerequisites

- Python 3.8+
- Conda (recommended) or virtualenv
- Git
- CUDA-enabled GPU (strongly recommended for performance; see paper Appendix A.8)
- Basic build tools (compilers etc. for some dependencies like `hdbscan`, `umap-learn`, `ctranslate2`)

### Steps

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/anonymoussubmission0000/vocsim.git
    cd vocsim
    ```

2.  **Set Up Environment**:
    VocSim includes a `environment.yml` file specifying most core dependencies including `pytorch`, `transformers`, `datasets`, `scikit-learn`, `scipy`, `pandas`, `numpy`, `yaml`, `h5py`, `tqdm`, `matplotlib`, `seaborn`, `torchaudio`, `librosa`, `soundfile`, `ctranslate2`, `tokenizers`, `umap-learn`, `hdbscan`, `torchmetrics`, and `joblib`.

    Using Conda (recommended):
    ```bash
    conda env create -f environment.yml
    conda activate vocsim_env
    ```
    Or with virtualenv:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    pip install -r requirements.txt # Note: environment.yml is more comprehensive
    # Manually install less common packages if needed based on your environment.yml
    # e.g., pip install ctranslate2 umap-learn hdbscan torchmetrics joblib
    ```
    *Note*: Ensure your PyTorch installation is compatible with your CUDA version for GPU acceleration.

3.  **Install VocSim**:
    ```bash
    pip install .
    ```

4.  **Download Datasets**:
    Download the datasets to a local directory (e.g., inside your `data_dir`). Update paths in `configs/base.yaml` accordingly.
    - **VocSim Core Dataset**: [Hugging Face](https://huggingface.co/datasets/anonymous-submission000/vocsim)
    - **Application Datasets**:
      - Avian Perception: [Hugging Face](https://huggingface.co/datasets/anonymous-submission000/vocsim-applications-avian-perception)
      - Mouse Strain: [Hugging Face](https://huggingface.co/datasets/anonymous-submission000/vocsim-applications-mouse-strain)
      - Mouse Identity: [Hugging Face](https://huggingface.co/datasets/anonymous-submission000/vocsim-applications-mouse-identity) (Note: The original MUPET data for mouse identity is in `.mat` and nested `.wav` format and requires preprocessing using `utils/dataset_utils.py` or specific configuration within the pipeline. See `reproducibility/configs/mouse_identity_paper.yaml`).

5.  **Configure Paths**:
    - Edit `configs/base.yaml`. This file defines base paths relative to the `project_root`:
      - `project_root`: The root directory of the cloned repository.
      - `data_dir`: Directory where downloaded/processed datasets are stored.
      - `results_dir`: Directory for final benchmark outputs (JSON, CSV, tables).
      - `features_dir`: Directory for caching extracted features and intermediate processing steps (HDF5 files).
      - `models_dir`: Directory for caching trained models (e.g., AE/VAE checkpoints, PCA models).
    - For paper reproduction, the primary config is often `reproducibility/configs/vocsim_paper.yaml` (or avian/mouse variants). These configs *override* values from `base.yaml`. Ensure paths specified here (like `results_dir` or paths within `dataset`) correctly point to your local setup, potentially overriding the base paths.

## Usage

VocSim’s evaluation pipeline is controlled by `vocsim.runner.PipelineRunner`. The pipeline consists of stages: `train`, `features`, `distances`, and `benchmarks`. You can run stages via top-level scripts in `reproducibility/scripts/`.

### Example: Reproduce Paper Benchmarks
To run the full pipeline for the main VocSim paper benchmarks:
```bash
python reproducibility/scripts/vocsim.py --steps all
```
This script loads `reproducibility/configs/vocsim_paper.yaml` (merged with `configs/base.yaml`) and executes all stages (`train` -> `features` -> `distances` -> `benchmarks`) for the datasets and features specified in that config.

For specific applications:
```bash
python reproducibility/scripts/avian_perception.py --steps all
python reproducibility/scripts/mouse_strain.py --steps all
python reproducibility/scripts/mouse_identity.py --steps all # Note: default steps exclude 'distances' here
```

### Command-Line Options
- The primary argument is `--steps`, allowing you to select pipeline stages:
    - `train`: Run configured model training jobs (`TrainerManager`). This stage outputs trained models to `models_dir`.
    - `features`: Extract features using configured extractors, perform intermediate processing (averaging/flattening), and apply PCA (`FeatureManager`). Outputs cached feature files (HDF5) and PCA models (Pickle/Joblib) to `features_dir` or `models_dir`.
    - `distances`: Compute pairwise distance matrices using configured metrics and features (`DistanceManager`). Outputs cached distance matrices (HDF5) to `features_dir`.
    - `benchmarks`: Run configured evaluation benchmarks using cached features and distances (`BenchmarkManager`). Outputs intermediate benchmark item results (JSON) to `features_dir/subset_id/benchmark_cache` and final aggregate results (JSON, CSV) to `results_dir/subset_id/`.
    - `all`: Run all stages in fixed order (`train` -> `features` -> `distances` -> `benchmarks`).
- If a step is skipped, subsequent steps will attempt to load necessary data from cache.

### Outputs
VocSim produces various output files:
-   **Logs**: Text logs of the run are saved to `logging:log_dir` (defaults to `logs/`) with timestamps.
-   **Cached Data**:
    -   **Features**: Extracted features and intermediate results are stored as HDF5 files (`.h5`) in `features_dir/subset_id/`. Filenames include hashes of the configuration to ensure deterministic caching. HDF5 files store features (`/features`) and corresponding original dataset indices (`/original_indices`) as datasets.
    -   **PCA Models**: Trained PCA models are stored as Pickle or Joblib files (`.pkl`/`.joblib`) in `models_dir/pca_models/`.
    -   **Distance Matrices**: Pairwise distance matrices are stored as HDF5 files (`.h5`) in `features_dir/subset_id/`.
    -   **Benchmark Item Cache**: Intermediate results for individual benchmark configurations are stored as JSON files (`.json`) in `features_dir/subset_id/benchmark_cache/`.
-   **Models**: Trained models (e.g., AE/VAE checkpoints) are saved as PyTorch `.pt` files in `models_dir/trainer_name_scope/checkpoints/`. The best/final model is typically saved as `final_model.pt`.
-   **Results**:
    -   Final benchmark results for each subset are aggregated into a JSON file (`bench_results_subset_id_run_id.json`) in `results_dir/subset_id/`.
    -   A summary CSV file (`bench_results_summary_subset_id_run_id.csv`) providing a flattened view of results is also saved in the same directory.
    -   If multiple subsets are processed in one run, a combined JSON/CSV is saved in the main `results_dir/`.
    -   LaTeX table (`.tex`) and plot (`.svg`) files can be generated by utility scripts and saved to configurable output directories (e.g., `reproducibility_outputs/`).

## Extending VocSim

VocSim’s modular design supports adding custom components like feature extractors, distance metrics, benchmarks, trainers, and datasets. The core pipeline uses base classes and relies on configurations to dynamically load and instantiate your custom implementations.

### Add a Feature Extractor
1.  Create a class in the `features/` directory (or a subdirectory like `features/my_family/`) inheriting from `features.base.FeatureExtractor`.
2.  Implement the `_initialize(**kwargs)` method for setup (loading models, etc.).
3.  Implement the `extract(audio_data: Union[np.ndarray, torch.Tensor], sample_rate: int, **kwargs: Any) -> Any` method to perform the extraction.
4.  Add the extractor configuration to your YAML config file under `feature_extractors`:
    ```yaml
    feature_extractors:
      - name: MyExtractorConfigName # A unique name for this config entry
        module: features.my_extractor_module # The Python module path (e.g., features.clap)
        class: MyExtractorClass # The name of your class in the module
        params:
          model_id: "my/model" # Specific parameters for your extractor's _initialize
          another_param: 123
        # Optional: Specify how to handle multi-dim output (default is flatten)
        # averaging: 'mean_row_col' # or 'first_row_col', 'first_row', 'first_col', or null for flatten
        # Optional: Add PCA
        # pca: 64 # Number of PCA components
        # pca_load_chunks: 0 # How many chunks to load for IncPCA fit (0=all for IncPCA, -1=all for std PCA)
        # Optional: Control which distances are computed for this feature
        # compute_distances_for: ['cosine', 'euclidean'] # If null, compute all configured distances
        # Optional: Control if this feature is benchmarked
        # benchmark_this: true # Default is true
        # Optional: Short name for tables
        # short_name: MyExt
    ```
5.  Ensure your extractor's module is discoverable (e.g., has an `__init__.py` in its package and subdirectories).

### Add a Distance Metric
1.  Create a class in the `distances/` directory inheriting from `distances.base.DistanceCalculator`.
2.  Implement the `_initialize(**kwargs)` method for setup.
3.  Implement the `compute_pairwise(features_X: Any, features_Y: Optional[Any] = None) -> Union[torch.Tensor, np.ndarray]` method.
4.  Add the distance configuration to your YAML config file under `distances`:
    ```yaml
    distances:
      - name: my_custom_dist # A unique name for this config entry
        module: distances.my_distance_module
        class: MyDistanceCalculatorClass
        params:
          param1: 0.5
    ```
5.  Ensure your distance module is discoverable.

### Add a Benchmark
1.  Create a class in the `benchmarks/` directory inheriting from `benchmarks.base.Benchmark`.
2.  Implement the `_initialize(**kwargs)` method for setup.
3.  Implement the `evaluate(**kwargs: Any) -> Dict[str, Any]` method. It receives keyword arguments including `feature_hdf5_path`, `distance_matrix`, `distance_matrix_path`, `labels`, `item_id_map`, `dataset`, `feature_config`, `distance_config`, etc.
4.  Add the benchmark configuration to your YAML config file under `benchmarks`:
    ```yaml
    benchmarks:
      - name: MyBenchmark # A unique name for this config entry
        module: benchmarks.my_benchmark_module
        class: MyBenchmarkClass
        params:
          threshold: 0.8
          label_source_key: 'my_specific_label_column' # Optional: Specify which column from item_id_map is used for labels
        # Optional: Specify which features/distances this benchmark applies to
        # target_features: ['MyExtractorConfigName']
        # target_distances: ['my_custom_dist', 'cosine'] # For distance-based benchmarks
    ```
5.  Ensure your benchmark module is discoverable.

### Add a Dataset
1.  Prepare your data in a format compatible with the Hugging Face `datasets` library (e.g., a directory with audio files and a metadata CSV, or a pre-built HF dataset on disk/hub). Ensure it has at least an `audio` column (as an `Audio` feature) and relevant metadata columns (e.g., `label`, `subset`, `speaker`, `original_name`). The `audio` column must be resampleable to the `target_sample_rate` configured in `base.yaml`.
2.  Add the dataset configuration to your YAML config file under the `datasets` section (this section is usually at the top level of your main run config, like `vocsim_paper.yaml`):
    ```yaml
    datasets:
      my_local_dataset_config_name: # A name you use in your run config's `dataset` section
        id: "/path/to/my/local_audio_folder" # Or "org/dataset_on_hub"
        # If using local files that aren't already HF datasets, you might need:
        # type: 'audiofolder' # or other supported HF dataset types
        # If metadata is separate:
        # metadata_file: "/path/to/my/metadata.csv"
        # audio_column: 'filepath_in_csv' # Column in metadata_file pointing to audio
        # label_column: 'class_label_in_csv' # Column in metadata_file for labels
        subset: null # If null, loads entire dataset. If a string, specify the subset name to load initially.
        split: train # Which split to use from the dataset (e.g., 'train', 'validation', 'test')
        # If you want to process specific subsets of this dataset in evaluation stages:
        # subsets_to_run: ['subset_name_A', 'subset_name_B'] # This overrides the top-level 'subset' key for run stages
        # default_label_column: 'my_label_key' # Specify the primary label column for benchmarks if not 'label'
    ```
3.  Ensure the `id` path is correctly resolved by the config loader (relative to `project_root` unless absolute).

## Reproducing the Paper

The `reproducibility/` directory contains specific configurations, scripts, and custom model/trainer implementations used for the paper.

-   **Configs**: `vocsim_paper.yaml`, `avian_paper.yaml`, `mouse_strain_paper.yaml`, `mouse_identity_paper.yaml`. These override `configs/base.yaml`.
-   **Scripts**: `vocsim.py`, `avian_perception.py`, `mouse_strain.py`, `mouse_identity.py`. These are the entry points to run the pipeline for each paper benchmark.
-   **Models/Trainers**: Custom AE/VAE models (`reproducibility/models/`) and their associated trainers (`reproducibility/trainers/`).

**Steps to Reproduce**:
1.  Update paths in `configs/base.yaml` and the relevant `reproducibility/configs/*.yaml` files to match your local setup.
2.  Ensure datasets are downloaded and accessible via the configured `data_dir`.
3.  Run the desired script(s):
    ```bash
    # Run full pipeline for main VocSim benchmarks
    python reproducibility/scripts/vocsim.py --steps all
    # Run full pipeline for Avian Perception benchmarks
    python reproducibility/scripts/avian_perception.py --steps all
    # Run features and benchmarks for Mouse Strain (no distances needed for classification)
    python reproducibility/scripts/mouse_strain.py --steps features benchmarks
    # Run features and benchmarks for Mouse Identity
    python reproducibility/scripts/mouse_identity.py --steps features benchmarks
    ```
    (The default `--steps` in each script matches the required stages for that specific benchmark).

## Utilities
The `reproducibility/scripts/misc/` directory contains various helper scripts:
-   `generate_dataset_characteristics.py`: Computes and saves dataset statistics (samples, classes, duration, SNR) to CSV and LaTeX tables.
-   `table_construction.py`: Parses benchmark results JSON files and generates formatted LaTeX tables for the paper appendix and application-specific results.
-   `generate_all_characteristic_plots.py`: Generates plots visualizing dataset characteristics and their correlation with benchmark performance.
-   `extract_FIRST_sample_features.py`: Extracts and saves the features for the *first* sample in a specified subset for inspection/debugging.
-   `vae_image.py`: Generates images of VAE/AE input spectrograms and their reconstructions from an example audio sample.

## License
VocSim is licensed under the MIT License (see `LICENSE`). Datasets and pretrained models are subject to their respective licenses (see paper Appendix A.1.1).

## Acknowledgements
Thanks to the dataset creators and pretrained model developers whose work enables VocSim. Special thanks to the developers of `pytorch`, `transformers`, `datasets`, `scikit-learn`, `scipy`, `pandas`, `numpy`, `yaml`, `h5py`, `tqdm`, `matplotlib`, `seaborn`, `torchaudio`, `librosa`, `soundfile`, `ctranslate2`, `tokenizers`, `umap-learn`, `hdbscan`, `torchmetrics`, and `joblib` for providing the foundational libraries.

---
*VocSim is under active development. Check [GitHub Issues](https://github.com/anonymoussubmission0000/vocsim/issues) for updates or to report bugs.*
```