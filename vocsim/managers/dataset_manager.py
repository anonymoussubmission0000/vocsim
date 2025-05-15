"""
Manages dataset loading, filtering, and metadata mapping for audio datasets.
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Audio, DatasetDict, load_dataset, load_from_disk

logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Handles dataset loading, subset filtering, and metadata map construction.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DatasetManager with configuration settings.

        Args:
            config: Configuration dictionary containing dataset parameters.
        """
        self.cfg = config
        self.target_sr = config.get("target_sample_rate", 16000)
        self.full_dataset_obj: Optional[Any] = None
        self.base_dataset_cache_id: Optional[str] = None
        self.current_subset_key: Optional[str] = None
        self.current_dataset_cache_id: Optional[str] = None
        self.item_id_to_metadata: Dict[str, Dict] = {}
        self.current_subset_dataset: Optional[Any] = None
        self.dataset_config: Dict[str, Any] = config.get("dataset", {})

    def get_labels_for_current_dataset(self, label_source_key: Optional[str] = None) -> Optional[List[Any]]:
        """
        Extracts labels from the current_subset_dataset based on the label_source_key.

        Args:
            label_source_key (Optional[str]): The key in the dataset items to use as the label source.

        Returns:
            Optional[List[Any]]: A list of labels corresponding to the order of items
                                 in the current_subset_dataset, or None if no dataset is loaded
                                 or extraction fails.
        """
        if self.current_subset_dataset is None:
            logger.error("No current_subset_dataset selected. Call get_subset_dataset() first.")
            return None

        if label_source_key:
            actual_label_key_to_use = label_source_key
            logger.info("Using explicitly provided label_source_key: '%s' for current subset '%s'.", actual_label_key_to_use, self.current_subset_key)
        elif self.dataset_config.get("default_label_column"):
            actual_label_key_to_use = self.dataset_config.get("default_label_column")
            logger.info("Using default_label_column from dataset config: '%s' for current subset '%s'.", actual_label_key_to_use, self.current_subset_key)
        else:
            actual_label_key_to_use = "label"
            logger.info("No specific label key provided, defaulting to '%s' for current subset '%s'.", actual_label_key_to_use, self.current_subset_key)

        labels_list = []
        missing_key_count = 0
        none_value_count = 0

        try:
            for i, item in enumerate(self.current_subset_dataset):
                label_value = item.get(actual_label_key_to_use)
                if label_value is not None:
                    labels_list.append(str(label_value))
                else:
                    if actual_label_key_to_use not in item:
                        missing_key_count += 1
                    else:
                        none_value_count += 1
                    labels_list.append(None)

            if missing_key_count > 0:
                logger.warning(
                    "Label key '%s' was missing in %d items out of %d for subset '%s'." " These will be treated as None/unlabeled.",
                    actual_label_key_to_use,
                    missing_key_count,
                    len(self.current_subset_dataset),
                    self.current_subset_key,
                )
            if none_value_count > 0:
                logger.warning(
                    "Label key '%s' had a None value in %d items for subset '%s'." " These are treated as unlabeled.",
                    actual_label_key_to_use,
                    none_value_count,
                    len(self.current_subset_dataset),
                    self.current_subset_key,
                )

            if not labels_list and len(self.current_subset_dataset) > 0:
                logger.warning(
                    "Extracted an empty list of labels, but current_subset_dataset has %d items. This is unexpected if items were expected to be labeled with key '%s'.",
                    len(self.current_subset_dataset),
                    actual_label_key_to_use,
                )
            elif not labels_list:
                logger.info("Extracted an empty list of labels for an empty current_subset_dataset for key '%s'.", actual_label_key_to_use)
            else:
                logger.info("Successfully extracted %d labels using key '%s'. First few: %s", len(labels_list), actual_label_key_to_use, labels_list[:5] if len(labels_list) > 5 else labels_list)

            return labels_list

        except Exception as e:
            logger.error("Error extracting labels with key '%s' for subset '%s': %s", actual_label_key_to_use, self.current_subset_key, e, exc_info=True)
            return None

    def load_full_dataset(self) -> bool:
        """
        Loads the full dataset as specified in the configuration.

        Returns:
            True if the dataset is loaded successfully, False otherwise.
        """
        dataset_config = self.cfg.get("dataset")
        if not dataset_config:
            logger.error("Dataset configuration is missing.")
            return False

        dataset_id = dataset_config.get("id")
        split = dataset_config.get("split", "train") or "all"
        if not dataset_id:
            logger.error("Dataset configuration requires an 'id' field.")
            return False

        safe_dataset_id = Path(dataset_id).name if Path(dataset_id).is_dir() else dataset_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        self.base_dataset_cache_id = f"{safe_dataset_id}_all_{split}"

        logger.info("Loading dataset '%s' (Split: %s, Cache ID: %s)", dataset_id, split, self.base_dataset_cache_id)
        try:
            load_path = Path(dataset_id)
            if load_path.is_dir():
                dataset = load_from_disk(str(load_path))
                if isinstance(dataset, DatasetDict):
                    if split not in dataset:
                        raise ValueError(f"Split '{split}' not found in DatasetDict at {load_path}")
                    dataset = dataset[split]
            else:
                dataset = load_dataset(dataset_id, name=None, split=split, trust_remote_code=True)

            if "audio" in dataset.column_names:
                if not isinstance(dataset.features["audio"], Audio):
                    dataset = dataset.cast_column("audio", Audio())
                current_sr = dataset.features["audio"].sampling_rate
                if current_sr != self.target_sr:
                    logger.info("Resampling audio to %d Hz", self.target_sr)
                    dataset = dataset.cast_column("audio", Audio(sampling_rate=self.target_sr))
                else:
                    logger.info("Audio sampling rate matches target: %d Hz", self.target_sr)
            else:
                logger.warning("Dataset lacks an 'audio' column.")

            self.full_dataset_obj = dataset
            logger.info("Dataset loaded successfully.")

            if dataset_config.get("subsets_to_run") and "subset" not in dataset.column_names:
                logger.error("Subsets requested but 'subset' column is missing.")
                self.full_dataset_obj = None
                return False
            return True

        except Exception as e:
            logger.error("Failed to load dataset '%s': %s", dataset_id, e, exc_info=True)
            self.full_dataset_obj = None
            return False

    def get_subset_dataset(self, subset_key: str) -> Optional[Tuple[Any, str]]:
        """
        Filters the full dataset to return a subset and its cache ID.

        Args:
            subset_key: Identifier for the subset to filter (e.g., 'all' for full dataset).

        Returns:
            A tuple of (subset dataset object, cache ID), or None if filtering fails.
        """
        if self.full_dataset_obj is None:
            logger.error("Full dataset not loaded.")
            return None

        self.current_subset_key = subset_key
        dataset_config = self.cfg.get("dataset", {})
        split = dataset_config.get("split", "train") or "all"
        base_id_part = self.base_dataset_cache_id.split("_all_")[0]

        if subset_key == "all":
            subset_dataset = self.full_dataset_obj
            self.current_dataset_cache_id = self.base_dataset_cache_id
            logger.info("Using full dataset for subset 'all'.")
        else:
            logger.info("Filtering dataset for subset '%s'", subset_key)
            try:
                if "subset" not in self.full_dataset_obj.column_names:
                    logger.error("Dataset lacks 'subset' column for filtering.")
                    return None
                subset_dataset = self.full_dataset_obj.filter(lambda x: x.get("subset") == subset_key, load_from_cache_file=False)
                self.current_dataset_cache_id = f"{base_id_part}_{subset_key}_{split}"
                num_samples = len(subset_dataset) if hasattr(subset_dataset, "__len__") else "iterable"
                if isinstance(num_samples, int) and num_samples == 0:
                    logger.warning("Subset '%s' contains 0 samples.", subset_key)
                    return None
                logger.info("Filtered subset '%s' with %s samples.", subset_key, num_samples)
            except Exception as e:
                logger.error("Failed to filter subset '%s': %s", subset_key, e)
                return None

        self.current_subset_dataset = subset_dataset
        self._build_item_id_map(subset_dataset)
        return subset_dataset, self.current_dataset_cache_id

    def _build_item_id_map(self, dataset: Any) -> None:
        """
        Constructs a metadata map for items in the dataset.

        Args:
            dataset: Dataset object to process.
        """
        self.item_id_to_metadata = {}
        logger.info("Building item ID map...")

        id_fields = ["original_name", "unique_id", "id", "filename", "path"]
        id_field = next(
            (field for field in id_fields if field in dataset.column_names or (field == "path" and "audio" in dataset.column_names)),
            None,
        )
        if id_field == "path" and "audio" in dataset.column_names:
            id_field = "audio.path"

        if id_field:
            logger.info("Using '%s' for item IDs.", id_field)
            try:
                ids = []
                if "." in id_field:
                    base, nested = id_field.split(".", 1)
                    ids = [str(item.get(base, {}).get(nested, i)) for i, item in enumerate(dataset)]
                else:
                    ids = [str(item.get(id_field, i)) for i, item in enumerate(dataset)]

                id_counts = Counter(ids)
                duplicates = {k: v for k, v in id_counts.items() if v > 1}
                if duplicates:
                    logger.warning("Found %d duplicate IDs. Appending suffixes.", len(duplicates))
                    final_ids = []
                    seen_counts = Counter()
                    for id_val in ids:
                        if id_counts[id_val] > 1:
                            final_ids.append(f"{id_val}_dup{seen_counts[id_val]}")
                            seen_counts[id_val] += 1
                        else:
                            final_ids.append(id_val)
                    ids = final_ids
                else:
                    logger.info("All item IDs are unique.")

                for i, (item_id, item) in enumerate(zip(ids, dataset)):
                    metadata = {
                        k: item.get(k) for k in ["label", "speaker", "subset", "original_name"] if item.get(k) is not None
                    }
                    metadata["index"] = i
                    self.item_id_to_metadata[item_id] = metadata

            except Exception as e:
                logger.error("Failed to build ID map using '%s': %s", id_field, e, exc_info=True)
                id_field = None

        if not id_field:
            logger.warning("No suitable ID field found. Using indices as IDs.")
            self.item_id_to_metadata = {
                str(i): {
                    "index": i,
                    "label": item.get("label"),
                    "speaker": item.get("speaker"),
                    "subset": item.get("subset"),
                    "original_name": item.get("original_name"),
                }
                for i, item in enumerate(dataset)
            }

        logger.info("Built item ID map for %d items.", len(self.item_id_to_metadata))

    def get_current_item_map(self) -> Dict[str, Dict]:
        """
        Retrieves the current item ID to metadata mapping.

        Returns:
            A dictionary mapping item IDs to their metadata.
        """
        return self.item_id_to_metadata