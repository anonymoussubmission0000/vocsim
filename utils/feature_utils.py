import logging
from typing import Union, Optional, Dict, List, Tuple, Any, Iterator
import torch
import numpy as np
from sklearn.decomposition import IncrementalPCA
from tqdm.auto import tqdm
from itertools import chain, tee
import gc

logger = logging.getLogger(__name__)

def apply_averaging(features_np: np.ndarray, method: Optional[str]) -> Optional[np.ndarray]:
    """
    Applies a specified averaging method to a NumPy feature array.

    Args:
        features_np (np.ndarray): Input feature array. Can be multi-dimensional.
        method (Optional[str]): The averaging method to apply. Supported methods:
                                'mean_row_col', 'first_row_col', 'first_row', 'first_col',
                                'mean_time_dim'. If None or unsupported, input is flattened.

    Returns:
        Optional[np.ndarray]: The averaged feature array (always 1D) or None on error.
    """
    original_shape = features_np.shape
    ndim = features_np.ndim
    func_name = "apply_averaging"
    
    allowed_methods = {"mean_row_col", "first_row_col", "first_row", "first_col", "mean_time_dim"}

    if method is None or method not in allowed_methods:
        if method is not None:
            logger.warning("[%s] Invalid method '%s'. Allowed: %s. Flattening input.", func_name, method, allowed_methods)
        else:
            logger.debug("[%s] No averaging method. Flattening input shape %s.", func_name, original_shape)
        if ndim == 0:
            return features_np.reshape(-1).astype(np.float32)
        return features_np.flatten().astype(np.float32)

    logger.debug("[%s] Applying method '%s' to input shape %s", func_name, method, original_shape)

    if method != 'mean_time_dim' and ndim != 2:
        logger.warning(
            "[%s] Method '%s' expects 2D input, got %dD shape %s. Reshaping to 2D.",
            func_name, method, ndim, original_shape
        )
        if any(s == 0 for s in original_shape):
            return np.empty((0,), dtype=np.float32)
        
        first_dim_size = np.prod(original_shape[:-1])
        last_dim_size = original_shape[-1]
        features_np = features_np.reshape(first_dim_size, last_dim_size)
        logger.debug("[%s] Reshaped input to %s", func_name, features_np.shape)

    result_1d = None
    try:
        if method == "mean_time_dim":
            if ndim < 1:
                 raise ValueError("Cannot apply mean_time_dim to a scalar.")
            result_1d = np.mean(features_np, axis=-1, dtype=np.float32)
        elif method == "mean_row_col":
            row_means = np.mean(features_np, axis=1, dtype=np.float32)
            col_means = np.mean(features_np, axis=0, dtype=np.float32)
            result_1d = np.concatenate([row_means, col_means])
        elif method == "first_row_col":
            first_row = features_np[0, :].astype(np.float32)
            first_col = features_np[:, 0].astype(np.float32)
            result_1d = np.concatenate([first_row, first_col])
        elif method == "first_row":
            result_1d = features_np[0, :].astype(np.float32)
        elif method == "first_col":
            result_1d = features_np[:, 0].astype(np.float32)

    except IndexError as e:
        logger.error("[%s] IndexError applying '%s' to shape %s: %s. Flattening.", func_name, method, features_np.shape, e)
        return features_np.flatten().astype(np.float32)
    except Exception as e:
        logger.error("[%s] Error applying '%s' to shape %s: %s. Flattening.", func_name, method, features_np.shape, e, exc_info=True)
        return features_np.flatten().astype(np.float32)

    if result_1d is None:
        logger.error("[%s] Result is None after applying method '%s'. Flattening.", func_name, method)
        return features_np.flatten().astype(np.float32)

    final_result = result_1d.flatten().astype(np.float32)
    logger.debug("[%s] Method '%s' successful. Input %s -> Output %s", func_name, method, original_shape, final_result.shape)
    return final_result


def apply_pca(
    features_iterator: Iterator[np.ndarray],
    n_components: int,
    pca_model: Optional[IncrementalPCA] = None,
    batch_size_hint: int = 1024,
) -> Tuple[Optional[List[np.ndarray]], Optional[IncrementalPCA]]:
    """
    Applies IncrementalPCA to features yielded by an iterator.

    Args:
        features_iterator: An iterator yielding NumPy arrays of shape [batch_size, feature_dim].
                           The iterator MUST yield 2D arrays.
        n_components: Number of PCA components. Must be positive.
        pca_model: A pre-fitted IncrementalPCA model to use for transformation, or None to fit a new one.
        batch_size_hint: Hint for IncrementalPCA's internal processing batch size during fitting.

    Returns:
        Tuple containing:
            - Optional[List[np.ndarray]]: List of transformed feature batches (if transforming).
                                         None if fitting or error.
            - Optional[IncrementalPCA]: The fitted or passed PCA model. None if fitting failed.
    """
    func_name = "apply_pca"
    if n_components <= 0:
        raise ValueError("n_components must be positive.")

    try:
        iter_peek, features_iterator_orig = tee(features_iterator)
        first_batch = next(iter_peek)
        features_iterator = chain([first_batch], iter_peek)
        if not isinstance(first_batch, np.ndarray) or first_batch.ndim != 2:
            logger.error(
                "[%s] Iterator must yield 2D NumPy arrays. Got %s with shape %s. Aborting PCA.",
                func_name,
                type(first_batch),
                getattr(first_batch, "shape", "N/A"),
            )
            return None, pca_model
        input_dim = first_batch.shape[1]
        actual_batch_size = first_batch.shape[0]
        logger.debug("[%s] Peeked first batch shape: %s. Input Dim: %d", func_name, first_batch.shape, input_dim)
    except StopIteration:
        logger.warning("[%s] Input iterator is empty.", func_name)
        return [], pca_model
    except Exception as peek_err:
        logger.warning("[%s] Failed to peek at iterator: %s. Cannot proceed without dimensionality.", func_name, peek_err)
        return None, pca_model

    if pca_model is None:
        logger.info("[%s] Fitting IncrementalPCA (n=%d, hint=%d)...", func_name, n_components, batch_size_hint)
        if n_components > input_dim:
            logger.warning("Requested n_components (%d) > feature dimension (%d). Using n_components=%d.", n_components, input_dim, input_dim)
            n_components = input_dim

        pca_model_inc = IncrementalPCA(n_components=n_components, batch_size=batch_size_hint)
        samples_processed = 0
        batch_num_fit = 0
        try:
            pbar_fit = tqdm(features_iterator, desc="Fitting IncrementalPCA", leave=False, unit="batch")
            for batch_2d_np in pbar_fit:
                batch_num_fit += 1
                if not isinstance(batch_2d_np, np.ndarray) or batch_2d_np.ndim != 2:
                    logger.warning(
                        "[%s] FIT: Skipping invalid batch %d (Expected 2D np.ndarray, got %s shape %s).",
                        func_name,
                        batch_num_fit,
                        type(batch_2d_np),
                        getattr(batch_2d_np, "shape", "N/A"),
                    )
                    continue
                batch_samples = batch_2d_np.shape[0]
                if batch_samples == 0:
                    continue

                has_nan = np.isnan(batch_2d_np).any()
                has_inf = np.isinf(batch_2d_np).any()
                if has_nan or has_inf:
                    logger.error("[%s] FIT: NaN/Inf in batch %d. Skipping!", func_name, batch_num_fit)
                    continue
                if batch_2d_np.shape[1] != input_dim:
                    logger.error("[%s] FIT: Inconsistent feature dim in batch %d (%d != %d). Skipping!", func_name, batch_num_fit, batch_2d_np.shape[1], input_dim)
                    continue
                if batch_2d_np.dtype != np.float32:
                    batch_2d_np = batch_2d_np.astype(np.float32)

                pca_model_inc.partial_fit(batch_2d_np)
                samples_processed += batch_samples
                fitted_components = getattr(pca_model_inc, "n_components_", 0)
                pbar_fit.set_postfix({"fitted_comp": fitted_components, "samples": samples_processed})

            if samples_processed == 0:
                logger.error("[%s] IncrementalPCA: No valid data processed during fit.", func_name)
                return None, None
            final_fitted_components = getattr(pca_model_inc, "n_components_", 0)
            if final_fitted_components == 0:
                logger.error("[%s] IncrementalPCA failed to fit any components.", func_name)
                return None, None
            var_sum = np.sum(pca_model_inc.explained_variance_ratio_) if hasattr(pca_model_inc, "explained_variance_ratio_") and pca_model_inc.explained_variance_ratio_ is not None else np.nan
            if final_fitted_components < n_components:
                logger.warning("[%s] PCA fitted %d/%d components. Var: %.4f", func_name, final_fitted_components, n_components, var_sum)
            else:
                logger.info("[%s] PCA fitted %d components from %d samples. Var: %.4f", func_name, final_fitted_components, samples_processed, var_sum)
            return None, pca_model_inc

        except ValueError as ve:
            logger.error("[%s] IncrementalPCA fitting failed (ValueError): %s", func_name, ve, exc_info=True)
            return None, None
        except Exception as e:
            logger.error("[%s] IncrementalPCA fitting failed unexpectedly: %s", func_name, e, exc_info=True)
            return None, None

    else:
        logger.info("[%s] Applying pre-fitted IncrementalPCA model batch-by-batch...", func_name)
        if not isinstance(pca_model, IncrementalPCA):
            logger.error("[%s] Expected IncrementalPCA model!", func_name)
            return None, pca_model

        transformed_data_batches: List[np.ndarray] = []
        n_features_expected = getattr(pca_model, "n_features_in_", -1)
        fitted_comps = getattr(pca_model, "n_components_", n_components)
        pca_out_dim = min(n_components, fitted_comps)

        if n_features_expected != -1 and n_features_expected != input_dim:
            logger.error("[%s] Input data dim (%d) does not match pre-fitted PCA model expected dim (%d). Aborting transform.", func_name, input_dim, n_features_expected)
            return None, pca_model

        if n_components > fitted_comps:
            logger.warning("Requested %d PCA components, but model only has %d. Outputting %d.", n_components, fitted_comps, pca_out_dim)
        elif pca_out_dim < n_components:
            logger.warning("[%s] Limiting transform output to %d components (Min of requested %d and fitted %d).", func_name, pca_out_dim, n_components, fitted_comps)

        samples_processed = 0
        batch_num_transform = 0
        try:
            pbar_transform = tqdm(features_iterator, desc="Transforming (IncrementalPCA)", leave=False, unit="batch")
            for batch_2d_np in pbar_transform:
                batch_num_transform += 1
                if not isinstance(batch_2d_np, np.ndarray) or batch_2d_np.ndim != 2:
                    logger.warning(
                        "[%s] Transform: Skipping invalid batch %d (Expected 2D np.ndarray, got %s shape %s).",
                        func_name,
                        batch_num_transform,
                        type(batch_2d_np),
                        getattr(batch_2d_np, "shape", "N/A"),
                    )
                    continue
                batch_samples = batch_2d_np.shape[0]
                if batch_samples == 0:
                    continue

                has_nan = np.isnan(batch_2d_np).any()
                has_inf = np.isinf(batch_2d_np).any()
                if has_nan or has_inf:
                    logger.error("[%s] Transform: NaN/Inf in batch %d. Skipping!", func_name, batch_num_transform)
                    continue
                if batch_2d_np.shape[1] != input_dim:
                    logger.error("[%s] Transform: Inconsistent feature dim in batch %d (%d != %d). Skipping!", func_name, batch_num_transform, batch_2d_np.shape[1], input_dim)
                    continue
                if batch_2d_np.dtype != np.float32:
                    batch_2d_np = batch_2d_np.astype(np.float32)

                transformed_batch = pca_model.transform(batch_2d_np)
                if transformed_batch.shape[1] != pca_out_dim:
                    logger.warning("PCA transform output dim %d != expected %d. Adjusting.", transformed_batch.shape[1], pca_out_dim)
                    transformed_batch = transformed_batch[:, :pca_out_dim]

                transformed_data_batches.append(transformed_batch.astype(np.float32))
                samples_processed += batch_samples
                pbar_transform.set_postfix({"samples": samples_processed})

            logger.info("[%s] PCA transform complete (%d batches, %d valid samples). Output dim: %d", func_name, len(transformed_data_batches), samples_processed, pca_out_dim)
            return transformed_data_batches, pca_model

        except ValueError as ve:
            logger.error("[%s] IncrementalPCA transform failed (ValueError): %s", func_name, ve, exc_info=True)
            return None, pca_model
        except Exception as e:
            logger.error("[%s] IncrementalPCA transform failed unexpectedly: %s", func_name, e, exc_info=True)
            return None, pca_model