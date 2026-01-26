# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Callable, List, Optional, Tuple, Union, cast

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

# TorchMetrics text helpers (same style as BERTScore)
from torchmetrics.functional.text.helper_embedding_metric import (
    TextDataset,
    TokenizedDataset,
    _check_shape_of_model_output,
    _get_progress_bar,
    _input_data_collator,
    _output_data_collator,
)
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import (
    _GEOMLOSS_AVAILABLE,
    _POT_AVAILABLE,
    _SKLEARN_AVAILABLE,
    _TQDM_AVAILABLE,
    _TRANSFORMERS_GREATER_EQUAL_4_4,
)

log = logging.getLogger(__name__)


@contextmanager
def _ignore_transformers_finetune_warning() -> Iterator[None]:
    """Temporarily silence common transformers loading warnings."""
    logger = logging.getLogger("transformers.modeling_utils")
    original_level = logger.getEffectiveLevel()
    try:
        logger.setLevel(logging.ERROR)
        yield
    finally:
        logger.setLevel(original_level)


# Default model recommended in the original implementation.
_DEFAULT_MODEL = "bert-base-uncased"

if _TRANSFORMERS_GREATER_EQUAL_4_4:
    from transformers import AutoModel, AutoTokenizer

    def _download_model_for_depth_score() -> None:
        """Download intensive operations."""
        with _ignore_transformers_finetune_warning():
            AutoTokenizer.from_pretrained(_DEFAULT_MODEL)
            AutoModel.from_pretrained(_DEFAULT_MODEL)

    if _SKIP_SLOW_DOCTEST and not _try_proceed_with_timeout(_download_model_for_depth_score):
        __doctest_skip__ = ["depth_score"]
else:
    __doctest_skip__ = ["depth_score"]


def _preprocess_multiple_references(
    preds: List[str], target: List[Union[str, Sequence[str]]]
) -> Tuple[List[str], List[str], Optional[List[Tuple[int, int]]]]:
    """Preprocess predictions and targets when dealing with multiple references.

    This function handles the case where a single prediction might have multiple
    reference targets (represented as a list/tuple of strings). It flattens the
    multi-reference structure into aligned (pred, ref) pairs and returns group
    boundaries so the final distance can later be reduced per original prediction.

    Args:
        preds: A list of predictions.
        target: A list of targets, where each item could be a string or a list/tuple of strings.

    Returns:
        Tuple: (preds, target, ref_group_boundaries)
            - preds: Flattened list of `str` where each prediction is repeated once per reference.
            - target: Flattened list of `str` containing all references.
            - ref_group_boundaries: List of tuples (start, end) indicating the boundaries of each
              original prediction's reference group in the flattened lists, or `None` if no
              multi-reference structure is present.

    Raises:
        ValueError:
            If `preds` is not a list of strings.

    """
    if not all(isinstance(item, str) for item in preds):
        raise ValueError("Invalid input provided.")

    has_nested = any(isinstance(item, (list, tuple)) for item in target)
    if not has_nested:
        return preds, cast(List[str], target), None

    ref_group_boundaries: List[Tuple[int, int]] = []
    new_preds: List[str] = []
    new_target: List[str] = []
    count = 0

    for pred, ref_group in zip(preds, target):
        if isinstance(ref_group, (list, tuple)):
            new_preds.extend([pred] * len(ref_group))
            new_target.extend(cast(List[str], ref_group))
            ref_group_boundaries.append((count, count + len(ref_group)))
            count += len(ref_group)
        else:
            new_preds.append(pred)
            new_target.append(cast(str, ref_group))
            ref_group_boundaries.append((count, count + 1))
            count += 1

    return new_preds, new_target, ref_group_boundaries


def _postprocess_multiple_references_distance(
    distances: Tensor,
    ref_group_boundaries: List[Tuple[int, int]],
    reduction: str = "min",
) -> Tensor:
    """Postprocess distances when dealing with multiple references.

    After `_preprocess_multiple_references` flattens multi-reference inputs, this function
    reduces the computed per-(pred, ref) distances back to a single distance per original
    prediction by aggregating within each reference group.

    Since DepthScore is a distance (lower is better), the default behavior uses `min`
    (best matching reference). Other reductions can be used for diagnostics.

    Args:
        distances: A 1D tensor of distances aligned with the flattened (pred, ref) pairs.
        ref_group_boundaries: List of tuples (start, end) indicating the boundaries of each
            original prediction's reference group in `distances`.
        reduction: Reduction to apply within each group. One of `{"min", "max", "mean"}`.
            - `"min"`: best reference match (default for distance metrics)
            - `"max"`: worst reference match
            - `"mean"`: average across references

    Returns:
        A 1D tensor of shape `(num_predictions,)` containing the reduced distance per prediction.

    Raises:
        ValueError:
            If `distances` is not 1D.
        ValueError:
            If `reduction` is not one of `{"min","max","mean"}`.

    """
    if distances.dim() != 1:
        raise ValueError("Expected 1D tensor of distances.")
    if reduction not in {"min", "max", "mean"}:
        raise ValueError("reduction must be one of {'min','max','mean'}.")

    out: List[Tensor] = []
    for start, end in ref_group_boundaries:
        chunk = distances[start:end]
        if reduction == "min":
            out.append(chunk.min())
        elif reduction == "max":
            out.append(chunk.max())
        else:
            out.append(chunk.mean())
    return torch.stack(out, dim=0)


def cov_matrix(x: np.ndarray, robust: bool = False) -> np.ndarray:
    """Covariance matrix (optionally robust)."""
    if robust:
        if not _SKLEARN_AVAILABLE:
            raise ModuleNotFoundError(
                "Robust covariance requires that `scikit-learn` is installed. "
                "Use `pip install scikit-learn` or `pip install torchmetrics[text]`."
            )
        from sklearn.covariance import MinCovDet as MCD  # noqa: N817

        return MCD().fit(x).covariance_
    return np.cov(x.T)


def standardize(x: np.ndarray, robust: bool = False) -> np.ndarray:
    """Affine standardization using inverse sqrt covariance."""
    sigma = cov_matrix(x, robust)
    _, n_features = x.shape
    rank = np.linalg.matrix_rank(x)

    if rank < n_features:
        if not _SKLEARN_AVAILABLE:
            raise ModuleNotFoundError(
                "Affine-invariant DepthScore requires that `scikit-learn` is installed. "
                "Use `pip install scikit-learn` or `pip install torchmetrics[text]`."
            )
        from sklearn.decomposition import PCA

        x = PCA(rank).fit_transform(x)
        sigma = cov_matrix(x)

    u, s, _ = np.linalg.svd(sigma)
    square_inv = u / np.sqrt(s)
    return x @ square_inv


def sampled_sphere(n_dirs: int, d: int) -> np.ndarray:
    """Uniform samples on unit sphere."""
    u = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=n_dirs)
    # The reference implementation uses `sklearn.preprocessing.normalize`. Here, that is mocked
    # so default irw metric runs without any additional dependencies being installed.
    return _normalize_l2_rows_exact(u)


def _normalize_l2_rows_exact(x: np.ndarray) -> np.ndarray:
    norms = np.sqrt(np.einsum("ij,ij->i", x, x))
    norms[norms == 0.0] = 1.0
    return x / norms[:, None]


def wasserstein(x: np.ndarray, y: np.ndarray) -> float:
    """Optimal transport cost with uniform weights."""
    if not _POT_AVAILABLE:
        raise ModuleNotFoundError(
            "The `wasserstein` backend requires that `POT` is installed. "
            "Use `pip install POT` or `pip install torchmetrics[text]`."
        )
    import ot  # pip install POT  # codespell:ignore ot

    m = ot.dist(x, y)  # codespell:ignore ot
    w_x = np.ones(len(x)) / len(x)
    w_y = np.ones(len(y)) / len(y)
    return float(ot.emd2(w_x, w_y, m))  # codespell:ignore ot


def sw(x: np.ndarray, y: np.ndarray, ndirs: int, p: int = 2) -> float:
    """Sliced Wasserstein distance."""
    if not _POT_AVAILABLE:
        raise ModuleNotFoundError(
            "The `sliced` backend requires that `POT` is installed. "
            "Use `pip install POT` or `pip install torchmetrics[text]`."
        )
    import ot  # pip install POT  # codespell:ignore ot

    n, d = x.shape
    u = sampled_sphere(ndirs, d)
    z_x = x @ u.T
    z_y = y @ u.T
    sliced = np.zeros(ndirs)
    for k in range(ndirs):
        sliced[k] = ot.emd2_1d(z_x[:, k], z_y[:, k], p=2)  # codespell:ignore ot
    return float((np.mean(sliced)) ** (1 / p))


def mmd(x: np.ndarray, y: np.ndarray) -> float:
    """Gaussian MMD via geomloss."""
    if not _GEOMLOSS_AVAILABLE:
        raise ModuleNotFoundError(
            "The `mmd` backend requires that `geomloss` is installed. "
            "Use `pip install geomloss` or `pip install torchmetrics[text]`."
        )
    import geomloss

    return float(geomloss.SamplesLoss("gaussian")(torch.tensor(x), torch.tensor(y)).item())


def ai_irw(
    x: np.ndarray, ai: bool = True, robust: bool = False, n_dirs: Optional[int] = None, random_state: int = 0
) -> np.ndarray:
    """(Affine-invariant) integrated rank-weighted depth."""
    np.random.seed(random_state)
    if ai:
        x = standardize(x, robust)

    n, d = x.shape
    n_dirs = d * 100 if n_dirs is None else n_dirs

    u = sampled_sphere(n_dirs, d)
    proj = x @ u.T
    ranks = np.argsort(proj, axis=0)

    depth = np.zeros_like(proj)
    for k in range(n_dirs):
        depth[ranks[:, k], k] = np.arange(1, n + 1)

    depth = depth / n
    depth = np.minimum(depth, 1 - depth)
    return np.mean(depth, axis=1)


def dr_distance(
    x: np.ndarray,
    y: np.ndarray,
    n_alpha: int = 5,
    n_dirs: int = 10000,
    data_depth: str = "irw",
    eps_min: float = 0.3,
    eps_max: float = 1.0,
    p: int = 5,
    random_state: int = 0,
) -> float:
    """Compute the depth-based pseudo-metric between two point clouds.

    This function implements the DepthScore "DR distance" between two empirical
    distributions represented by token-embedding point clouds `x` and `y`. The distance
    is computed by (1) choosing a data depth / distributional discrepancy backend
    (e.g., IRW depth, affine-invariant IRW, Wasserstein, sliced Wasserstein, or MMD),
    and (2) integrating over depth level sets between `eps_min` and `eps_max`, while
    approximating the supremum over directions on the unit sphere by Monte Carlo.

    Args:
        x: Array of shape `(n_samples, n_features)` representing the first point cloud.
        y: Array of shape `(n_samples, n_features)` representing the second point cloud.
        n_alpha: Monte-Carlo parameter controlling the approximation of the integral
            over alpha (number of level-set thresholds between `eps_min` and `eps_max`).
        n_dirs: Number of random directions used to approximate the supremum over the
            unit sphere (and for depth estimation when applicable).
        data_depth: Depth / discrepancy measure to use. One of
            `{"irw", "ai_irw", "wasserstein", "sliced", "mmd"}`.
            - `"irw"` / `"ai_irw"` compute depth values and then integrate level sets.
            - `"wasserstein"` returns the (unsliced) OT cost directly.  # codespell:ignore ot
            - `"sliced"` returns the sliced Wasserstein distance directly.
            - `"mmd"` returns the Gaussian MMD directly.
        eps_min: Lower level-set bound in `[0, eps_max]` (lowest alpha / quantile level).
        eps_max: Upper level-set bound in `[eps_min, 1]` (highest alpha / quantile level).
        p: Power used in the ground cost aggregation (corresponds to the exponent in the
            reference implementation).
        random_state: Random seed controlling direction sampling and any stochastic steps.

    Returns:
        The computed pseudo-metric score as a Python `float`.

    Raises:
        ValueError:
            If `data_depth` is unsupported.
        ValueError:
            If `eps_min` and `eps_max` do not satisfy `0 <= eps_min <= eps_max <= 1`.

    """
    np.random.seed(random_state)

    # Match reference numerics: many reference code paths end up in float64.
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if data_depth == "irw":
        depth_x = ai_irw(x, ai=False, n_dirs=n_dirs, random_state=random_state)
        depth_y = ai_irw(y, ai=False, n_dirs=n_dirs, random_state=random_state)
    elif data_depth == "ai_irw":
        depth_x = ai_irw(x, ai=True, n_dirs=n_dirs, random_state=random_state)
        depth_y = ai_irw(y, ai=True, n_dirs=n_dirs, random_state=random_state)
    elif data_depth == "wasserstein":
        return wasserstein(x, y)
    elif data_depth == "sliced":
        return sw(x, y, ndirs=n_dirs)
    elif data_depth == "mmd":
        return mmd(x, y)
    else:
        raise ValueError("Unsupported depth")

    if not (0.0 <= eps_min <= eps_max <= 1.0):
        raise ValueError("Expected 0 <= eps_min <= eps_max <= 1")

    _, d = x.shape
    u = sampled_sphere(n_dirs, d)
    proj_x = x @ u.T
    proj_y = y @ u.T

    alphas = np.linspace(int(eps_min * 100), int(eps_max * 100), n_alpha)
    q_x = [np.percentile(depth_x, a) for a in alphas]
    q_y = [np.percentile(depth_y, a) for a in alphas]

    score = 0.0
    for i in range(n_alpha):
        idx_x = np.where(depth_x >= q_x[i])[0]
        idx_y = np.where(depth_y >= q_y[i])[0]
        supp_x = np.max(proj_x[idx_x], axis=0)
        supp_y = np.max(proj_y[idx_y], axis=0)
        score += float(np.max((supp_x - supp_y) ** p))

    return float((score / n_alpha) ** (1 / p))


def _get_embeddings_and_mask(
    dataloader: DataLoader,
    target_len: int,
    model: Module,
    device: Optional[Union[str, torch.device]] = None,
    num_layers: Optional[int] = None,
    all_layers: bool = False,
    verbose: bool = False,
    user_forward_fn: Optional[Callable[[Module, dict[str, Tensor]], Tensor]] = None,
) -> Tuple[Tensor, Tensor]:
    """Compute normalized token embeddings and the corresponding attention mask.

    Args:
        dataloader: Dataloader over `TextDataset` or `TokenizedDataset`.
        target_len: Length of the longest sequence in the dataset (used for output collation/padding).
        model: Transformer model used for embedding extraction.
        device: Device to run inference on.
        num_layers: Which hidden layer to use from `output_hidden_states`.
            If `None`, the last layer is used.
        all_layers: Whether to use representations from all layers.
            If `True`, `num_layers` is ignored.
        verbose: Whether to show a progress bar during embedding extraction.
        user_forward_fn:
            Optional user-defined forward function. If provided, it must:
            - accept `(model, batch_dict)` where `batch_dict` contains `"input_ids"` and `"attention_mask"`
            - return a tensor shaped like `(batch, seq_len, hidden_dim)`.

    Returns:
        A tuple `(embeddings, attention_mask)` where:
            - embeddings: Tensor shaped `(batch, 1, seq_len, hidden_dim)` when `all_layers=False`,
              or `(batch, num_layers, seq_len, hidden_dim)` when `all_layers=True`.
              Embeddings are L2-normalized over the hidden dimension and masked by `attention_mask`.
            - attention_mask: Tensor shaped `(batch, seq_len)` aligned with `embeddings`.

    Raises:
        ValueError:
            If `user_forward_fn` output shape does not match the expected model output shape.
        ValueError:
            If `all_layers=True` is used with a custom `user_forward_fn`.

    """
    embeddings_list: List[Tensor] = []
    mask_list: List[Tensor] = []

    for batch in _get_progress_bar(dataloader, verbose):
        with torch.no_grad():
            batch = _input_data_collator(batch, device)

            if not all_layers:
                if user_forward_fn is None:
                    out = model(batch["input_ids"], batch["attention_mask"], output_hidden_states=True)
                    hs = out.hidden_states[num_layers if num_layers is not None else -1]
                else:
                    hs = user_forward_fn(model, batch)
                    _check_shape_of_model_output(hs, batch["input_ids"])
                # unify to (b, 1, s, d) like BERTScore's internal shape
                hs = hs.unsqueeze(1)
            else:
                if user_forward_fn is not None:
                    raise ValueError(
                        "The option `all_layers=True` can be used only with default `transformers` models."
                    )
                out = model(batch["input_ids"], batch["attention_mask"], output_hidden_states=True)
                hs = torch.cat([o.unsqueeze(1) for o in out.hidden_states], dim=1)

        # normalize embeddings (safe)
        denom = hs.norm(dim=-1).unsqueeze(-1).clamp_min(1e-12)
        hs = hs / denom

        hs, attention_mask = _output_data_collator(hs, batch["attention_mask"], target_len)

        # mask out padding/special tokens
        hs = torch.einsum("blsd, bs -> blsd", hs, attention_mask)

        embeddings_list.append(hs.cpu())
        mask_list.append(attention_mask.cpu())

    return torch.cat(embeddings_list, dim=0), torch.cat(mask_list, dim=0)


def depth_score(
    preds: Union[str, Sequence[str], dict[str, Tensor]],
    target: Union[str, Sequence[str], Sequence[Sequence[str]], dict[str, Tensor]],
    model_name_or_path: Optional[str] = None,
    num_layers: Optional[int] = None,
    all_layers: bool = False,
    model: Optional[Module] = None,
    user_tokenizer: Any = None,
    user_forward_fn: Optional[Callable[[Module, dict[str, Tensor]], Tensor]] = None,
    verbose: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    max_length: int = 512,
    batch_size: int = 64,
    num_threads: int = 0,
    truncation: bool = False,
    # DepthScore-specific knobs
    n_alpha: int = 5,
    n_dirs: int = 10000,
    eps: float = 0.3,
    p: int = 5,
    depth_measure: str = "irw",
    # Multi-ref postprocess for a distance metric (best = min by default)
    multi_ref_reduction: str = "min",
) -> Tensor:
    """`DepthScore Evaluating Text Generation`_ for text similarity matching.

    DepthScore measures the distance between two sentences by comparing the distributions
    of their contextual token embeddings using a depth-based pseudo-metric. Lower values
    indicate that the predicted sentence is closer to the reference sentence.

    This implementation follows the original implementation from `DEPTH_score`_.

    Args:
        preds: Predicted sentence(s) as `str`, `Sequence[str]`, or tokenized dict
            containing `"input_ids"` and `"attention_mask"`.
        target: Reference sentence(s) as `str`, `Sequence[str]`, multi-reference
            `Sequence[Sequence[str]]`, or tokenized dict containing `"input_ids"` and `"attention_mask"`.
        model_name_or_path: Hugging Face model name/path used when `model` is not provided.
        num_layers: Hidden layer index to use for contextual embeddings. If `None`, the last layer is used.
        all_layers:
            An indication of whether the representation from all model's layers should be used.
            If ``all_layers=True``, the argument ``num_layers`` is ignored.
        model: Optional user-provided model. If provided, `user_tokenizer` must also be provided.
        user_tokenizer: Tokenizer to use with a user-provided model. Ignored when `model` is `None`.
        user_forward_fn:
            Optional user-defined forward function producing embeddings from `(model, batch_dict)`.
        verbose: Whether to show a progress bar during embedding extraction.
        device: Device to run embedding extraction on.
        max_length: Maximum input sequence length. Longer sequences are trimmed if `truncation=True`.
        batch_size: Batch size used for model processing.
        num_threads: Number of dataloader workers.
        truncation: Whether to truncate input sequences to `max_length`.
        n_alpha: Number of alpha levels used by the depth-based distance computation.
        n_dirs: Number of random projection directions used by depth/sliced computations.
        eps: Lower quantile bound (eps_min) used in the depth distance integration (upper bound fixed at 1.0).
        p: Power used in the distance aggregation.
        depth_measure: Depth/distance backend to use. One of:
            `"irw"`, `"ai_irw"`, `"wasserstein"`, `"sliced"`, `"mmd"`.
        multi_ref_reduction: Reduction to apply across multiple references per prediction.
            Default `"min"` (best match) since this is a distance metric.

    Returns:
        A 1D tensor of distances of shape `(num_predictions,)`. For multi-reference input,
        the output is reduced per original prediction according to `multi_ref_reduction`.

    Raises:
        ValueError:
            If `len(preds) != len(target)`.
        ModuleNotFoundError:
            If `verbose=True` but `tqdm` is not installed.
        ModuleNotFoundError:
            If default transformers model is required but `transformers` is not installed.
        ValueError:
            If invalid input is provided for `preds`/`target`.
        ValueError:
            If `num_layers` is larger than the number of model layers (when detectable).

    Example:
        >>> from torchmetrics.functional.text.depth_score import depth_score
        >>> preds = ["hello there", "general kenobi"]
        >>> target = ["hello there", "master kenobi"]
        >>> depth_score(preds, target, model_name_or_path="distilbert-base-uncased", num_layers=4, device="cpu")
        tensor([...])

    Example:
        >>> from torchmetrics.functional.text.depth_score import depth_score
        >>> preds = ["hello there", "general kenobi"]
        >>> target = [["hello there", "master kenobi"], ["hello there", "master kenobi"]]
        >>> depth_score(preds, target, model_name_or_path="distilbert-base-uncased", num_layers=4, device="cpu")
        tensor([...])

    """
    ref_group_boundaries: Optional[List[Tuple[int, int]]] = None

    if isinstance(preds, str):
        preds = [preds]
    if isinstance(target, str):
        target = [target]
    if not isinstance(preds, (list, dict)):
        preds = list(preds)
    if not isinstance(target, (list, dict)):
        target = list(target)

    if len(preds) != len(target):
        raise ValueError(
            "Expected number of predicted and reference sentences to be the same, but got"
            f" {len(preds)} and {len(target)}"
        )

    if isinstance(preds, list) and len(preds) > 0 and isinstance(target, list) and len(target) > 0:
        preds, target, ref_group_boundaries = _preprocess_multiple_references(preds, target)

    if verbose and (not _TQDM_AVAILABLE):
        raise ModuleNotFoundError(
            "An argument `verbose = True` requires `tqdm` package be installed. Install with `pip install tqdm`."
        )

    if model is None:
        if not _TRANSFORMERS_GREATER_EQUAL_4_4:
            raise ModuleNotFoundError(
                "`depth_score` metric with default models requires `transformers` package be installed."
                " Either install with `pip install transformers>=4.4` or `pip install torchmetrics[text]`."
            )
        if model_name_or_path is None:
            rank_zero_warn(
                "The argument `model_name_or_path` was not specified while it is required when default"
                " `transformers` model are used."
                f" It is, therefore, used the default recommended model - {_DEFAULT_MODEL}."
            )
        from transformers import AutoModel, AutoTokenizer

        with _ignore_transformers_finetune_warning():
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path or _DEFAULT_MODEL)
            model = AutoModel.from_pretrained(model_name_or_path or _DEFAULT_MODEL)
    else:
        if user_tokenizer is None:
            raise ValueError("When `model` is provided, `user_tokenizer` must also be provided.")
        tokenizer = user_tokenizer

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    try:
        if hasattr(model.config, "num_hidden_layers") and isinstance(model.config.num_hidden_layers, int):
            if num_layers and num_layers > model.config.num_hidden_layers:
                raise ValueError(
                    f"num_layers={num_layers} is forbidden for {model_name_or_path}."
                    f" Please use num_layers <= {model.config.num_hidden_layers}"
                )
        else:
            rank_zero_warn(
                "Model config does not have `num_hidden_layers` as an integer attribute. "
                "Unable to validate `num_layers`."
            )
    except AttributeError:
        rank_zero_warn("It was not possible to retrieve the parameter `num_layers` from the model specification.")

    _are_empty_lists = all(isinstance(text, list) and len(text) == 0 for text in (preds, target))
    _are_valid_lists = all(
        isinstance(text, list) and len(text) > 0 and isinstance(text[0], str) for text in (preds, target)
    )
    _are_valid_tensors = all(
        isinstance(text, dict) and isinstance(text["input_ids"], Tensor) for text in (preds, target)
    )

    if _are_empty_lists:
        rank_zero_warn("Predictions and references are empty.")
        return torch.zeros(1, dtype=torch.float32)

    if _are_valid_lists:
        target_dataset = TextDataset(target, tokenizer, max_length, truncation=truncation)
        preds_dataset = TextDataset(preds, tokenizer, max_length, truncation=truncation)

    elif _are_valid_tensors:
        target_dataset = TokenizedDataset(**cast(dict, target))
        preds_dataset = TokenizedDataset(**cast(dict, preds))
    else:
        raise ValueError("Invalid input provided.")

    target_loader = DataLoader(target_dataset, batch_size=batch_size, num_workers=num_threads)
    preds_loader = DataLoader(preds_dataset, batch_size=batch_size, num_workers=num_threads)

    target_embeddings, target_mask = _get_embeddings_and_mask(
        target_loader,
        target_dataset.max_length,
        model,
        device=device,
        num_layers=num_layers,
        all_layers=all_layers,
        verbose=verbose,
        user_forward_fn=user_forward_fn,
    )
    preds_embeddings, preds_mask = _get_embeddings_and_mask(
        preds_loader,
        preds_dataset.max_length,
        model,
        device=device,
        num_layers=num_layers,
        all_layers=all_layers,
        verbose=verbose,
        user_forward_fn=user_forward_fn,
    )

    # Reorder back (TextDataset sorts by length internally)
    target_embeddings = target_embeddings[target_loader.dataset.sorting_indices]
    preds_embeddings = preds_embeddings[preds_loader.dataset.sorting_indices]
    target_mask = target_mask[target_loader.dataset.sorting_indices]
    preds_mask = preds_mask[preds_loader.dataset.sorting_indices]

    # Pairwise (same index) distances
    distances: List[float] = []
    n = preds_embeddings.shape[0]

    for i in range(n):
        pm = preds_mask[i].bool()
        tm = target_mask[i].bool()

        x = preds_embeddings[i, 0, pm, :].numpy()
        y = target_embeddings[i, 0, tm, :].numpy()

        if x.shape[0] == 0 or y.shape[0] == 0:
            distances.append(float("inf"))
            continue

        distances.append(
            dr_distance(
                x,
                y,
                n_alpha=n_alpha,
                n_dirs=n_dirs,
                data_depth=depth_measure,
                eps_min=eps,
                eps_max=1.0,
                p=p,
                random_state=0,
            )
        )

    out = torch.tensor(distances, dtype=torch.float32)

    # Multi-reference reduction (distance metric: default "min" = best ref)
    if ref_group_boundaries is not None:
        out = _postprocess_multiple_references_distance(out, ref_group_boundaries, reduction=multi_ref_reduction)

    return out
