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
from typing import Optional, Union

import torch
from torch import Tensor, tensor
from typing_extensions import Literal

from torchmetrics.utilities.checks import _check_same_shape


def is_nonnegative(x: Tensor, atol: float = 1e-5) -> Tensor:
    """Return True if all elements of tensor are nonnegative within certain tolerance.

    Args:
        x: tensor
        atol: absolute tolerance

    Returns:
        Boolean tensor indicating if all values are nonnegative

    """
    return torch.logical_or(x > 0.0, torch.abs(x) < atol).all()


def _validate_average_method_arg(
    average_method: Literal["min", "geometric", "arithmetic", "max"] = "arithmetic",
) -> None:
    if average_method not in ("min", "geometric", "arithmetic", "max"):
        raise ValueError(
            "Expected argument `average_method` to be one of  `min`, `geometric`, `arithmetic`, `max`,"
            f"but got {average_method}"
        )


def calculate_entropy(x: Tensor) -> Tensor:
    """Calculate entropy for a tensor of labels.

    Final calculation of entropy is performed in log form to account for roundoff error.

    Args:
        x: labels

    Returns:
        entropy: entropy of tensor

    Example:
        >>> from torchmetrics.functional.clustering.utils import calculate_entropy
        >>> labels = torch.tensor([1, 3, 2, 2, 1])
        >>> calculate_entropy(labels)
        tensor(1.0549)

    """
    if len(x) == 0:
        return tensor(1.0, device=x.device)

    p = torch.bincount(torch.unique(x, return_inverse=True)[1])
    p = p[p > 0]

    if p.size() == 1:
        return tensor(0.0, device=x.device)

    n = p.sum()
    return -torch.sum((p / n) * (torch.log(p) - torch.log(n)))


def calculate_generalized_mean(x: Tensor, p: Union[int, Literal["min", "geometric", "arithmetic", "max"]]) -> Tensor:
    """Return generalized (power) mean of a tensor.

    Args:
        x: tensor
        p: power

    Returns:
        generalized_mean: generalized mean

    Example (p="min"):
        >>> from torchmetrics.functional.clustering.utils import calculate_generalized_mean
        >>> x = torch.tensor([1, 3, 2, 2, 1])
        >>> calculate_generalized_mean(x, "min")
        tensor(1)

    Example (p="geometric"):
        >>> from torchmetrics.functional.clustering.utils import calculate_generalized_mean
        >>> x = torch.tensor([1, 3, 2, 2, 1])
        >>> calculate_generalized_mean(x, "geometric")
        tensor(1.6438)

    """
    if torch.is_complex(x) or not is_nonnegative(x):
        raise ValueError("`x` must contain positive real numbers")

    if isinstance(p, str):
        if p == "min":
            return x.min()
        if p == "geometric":
            return torch.exp(torch.mean(x.log()))
        if p == "arithmetic":
            return x.mean()
        if p == "max":
            return x.max()

        raise ValueError("'method' must be 'min', 'geometric', 'arirthmetic', or 'max'")

    return torch.mean(torch.pow(x, p)) ** (1.0 / p)


def calculate_contingency_matrix(
    preds: Tensor, target: Tensor, eps: Optional[float] = None, sparse: bool = False
) -> Tensor:
    """Calculate contingency matrix.

    Args:
        preds: predicted labels
        target: ground truth labels
        eps: value added to contingency matrix
        sparse: If True, returns contingency matrix as a sparse matrix. Else, return as dense matrix.
            `eps` must be `None` if `sparse` is `True`.

    Returns:
        contingency: contingency matrix of shape (n_classes_target, n_classes_preds)

    Example:
        >>> import torch
        >>> from torchmetrics.functional.clustering.utils import calculate_contingency_matrix
        >>> preds = torch.tensor([2, 1, 0, 1, 0])
        >>> target = torch.tensor([0, 2, 1, 1, 0])
        >>> calculate_contingency_matrix(preds, target, eps=1e-16)
        tensor([[1.0000e+00, 1.0000e-16, 1.0000e+00],
                [1.0000e+00, 1.0000e+00, 1.0000e-16],
                [1.0000e-16, 1.0000e+00, 1.0000e-16]])

    """
    if eps is not None and sparse is True:
        raise ValueError("Cannot specify `eps` and return sparse tensor.")
    if preds.ndim != 1 or target.ndim != 1:
        raise ValueError(f"Expected 1d `preds` and `target` but got {preds.ndim} and {target.dim}.")

    preds_classes, preds_idx = torch.unique(preds, return_inverse=True)
    target_classes, target_idx = torch.unique(target, return_inverse=True)

    num_classes_preds = preds_classes.size(0)
    num_classes_target = target_classes.size(0)

    contingency = torch.sparse_coo_tensor(
        torch.stack((
            target_idx,
            preds_idx,
        )),
        torch.ones(target_idx.shape[0], dtype=preds_idx.dtype, device=preds_idx.device),
        (
            num_classes_target,
            num_classes_preds,
        ),
    )

    if not sparse:
        contingency = contingency.to_dense()
        if eps:
            contingency = contingency + eps

    return contingency


def _is_real_discrete_label(x: Tensor) -> bool:
    """Check if tensor of labels is real and discrete."""
    if x.ndim != 1:
        raise ValueError(f"Expected arguments to be 1-d tensors but got {x.ndim}-d tensors.")
    return not (torch.is_floating_point(x) or torch.is_complex(x))


def check_cluster_labels(preds: Tensor, target: Tensor) -> None:
    """Check shape of input tensors and if they are real, discrete tensors.

    Args:
        preds: predicted labels
        target: ground truth labels

    """
    _check_same_shape(preds, target)
    if not (_is_real_discrete_label(preds) and _is_real_discrete_label(target)):
        raise ValueError(f"Expected real, discrete values for x but received {preds.dtype} and {target.dtype}.")


def _validate_intrinsic_cluster_data(data: Tensor, labels: Tensor) -> None:
    """Validate that the input data and labels have correct shape and type."""
    if data.ndim != 2:
        raise ValueError(f"Expected 2D data, got {data.ndim}D data instead")
    if not data.is_floating_point():
        raise ValueError(f"Expected floating point data, got {data.dtype} data instead")
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels, got {labels.ndim}D labels instead")


def _validate_intrinsic_labels_to_samples(num_labels: int, num_samples: int) -> None:
    """Validate that the number of labels are in the correct range."""
    if not 1 < num_labels < num_samples:
        raise ValueError(
            "Number of detected clusters must be greater than one and less than the number of samples."
            f"Got {num_labels} clusters and {num_samples} samples."
        )


def calculate_pair_cluster_confusion_matrix(
    preds: Optional[Tensor] = None,
    target: Optional[Tensor] = None,
    contingency: Optional[Tensor] = None,
) -> Tensor:
    """Calculates the pair cluster confusion matrix.

    Can either be calculated from predicted cluster labels and target cluster labels or from a pre-computed
    contingency matrix. The pair cluster confusion matrix is a 2x2 matrix where that defines the similarity between
    two clustering by considering all pairs of samples and counting pairs that are assigned into same or different
    clusters in the predicted and target clusterings.

    Note that the matrix is not symmetric.

    Inspired by:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cluster.pair_confusion_matrix.html

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels
        contingency: contingency matrix

    Returns:
        A 2x2 tensor containing the pair cluster confusion matrix.

    Raises:
        ValueError:
            If neither `preds` and `target` nor `contingency` are provided.
        ValueError:
            If both `preds` and `target` and `contingency` are provided.

    Example:
        >>> import torch
        >>> from torchmetrics.functional.clustering.utils import calculate_pair_cluster_confusion_matrix
        >>> preds = torch.tensor([0, 0, 1, 1])
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> calculate_pair_cluster_confusion_matrix(preds, target)
        tensor([[8, 0],
                [0, 4]])
        >>> preds = torch.tensor([0, 0, 1, 2])
        >>> target = torch.tensor([0, 0, 1, 1])
        >>> calculate_pair_cluster_confusion_matrix(preds, target)
        tensor([[8, 2],
                [0, 2]])

    """
    if preds is None and target is None and contingency is None:
        raise ValueError("Must provide either `preds` and `target` or `contingency`.")
    if preds is not None and target is not None and contingency is not None:
        raise ValueError("Must provide either `preds` and `target` or `contingency`, not both.")

    if preds is not None and target is not None:
        contingency = calculate_contingency_matrix(preds, target)

    if contingency is None:
        raise ValueError("Must provide `contingency` if `preds` and `target` are not provided.")

    num_samples = contingency.sum()
    sum_c = contingency.sum(dim=1)
    sum_k = contingency.sum(dim=0)
    sum_squared = (contingency**2).sum()

    pair_matrix = torch.zeros(2, 2, dtype=contingency.dtype, device=contingency.device)
    pair_matrix[1, 1] = sum_squared - num_samples
    pair_matrix[1, 0] = (contingency * sum_k).sum() - sum_squared
    pair_matrix[0, 1] = (contingency.T * sum_c).sum() - sum_squared
    pair_matrix[0, 0] = num_samples**2 - pair_matrix[0, 1] - pair_matrix[1, 0] - sum_squared
    return pair_matrix
