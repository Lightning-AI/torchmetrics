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
from functools import partial
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pytest
import torch
from numpy import array
from torch import Tensor, tensor
from typing_extensions import Literal

from unittests.helpers import seed_all
from unittests.helpers.testers import Metric, MetricTester
from unittests.retrieval.inputs import _input_retrieval_scores as _irs
from unittests.retrieval.inputs import _input_retrieval_scores_all_target as _irs_all
from unittests.retrieval.inputs import _input_retrieval_scores_empty as _irs_empty
from unittests.retrieval.inputs import _input_retrieval_scores_extra as _irs_extra
from unittests.retrieval.inputs import _input_retrieval_scores_float_target as _irs_float_tgt
from unittests.retrieval.inputs import _input_retrieval_scores_for_adaptive_k as _irs_adpt_k
from unittests.retrieval.inputs import _input_retrieval_scores_int_target as _irs_int_tgt
from unittests.retrieval.inputs import _input_retrieval_scores_mismatching_sizes as _irs_bad_sz
from unittests.retrieval.inputs import _input_retrieval_scores_mismatching_sizes_func as _irs_bad_sz_fn
from unittests.retrieval.inputs import _input_retrieval_scores_no_target as _irs_no_tgt
from unittests.retrieval.inputs import _input_retrieval_scores_with_ignore_index as _irs_ii
from unittests.retrieval.inputs import _input_retrieval_scores_wrong_targets as _irs_bad_tgt

seed_all(42)

# a version of get_group_indexes that depends on NumPy is here to avoid this dependency for the full library


def _retrieval_aggregate(
    values: Tensor,
    aggregation: Union[Literal["mean", "median", "min", "max"], Callable] = "mean",
    dim: Optional[int] = None,
) -> Tensor:
    """Aggregate the final retrieval values into a single value."""
    if aggregation == "mean":
        return values.mean() if dim is None else values.mean(dim=dim)
    if aggregation == "median":
        return values.median() if dim is None else values.median(dim=dim).values
    if aggregation == "min":
        return values.min() if dim is None else values.min(dim=dim).values
    if aggregation == "max":
        return values.max() if dim is None else values.max(dim=dim).values
    return aggregation(values, dim=dim)


def get_group_indexes(indexes: Union[Tensor, np.ndarray]) -> List[Union[Tensor, np.ndarray]]:
    """Extract group indexes.

    Given an integer :class:`~torch.Tensor` or `np.ndarray` `indexes`, return a :class:`~torch.Tensor` or
    `np.ndarray` of indexes for each different value in `indexes`.

    Args:
        indexes: a :class:`~torch.Tensor` or `np.ndarray` of integers

    Return:
        A list of integer :class:`~torch.Tensor`s or `np.ndarray`s

    Example:
        >>> indexes = torch.tensor([0, 0, 0, 1, 1, 1, 1])
        >>> get_group_indexes(indexes)
        [tensor([0, 1, 2]), tensor([3, 4, 5, 6])]

    """
    structure, dtype = (tensor, torch.long) if isinstance(indexes, Tensor) else (np.array, np.int64)

    res = {}
    for i, _id in enumerate(indexes):
        _id = _id.item()
        if _id in res:
            res[_id] += [i]
        else:
            res[_id] = [i]

    return [structure(x, dtype=dtype) for x in res.values()]


def _custom_aggregate_fn(val: Tensor, dim=None) -> Tensor:
    return (val**2).mean() if dim is None else (val**2).mean(dim=dim)


def _compute_sklearn_metric(
    preds: Union[Tensor, array],
    target: Union[Tensor, array],
    indexes: Optional[np.ndarray] = None,
    metric: Optional[Callable] = None,
    empty_target_action: str = "skip",
    ignore_index: Optional[int] = None,
    reverse: bool = False,
    aggregation: Union[Literal["mean", "median", "min", "max"], Callable] = "mean",
    **kwargs: Any,
) -> Tensor:
    """Compute metric with multiple iterations over every query predictions set."""
    if indexes is None:
        indexes = np.full_like(preds, fill_value=0, dtype=np.int64)
    if isinstance(indexes, Tensor):
        indexes = indexes.cpu().numpy()
    if isinstance(preds, Tensor):
        preds = preds.cpu().numpy()
    if isinstance(target, Tensor):
        target = target.cpu().numpy()

    assert isinstance(indexes, np.ndarray)
    assert isinstance(preds, np.ndarray)
    assert isinstance(target, np.ndarray)

    if ignore_index is not None:
        valid_positions = target != ignore_index
        indexes, preds, target = indexes[valid_positions], preds[valid_positions], target[valid_positions]

    indexes = indexes.flatten()
    preds = preds.flatten()
    target = target.flatten()
    groups = get_group_indexes(indexes)

    sk_results = []
    for group in groups:
        trg, pds = target[group], preds[group]

        if ((1 - trg) if reverse else trg).sum() == 0:
            if empty_target_action == "skip":
                pass
            elif empty_target_action == "pos":
                sk_results.append(1.0)
            else:
                sk_results.append(0.0)
        else:
            res = metric(trg, pds, **kwargs)
            sk_results.append(res)

    sk_results = np.array(sk_results)
    sk_results[np.isnan(sk_results)] = 0.0  # this is needed with old versions of sklearn

    if len(sk_results) > 0:
        return _retrieval_aggregate(torch.from_numpy(sk_results), aggregation=aggregation).numpy()
    return np.array(0.0)


def _concat_tests(*tests: Tuple[Dict]) -> Dict:
    """Concat tests composed by a string and a list of arguments."""
    assert len(tests), "`_concat_tests` expects at least an argument"
    assert all(tests[0]["argnames"] == x["argnames"] for x in tests[1:]), "the header must be the same for all tests"
    return {"argnames": tests[0]["argnames"], "argvalues": list(chain.from_iterable(x["argvalues"] for x in tests))}


_errors_test_functional_metric_parameters_default = {
    "argnames": "preds,target,message,metric_args",
    "argvalues": [
        # check input shapes are consistent (func)
        (_irs_bad_sz_fn.preds, _irs_bad_sz_fn.target, "`preds` and `target` must be of the same shape", {}),
        # check input tensors are not empty
        (_irs_empty.preds, _irs_empty.target, "`preds` and `target` must be non-empty and non-scalar tensors", {}),
        # check on input dtypes
        (_irs.preds.bool(), _irs.target, "`preds` must be a tensor of floats", {}),
        # check targets are between 0 and 1
        (_irs_bad_tgt.preds, _irs_bad_tgt.target, "`target` must contain `binary` values", {}),
    ],
}

_errors_test_functional_metric_parameters_with_nonbinary = {
    "argnames": "preds,target,message,metric_args",
    "argvalues": [
        # check input shapes are consistent (func)
        (_irs_bad_sz_fn.preds, _irs_bad_sz_fn.target, "`preds` and `target` must be of the same shape", {}),
        # check input tensors are not empty
        (_irs_empty.preds, _irs_empty.target, "`preds` and `target` must be non-empty and non-scalar tensors", {}),
        # check on input dtypes
        (_irs.preds.bool(), _irs.target, "`preds` must be a tensor of floats", {}),
    ],
}

_errors_test_functional_metric_parameters_k = {
    "argnames": "preds,target,message,metric_args",
    "argvalues": [
        (_irs.preds, _irs.target, "`top_k` has to be a positive integer or None", {"top_k": -10}),
        (_irs.preds, _irs.target, "`top_k` has to be a positive integer or None", {"top_k": 4.0}),
    ],
}

_errors_test_functional_metric_parameters_adaptive_k = {
    "argnames": "preds,target,message,metric_args",
    "argvalues": [
        (_irs.preds, _irs.target, "`adaptive_k` has to be a boolean", {"adaptive_k": 10}),
        (_irs.preds, _irs.target, "`adaptive_k` has to be a boolean", {"adaptive_k": None}),
    ],
}

_errors_test_class_metric_parameters_no_pos_target = {
    "argnames": "indexes,preds,target,message,metric_args",
    "argvalues": [
        # check when error when there are no positive targets
        (
            _irs_no_tgt.indexes,
            _irs_no_tgt.preds,
            _irs_no_tgt.target,
            "`compute` method was provided with a query with no positive target.",
            {"empty_target_action": "error"},
        ),
    ],
}

_errors_test_class_metric_parameters_no_neg_target = {
    "argnames": "indexes,preds,target,message,metric_args",
    "argvalues": [
        # check when error when there are no negative targets
        (
            _irs_all.indexes,
            _irs_all.preds,
            _irs_all.target,
            "`compute` method was provided with a query with no negative target.",
            {"empty_target_action": "error"},
        ),
    ],
}

_errors_test_class_metric_parameters_with_nonbinary = {
    "argnames": "indexes,preds,target,message,metric_args",
    "argvalues": [
        (None, _irs.preds, _irs.target, "`indexes` cannot be None", {"empty_target_action": "error"}),
        # check when input arguments are invalid
        (
            _irs.indexes,
            _irs.preds,
            _irs.target,
            "`empty_target_action` received a wrong value `casual_argument`.",
            {"empty_target_action": "casual_argument"},
        ),
        # check ignore_index is valid
        (
            _irs.indexes,
            _irs.preds,
            _irs.target,
            "Argument `ignore_index` must be an integer or None.",
            {"ignore_index": -100.0},
        ),
        # check input shapes are consistent
        (
            _irs_bad_sz.indexes,
            _irs_bad_sz.preds,
            _irs_bad_sz.target,
            "`indexes`, `preds` and `target` must be of the same shape",
            {"empty_target_action": "skip"},
        ),
        # check input tensors are not empty
        (
            _irs_empty.indexes,
            _irs_empty.preds,
            _irs_empty.target,
            "`indexes`, `preds` and `target` must be non-empty and non-scalar tensors",
            {"empty_target_action": "skip"},
        ),
        # check on input dtypes
        (
            _irs.indexes.bool(),
            _irs.preds,
            _irs.target,
            "`indexes` must be a tensor of long integers",
            {"empty_target_action": "skip"},
        ),
        (
            _irs.indexes,
            _irs.preds.bool(),
            _irs.target,
            "`preds` must be a tensor of floats",
            {"empty_target_action": "skip"},
        ),
    ],
}

_errors_test_class_metric_parameters_default = {
    "argnames": "indexes,preds,target,message,metric_args",
    "argvalues": [
        (None, _irs.preds, _irs.target, "`indexes` cannot be None", {"empty_target_action": "error"}),
        # check when input arguments are invalid
        (
            _irs.indexes,
            _irs.preds,
            _irs.target,
            "`empty_target_action` received a wrong value `casual_argument`.",
            {"empty_target_action": "casual_argument"},
        ),
        # check ignore_index is valid
        (
            _irs.indexes,
            _irs.preds,
            _irs.target,
            "Argument `ignore_index` must be an integer or None.",
            {"ignore_index": -100.0},
        ),
        # check input shapes are consistent
        (
            _irs_bad_sz.indexes,
            _irs_bad_sz.preds,
            _irs_bad_sz.target,
            "`indexes`, `preds` and `target` must be of the same shape",
            {"empty_target_action": "skip"},
        ),
        # check input tensors are not empty
        (
            _irs_empty.indexes,
            _irs_empty.preds,
            _irs_empty.target,
            "`indexes`, `preds` and `target` must be non-empty and non-scalar tensors",
            {"empty_target_action": "skip"},
        ),
        # check on input dtypes
        (
            _irs.indexes.bool(),
            _irs.preds,
            _irs.target,
            "`indexes` must be a tensor of long integers",
            {"empty_target_action": "skip"},
        ),
        (
            _irs.indexes,
            _irs.preds.bool(),
            _irs.target,
            "`preds` must be a tensor of floats",
            {"empty_target_action": "skip"},
        ),
    ],
}

_errors_test_class_metric_parameters_k = {
    "argnames": "indexes,preds,target,message,metric_args",
    "argvalues": [
        (_irs.index, _irs.preds, _irs.target, "`top_k` has to be a positive integer or None", {"top_k": -10}),
        (_irs.index, _irs.preds, _irs.target, "`top_k` has to be a positive integer or None", {"top_k": 4.0}),
    ],
}

_errors_test_class_metric_parameters_adaptive_k = {
    "argnames": "indexes,preds,target,message,metric_args",
    "argvalues": [
        (_irs.index, _irs.preds, _irs.target, "`adaptive_k` has to be a boolean", {"adaptive_k": 10}),
        (_irs.index, _irs.preds, _irs.target, "`adaptive_k` has to be a boolean", {"adaptive_k": None}),
    ],
}

_default_metric_class_input_arguments = {
    "argnames": "indexes,preds,target",
    "argvalues": [
        (_irs.indexes, _irs.preds, _irs.target),
        (_irs_extra.indexes, _irs_extra.preds, _irs_extra.target),
        (_irs_no_tgt.indexes, _irs_no_tgt.preds, _irs_no_tgt.target),
        (_irs_adpt_k.indexes, _irs_adpt_k.preds, _irs_adpt_k.target),
    ],
}

_default_metric_class_input_arguments_ignore_index = {
    "argnames": "indexes,preds,target",
    "argvalues": [
        (_irs_ii.indexes, _irs_ii.preds, _irs_ii.target),
    ],
}

_default_metric_class_input_arguments_with_non_binary_target = {
    "argnames": "indexes,preds,target",
    "argvalues": [
        (_irs.indexes, _irs.preds, _irs.target),
        (_irs_extra.indexes, _irs_extra.preds, _irs_extra.target),
        (_irs_no_tgt.indexes, _irs_no_tgt.preds, _irs_no_tgt.target),
        (_irs_int_tgt.indexes, _irs_int_tgt.preds, _irs_int_tgt.target),
        (_irs_float_tgt.indexes, _irs_float_tgt.preds, _irs_float_tgt.target),
    ],
}

_default_metric_functional_input_arguments = {
    "argnames": "preds,target",
    "argvalues": [
        (_irs.preds, _irs.target),
        (_irs_extra.preds, _irs_extra.target),
        (_irs_no_tgt.preds, _irs_no_tgt.target),
    ],
}

_default_metric_functional_input_arguments_with_non_binary_target = {
    "argnames": "preds,target",
    "argvalues": [
        (_irs.preds, _irs.target),
        (_irs_extra.preds, _irs_extra.target),
        (_irs_no_tgt.preds, _irs_no_tgt.target),
        (_irs_int_tgt.preds, _irs_int_tgt.target),
        (_irs_float_tgt.preds, _irs_float_tgt.target),
    ],
}


def _errors_test_class_metric(
    indexes: Tensor,
    preds: Tensor,
    target: Tensor,
    metric_class: Metric,
    message: str = "",
    metric_args: Optional[dict] = None,
    exception_type: Type[Exception] = ValueError,
    kwargs_update: Optional[dict] = None,
):
    """Check types, parameters and errors.

    Args:
        indexes: torch tensor with indexes
        preds: torch tensor with predictions
        target: torch tensor with targets
        metric_class: metric class that should be tested
        message: message that exception should return
        metric_args: arguments for class initialization
        exception_type: callable function that is used for comparison
        kwargs_update: Additional keyword arguments that will be passed with indexes, preds and
            target when running update on the metric.

    """
    metric_args = metric_args or {}
    kwargs_update = kwargs_update or {}
    with pytest.raises(exception_type, match=message):  # noqa: PT012
        metric = metric_class(**metric_args)
        metric(preds, target, indexes=indexes, **kwargs_update)


def _errors_test_functional_metric(
    preds: Tensor,
    target: Tensor,
    metric_functional: Metric,
    message: str = "",
    exception_type: Type[Exception] = ValueError,
    kwargs_update: Optional[dict] = None,
):
    """Check types, parameters and errors.

    Args:
        preds: torch tensor with predictions
        target: torch tensor with targets
        metric_functional: functional metric that should be tested
        message: message that exception should return
        exception_type: callable function that is used for comparison
        kwargs_update: Additional keyword arguments that will be passed with indexes, preds and
            target when running update on the metric.

    """
    kwargs_update = kwargs_update or {}
    with pytest.raises(exception_type, match=message):
        metric_functional(preds, target, **kwargs_update)


class RetrievalMetricTester(MetricTester):
    """General tester class for retrieval metrics."""

    atol: float = 1e-6

    def run_class_metric_test(
        self,
        ddp: bool,
        indexes: Tensor,
        preds: Tensor,
        target: Tensor,
        metric_class: Metric,
        reference_metric: Callable,
        metric_args: dict,
        reverse: bool = False,
    ):
        """Test class implementation of metric."""
        _ref_metric_adapted = partial(_compute_sklearn_metric, metric=reference_metric, reverse=reverse, **metric_args)

        super().run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=metric_class,
            reference_metric=_ref_metric_adapted,
            metric_args=metric_args,
            fragment_kwargs=True,
            indexes=indexes,  # every additional argument will be passed to metric_class and _ref_metric_adapted
        )

    def run_functional_metric_test(
        self,
        preds: Tensor,
        target: Tensor,
        metric_functional: Callable,
        reference_metric: Callable,
        metric_args: dict,
        reverse: bool = False,
        **kwargs: Any,
    ):
        """Test functional implementation of metric."""
        _ref_metric_adapted = partial(_compute_sklearn_metric, metric=reference_metric, reverse=reverse, **metric_args)

        super().run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=metric_functional,
            reference_metric=_ref_metric_adapted,
            metric_args=metric_args,
            fragment_kwargs=True,
            **kwargs,
        )

    def run_precision_test_cpu(
        self,
        indexes: Tensor,
        preds: Tensor,
        target: Tensor,
        metric_module: Metric,
        metric_functional: Callable,
    ):
        """Test dtype support of the metric on CPU."""

        def metric_functional_ignore_indexes(preds, target, indexes, empty_target_action):
            return metric_functional(preds, target)

        super().run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=metric_module,
            metric_functional=metric_functional_ignore_indexes,
            metric_args={"empty_target_action": "neg"},
            indexes=indexes,  # every additional argument will be passed to the retrieval metric and _ref_metric_adapted
        )

    def run_precision_test_gpu(
        self,
        indexes: Tensor,
        preds: Tensor,
        target: Tensor,
        metric_module: Metric,
        metric_functional: Callable,
    ):
        """Test dtype support of the metric on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("Test requires GPU")

        def metric_functional_ignore_indexes(preds, target, indexes, empty_target_action):
            return metric_functional(preds, target)

        super().run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=metric_module,
            metric_functional=metric_functional_ignore_indexes,
            metric_args={"empty_target_action": "neg"},
            indexes=indexes,  # every additional argument will be passed to retrieval metric and _ref_metric_adapted
        )

    @staticmethod
    def run_metric_class_arguments_test(
        indexes: Tensor,
        preds: Tensor,
        target: Tensor,
        metric_class: Metric,
        message: str = "",
        metric_args: Optional[dict] = None,
        exception_type: Type[Exception] = ValueError,
        kwargs_update: Optional[dict] = None,
    ) -> None:
        """Test that specific errors are raised for incorrect input."""
        _errors_test_class_metric(
            indexes=indexes,
            preds=preds,
            target=target,
            metric_class=metric_class,
            message=message,
            metric_args=metric_args,
            exception_type=exception_type,
            **kwargs_update,
        )

    @staticmethod
    def run_functional_metric_arguments_test(
        preds: Tensor,
        target: Tensor,
        metric_functional: Callable,
        message: str = "",
        exception_type: Type[Exception] = ValueError,
        kwargs_update: Optional[dict] = None,
    ) -> None:
        """Test that specific errors are raised for incorrect input."""
        _errors_test_functional_metric(
            preds=preds,
            target=target,
            metric_functional=metric_functional,
            message=message,
            exception_type=exception_type,
            kwargs_update=kwargs_update,
        )
