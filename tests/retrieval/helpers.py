# Copyright The PyTorch Lightning team.
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
from typing import Callable, Tuple, Union

import numpy as np
import pytest
import torch
from numpy import array
from torch import Tensor

from tests.helpers import seed_all
from tests.helpers.testers import Metric, MetricTester
from tests.retrieval.inputs import (
    _input_retrieval_scores,
    _input_retrieval_scores_all_target,
    _input_retrieval_scores_empty,
    _input_retrieval_scores_extra,
    _input_retrieval_scores_mismatching_sizes,
    _input_retrieval_scores_mismatching_sizes_func,
    _input_retrieval_scores_no_target,
    _input_retrieval_scores_wrong_targets,
)
from torchmetrics.utilities.data import get_group_indexes

seed_all(42)


def _compute_sklearn_metric(
    preds: Union[Tensor, array],
    target: Union[Tensor, array],
    indexes: np.ndarray = None,
    metric: Callable = None,
    empty_target_action: str = "skip",
    reverse: bool = False,
    **kwargs
) -> Tensor:
    """ Compute metric with multiple iterations over every query predictions set. """

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

    indexes = indexes.flatten()
    preds = preds.flatten()
    target = target.flatten()
    groups = get_group_indexes(indexes)

    sk_results = []
    for group in groups:
        trg, pds = target[group], preds[group]

        if ((1 - trg) if reverse else trg).sum() == 0:
            if empty_target_action == 'skip':
                pass
            elif empty_target_action == 'pos':
                sk_results.append(1.0)
            else:
                sk_results.append(0.0)
        else:
            res = metric(trg, pds, **kwargs)
            sk_results.append(res)

    if len(sk_results) > 0:
        return np.mean(sk_results)
    return np.array(0.0)


def _concat_tests(*tests: Tuple[str, Tuple]) -> Tuple[str, Tuple]:
    """Concat tests composed by a string and a list of arguments."""
    assert len(tests), "cannot concatenate "
    assert all([tests[0][0] == x[0] for x in tests[1:]]), "the header must be the same for all tests"
    return (tests[0][0], sum([x[1] for x in tests], []))


_errors_test_functional_metric_parameters_default = [
    "preds, target, message, metric_args", [
        # check input shapes are consistent (func)
        (
            _input_retrieval_scores_mismatching_sizes_func.preds,
            _input_retrieval_scores_mismatching_sizes_func.target,
            "`preds` and `target` must be of the same shape",
            {},
        ),
        # check input tensors are not empty
        (
            _input_retrieval_scores_empty.preds,
            _input_retrieval_scores_empty.target,
            "`preds` and `target` must be non-empty and non-scalar tensors",
            {},
        ),
        # check on input dtypes
        (
            _input_retrieval_scores.preds.bool(),
            _input_retrieval_scores.target,
            "`preds` must be a tensor of floats",
            {},
        ),
        (
            _input_retrieval_scores.preds,
            _input_retrieval_scores.target.float(),
            "`target` must be a tensor of booleans or integers",
            {},
        ),
        # check targets are between 0 and 1
        (
            _input_retrieval_scores_wrong_targets.preds,
            _input_retrieval_scores_wrong_targets.target,
            "`target` must contain `binary` values",
            {},
        ),
    ]
]


_errors_test_functional_metric_parameters_k = [
    "preds, target, message, metric_args", [
        (
            _input_retrieval_scores.preds,
            _input_retrieval_scores.target,
            "`k` has to be a positive integer or None",
            {'k': -10},
        ),
        (
            _input_retrieval_scores.preds,
            _input_retrieval_scores.target,
            "`k` has to be a positive integer or None",
            {'k': 4.0},
        ),
    ]
]


_errors_test_class_metric_parameters_no_pos_target = [
    "indexes, preds, target, message, metric_args", [
        # check when error when there are no positive targets
        (
            _input_retrieval_scores_no_target.indexes,
            _input_retrieval_scores_no_target.preds,
            _input_retrieval_scores_no_target.target,
            "`compute` method was provided with a query with no positive target.",
            {'empty_target_action': "error"},
        ),
    ]
]


_errors_test_class_metric_parameters_no_neg_target = [
    "indexes, preds, target, message, metric_args", [
        # check when error when there are no negative targets
        (
            _input_retrieval_scores_all_target.indexes,
            _input_retrieval_scores_all_target.preds,
            _input_retrieval_scores_all_target.target,
            "`compute` method was provided with a query with no negative target.",
            {'empty_target_action': "error"},
        ),
    ]
]


_errors_test_class_metric_parameters_default = [
    "indexes, preds, target, message, metric_args", [
        (
            None,
            _input_retrieval_scores.preds,
            _input_retrieval_scores.target,
            "`indexes` cannot be None",
            {'empty_target_action': "error"},
        ),
        # check when input arguments are invalid
        (
            _input_retrieval_scores.indexes,
            _input_retrieval_scores.preds,
            _input_retrieval_scores.target,
            "`empty_target_action` received a wrong value `casual_argument`.",
            {'empty_target_action': "casual_argument"},
        ),
        # check input shapes are consistent
        (
            _input_retrieval_scores_mismatching_sizes.indexes,
            _input_retrieval_scores_mismatching_sizes.preds,
            _input_retrieval_scores_mismatching_sizes.target,
            "`indexes`, `preds` and `target` must be of the same shape",
            {'empty_target_action': "skip"},
        ),
        # check input tensors are not empty
        (
            _input_retrieval_scores_empty.indexes,
            _input_retrieval_scores_empty.preds,
            _input_retrieval_scores_empty.target,
            "`indexes`, `preds` and `target` must be non-empty and non-scalar tensors",
            {'empty_target_action': "skip"},
        ),
        # check on input dtypes
        (
            _input_retrieval_scores.indexes.bool(),
            _input_retrieval_scores.preds,
            _input_retrieval_scores.target,
            "`indexes` must be a tensor of long integers",
            {'empty_target_action': "skip"},
        ),
        (
            _input_retrieval_scores.indexes,
            _input_retrieval_scores.preds.bool(),
            _input_retrieval_scores.target,
            "`preds` must be a tensor of floats",
            {'empty_target_action': "skip"},
        ),
        (
            _input_retrieval_scores.indexes,
            _input_retrieval_scores.preds,
            _input_retrieval_scores.target.float(),
            "`target` must be a tensor of booleans or integers",
            {'empty_target_action': "skip"},
        ),
        # check targets are between 0 and 1
        (
            _input_retrieval_scores_wrong_targets.indexes,
            _input_retrieval_scores_wrong_targets.preds,
            _input_retrieval_scores_wrong_targets.target,
            "`target` must contain `binary` values",
            {'empty_target_action': "skip"},
        ),
    ]
]


_errors_test_class_metric_parameters_k = [
    "indexes, preds, target, message, metric_args", [
        (
            _input_retrieval_scores.index,
            _input_retrieval_scores.preds,
            _input_retrieval_scores.target,
            "`k` has to be a positive integer or None",
            {'k': -10},
        ),
    ]
]


_default_metric_class_input_arguments = [
    "indexes, preds, target", [
        (
            _input_retrieval_scores.indexes,
            _input_retrieval_scores.preds,
            _input_retrieval_scores.target
        ),
        (
            _input_retrieval_scores_extra.indexes,
            _input_retrieval_scores_extra.preds,
            _input_retrieval_scores_extra.target,
        ),
        (
            _input_retrieval_scores_no_target.indexes,
            _input_retrieval_scores_no_target.preds,
            _input_retrieval_scores_no_target.target,
        ),
    ]
]


_default_metric_functional_input_arguments = [
    "preds, target", [
        (
            _input_retrieval_scores.preds,
            _input_retrieval_scores.target
        ),
        (
            _input_retrieval_scores_extra.preds,
            _input_retrieval_scores_extra.target
        ),
        (
            _input_retrieval_scores_no_target.preds,
            _input_retrieval_scores_no_target.target
        ),
    ]
]


def _errors_test_class_metric(
    indexes: Tensor,
    preds: Tensor,
    target: Tensor,
    metric_class: Metric,
    message: str = "",
    metric_args: dict = {},
    exception_type: Exception = ValueError,
    kwargs_update: dict = {},
):
    """Utility function doing checks about types, parameters and errors.

    Args:
        indexes: torch tensor with indexes
        preds: torch tensor with predictions
        target: torch tensor with targets
        metric_class: lightning metric class that should be tested
        message: message that exception should return
        metric_args: arguments for class initialization
        exception_type: callable function that is used for comparison
        kwargs_update: Additional keyword arguments that will be passed with indexes, preds and
            target when running update on the metric.
    """
    with pytest.raises(exception_type, match=message):
        metric = metric_class(**metric_args)
        metric(preds, target, indexes=indexes, **kwargs_update)


def _errors_test_functional_metric(
    preds: Tensor,
    target: Tensor,
    metric_functional: Metric,
    message: str = "",
    exception_type: Exception = ValueError,
    kwargs_update: dict = {},
):
    """Utility function doing checks about types, parameters and errors.

    Args:
        preds: torch tensor with predictions
        target: torch tensor with targets
        metric_functional: lightning functional metric that should be tested
        message: message that exception should return
        exception_type: callable function that is used for comparison
        kwargs_update: Additional keyword arguments that will be passed with indexes, preds and
            target when running update on the metric.
    """
    with pytest.raises(exception_type, match=message):
        metric_functional(preds, target, **kwargs_update)


class RetrievalMetricTester(MetricTester):

    def run_class_metric_test(
        self,
        ddp: bool,
        indexes: Tensor,
        preds: Tensor,
        target: Tensor,
        metric_class: Metric,
        sk_metric: Callable,
        dist_sync_on_step: bool,
        metric_args: dict,
        reverse: bool = True,
    ):
        _sk_metric_adapted = partial(_compute_sklearn_metric, metric=sk_metric, reverse=reverse,  **metric_args)

        super().run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=metric_class,
            sk_metric=_sk_metric_adapted,
            dist_sync_on_step=dist_sync_on_step,
            metric_args=metric_args,
            fragment_kwargs=True,
            indexes=indexes,  # every additional argument will be passed to metric_class and _sk_metric_adapted
        )

    def run_functional_metric_test(
        self,
        preds: Tensor,
        target: Tensor,
        metric_functional: Callable,
        sk_metric: Callable,
        metric_args: dict,
        reverse: bool = True,
        **kwargs,
    ):
        # action on functional version of IR metrics is to return `tensor(0.0)` if not target is positive.
        _sk_metric_adapted = partial(_compute_sklearn_metric, metric=sk_metric, reverse=reverse, **metric_args)

        super().run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=metric_functional,
            sk_metric=_sk_metric_adapted,
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
        # action on functional version of IR metrics is to return `tensor(0.0)` if not target is positive.
        def metric_functional_ignore_indexes(preds, target, indexes):
            return metric_functional(preds, target)

        super().run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=metric_module,
            metric_functional=metric_functional_ignore_indexes,
            metric_args={'empty_target_action': 'neg'},
            indexes=indexes,  # every additional argument will be passed to RetrievalMAP and _sk_metric_adapted
        )

    def run_precision_test_gpu(
        self,
        indexes: Tensor,
        preds: Tensor,
        target: Tensor,
        metric_module: Metric,
        metric_functional: Callable,
    ):
        if not torch.cuda.is_available():
            pytest.skip()

        # action on functional version of IR metrics is to return `tensor(0.0)` if not target is positive.
        def metric_functional_ignore_indexes(preds, target, indexes):
            return metric_functional(preds, target)

        super().run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=metric_module,
            metric_functional=metric_functional_ignore_indexes,
            metric_args={'empty_target_action': 'neg'},
            indexes=indexes,  # every additional argument will be passed to RetrievalMAP and _sk_metric_adapted
        )

    def run_metric_class_arguments_test(
        self,
        indexes: Tensor,
        preds: Tensor,
        target: Tensor,
        metric_class: Metric,
        message: str = "",
        metric_args: dict = {},
        exception_type: Exception = ValueError,
        kwargs_update: dict = {},
    ):
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

    def run_functional_metric_arguments_test(
        self,
        preds: Tensor,
        target: Tensor,
        metric_functional: Callable,
        message: str = "",
        exception_type: Exception = ValueError,
        kwargs_update: dict = {},
    ):
        _errors_test_functional_metric(
            preds=preds,
            target=target,
            metric_functional=metric_functional,
            message=message,
            exception_type=exception_type,
            kwargs_update=kwargs_update,
        )
