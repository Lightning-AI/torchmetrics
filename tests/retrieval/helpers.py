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
from typing import Callable, Dict, List, Tuple, Type, Union

import numpy as np
import pytest
import torch
from numpy import array
from torch import Tensor, tensor

from tests.helpers import seed_all
from tests.helpers.testers import Metric, MetricTester
from tests.retrieval.inputs import _input_retrieval_scores as _irs
from tests.retrieval.inputs import _input_retrieval_scores_all_target as _irs_all
from tests.retrieval.inputs import _input_retrieval_scores_empty as _irs_empty
from tests.retrieval.inputs import _input_retrieval_scores_extra as _irs_extra
from tests.retrieval.inputs import _input_retrieval_scores_mismatching_sizes as _irs_mis_sz
from tests.retrieval.inputs import _input_retrieval_scores_mismatching_sizes_func as _irs_mis_sz_fn
from tests.retrieval.inputs import _input_retrieval_scores_no_target as _irs_no_tgt
from tests.retrieval.inputs import _input_retrieval_scores_wrong_targets as _irs_bad_tgt

seed_all(42)

# a version of get_group_indexes that depends on NumPy is here to avoid this dependency for the full library


def get_group_indexes(indexes: Union[Tensor, np.ndarray]) -> List[Union[Tensor, np.ndarray]]:
    """
    Given an integer `torch.Tensor` or `np.ndarray` `indexes`, return a `torch.Tensor` or `np.ndarray` of indexes for
    each different value in `indexes`.

    Args:
        indexes: a `torch.Tensor` or `np.ndarray` of integers

    Return:
        A list of integer `torch.Tensor`s or `np.ndarray`s

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


def _concat_tests(*tests: Tuple[Dict]) -> Dict:
    """Concat tests composed by a string and a list of arguments."""
    assert len(tests), "`_concat_tests` expects at least an argument"
    assert all(tests[0]['argnames'] == x['argnames'] for x in tests[1:]), "the header must be the same for all tests"
    return dict(argnames=tests[0]['argnames'], argvalues=sum([x['argvalues'] for x in tests], []))


_errors_test_functional_metric_parameters_default = dict(
    argnames="preds,target,message,metric_args",
    argvalues=[
        # check input shapes are consistent (func)
        (_irs_mis_sz_fn.preds, _irs_mis_sz_fn.target, "`preds` and `target` must be of the same shape", {}),
        # check input tensors are not empty
        (_irs_empty.preds, _irs_empty.target, "`preds` and `target` must be non-empty and non-scalar tensors", {}),
        # check on input dtypes
        (_irs.preds.bool(), _irs.target, "`preds` must be a tensor of floats", {}),
        (_irs.preds, _irs.target.float(), "`target` must be a tensor of booleans or integers", {}),
        # check targets are between 0 and 1
        (_irs_bad_tgt.preds, _irs_bad_tgt.target, "`target` must contain `binary` values", {}),
    ]
)

_errors_test_functional_metric_parameters_k = dict(
    argnames="preds,target,message,metric_args",
    argvalues=[
        (_irs.preds, _irs.target, "`k` has to be a positive integer or None", dict(k=-10)),
        (_irs.preds, _irs.target, "`k` has to be a positive integer or None", dict(k=4.0)),
    ]
)

_errors_test_class_metric_parameters_no_pos_target = dict(
    argnames="indexes,preds,target,message,metric_args",
    argvalues=[
        # check when error when there are no positive targets
        (
            _irs_no_tgt.indexes, _irs_no_tgt.preds, _irs_no_tgt.target,
            "`compute` method was provided with a query with no positive target.", dict(empty_target_action="error")
        ),
    ]
)

_errors_test_class_metric_parameters_no_neg_target = dict(
    argnames="indexes,preds,target,message,metric_args",
    argvalues=[
        # check when error when there are no negative targets
        (
            _irs_all.indexes, _irs_all.preds, _irs_all.target,
            "`compute` method was provided with a query with no negative target.", dict(empty_target_action="error")
        ),
    ]
)

_errors_test_class_metric_parameters_default = dict(
    argnames="indexes,preds,target,message,metric_args",
    argvalues=[
        (None, _irs.preds, _irs.target, "`indexes` cannot be None", dict(empty_target_action="error")),
        # check when input arguments are invalid
        (
            _irs.indexes, _irs.preds, _irs.target, "`empty_target_action` received a wrong value `casual_argument`.",
            dict(empty_target_action="casual_argument")
        ),
        # check input shapes are consistent
        (
            _irs_mis_sz.indexes, _irs_mis_sz.preds, _irs_mis_sz.target,
            "`indexes`, `preds` and `target` must be of the same shape", dict(empty_target_action="skip")
        ),
        # check input tensors are not empty
        (
            _irs_empty.indexes, _irs_empty.preds,
            _irs_empty.target, "`indexes`, `preds` and `target` must be non-empty and non-scalar tensors",
            dict(empty_target_action="skip")
        ),
        # check on input dtypes
        (
            _irs.indexes.bool(), _irs.preds, _irs.target, "`indexes` must be a tensor of long integers",
            dict(empty_target_action="skip")
        ),
        (
            _irs.indexes, _irs.preds.bool(), _irs.target, "`preds` must be a tensor of floats",
            dict(empty_target_action="skip")
        ),
        (
            _irs.indexes, _irs.preds, _irs.target.float(), "`target` must be a tensor of booleans or integers",
            dict(empty_target_action="skip")
        ),
        # check targets are between 0 and 1
        (
            _irs_bad_tgt.indexes, _irs_bad_tgt.preds, _irs_bad_tgt.target, "`target` must contain `binary` values",
            dict(empty_target_action="skip")
        ),
    ]
)

_errors_test_class_metric_parameters_k = dict(
    argnames="indexes,preds,target,message,metric_args",
    argvalues=[
        (_irs.index, _irs.preds, _irs.target, "`k` has to be a positive integer or None", dict(k=-10)),
    ]
)

_default_metric_class_input_arguments = dict(
    argnames="indexes,preds,target",
    argvalues=[
        (_irs.indexes, _irs.preds, _irs.target),
        (_irs_extra.indexes, _irs_extra.preds, _irs_extra.target),
        (_irs_no_tgt.indexes, _irs_no_tgt.preds, _irs_no_tgt.target),
    ]
)

_default_metric_functional_input_arguments = dict(
    argnames="preds,target",
    argvalues=[
        (_irs.preds, _irs.target),
        (_irs_extra.preds, _irs_extra.target),
        (_irs_no_tgt.preds, _irs_no_tgt.target),
    ]
)


def _errors_test_class_metric(
    indexes: Tensor,
    preds: Tensor,
    target: Tensor,
    metric_class: Metric,
    message: str = "",
    metric_args: dict = None,
    exception_type: Type[Exception] = ValueError,
    kwargs_update: dict = None,
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
    metric_args = metric_args or {}
    kwargs_update = kwargs_update or {}
    with pytest.raises(exception_type, match=message):
        metric = metric_class(**metric_args)
        metric(preds, target, indexes=indexes, **kwargs_update)


def _errors_test_functional_metric(
    preds: Tensor,
    target: Tensor,
    metric_functional: Metric,
    message: str = "",
    exception_type: Type[Exception] = ValueError,
    kwargs_update: dict = None,
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
    kwargs_update = kwargs_update or {}
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
        reverse: bool = False,
    ):
        _sk_metric_adapted = partial(_compute_sklearn_metric, metric=sk_metric, reverse=reverse, **metric_args)

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
        reverse: bool = False,
        **kwargs,
    ):
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

    @staticmethod
    def run_metric_class_arguments_test(
        indexes: Tensor,
        preds: Tensor,
        target: Tensor,
        metric_class: Metric,
        message: str = "",
        metric_args: dict = None,
        exception_type: Type[Exception] = ValueError,
        kwargs_update: dict = None,
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

    @staticmethod
    def run_functional_metric_arguments_test(
        preds: Tensor,
        target: Tensor,
        metric_functional: Callable,
        message: str = "",
        exception_type: Type[Exception] = ValueError,
        kwargs_update: dict = None,
    ):
        _errors_test_functional_metric(
            preds=preds,
            target=target,
            metric_functional=metric_functional,
            message=message,
            exception_type=exception_type,
            kwargs_update=kwargs_update,
        )
