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
from operator import neg, pos

import pytest
import torch
from torch import tensor

from torchmetrics.metric import CompositionalMetric, Metric
from unittests.helpers import _MARK_TORCH_MIN_1_4, _MARK_TORCH_MIN_1_5, _MARK_TORCH_MIN_1_6


class DummyMetric(Metric):
    full_state_update = True

    def __init__(self, val_to_return):
        super().__init__()
        self.add_state("_num_updates", tensor(0), dist_reduce_fx="sum")
        self._val_to_return = val_to_return

    def update(self, *args, **kwargs) -> None:
        self._num_updates += 1

    def compute(self):
        return tensor(self._val_to_return)


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        (DummyMetric(2), tensor(4)),
        (2, tensor(4)),
        (2.0, tensor(4.0)),
        pytest.param(tensor(2), tensor(4), marks=pytest.mark.skipif(**_MARK_TORCH_MIN_1_4)),
    ],
)
def test_metrics_add(second_operand, expected_result):
    first_metric = DummyMetric(2)

    final_add = first_metric + second_operand
    final_radd = second_operand + first_metric

    assert isinstance(final_add, CompositionalMetric)
    assert isinstance(final_radd, CompositionalMetric)

    final_add.update()
    final_radd.update()

    assert torch.allclose(expected_result, final_add.compute())
    assert torch.allclose(expected_result, final_radd.compute())


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [(DummyMetric(3), tensor(2)), (3, tensor(2)), (3, tensor(2)), (tensor(3), tensor(2))],
)
@pytest.mark.skipif(**_MARK_TORCH_MIN_1_5)
def test_metrics_and(second_operand, expected_result):
    first_metric = DummyMetric(2)

    final_and = first_metric & second_operand
    final_rand = second_operand & first_metric

    assert isinstance(final_and, CompositionalMetric)
    assert isinstance(final_rand, CompositionalMetric)

    final_and.update()
    final_rand.update()
    assert torch.allclose(expected_result, final_and.compute())
    assert torch.allclose(expected_result, final_rand.compute())


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        (DummyMetric(2), tensor(True)),
        (2, tensor(True)),
        (2.0, tensor(True)),
        (tensor(2), tensor(True)),
    ],
)
def test_metrics_eq(second_operand, expected_result):
    first_metric = DummyMetric(2)

    final_eq = first_metric == second_operand

    assert isinstance(final_eq, CompositionalMetric)

    final_eq.update()
    # can't use allclose for bool tensors
    assert (expected_result == final_eq.compute()).all()


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        (DummyMetric(2), tensor(2)),
        (2, tensor(2)),
        (2.0, tensor(2.0)),
        (tensor(2), tensor(2)),
    ],
)
@pytest.mark.skipif(**_MARK_TORCH_MIN_1_5)
def test_metrics_floordiv(second_operand, expected_result):
    first_metric = DummyMetric(5)

    final_floordiv = first_metric // second_operand

    assert isinstance(final_floordiv, CompositionalMetric)

    final_floordiv.update()
    assert torch.allclose(expected_result, final_floordiv.compute())


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        (DummyMetric(2), tensor(True)),
        (2, tensor(True)),
        (2.0, tensor(True)),
        (tensor(2), tensor(True)),
    ],
)
def test_metrics_ge(second_operand, expected_result):
    first_metric = DummyMetric(5)

    final_ge = first_metric >= second_operand

    assert isinstance(final_ge, CompositionalMetric)

    final_ge.update()
    # can't use allclose for bool tensors
    assert (expected_result == final_ge.compute()).all()


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        (DummyMetric(2), tensor(True)),
        (2, tensor(True)),
        (2.0, tensor(True)),
        (tensor(2), tensor(True)),
    ],
)
def test_metrics_gt(second_operand, expected_result):
    first_metric = DummyMetric(5)

    final_gt = first_metric > second_operand

    assert isinstance(final_gt, CompositionalMetric)

    final_gt.update()
    # can't use allclose for bool tensors
    assert (expected_result == final_gt.compute()).all()


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        (DummyMetric(2), tensor(False)),
        (2, tensor(False)),
        (2.0, tensor(False)),
        (tensor(2), tensor(False)),
    ],
)
def test_metrics_le(second_operand, expected_result):
    first_metric = DummyMetric(5)

    final_le = first_metric <= second_operand

    assert isinstance(final_le, CompositionalMetric)

    final_le.update()
    # can't use allclose for bool tensors
    assert (expected_result == final_le.compute()).all()


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        (DummyMetric(2), tensor(False)),
        (2, tensor(False)),
        (2.0, tensor(False)),
        (tensor(2), tensor(False)),
    ],
)
def test_metrics_lt(second_operand, expected_result):
    first_metric = DummyMetric(5)

    final_lt = first_metric < second_operand

    assert isinstance(final_lt, CompositionalMetric)

    final_lt.update()
    # can't use allclose for bool tensors
    assert (expected_result == final_lt.compute()).all()


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [(DummyMetric([2, 2, 2]), tensor(12)), (tensor([2, 2, 2]), tensor(12))],
)
def test_metrics_matmul(second_operand, expected_result):
    first_metric = DummyMetric([2, 2, 2])

    final_matmul = first_metric @ second_operand

    assert isinstance(final_matmul, CompositionalMetric)

    final_matmul.update()
    assert torch.allclose(expected_result, final_matmul.compute())


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        (DummyMetric(2), tensor(1)),
        (2, tensor(1)),
        (2.0, tensor(1)),
        (tensor(2), tensor(1)),
    ],
)
def test_metrics_mod(second_operand, expected_result):
    first_metric = DummyMetric(5)

    final_mod = first_metric % second_operand

    assert isinstance(final_mod, CompositionalMetric)

    final_mod.update()
    # prevent Runtime error for PT 1.8 - Long did not match Float
    assert torch.allclose(expected_result.to(float), final_mod.compute().to(float))


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        (DummyMetric(2), tensor(4)),
        (2, tensor(4)),
        (2.0, tensor(4.0)),
        pytest.param(tensor(2), tensor(4), marks=pytest.mark.skipif(**_MARK_TORCH_MIN_1_4)),
    ],
)
def test_metrics_mul(second_operand, expected_result):
    first_metric = DummyMetric(2)

    final_mul = first_metric * second_operand
    final_rmul = second_operand * first_metric

    assert isinstance(final_mul, CompositionalMetric)
    assert isinstance(final_rmul, CompositionalMetric)

    final_mul.update()
    final_rmul.update()
    assert torch.allclose(expected_result, final_mul.compute())
    assert torch.allclose(expected_result, final_rmul.compute())


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        (DummyMetric(2), tensor(False)),
        (2, tensor(False)),
        (2.0, tensor(False)),
        (tensor(2), tensor(False)),
    ],
)
def test_metrics_ne(second_operand, expected_result):
    first_metric = DummyMetric(2)

    final_ne = first_metric != second_operand

    assert isinstance(final_ne, CompositionalMetric)

    final_ne.update()
    # can't use allclose for bool tensors
    assert (expected_result == final_ne.compute()).all()


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [(DummyMetric([1, 0, 3]), tensor([-1, -2, 3])), (tensor([1, 0, 3]), tensor([-1, -2, 3]))],
)
@pytest.mark.skipif(**_MARK_TORCH_MIN_1_5)
def test_metrics_or(second_operand, expected_result):
    first_metric = DummyMetric([-1, -2, 3])

    final_or = first_metric | second_operand
    final_ror = second_operand | first_metric

    assert isinstance(final_or, CompositionalMetric)
    assert isinstance(final_ror, CompositionalMetric)

    final_or.update()
    final_ror.update()
    assert torch.allclose(expected_result, final_or.compute())
    assert torch.allclose(expected_result, final_ror.compute())


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        (DummyMetric(2), tensor(4)),
        (2, tensor(4)),
        pytest.param(2.0, tensor(4.0), marks=pytest.mark.skipif(**_MARK_TORCH_MIN_1_6)),
        (tensor(2), tensor(4)),
    ],
)
def test_metrics_pow(second_operand, expected_result):
    first_metric = DummyMetric(2)

    final_pow = first_metric**second_operand

    assert isinstance(final_pow, CompositionalMetric)

    final_pow.update()
    assert torch.allclose(expected_result, final_pow.compute())


@pytest.mark.parametrize(
    ["first_operand", "expected_result"],
    [(5, tensor(2)), (5.0, tensor(2.0)), (tensor(5), tensor(2))],
)
@pytest.mark.skipif(**_MARK_TORCH_MIN_1_5)
def test_metrics_rfloordiv(first_operand, expected_result):
    second_operand = DummyMetric(2)

    final_rfloordiv = first_operand // second_operand

    assert isinstance(final_rfloordiv, CompositionalMetric)

    final_rfloordiv.update()
    assert torch.allclose(expected_result, final_rfloordiv.compute())


@pytest.mark.parametrize(
    ["first_operand", "expected_result"],
    [pytest.param(tensor([2, 2, 2]), tensor(12), marks=pytest.mark.skipif(**_MARK_TORCH_MIN_1_4))],
)
def test_metrics_rmatmul(first_operand, expected_result):
    second_operand = DummyMetric([2, 2, 2])

    final_rmatmul = first_operand @ second_operand

    assert isinstance(final_rmatmul, CompositionalMetric)

    final_rmatmul.update()
    assert torch.allclose(expected_result, final_rmatmul.compute())


@pytest.mark.parametrize(
    ["first_operand", "expected_result"],
    [pytest.param(tensor(2), tensor(2), marks=pytest.mark.skipif(**_MARK_TORCH_MIN_1_4))],
)
def test_metrics_rmod(first_operand, expected_result):
    second_operand = DummyMetric(5)

    final_rmod = first_operand % second_operand

    assert isinstance(final_rmod, CompositionalMetric)

    final_rmod.update()
    assert torch.allclose(expected_result, final_rmod.compute())


@pytest.mark.parametrize(
    "first_operand,expected_result",
    [
        (DummyMetric(2), tensor(4)),
        (2, tensor(4)),
        pytest.param(2.0, tensor(4.0), marks=pytest.mark.skipif(**_MARK_TORCH_MIN_1_6)),
    ],
)
def test_metrics_rpow(first_operand, expected_result):
    second_operand = DummyMetric(2)

    final_rpow = first_operand**second_operand

    assert isinstance(final_rpow, CompositionalMetric)
    final_rpow.update()
    assert torch.allclose(expected_result, final_rpow.compute())


@pytest.mark.parametrize(
    ["first_operand", "expected_result"],
    [
        (DummyMetric(3), tensor(1)),
        (3, tensor(1)),
        (3.0, tensor(1.0)),
        pytest.param(tensor(3), tensor(1), marks=pytest.mark.skipif(**_MARK_TORCH_MIN_1_4)),
    ],
)
def test_metrics_rsub(first_operand, expected_result):
    second_operand = DummyMetric(2)

    final_rsub = first_operand - second_operand

    assert isinstance(final_rsub, CompositionalMetric)
    final_rsub.update()
    assert torch.allclose(expected_result, final_rsub.compute())


@pytest.mark.parametrize(
    ["first_operand", "expected_result"],
    [
        (DummyMetric(6), tensor(2.0)),
        (6, tensor(2.0)),
        (6.0, tensor(2.0)),
        (tensor(6), tensor(2.0)),
    ],
)
@pytest.mark.skipif(**_MARK_TORCH_MIN_1_5)
def test_metrics_rtruediv(first_operand, expected_result):
    second_operand = DummyMetric(3)

    final_rtruediv = first_operand / second_operand

    assert isinstance(final_rtruediv, CompositionalMetric)
    final_rtruediv.update()
    assert torch.allclose(expected_result, final_rtruediv.compute())


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        (DummyMetric(2), tensor(1)),
        (2, tensor(1)),
        (2.0, tensor(1.0)),
        (tensor(2), tensor(1)),
    ],
)
def test_metrics_sub(second_operand, expected_result):
    first_metric = DummyMetric(3)

    final_sub = first_metric - second_operand

    assert isinstance(final_sub, CompositionalMetric)
    final_sub.update()
    assert torch.allclose(expected_result, final_sub.compute())


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [
        (DummyMetric(3), tensor(2.0)),
        (3, tensor(2.0)),
        (3.0, tensor(2.0)),
        (tensor(3), tensor(2.0)),
    ],
)
@pytest.mark.skipif(**_MARK_TORCH_MIN_1_5)
def test_metrics_truediv(second_operand, expected_result):
    first_metric = DummyMetric(6)

    final_truediv = first_metric / second_operand

    assert isinstance(final_truediv, CompositionalMetric)
    final_truediv.update()
    assert torch.allclose(expected_result, final_truediv.compute())


@pytest.mark.parametrize(
    ["second_operand", "expected_result"],
    [(DummyMetric([1, 0, 3]), tensor([-2, -2, 0])), (tensor([1, 0, 3]), tensor([-2, -2, 0]))],
)
@pytest.mark.skipif(**_MARK_TORCH_MIN_1_5)
def test_metrics_xor(second_operand, expected_result):
    first_metric = DummyMetric([-1, -2, 3])

    final_xor = first_metric ^ second_operand
    final_rxor = second_operand ^ first_metric

    assert isinstance(final_xor, CompositionalMetric)
    assert isinstance(final_rxor, CompositionalMetric)

    final_xor.update()
    final_rxor.update()
    assert torch.allclose(expected_result, final_xor.compute())
    assert torch.allclose(expected_result, final_rxor.compute())


def test_metrics_abs():
    first_metric = DummyMetric(-1)

    final_abs = abs(first_metric)

    assert isinstance(final_abs, CompositionalMetric)
    final_abs.update()
    assert torch.allclose(tensor(1), final_abs.compute())


def test_metrics_invert():
    first_metric = DummyMetric(1)

    final_inverse = ~first_metric
    assert isinstance(final_inverse, CompositionalMetric)
    final_inverse.update()
    assert torch.allclose(tensor(-2), final_inverse.compute())


def test_metrics_neg():
    first_metric = DummyMetric(1)

    final_neg = neg(first_metric)
    assert isinstance(final_neg, CompositionalMetric)
    final_neg.update()
    assert torch.allclose(tensor(-1), final_neg.compute())


def test_metrics_pos():
    first_metric = DummyMetric(-1)

    final_pos = pos(first_metric)
    assert isinstance(final_pos, CompositionalMetric)
    final_pos.update()
    assert torch.allclose(tensor(1), final_pos.compute())


@pytest.mark.parametrize(
    ["value", "idx", "expected_result"],
    [([1, 2, 3], 1, tensor(2)), ([[0, 1], [2, 3]], (1, 0), tensor(2)), ([[0, 1], [2, 3]], 1, tensor([2, 3]))],
)
def test_metrics_getitem(value, idx, expected_result):
    first_metric = DummyMetric(value)

    final_getitem = first_metric[idx]
    assert isinstance(final_getitem, CompositionalMetric)
    final_getitem.update()
    assert torch.allclose(expected_result, final_getitem.compute())


def test_compositional_metrics_update():
    """test update method for compositional metrics."""
    compos = DummyMetric(5) + DummyMetric(4)

    assert isinstance(compos, CompositionalMetric)
    compos.update()
    compos.update()
    compos.update()

    assert isinstance(compos.metric_a, DummyMetric)
    assert isinstance(compos.metric_b, DummyMetric)

    assert compos.metric_a._num_updates == 3
    assert compos.metric_b._num_updates == 3
