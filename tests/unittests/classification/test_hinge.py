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

import numpy as np
import pytest
import torch
from sklearn.metrics import hinge_loss as sk_hinge
from sklearn.preprocessing import OneHotEncoder

from torchmetrics import HingeLoss
from torchmetrics.functional import hinge_loss
from torchmetrics.functional.classification.hinge import MulticlassMode
from unittests.classification.inputs import Input
from unittests.helpers.testers import BATCH_SIZE, NUM_BATCHES, NUM_CLASSES, MetricTester

torch.manual_seed(42)


# def _sk_binary_hinge_loss(preds, target, n_bins, norm, ignore_index):
#     preds = preds.numpy().flatten()
#     target = target.numpy().flatten()
#     if not ((0 < preds) & (preds < 1)).all():
#         preds = sigmoid(preds)
#     target, preds = remove_ignore_index(target, preds, ignore_index)
#     metric = ECE if norm == "l1" else MCE
#     return metric(n_bins).measure(preds, target)


# @pytest.mark.parametrize("input", (_binary_cases[1], _binary_cases[2], _binary_cases[4], _binary_cases[5]))
# class TestBinaryHingeLoss(MetricTester):
#     @pytest.mark.parametrize("n_bins", [10, 15, 20])
#     @pytest.mark.parametrize("norm", ["l1", "max"])
#     @pytest.mark.parametrize("ignore_index", [None, -1, 0])
#     @pytest.mark.parametrize("ddp", [True, False])
#     def test_binary_hinge_loss(self, input, ddp, n_bins, norm, ignore_index):
#         preds, target = input
#         if ignore_index is not None:
#             target = inject_ignore_index(target, ignore_index)
#         self.run_class_metric_test(
#             ddp=ddp,
#             preds=preds,
#             target=target,
#             metric_class=BinaryHingeLoss,
#             sk_metric=partial(_sk_binary_hinge_loss, n_bins=n_bins, norm=norm, ignore_index=ignore_index),
#             metric_args={
#                 "n_bins": n_bins,
#                 "norm": norm,
#                 "ignore_index": ignore_index,
#             },
#         )

#     @pytest.mark.parametrize("n_bins", [10, 15, 20])
#     @pytest.mark.parametrize("norm", ["l1", "max"])
#     @pytest.mark.parametrize("ignore_index", [None, -1, 0])
#     def test_binary_hinge_loss_functional(self, input, n_bins, norm, ignore_index):
#         preds, target = input
#         if ignore_index is not None:
#             target = inject_ignore_index(target, ignore_index)
#         self.run_functional_metric_test(
#             preds=preds,
#             target=target,
#             metric_functional=binary_hinge_loss,
#             sk_metric=partial(_sk_binary_hinge_loss, n_bins=n_bins, norm=norm, ignore_index=ignore_index),
#             metric_args={
#                 "n_bins": n_bins,
#                 "norm": norm,
#                 "ignore_index": ignore_index,
#             },
#         )

#     def test_binary_hinge_loss_differentiability(self, input):
#         preds, target = input
#         self.run_differentiability_test(
#             preds=preds,
#             target=target,
#             metric_module=BinaryHingeLoss,
#             metric_functional=binary_hinge_loss,
#         )

#     @pytest.mark.parametrize("dtype", [torch.half, torch.double])
#     def test_binary_hinge_loss_dtype_cpu(self, input, dtype):
#         preds, target = input
#         if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
#             pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
#         if (preds < 0).any() and dtype == torch.half:
#             pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
#         self.run_precision_test_cpu(
#             preds=preds,
#             target=target,
#             metric_module=BinaryHingeLoss,
#             metric_functional=binary_hinge_loss,
#             dtype=dtype,
#         )

#     @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
#     @pytest.mark.parametrize("dtype", [torch.half, torch.double])
#     def test_binary_hinge_loss_dtype_gpu(self, input, dtype):
#         preds, target = input
#         self.run_precision_test_gpu(
#             preds=preds,
#             target=target,
#             metric_module=BinaryHingeLoss,
#             metric_functional=binary_hinge_loss,
#             dtype=dtype,
#         )


# def _sk_multiclass_hinge_loss(preds, target, n_bins, norm, ignore_index):
#     preds = preds.numpy()
#     target = target.numpy().flatten()
#     if not ((0 < preds) & (preds < 1)).all():
#         preds = softmax(preds, 1)
#     preds = np.moveaxis(preds, 1, -1).reshape((-1, preds.shape[1]))
#     target, preds = remove_ignore_index(target, preds, ignore_index)
#     metric = ECE if norm == "l1" else MCE
#     return metric(n_bins).measure(preds, target)


# @pytest.mark.parametrize(
#     "input", (_multiclass_cases[1], _multiclass_cases[2], _multiclass_cases[4], _multiclass_cases[5])
# )
# class TestMulticlassHingeLoss(MetricTester):
#     @pytest.mark.parametrize("n_bins", [10, 15, 20])
#     @pytest.mark.parametrize("norm", ["l1", "max"])
#     @pytest.mark.parametrize("ignore_index", [None, -1, 0])
#     @pytest.mark.parametrize("ddp", [True, False])
#     def test_multiclass_hinge_loss(self, input, ddp, n_bins, norm, ignore_index):
#         preds, target = input
#         if ignore_index is not None:
#             target = inject_ignore_index(target, ignore_index)
#         self.run_class_metric_test(
#             ddp=ddp,
#             preds=preds,
#             target=target,
#             metric_class=MulticlassHingeLoss,
#             sk_metric=partial(_sk_multiclass_hinge_loss, n_bins=n_bins, norm=norm, ignore_index=ignore_index),
#             metric_args={
#                 "num_classes": NUM_CLASSES,
#                 "n_bins": n_bins,
#                 "norm": norm,
#                 "ignore_index": ignore_index,
#             },
#         )

#     @pytest.mark.parametrize("n_bins", [10, 15, 20])
#     @pytest.mark.parametrize("norm", ["l1", "max"])
#     @pytest.mark.parametrize("ignore_index", [None, -1, 0])
#     def test_multiclass_hinge_loss_functional(self, input, n_bins, norm, ignore_index):
#         preds, target = input
#         if ignore_index is not None:
#             target = inject_ignore_index(target, ignore_index)
#         self.run_functional_metric_test(
#             preds=preds,
#             target=target,
#             metric_functional=multiclass_hinge_loss,
#             sk_metric=partial(_sk_multiclass_hinge_loss, n_bins=n_bins, norm=norm, ignore_index=ignore_index),
#             metric_args={
#                 "num_classes": NUM_CLASSES,
#                 "n_bins": n_bins,
#                 "norm": norm,
#                 "ignore_index": ignore_index,
#             },
#         )

#     def test_multiclass_hinge_loss_differentiability(self, input):
#         preds, target = input
#         self.run_differentiability_test(
#             preds=preds,
#             target=target,
#             metric_module=MulticlassHingeLoss,
#             metric_functional=multiclass_hinge_loss,
#             metric_args={"num_classes": NUM_CLASSES},
#         )

#     @pytest.mark.parametrize("dtype", [torch.half, torch.double])
#     def test_multiclass_hinge_loss_dtype_cpu(self, input, dtype):
#         preds, target = input
#         if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
#             pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
#         self.run_precision_test_cpu(
#             preds=preds,
#             target=target,
#             metric_module=MulticlassHingeLoss,
#             metric_functional=multiclass_hinge_loss,
#             metric_args={"num_classes": NUM_CLASSES},
#             dtype=dtype,
#         )

#     @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
#     @pytest.mark.parametrize("dtype", [torch.half, torch.double])
#     def test_multiclass_hinge_loss_dtype_gpu(self, input, dtype):
#         preds, target = input
#         self.run_precision_test_gpu(
#             preds=preds,
#             target=target,
#             metric_module=MulticlassHingeLoss,
#             metric_functional=multiclass_hinge_loss,
#             metric_args={"num_classes": NUM_CLASSES},
#             dtype=dtype,
#         )


# -------------------------- Old stuff --------------------------

# _input_binary = Input(
#     preds=torch.randn(NUM_BATCHES, BATCH_SIZE), target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE))
# )

# _input_binary_single = Input(preds=torch.randn((NUM_BATCHES, 1)), target=torch.randint(high=2, size=(NUM_BATCHES, 1)))

# _input_multiclass = Input(
#     preds=torch.randn(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES),
#     target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE)),
# )


# def _sk_hinge(preds, target, squared, multiclass_mode):
#     sk_preds, sk_target = preds.numpy(), target.numpy()

#     if multiclass_mode == MulticlassMode.ONE_VS_ALL:
#         enc = OneHotEncoder()
#         enc.fit(sk_target.reshape(-1, 1))
#         sk_target = enc.transform(sk_target.reshape(-1, 1)).toarray()

#     if sk_preds.ndim == 1 or multiclass_mode == MulticlassMode.ONE_VS_ALL:
#         sk_target = 2 * sk_target - 1

#     if squared or sk_target.max() != 1 or sk_target.min() != -1:
#         # Squared not an option in sklearn and infers classes incorrectly with single element, so adapted from source
#         if sk_preds.ndim == 1 or multiclass_mode == MulticlassMode.ONE_VS_ALL:
#             margin = sk_target * sk_preds
#         else:
#             mask = np.ones_like(sk_preds, dtype=bool)
#             mask[np.arange(sk_target.shape[0]), sk_target] = False
#             margin = sk_preds[~mask]
#             margin -= np.max(sk_preds[mask].reshape(sk_target.shape[0], -1), axis=1)
#         measures = 1 - margin
#         measures = np.clip(measures, 0, None)

#         if squared:
#             measures = measures**2
#         return measures.mean(axis=0)
#     if multiclass_mode == MulticlassMode.ONE_VS_ALL:
#         result = np.zeros(sk_preds.shape[1])
#         for i in range(result.shape[0]):
#             result[i] = sk_hinge(y_true=sk_target[:, i], pred_decision=sk_preds[:, i])
#         return result

#     return sk_hinge(y_true=sk_target, pred_decision=sk_preds)


# @pytest.mark.parametrize(
#     "preds, target, squared, multiclass_mode",
#     [
#         (_input_binary.preds, _input_binary.target, False, None),
#         (_input_binary.preds, _input_binary.target, True, None),
#         (_input_binary_single.preds, _input_binary_single.target, False, None),
#         (_input_binary_single.preds, _input_binary_single.target, True, None),
#         (_input_multiclass.preds, _input_multiclass.target, False, MulticlassMode.CRAMMER_SINGER),
#         (_input_multiclass.preds, _input_multiclass.target, True, MulticlassMode.CRAMMER_SINGER),
#         (_input_multiclass.preds, _input_multiclass.target, False, MulticlassMode.ONE_VS_ALL),
#         (_input_multiclass.preds, _input_multiclass.target, True, MulticlassMode.ONE_VS_ALL),
#     ],
# )
# class TestHinge(MetricTester):
#     @pytest.mark.parametrize("ddp", [True, False])
#     @pytest.mark.parametrize("dist_sync_on_step", [True, False])
#     def test_hinge_class(self, ddp, dist_sync_on_step, preds, target, squared, multiclass_mode):
#         self.run_class_metric_test(
#             ddp=ddp,
#             preds=preds,
#             target=target,
#             metric_class=HingeLoss,
#             sk_metric=partial(_sk_hinge, squared=squared, multiclass_mode=multiclass_mode),
#             dist_sync_on_step=dist_sync_on_step,
#             metric_args={
#                 "squared": squared,
#                 "multiclass_mode": multiclass_mode,
#             },
#         )

#     def test_hinge_fn(self, preds, target, squared, multiclass_mode):
#         self.run_functional_metric_test(
#             preds=preds,
#             target=target,
#             metric_functional=partial(hinge_loss, squared=squared, multiclass_mode=multiclass_mode),
#             sk_metric=partial(_sk_hinge, squared=squared, multiclass_mode=multiclass_mode),
#         )

#     def test_hinge_differentiability(self, preds, target, squared, multiclass_mode):
#         self.run_differentiability_test(
#             preds=preds,
#             target=target,
#             metric_module=HingeLoss,
#             metric_functional=partial(hinge_loss, squared=squared, multiclass_mode=multiclass_mode),
#         )


# _input_multi_target = Input(preds=torch.randn(BATCH_SIZE), target=torch.randint(high=2, size=(BATCH_SIZE, 2)))

# _input_binary_different_sizes = Input(
#     preds=torch.randn(BATCH_SIZE * 2), target=torch.randint(high=2, size=(BATCH_SIZE,))
# )

# _input_multi_different_sizes = Input(
#     preds=torch.randn(BATCH_SIZE * 2, NUM_CLASSES), target=torch.randint(high=NUM_CLASSES, size=(BATCH_SIZE,))
# )

# _input_extra_dim = Input(
#     preds=torch.randn(BATCH_SIZE, NUM_CLASSES, 2), target=torch.randint(high=2, size=(BATCH_SIZE,))
# )


# @pytest.mark.parametrize(
#     "preds, target, multiclass_mode",
#     [
#         (_input_multi_target.preds, _input_multi_target.target, None),
#         (_input_binary_different_sizes.preds, _input_binary_different_sizes.target, None),
#         (_input_multi_different_sizes.preds, _input_multi_different_sizes.target, None),
#         (_input_extra_dim.preds, _input_extra_dim.target, None),
#         (_input_multiclass.preds[0], _input_multiclass.target[0], "invalid_mode"),
#     ],
# )
# def test_bad_inputs_fn(preds, target, multiclass_mode):
#     with pytest.raises(ValueError):
#         _ = hinge_loss(preds, target, multiclass_mode=multiclass_mode)


# def test_bad_inputs_class():
#     with pytest.raises(ValueError):
#         HingeLoss(multiclass_mode="invalid_mode")
