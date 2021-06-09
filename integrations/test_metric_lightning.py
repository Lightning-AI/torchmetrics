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
from unittest import mock

import pytest
import torch
from pytorch_lightning import LightningModule, Trainer
from torch import tensor
from torch.utils.data import DataLoader

from integrations.lightning_models import BoringModel, RandomDataset
from tests.helpers import _LIGHTNING_GREATER_EQUAL_1_3
from torchmetrics import Accuracy, AveragePrecision, Metric


class SumMetric(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("x", tensor(0.0), dist_reduce_fx="sum")

    def update(self, x):
        self.x += x

    def compute(self):
        return self.x


class DiffMetric(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("x", tensor(0.0), dist_reduce_fx="sum")

    def update(self, x):
        self.x -= x

    def compute(self):
        return self.x


def test_metric_lightning(tmpdir):

    class TestModel(BoringModel):

        def __init__(self):
            super().__init__()
            self.metric = SumMetric()
            self.sum = 0.0

        def training_step(self, batch, batch_idx):
            x = batch
            self.metric(x.sum())
            self.sum += x.sum()

            return self.step(x)

        def training_epoch_end(self, outs):
            if not torch.allclose(self.sum, self.metric.compute()):
                raise ValueError('Sum and computed value must be equal')
            self.sum = 0.0
            self.metric.reset()

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.fit(model)


@pytest.mark.skipif(not _LIGHTNING_GREATER_EQUAL_1_3, reason='test requires lightning v1.3 or higher')
def test_metrics_reset(tmpdir):
    """Tests that metrics are reset correctly after the end of the train/val/test epoch.
    Taken from:
        https://github.com/PyTorchLightning/pytorch-lightning/pull/7055
    """

    class TestModel(LightningModule):

        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(32, 1)

            for stage in ['train', 'val', 'test']:
                acc = Accuracy()
                acc.reset = mock.Mock(side_effect=acc.reset)
                ap = AveragePrecision(num_classes=1, pos_label=1)
                ap.reset = mock.Mock(side_effect=ap.reset)
                self.add_module(f"acc_{stage}", acc)
                self.add_module(f"ap_{stage}", ap)

        def forward(self, x):
            return self.layer(x)

        def _step(self, stage, batch):
            labels = (batch.detach().sum(1) > 0).float()  # Fake some targets
            logits = self.forward(batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.unsqueeze(1))
            probs = torch.sigmoid(logits.detach())
            self.log(f"loss/{stage}", loss)

            acc = self._modules[f"acc_{stage}"]
            ap = self._modules[f"ap_{stage}"]

            labels_int = labels.to(torch.long)
            acc(probs.flatten(), labels_int)
            ap(probs.flatten(), labels_int)

            # Metric.forward calls reset so reset the mocks here
            acc.reset.reset_mock()
            ap.reset.reset_mock()

            self.log(f"{stage}/accuracy", acc)
            self.log(f"{stage}/ap", ap)

            return loss

        def training_step(self, batch, batch_idx, *args, **kwargs):
            return self._step('train', batch)

        def validation_step(self, batch, batch_idx, *args, **kwargs):
            return self._step('val', batch)

        def test_step(self, batch, batch_idx, *args, **kwargs):
            return self._step('test', batch)

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return [optimizer], [lr_scheduler]

        @staticmethod
        def train_dataloader():
            return DataLoader(RandomDataset(32, 64), batch_size=2)

        @staticmethod
        def val_dataloader():
            return DataLoader(RandomDataset(32, 64), batch_size=2)

        @staticmethod
        def test_dataloader():
            return DataLoader(RandomDataset(32, 64), batch_size=2)

        def _assert_epoch_end(self, stage):
            acc = self._modules[f"acc_{stage}"]
            ap = self._modules[f"ap_{stage}"]

            acc.reset.asset_not_called()
            ap.reset.assert_not_called()

        def train_epoch_end(self, outputs):
            self._assert_epoch_end('train')

        def validation_epoch_end(self, outputs):
            self._assert_epoch_end('val')

        def test_epoch_end(self, outputs):
            self._assert_epoch_end('test')

    def _assert_called(model, stage):
        acc = model._modules[f"acc_{stage}"]
        ap = model._modules[f"ap_{stage}"]

        acc.reset.assert_called_once()
        acc.reset.reset_mock()

        ap.reset.assert_called_once()
        ap.reset.reset_mock()

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        max_epochs=1,
        progress_bar_refresh_rate=0,
    )

    trainer.fit(model)
    _assert_called(model, 'train')
    _assert_called(model, 'val')

    trainer.validate(model)
    _assert_called(model, 'val')

    trainer.test(model)
    _assert_called(model, 'test')


# todo: reconsider if it make sense to keep here
# def test_metric_lightning_log(tmpdir):
#     """ Test logging a metric object and that the metric state gets reset after each epoch."""
#     class TestModel(BoringModel):
#         def __init__(self):
#             super().__init__()
#             self.metric_step = SumMetric()
#             self.metric_epoch = SumMetric()
#             self.sum = 0.0
#
#         def on_epoch_start(self):
#             self.sum = 0.0
#
#         def training_step(self, batch, batch_idx):
#             x = batch
#             self.metric_step(x.sum())
#             self.sum += x.sum()
#             self.log("sum_step", self.metric_step, on_epoch=True, on_step=False)
#             return {'loss': self.step(x), 'data': x}
#
#         def training_epoch_end(self, outs):
#             self.log("sum_epoch", self.metric_epoch(torch.stack([o['data'] for o in outs]).sum()))
#
#     model = TestModel()
#     model.val_dataloader = None
#
#     trainer = Trainer(
#         default_root_dir=tmpdir,
#         limit_train_batches=2,
#         limit_val_batches=2,
#         max_epochs=2,
#         log_every_n_steps=1,
#         weights_summary=None,
#     )
#     trainer.fit(model)
#
#     logged = trainer.logged_metrics
#     assert torch.allclose(tensor(logged["sum_step"]), model.sum)
#     assert torch.allclose(tensor(logged["sum_epoch"]), model.sum)

# todo: need to be fixed
# def test_scriptable(tmpdir):
#     class TestModel(BoringModel):
#         def __init__(self):
#             super().__init__()
#             # the metric is not used in the module's `forward`
#             # so the module should be exportable to TorchScript
#             self.metric = SumMetric()
#             self.sum = 0.0
#
#         def training_step(self, batch, batch_idx):
#             x = batch
#             self.metric(x.sum())
#             self.sum += x.sum()
#             self.log("sum", self.metric, on_epoch=True, on_step=False)
#             return self.step(x)
#
#     model = TestModel()
#     trainer = Trainer(
#         default_root_dir=tmpdir,
#         limit_train_batches=2,
#         limit_val_batches=2,
#         max_epochs=1,
#         log_every_n_steps=1,
#         weights_summary=None,
#         logger=False,
#         checkpoint_callback=False,
#     )
#     trainer.fit(model)
#     rand_input = torch.randn(10, 32)
#
#     script_model = model.to_torchscript()
#
#     # test that we can still do inference
#     output = model(rand_input)
#     script_output = script_model(rand_input)
#     assert torch.allclose(output, script_output)

# def test_metric_collection_lightning_log(tmpdir):
#
#     class TestModel(BoringModel):
#
#         def __init__(self):
#             super().__init__()
#             self.metric = MetricCollection([SumMetric(), DiffMetric()])
#             self.sum = 0.0
#             self.diff = 0.0
#
#         def training_step(self, batch, batch_idx):
#             x = batch
#             metric_vals = self.metric(x.sum())
#             self.sum += x.sum()
#             self.diff -= x.sum()
#             self.log_dict({f'{k}_step': v for k, v in metric_vals.items()})
#             return self.step(x)
#
#         def training_epoch_end(self, outputs):
#             metric_vals = self.metric.compute()
#             self.log_dict({f'{k}_epoch': v for k, v in metric_vals.items()})
#
#     model = TestModel()
#     model.val_dataloader = None
#
#     trainer = Trainer(
#         default_root_dir=tmpdir,
#         limit_train_batches=2,
#         limit_val_batches=2,
#         max_epochs=1,
#         log_every_n_steps=1,
#         weights_summary=None,
#     )
#     trainer.fit(model)
#
#     logged = trainer.logged_metrics
#     assert torch.allclose(tensor(logged["SumMetric_epoch"]), model.sum)
#     assert torch.allclose(tensor(logged["DiffMetric_epoch"]), model.diff)
