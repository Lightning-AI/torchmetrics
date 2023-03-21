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
from unittest import mock

import torch
from lightning_utilities import module_available
from torch import tensor
from torch.nn import Linear

if module_available("lightning"):
    from lightning import LightningModule, Trainer
else:
    from pytorch_lightning import LightningModule, Trainer

from integrations.helpers import no_warning_call
from integrations.lightning.boring_model import BoringModel
from torchmetrics import MetricCollection, SumMetric
from torchmetrics.classification import BinaryAccuracy, BinaryAveragePrecision


class DiffMetric(SumMetric):
    """DiffMetric inheritted from `SumMetric` by overidding its `update` method."""

    def update(self, value):
        """Update state."""
        super().update(-value)


def test_metric_lightning(tmpdir):
    """Test that including a metric inside a lightning module calculates a simple sum correctly."""

    class TestModel(BoringModel):
        def __init__(self) -> None:
            super().__init__()
            self.metric = SumMetric()
            self.sum = 0.0

        def training_step(self, batch, batch_idx):
            x = batch
            self.metric(x.sum())
            self.sum += x.sum()

            return self.step(x)

        def on_training_epoch_end(self):
            if not torch.allclose(self.sum, self.metric.compute()):
                raise ValueError("Sum and computed value must be equal")
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
    )
    trainer.fit(model)


def test_metrics_reset(tmpdir):
    """Tests that metrics are reset correctly after the end of the train/val/test epoch.

    Taken from: `Metric Test for Reset`_
    """

    class TestModel(BoringModel):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.Linear(32, 1)

            for stage in ["train", "val", "test"]:
                acc = BinaryAccuracy()
                acc.reset = mock.Mock(side_effect=acc.reset)
                ap = BinaryAveragePrecision()
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
            return self._step("train", batch)

        def validation_step(self, batch, batch_idx, *args, **kwargs):
            return self._step("val", batch)

        def test_step(self, batch, batch_idx, *args, **kwargs):
            return self._step("test", batch)

        def _assert_epoch_end(self, stage):
            acc = self._modules[f"acc_{stage}"]
            ap = self._modules[f"ap_{stage}"]

            acc.reset.asset_not_called()
            ap.reset.assert_not_called()

        def on_train_epoch_end(self):
            self._assert_epoch_end("train")

        def on_validation_epoch_end(self):
            self._assert_epoch_end("val")

        def on_test_epoch_end(self):
            self._assert_epoch_end("test")

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
    )

    trainer.fit(model)
    _assert_called(model, "train")
    _assert_called(model, "val")

    trainer.validate(model)
    _assert_called(model, "val")

    trainer.test(model)
    _assert_called(model, "test")


def test_metric_lightning_log(tmpdir):
    """Test logging a metric object and that the metric state gets reset after each epoch."""

    class TestModel(BoringModel):
        def __init__(self) -> None:
            super().__init__()
            self.metric_step = SumMetric()
            self.metric_epoch = SumMetric()
            self.sum = torch.tensor(0.0)
            self.outs = []

        def on_train_epoch_start(self):
            self.sum = torch.tensor(0.0)

        def training_step(self, batch, batch_idx):
            x = batch
            self.metric_step(x.sum())
            self.sum += x.sum()
            self.log("sum_step", self.metric_step, on_epoch=True, on_step=False)
            self.outs.append(x)
            return self.step(x)

        def on_train_epoch_end(self):
            self.log("sum_epoch", self.metric_epoch(torch.stack(self.outs)))
            self.outs = []

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=0,
        max_epochs=2,
        log_every_n_steps=1,
    )
    with no_warning_call(
        UserWarning,
        match="Torchmetrics v0.9 introduced a new argument class property called.*",
    ):
        trainer.fit(model)

    logged = trainer.logged_metrics
    assert torch.allclose(tensor(logged["sum_step"]), model.sum, atol=2e-4)
    assert torch.allclose(tensor(logged["sum_epoch"]), model.sum, atol=2e-4)


def test_metric_collection_lightning_log(tmpdir):
    """Test that MetricCollection works with Lightning modules."""

    class TestModel(BoringModel):
        def __init__(self) -> None:
            super().__init__()
            self.metric = MetricCollection([SumMetric(), DiffMetric()])
            self.sum = torch.tensor(0.0)
            self.diff = torch.tensor(0.0)

        def training_step(self, batch, batch_idx):
            x = batch
            metric_vals = self.metric(x.sum())
            self.sum += x.sum()
            self.diff -= x.sum()
            self.log_dict({f"{k}_step": v for k, v in metric_vals.items()})
            return self.step(x)

        def on_train_epoch_end(self):
            metric_vals = self.metric.compute()
            self.log_dict({f"{k}_epoch": v for k, v in metric_vals.items()})

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=0,
        max_epochs=1,
        log_every_n_steps=1,
    )
    with no_warning_call(
        UserWarning,
        match="Torchmetrics v0.9 introduced a new argument class property called.*",
    ):
        trainer.fit(model)

    logged = trainer.logged_metrics
    assert torch.allclose(tensor(logged["SumMetric_epoch"]), model.sum, atol=2e-4)
    assert torch.allclose(tensor(logged["DiffMetric_epoch"]), model.diff, atol=2e-4)


def test_scriptable(tmpdir):
    """Test that lightning modules can still be scripted even if metrics cannot."""

    class TestModel(BoringModel):
        def __init__(self) -> None:
            super().__init__()
            # the metric is not used in the module's `forward`
            # so the module should be exportable to TorchScript
            self.metric = SumMetric()
            self.sum = torch.tensor(0.0)

        def training_step(self, batch, batch_idx):
            x = batch
            self.metric(x.sum())
            self.sum += x.sum()
            self.log("sum", self.metric, on_epoch=True, on_step=False)
            return self.step(x)

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        logger=False,
    )
    trainer.fit(model)
    rand_input = torch.randn(10, 32)

    script_model = model.to_torchscript()

    # test that we can still do inference
    output = model(rand_input)
    script_output = script_model(rand_input)
    assert torch.allclose(output, script_output)


def test_dtype_in_pl_module_transfer(tmpdir):
    """Test that metric states don't change dtype when .half() or .float() is called on the LightningModule."""

    class BoringModel(LightningModule):
        def __init__(self, metric_dtype=torch.float32) -> None:
            super().__init__()
            self.layer = Linear(32, 32)
            self.metric = SumMetric()
            self.metric.set_dtype(metric_dtype)

        def forward(self, x):
            return self.layer(x)

        def training_step(self, batch, batch_idx):
            pred = self.forward(batch)
            loss = self(batch).sum()
            self.metric.update(torch.flatten(pred), torch.flatten(batch))

            return {"loss": loss}

        def configure_optimizers(self):
            return torch.optim.SGD(self.layer.parameters(), lr=0.1)

    model = BoringModel()
    assert model.metric.value.dtype == torch.float32
    model = model.half()
    assert model.metric.value.dtype == torch.float32

    model = BoringModel()
    assert model.metric.value.dtype == torch.float32
    model = model.double()
    assert model.metric.value.dtype == torch.float32

    model = BoringModel(metric_dtype=torch.float16)
    assert model.metric.value.dtype == torch.float16
    model = model.float()
    assert model.metric.value.dtype == torch.float16

    model = BoringModel()
    assert model.metric.value.dtype == torch.float32

    model = model.type(torch.half)
    assert model.metric.value.dtype == torch.float32
