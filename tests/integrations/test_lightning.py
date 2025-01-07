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
    from lightning.pytorch import LightningModule, Trainer, seed_everything
    from lightning.pytorch.loggers import CSVLogger
else:
    from pytorch_lightning import LightningModule, Trainer, seed_everything
    from pytorch_lightning.loggers import CSVLogger

from integrations.lightning.boring_model import BoringModel
from torchmetrics import MetricCollection
from torchmetrics.aggregation import SumMetric
from torchmetrics.classification import BinaryAccuracy, BinaryAveragePrecision, MulticlassAccuracy
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchmetrics.utilities.prints import rank_zero_only
from torchmetrics.wrappers import ClasswiseWrapper, MinMaxMetric, MultitaskWrapper

seed_everything(42)


class DiffMetric(SumMetric):
    """DiffMetric inherited from `SumMetric` by overriding its `update` method."""

    def update(self, value):
        """Update state."""
        super().update(-value)


def test_metric_lightning(tmpdir):
    """Test that including a metric inside a lightning module calculates a simple sum correctly."""

    class TestModel(BoringModel):
        def __init__(self) -> None:
            super().__init__()
            self.metric = SumMetric()
            self.register_buffer("sum", torch.tensor(0.0))

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

        def training_step(self, batch, batch_idx):
            return self._step("train", batch)

        def validation_step(self, batch, batch_idx):
            return self._step("val", batch)

        def test_step(self, batch, batch_idx):
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

            # initialize one metric for every combination of `on_step` and `on_epoch` and `forward` and `update`
            self.metric_update = SumMetric()
            self.metric_update_step = SumMetric()
            self.metric_update_epoch = SumMetric()

            self.metric_forward = SumMetric()
            self.metric_forward_step = SumMetric()
            self.metric_forward_epoch = SumMetric()

            self.compo_update = SumMetric() + SumMetric()
            self.compo_update_step = SumMetric() + SumMetric()
            self.compo_update_epoch = SumMetric() + SumMetric()

            self.compo_forward = SumMetric() + SumMetric()
            self.compo_forward_step = SumMetric() + SumMetric()
            self.compo_forward_epoch = SumMetric() + SumMetric()

            self.sum = []

        def training_step(self, batch, batch_idx):
            x = batch
            s = x.sum()

            for metric in [self.metric_update, self.metric_update_step, self.metric_update_epoch]:
                metric.update(s)
            for metric in [self.metric_forward, self.metric_forward_step, self.metric_forward_epoch]:
                _ = metric(s)
            for metric in [self.compo_update, self.compo_update_step, self.compo_update_epoch]:
                metric.update(s)
            for metric in [self.compo_forward, self.compo_forward_step, self.compo_forward_epoch]:
                _ = metric(s)

            self.sum.append(s)

            self.log("metric_update", self.metric_update)
            self.log("metric_update_step", self.metric_update_step, on_epoch=False, on_step=True)
            self.log("metric_update_epoch", self.metric_update_epoch, on_epoch=True, on_step=False)

            self.log("metric_forward", self.metric_forward)
            self.log("metric_forward_step", self.metric_forward_step, on_epoch=False, on_step=True)
            self.log("metric_forward_epoch", self.metric_forward_epoch, on_epoch=True, on_step=False)

            self.log("compo_update", self.compo_update)
            self.log("compo_update_step", self.compo_update_step, on_epoch=False, on_step=True)
            self.log("compo_update_epoch", self.compo_update_epoch, on_epoch=True, on_step=False)

            self.log("compo_forward", self.compo_forward)
            self.log("compo_forward_step", self.compo_forward_step, on_epoch=False, on_step=True)
            self.log("compo_forward_epoch", self.compo_forward_epoch, on_epoch=True, on_step=False)

            return self.step(x)

    model = TestModel()

    class CustomCSVLogger(CSVLogger):
        """Custom CSVLogger that does not call `experiment.save()` to prevent state being reset."""

        @rank_zero_only
        def save(self) -> None:
            pass

    logger = CustomCSVLogger("tmpdir/logs")
    # is_cuda = torch.cuda.is_available()
    # cuda_extra = {"devices": int(is_cuda)} if is_cuda else {}
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=0,
        max_epochs=2,
        log_every_n_steps=1,
        logger=logger,
        # **cuda_extra,
    )
    trainer.fit(model)

    logged_metrics = logger.experiment.metrics

    epoch_0_step_0 = logged_metrics[0]
    assert "metric_forward" in epoch_0_step_0
    assert epoch_0_step_0["metric_forward"] == model.sum[0]
    assert "metric_forward_step" in epoch_0_step_0
    assert epoch_0_step_0["metric_forward_step"] == model.sum[0]
    assert "compo_forward" in epoch_0_step_0
    assert epoch_0_step_0["compo_forward"] == 2 * model.sum[0]
    assert "compo_forward_step" in epoch_0_step_0
    assert epoch_0_step_0["compo_forward_step"] == 2 * model.sum[0]

    epoch_0_step_1 = logged_metrics[1]
    assert "metric_forward" in epoch_0_step_1
    assert epoch_0_step_1["metric_forward"] == model.sum[1]
    assert "metric_forward_step" in epoch_0_step_1
    assert epoch_0_step_1["metric_forward_step"] == model.sum[1]
    assert "compo_forward" in epoch_0_step_1
    assert epoch_0_step_1["compo_forward"] == 2 * model.sum[1]
    assert "compo_forward_step" in epoch_0_step_1
    assert epoch_0_step_1["compo_forward_step"] == 2 * model.sum[1]

    epoch_0 = logged_metrics[2]
    assert "metric_update_epoch" in epoch_0
    assert epoch_0["metric_update_epoch"] == sum([model.sum[0], model.sum[1]])
    assert "metric_forward_epoch" in epoch_0
    assert epoch_0["metric_forward_epoch"] == sum([model.sum[0], model.sum[1]])
    assert "compo_update_epoch" in epoch_0
    assert epoch_0["compo_update_epoch"] == 2 * sum([model.sum[0], model.sum[1]])
    assert "compo_forward_epoch" in epoch_0
    assert epoch_0["compo_forward_epoch"] == 2 * sum([model.sum[0], model.sum[1]])

    epoch_1_step_0 = logged_metrics[3]
    assert "metric_forward" in epoch_1_step_0
    assert epoch_1_step_0["metric_forward"] == model.sum[2]
    assert "metric_forward_step" in epoch_1_step_0
    assert epoch_1_step_0["metric_forward_step"] == model.sum[2]
    assert "compo_forward" in epoch_1_step_0
    assert epoch_1_step_0["compo_forward"] == 2 * model.sum[2]
    assert "compo_forward_step" in epoch_1_step_0
    assert epoch_1_step_0["compo_forward_step"] == 2 * model.sum[2]

    epoch_1_step_1 = logged_metrics[4]
    assert "metric_forward" in epoch_1_step_1
    assert epoch_1_step_1["metric_forward"] == model.sum[3]
    assert "metric_forward_step" in epoch_1_step_1
    assert epoch_1_step_1["metric_forward_step"] == model.sum[3]
    assert "compo_forward" in epoch_1_step_1
    assert epoch_1_step_1["compo_forward"] == 2 * model.sum[3]
    assert "compo_forward_step" in epoch_1_step_1
    assert epoch_1_step_1["compo_forward_step"] == 2 * model.sum[3]

    epoch_1 = logged_metrics[5]
    assert "metric_update_epoch" in epoch_1
    assert epoch_1["metric_update_epoch"] == sum([model.sum[2], model.sum[3]])
    assert "metric_forward_epoch" in epoch_1
    assert epoch_1["metric_forward_epoch"] == sum([model.sum[2], model.sum[3]])
    assert "compo_update_epoch" in epoch_1
    assert epoch_1["compo_update_epoch"] == 2 * sum([model.sum[2], model.sum[3]])
    assert "compo_forward_epoch" in epoch_1
    assert epoch_1["compo_forward_epoch"] == 2 * sum([model.sum[2], model.sum[3]])


def test_metric_collection_lightning_log(tmpdir):
    """Test that MetricCollection works with Lightning modules."""

    class TestModel(BoringModel):
        def __init__(self) -> None:
            super().__init__()
            self.metric = MetricCollection([SumMetric(), DiffMetric()])
            self.register_buffer("sum", torch.tensor(0.0))
            self.register_buffer("diff", torch.tensor(0.0))

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
    # is_cuda = torch.cuda.is_available()
    # cuda_extra = {"devices": int(is_cuda)} if is_cuda else {}

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=0,
        max_epochs=1,
        log_every_n_steps=1,
        # **cuda_extra,
    )
    trainer.fit(model)

    logged = trainer.logged_metrics
    assert torch.allclose(tensor(logged["SumMetric_epoch"]), model.sum, atol=2e-4)
    assert torch.allclose(tensor(logged["DiffMetric_epoch"]), model.diff, atol=2e-4)


def test_task_wrapper_lightning_logging(tmpdir):
    """Test that MultiTaskWrapper works with Lightning modules."""

    class TestModel(BoringModel):
        def __init__(self) -> None:
            super().__init__()
            self.multitask = MultitaskWrapper({"classification": BinaryAccuracy(), "regression": MeanSquaredError()})
            self.multitask_collection = MultitaskWrapper({
                "classification": MetricCollection([BinaryAccuracy(), BinaryAveragePrecision()]),
                "regression": MetricCollection([MeanSquaredError(), MeanAbsoluteError()]),
            })

            self.accuracy = BinaryAccuracy()
            self.mse = MeanSquaredError()

        def training_step(self, batch, batch_idx):
            preds = torch.rand(10)
            target = torch.rand(10)
            self.multitask(
                {"classification": preds, "regression": preds},
                {"classification": target.round().int(), "regression": target},
            )
            self.multitask_collection(
                {"classification": preds, "regression": preds},
                {"classification": target.round().int(), "regression": target},
            )
            self.accuracy(preds.round(), target.round())
            self.mse(preds, target)
            self.log("accuracy", self.accuracy, on_epoch=True)
            self.log("mse", self.mse, on_epoch=True)
            self.log_dict(self.multitask, on_epoch=True)
            self.log_dict(self.multitask_collection, on_epoch=True)
            return self.step(batch)

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=0,
        max_epochs=1,
        log_every_n_steps=1,
    )
    trainer.fit(model)

    logged = trainer.logged_metrics
    assert torch.allclose(logged["accuracy_step"], logged["classification_step"])
    assert torch.allclose(logged["accuracy_epoch"], logged["classification_epoch"])
    assert torch.allclose(logged["mse_step"], logged["regression_step"])
    assert torch.allclose(logged["mse_epoch"], logged["regression_epoch"])
    assert "regression_MeanAbsoluteError_epoch" in logged
    assert "regression_MeanSquaredError_epoch" in logged
    assert "classification_BinaryAccuracy_epoch" in logged
    assert "classification_BinaryAveragePrecision_epoch" in logged


def test_scriptable(tmpdir):
    """Test that lightning modules can still be scripted even if metrics cannot."""

    class TestModel(BoringModel):
        def __init__(self) -> None:
            super().__init__()
            # the metric is not used in the module's `forward`
            # so the module should be exportable to TorchScript
            self.metric = SumMetric()
            self.register_buffer("sum", torch.tensor(0.0))

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
    assert model.metric.sum_value.dtype == torch.float32
    model = model.half()
    assert model.metric.sum_value.dtype == torch.float32

    model = BoringModel()
    assert model.metric.sum_value.dtype == torch.float32
    model = model.double()
    assert model.metric.sum_value.dtype == torch.float32

    model = BoringModel(metric_dtype=torch.float16)
    assert model.metric.sum_value.dtype == torch.float16
    model = model.float()
    assert model.metric.sum_value.dtype == torch.float16

    model = BoringModel()
    assert model.metric.sum_value.dtype == torch.float32

    model = model.type(torch.half)
    assert model.metric.sum_value.dtype == torch.float32


def test_collection_classwise_lightning_integration(tmpdir):
    """Check the integration of ClasswiseWrapper, MetricCollection and LightningModule.

    See issue: https://github.com/Lightning-AI/torchmetrics/issues/2683

    """

    class TestModel(BoringModel):
        def __init__(self) -> None:
            super().__init__()
            self.train_metrics = MetricCollection(
                {
                    "macro_accuracy": MulticlassAccuracy(num_classes=5, average="macro"),
                    "classwise_accuracy": ClasswiseWrapper(MulticlassAccuracy(num_classes=5, average=None)),
                },
                prefix="train_",
            )
            self.val_metrics = self.train_metrics.clone(prefix="val_")

        def training_step(self, batch, batch_idx):
            loss = self(batch).sum()
            preds = torch.randint(0, 5, (100,), device=batch.device)
            target = torch.randint(0, 5, (100,), device=batch.device)

            batch_values = self.train_metrics(preds, target)
            self.log_dict(batch_values, on_step=True, on_epoch=False)
            return {"loss": loss}

        def validation_step(self, batch, batch_idx):
            preds = torch.randint(0, 5, (100,), device=batch.device)
            target = torch.randint(0, 5, (100,), device=batch.device)
            self.val_metrics.update(preds, target)

        def on_validation_epoch_end(self):
            self.log_dict(self.val_metrics.compute(), on_step=False, on_epoch=True)
            self.val_metrics.reset()

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
    )
    trainer.fit(model)

    logged = trainer.logged_metrics

    # check that all metrics are logged
    assert "train_macro_accuracy" in logged
    assert "val_macro_accuracy" in logged
    for i in range(5):
        assert f"train_multiclassaccuracy_{i}" in logged
        assert f"val_multiclassaccuracy_{i}" in logged


def test_collection_minmax_lightning_integration(tmpdir):
    """Check the integration of MinMaxWrapper, MetricCollection and LightningModule.

    See issue: https://github.com/Lightning-AI/torchmetrics/issues/2763

    """

    class TestModel(BoringModel):
        def __init__(self) -> None:
            super().__init__()
            self.train_metrics = MetricCollection(
                {
                    "macro_accuracy": MinMaxMetric(MulticlassAccuracy(num_classes=5, average="macro")),
                    "weighted_accuracy": MinMaxMetric(MulticlassAccuracy(num_classes=5, average="weighted")),
                },
                prefix="train_",
            )
            self.val_metrics = self.train_metrics.clone(prefix="val_")

        def training_step(self, batch, batch_idx):
            loss = self(batch).sum()
            preds = torch.randint(0, 5, (100,), device=batch.device)
            target = torch.randint(0, 5, (100,), device=batch.device)

            batch_values = self.train_metrics(preds, target)
            self.log_dict(batch_values, on_step=True, on_epoch=False)
            return {"loss": loss}

        def validation_step(self, batch, batch_idx):
            preds = torch.randint(0, 5, (100,), device=batch.device)
            target = torch.randint(0, 5, (100,), device=batch.device)
            self.val_metrics.update(preds, target)

        def on_validation_epoch_end(self):
            self.log_dict(self.val_metrics.compute(), on_step=False, on_epoch=True)
            self.val_metrics.reset()

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
    )
    trainer.fit(model)

    logged = trainer.logged_metrics

    # check that all metrics are logged
    for prefix in ["train_", "val_"]:
        for metric in ["macro_accuracy", "weighted_accuracy"]:
            for key in ["max", "min", "raw"]:
                assert f"{prefix}{metric}_{key}" in logged
