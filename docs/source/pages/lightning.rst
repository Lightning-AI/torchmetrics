.. testsetup:: *

    import torch
    from torch.nn import Module
    from lightning.pytorch import LightningModule
    from torchmetrics import Metric

#################################
TorchMetrics in PyTorch Lightning
#################################

TorchMetrics was originally created as part of `PyTorch Lightning <https://github.com/Lightning-AI/pytorch-lightning>`_, a powerful deep learning research
framework designed for scaling models without boilerplate.

.. note::

    TorchMetrics always offers compatibility with the last 2 major PyTorch Lightning versions, but we recommend to always keep both frameworks
    up-to-date for the best experience.

While TorchMetrics was built to be used with native PyTorch, using TorchMetrics with Lightning offers additional benefits:

* Modular metrics are automatically placed on the correct device when properly defined inside a LightningModule.
  This means that your data will always be placed on the same device as your metrics. No need to call ``.to(device)`` anymore!
* Native support for logging metrics in Lightning using
  `self.log <https://lightning.ai/docs/pytorch/stable/extensions/logging.html#logging-from-a-lightningmodule>`_ inside
  your LightningModule.
* The ``.reset()`` method of the metric will automatically be called at the end of an epoch.

The example below shows how to use a metric in your `LightningModule <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html>`_:

.. testcode:: python

    class MyModel(LightningModule):

        def __init__(self, num_classes):
            ...
            self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)

        def training_step(self, batch, batch_idx):
            x, y = batch
            preds = self(x)
            ...
            # log step metric
            self.accuracy(preds, y)
            self.log('train_acc_step', self.accuracy)
            ...

        def on_train_epoch_end(self):
            # log epoch metric
            self.log('train_acc_epoch', self.accuracy)

Metric logging in Lightning happens through the ``self.log`` or ``self.log_dict`` method. Both methods only support the
logging of *scalar-tensors*. While the vast majority of metrics in torchmetrics returns a scalar tensor, some metrics
such as :class:`~torchmetrics.classification.confusion_matrix.ConfusionMatrix`,
:class:`~torchmetrics.classification.roc.ROC`,
:class:`~torchmetrics.detection.mean_ap.MeanAveragePrecision`, :class:`~torchmetrics.text.rouge.ROUGEScore` return
outputs that are non-scalar tensors (often dicts or list of tensors) and should therefore be dealt with separately.
For info about the return type and shape please look at the documentation for the ``compute`` method for each metric
you want to log.

********************
Logging TorchMetrics
********************

Logging metrics can be done in two ways: either logging the metric object directly or the computed metric values.
When :class:`~torchmetrics.Metric` objects, which return a scalar tensor are logged directly in Lightning using the
LightningModule `self.log <https://lightning.ai/docs/pytorch/stable/extensions/logging.html#logging-from-a-lightningmodule>`_
method, Lightning will log the metric based on ``on_step`` and ``on_epoch`` flags present in ``self.log(...)``. If
``on_epoch`` is True, the logger automatically logs the end of epoch metric value by calling ``.compute()``.

.. note::

    ``sync_dist``, ``sync_dist_group`` and ``reduce_fx`` flags from ``self.log(...)`` don't affect the metric logging
    in any manner. The metric class contains its own distributed synchronization logic.

    This however is only true for metrics that inherit the base class ``Metric``,
    and thus the functional metric API provides no support for in-built distributed synchronization
    or reduction functions.


.. testcode:: python

    class MyModule(LightningModule):

        def __init__(self, num_classes):
            ...
            self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
            self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)

        def training_step(self, batch, batch_idx):
            x, y = batch
            preds = self(x)
            ...
            self.train_acc(preds, y)
            self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)

        def validation_step(self, batch, batch_idx):
            logits = self(x)
            ...
            self.valid_acc(logits, y)
            self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)

As an alternative to logging the metric object and letting Lightning take care of when to reset the metric etc. you can
also manually log the output of the metrics.

.. testcode:: python

    class MyModule(LightningModule):

        def __init__(self):
            ...
            self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
            self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)

        def training_step(self, batch, batch_idx):
            x, y = batch
            preds = self(x)
            ...
            batch_value = self.train_acc(preds, y)
            self.log('train_acc_step', batch_value)

        def on_train_epoch_end(self):
            self.train_acc.reset()

        def validation_step(self, batch, batch_idx):
            logits = self(x)
            ...
            self.valid_acc.update(logits, y)

        def on_validation_epoch_end(self, outputs):
            self.log('valid_acc_epoch', self.valid_acc.compute())
            self.valid_acc.reset()

Note that logging metrics this way will require you to manually reset the metrics at the end of the epoch yourself.
In general, we recommend logging the metric object to make sure that metrics are correctly computed and reset.
Additionally, we highly recommend that the two ways of logging are not mixed as it can lead to wrong results.

.. note::

    When using any Modular metric, calling ``self.metric(...)`` or ``self.metric.forward(...)`` serves the dual purpose
    of calling ``self.metric.update()`` on its input and simultaneously returning the metric value over the provided
    input. So if you are logging a metric *only* on epoch-level (as in the example above), it is recommended to call
    ``self.metric.update()`` directly to avoid the extra computation.

    .. testcode:: python

        class MyModule(LightningModule):

            def __init__(self, num_classes):
                ...
                self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)

            def validation_step(self, batch, batch_idx):
                logits = self(x)
                ...
                self.valid_acc.update(logits, y)
                self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)


***************
Common Pitfalls
***************

The following contains a list of pitfalls to be aware of:

* Modular metrics contain internal states that should belong to only one DataLoader. In case you are using multiple DataLoaders,
  it is recommended to initialize a separate modular metric instances for each DataLoader and use them separately. The same holds
  for using separate metrics for training, validation and testing.

.. testcode:: python

    class MyModule(LightningModule):

        def __init__(self, num_classes):
            ...
            self.val_acc = nn.ModuleList(
                [torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes) for _ in range(2)]
            )

        def val_dataloader(self):
            return [DataLoader(...), DataLoader(...)]

        def validation_step(self, batch, batch_idx, dataloader_idx):
            x, y = batch
            preds = self(x)
            ...
            self.val_acc[dataloader_idx](preds, y)
            self.log('val_acc', self.val_acc[dataloader_idx])

* Mixing the two logging methods by calling ``self.log("val", self.metric)`` in ``{training|validation|test}_step``
  method and then calling ``self.log("val", self.metric.compute())`` in the corresponding
  ``on_{train|validation|test}_epoch_end`` method.
  Because the object is logged in the first case, Lightning will reset the metric before calling the second line leading
  to errors or nonsense results.

* Calling ``self.log("val", self.metric(preds, target))`` with the intention of logging the metric object. Because
  ``self.metric(preds, target)`` corresponds to calling the forward method, this will return a tensor and not the
  metric object. Such logging will be wrong in this case. Instead, it is essential to separate into several lines:

.. testcode:: python

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        ...
        # log step metric
        self.accuracy(preds, y)  # compute metrics
        self.log('train_acc_step', self.accuracy)  # log metric object

* Using :class:`~torchmetrics.wrappers.MetricTracker` wrapper with Lightning is a special case, because the wrapper in itself is not a metric
  i.e. it does not inherit from the base :class:`~torchmetrics.Metric` class but instead from :class:`~torch.nn.ModuleList`. Thus,
  to log the output of this metric one needs to manually log the returned values (not the object) using ``self.log``
  and for epoch level logging this should be done in the appropriate ``on_{train|validation|test}_epoch_end`` method.
