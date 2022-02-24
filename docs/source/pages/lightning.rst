.. testsetup:: *

    import torch
    from torch.nn import Module
    from pytorch_lightning.core.lightning import LightningModule
    from torchmetrics import Metric

#################################
TorchMetrics in PyTorch Lightning
#################################

TorchMetrics was originaly created as part of `PyTorch Lightning <https://github.com/PyTorchLightning/pytorch-lightning>`_, a powerful deep learning research framework designed for scaling models without boilerplate.

..note::

    TorchMetrics always offers compatibility with the last 2 major PyTorch Lightning versions, but we recommend to always keep both frameworks
    up-to-date for the best experience.

While TorchMetrics was built to be used with native PyTorch, using TorchMetrics with Lightning offers additional benefits:

* Module metrics are automatically placed on the correct device when properly defined inside a LightningModule. This means that your data will always be placed on the same device as your metrics.
* Native support for logging metrics in Lightning using `self.log <https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#logging-from-a-lightningmodule>`_ inside your LightningModule.
* The ``.reset()`` method of the metric will automatically be called at the end of an epoch.

The example below shows how to use a metric in your `LightningModule <https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html>`_:

.. testcode:: python

    class MyModel(LightningModule):

        def __init__(self):
            ...
            self.accuracy = torchmetrics.Accuracy()

        def training_step(self, batch, batch_idx):
            x, y = batch
            preds = self(x)
            ...
            # log step metric
            self.accuracy(preds, y)
            self.log('train_acc_step', self.accuracy)
            ...

        def training_epoch_end(self, outs):
            # log epoch metric
            self.log('train_acc_epoch', self.accuracy)

.. note::

    ``self.log`` in Lightning only supports logging of *scalar-tensors*. While the vast majority of metrics in torchmetrics returns a scalar tensor, some metrics such as
    :class:`~torchmetrics.ConfusionMatrix`, :class:`~torchmetrics.ROC`, :class:`~torchmetrics.MAP`, :class:`~torchmetrics.RougeScore` return outputs that are non-scalar
    tensors (often dicts or list of tensors) and should therefore be dealt with separately. For info about the return type and shape please look at the documentation for
    the ``compute`` method for each metric you want to log.

.. note::

    Modular metrics contain internal states that should belong to only one DataLoader. In case you are using multiple DataLoaders,
    it is recommended to initialize a separate modular metric instances for each DataLoader and use them separately.

    .. testcode:: python

        class MyModule(LightningModule):

            def __init__(self):
                ...
                self.val_acc = nn.ModuleList([torchmetrics.Accuracy() for _ in range(2)])

            def val_dataloader(self):
                return [DataLoader(...), DataLoader(...)]

            def validation_step(self, batch, batch_idx, dataloader_idx):
                x, y = batch
                preds = self(x)
                ...
                self.val_acc[dataloader_idx](preds, y)
                self.log('val_acc', self.val_acc[dataloader_idx])


********************
Logging TorchMetrics
********************

When :class:`~torchmetrics.Metric` objects, which return a scalar tensor,  are logged directly in Lightning using the LightningModule `self.log <https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#logging-from-a-lightningmodule>`_ method. Lightning will log
the metric based on ``on_step`` and ``on_epoch`` flags present in ``self.log(...)``.
If ``on_epoch`` is True, the logger automatically logs the end of epoch metric value by calling
``.compute()``.

.. note::

    ``sync_dist``, ``sync_dist_op``, ``sync_dist_group``, ``reduce_fx`` and ``tbptt_reduce_fx``
    flags from ``self.log(...)`` don't affect the metric logging in any manner. The metric class
    contains its own distributed synchronization logic.

    This however is only true for metrics that inherit the base class ``Metric``,
    and thus the functional metric API provides no support for in-built distributed synchronization
    or reduction functions.


.. testcode:: python

    class MyModule(LightningModule):

        def __init__(self):
            ...
            self.train_acc = torchmetrics.Accuracy()
            self.valid_acc = torchmetrics.Accuracy()

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

.. note::

    The ``.reset()`` method of the metric will automatically be called at the end of an epoch within lightning only if you pass
    the metric instance inside `self.log <https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#logging-from-a-lightningmodule>`_.
    Also if you are calling ``.compute`` by yourself, you need to call the ``.reset()`` too.

    .. testcode:: python

        class MyModule(LightningModule):

            def __init__(self):
                ...
                self.train_acc = torchmetrics.Accuracy()
                self.train_precision = torchmetrics.Precision()

            def training_step(self, batch, batch_idx):
                x, y = batch
                preds = self(x)
                ...

                # this will reset the metric automatically at the epoch end
                self.train_acc(preds, y)
                self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)

                # this will not reset the metric automatically at the epoch end
                precision = self.train_precision(preds, y)
                self.log('train_precision', precision, on_step=True, on_epoch=False)

            def training_epoch_end(self, outputs):
                # this will compute and reset the metric automatically at the epoch end
                self.log('train_epoch_accuracy', self.training_acc)

                # this will not reset the metric automatically at the epoch end so you
                # need to call it yourself
                mean_precision = self.precision.compute()
                self.log('train_epoch_precision', mean_precision)
                self.precision.reset()


.. note::

    If using metrics in data parallel mode (dp), the metric update/logging should be done
    in the ``<mode>_step_end`` method (where ``<mode>`` is either ``training``, ``validation``
    or ``test``). This is due to metric states else being destroyed after each forward pass,
    leading to wrong accumulation. In practice do the following:

    .. testcode:: python

        class MyModule(LightningModule):

            def training_step(self, batch, batch_idx):
                data, target = batch
                preds = self(data)
                # ...
                return {'loss': loss, 'preds': preds, 'target': target}

            def training_step_end(self, outputs):
                #update and log
                self.metric(outputs['preds'], outputs['target'])
                self.log('metric', self.metric)

.. note::

    When using any Modular metric, calling ``self.metric(...)`` or ``self.metric.forward(...)`` serves the dual purpose of calling ``self.metric.update()``
    on its input and simultaneously returning the metric value over the provided input. So if you are logging a metric *only* on epoch-level, it
    is recommended to call ``self.metric.update()`` directly to avoid the extra computation.

    .. testcode:: python

        class MyModule(LightningModule):

            def __init__(self):
                ...
                self.valid_acc = torchmetrics.Accuracy()

            def validation_step(self, batch, batch_idx):
                logits = self(x)
                ...
                self.valid_acc.update(logits, y)
                self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)

For more details see `Lightning Docs <https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#logging-from-a-lightningmodule>`_
