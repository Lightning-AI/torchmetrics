.. testsetup:: *

    import torch
    from torch.nn import Module
    from pytorch_lightning.core.lightning import LightningModule
    from torchmetrics import Metric

#################################
TorchMetrics in PyTorch Lightning
#################################

TorchMetrics was originaly created as part of `PyTorch Lightning <https://github.com/PyTorchLightning/pytorch-lightning>`_, a powerful deep learning research framework designed for scaling models without boilerplate.

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
            self.log('train_acc_step', self.accuracy(preds, y))
            ...

        def training_epoch_end(self, outs):
            # log epoch metric
            self.log('train_acc_epoch', self.accuracy.compute())

********************
Logging TorchMetrics
********************

:class:`~torchmetrics.Metric` objects can also be directly logged in Lightning using the LightningModule `self.log <https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#logging-from-a-lightningmodule>`_ method. Lightning will log
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
                return {'loss' : loss, 'preds' : preds, 'target' : target}

            def training_step_end(self, outputs):
                #update and log
                self.metric(outputs['preds'], outputs['target'])
                self.log('metric', self.metric)

For more details see `Lightning Docs <https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#logging-from-a-lightningmodule>`_
