###########
Quick Start
###########

TorchMetrics is a collection of PyTorch metrics implementaions and an easy to use API to create custom metrics.
It is designed to be distrubuted-training compatible and offers:

* Automatic accumulation over batches
* Automatic synchronization between multiple devices

You can use TorchMetrics in any PyTorch model, or with in `PyTorch Lightning <https://pytorch-lightning.readthedocs.io/en/stable/>`_ to enjoy additional features:

* This means that your data will always be placed on the same device as your metrics.
* Native support for logging metrics in Lightning using self.log inside your ``LightningModule``. Lightning will log the metric based on on_step and on_epoch flags present in ``self.log(…)``. If ``on_epoch=True``, the logger automatically logs the end of epoch metric value by calling ``.compute()``.
* The ``.reset()`` method of the metric will automatically be called at the end of an epoch.

Install
*******

You can install TorchMetrics using pip or conda:

.. code-block:: bash

    pip install torchmetrics


Using TorchMetrics
******************

Functional metrics
~~~~~~~~~~~~~~~~~~

Similar to `torch.nn <https://pytorch.org/docs/stable/nn>`_, most metrics have both a class-based and a functional version. 
The functional versions implement the basic operations required for computing each metric. 
They are simple python functions that as input take `torch.tensors <https://pytorch.org/docs/stable/tensors.html>`_ 
and return the corresponding metric as a ``torch.tensor``. 
The code-snippet below shows a simple example for calculating the accuracy using the functional interface:

.. testcode::

    import torch
    # import our library
    import torchmetrics

    # simulate a classification problem
    preds = torch.randn(10, 5).softmax(dim=-1)
    target = torch.randint(5, (10,))

    acc = torchmetrics.functional.accuracy(preds, target)

Module metrics
~~~~~~~~~~~~~~

Nearly all functional metrics have a corresponding class-based metric that calls it a functional counterpart underneath. The class-based metrics are characterized by having one or more internal metrics states (similar to the parameters of the PyTorch module) that allow them to offer additional functionalities:

* Accumulation of multiple batches
* Automatic synchronization between multiple devices
* Metric arithmetic

The code below shows how to use the class-based interface:

.. testcode::

    import torch
    # import our library
    import torchmetrics

    # initialize metric
    metric = torchmetrics.Accuracy()

    n_batches = 10
    for i in range(n_batches):
        # simulate a classification problem
        preds = torch.randn(10, 5).softmax(dim=-1)
        target = torch.randint(5, (10,))
        # metric on current batch
        acc = metric(preds, target)
        print(f"Accuracy on batch {i}: {acc}")

    # metric on all batches using custom accumulation
    acc = metric.compute()
    print(f"Accuracy on all data: {acc}")

    # Reseting internal state such that metric ready for new data
    metric.reset()

.. testoutput::
   :hide:
   :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

    Accuracy on batch ...


Implementing your own metric
****************************

Implementing your own metric is as easy as subclassing an :class:`torch.nn.Module`. Simply, subclass :class:`~torchmetrics.Metric` and do the following:

1. Implement ``__init__`` where you call ``self.add_state`` for every internal state that is needed for the metrics computations
2. Implement ``update`` method, where all logic that is necessary for updating metric states go
3. Implement ``compute`` method, where the final metric computations happens

For practical examples and more info about implementing a metric, please see this :ref:`page <implement>`.