TorchMetrics is a collection of 25+ PyTorch metrics implementations and an easy-to-use API to create custom metrics. It offers:

* A standardized interface to increase reproducability
* Reduces Boilerplate
* Distrubuted-training compatible
* Rigorously tested
* Automatic accumulation over batches
* Automatic synchronization between multiple devices

You can use TorchMetrics in any PyTorch model, or with in `PyTorch Lightning <https://pytorch-lightning.readthedocs.io/en/stable/>`_ to enjoy additional features:

* This means that your data will always be placed on the same device as your metrics.
* Native support for logging metrics in Lightning to reduce even more boilerplate.


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
    import torchmetrics

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

    # metric on all batches using custom accumulation
    acc = metric.compute()
    print(f"Accuracy on all data: {acc}")