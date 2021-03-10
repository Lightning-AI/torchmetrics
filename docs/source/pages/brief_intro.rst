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

.. testcode::

    import torch
    import torchmetrics

    # simulate a classification problem
    preds = torch.randn(10, 5).softmax(dim=-1)
    target = torch.randint(5, (10,))

    acc = torchmetrics.functional.accuracy(preds, target)

Module metrics
~~~~~~~~~~~~~~

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
        print(f"Accuracy on batch {i}: {acc}")

    # metric on all batches using custom accumulation
    acc = metric.compute()
    print(f"Accuracy on all data: {acc}")

.. testoutput::
   :hide:
   :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

    Accuracy on batch ...
