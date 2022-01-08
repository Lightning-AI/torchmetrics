TorchMetrics is a collection of Machine learning metrics for distributed, scalable PyTorch models and an easy-to-use API to create custom metrics. It offers the following benefits:

* Optimized for distributed-training
* A standardized interface to increase reproducibility
* Reduces Boilerplate
* Distributed-training compatible
* Rigorously tested
* Automatic accumulation over batches
* Automatic synchronization between multiple devices

You can use TorchMetrics in any PyTorch model, or with in `PyTorch Lightning <https://pytorch-lightning.readthedocs.io/en/stable/>`_ to enjoy additional features:

* This means that your data will always be placed on the same device as your metrics.
* Native support for logging metrics in Lightning to reduce even more boilerplate.

Using TorchMetrics
******************

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

Module metric usage remains the same when using multiple GPUs or multiple nodes.


Functional metrics
~~~~~~~~~~~~~~~~~~

.. testcode::

    import torch
    import torchmetrics

    # simulate a classification problem
    preds = torch.randn(10, 5).softmax(dim=-1)
    target = torch.randint(5, (10,))

    acc = torchmetrics.functional.accuracy(preds, target)


Implementing a metric
~~~~~~~~~~~~~~~~~~~~~

.. testcode::

    class MyAccuracy(Metric):
        def __init__(self, dist_sync_on_step=False):
            # call `self.add_state`for every internal state that is needed for the metrics computations
            # dist_reduce_fx indicates the function that should be used to reduce
            # state from multiple processes
            super().__init__(dist_sync_on_step=dist_sync_on_step)

            self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        def update(self, preds: torch.Tensor, target: torch.Tensor):
            # update metric states
            preds, target = self._input_format(preds, target)
            assert preds.shape == target.shape

            self.correct += torch.sum(preds == target)
            self.total += target.numel()

        def compute(self):
            # compute final result
            return self.correct.float() / self.total
