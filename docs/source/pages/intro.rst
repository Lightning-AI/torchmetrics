
The Torchmetrics is a metrics API created for easy metric development and usage in both PyTorch and
`PyTorch Lightning <https://pytorch-lightning.readthedocs.io/en/stable/>`_. It was originally a part of
Pytorch Lightning, but got split off so all PyTorch users could take advantage of the large collection of metrics
implemented.
We currently have around 25+ metrics implemented and we continuesly is adding more metrics, both within
already covered domains (classification, regression ect.) but also new domains (object detection ect.).
We make sure that all our metrics are rigorously tested against other popular implementations.


Build-in metrics
****************

Similar to `torch.nn` most metrics comes as both a Module based version and simple functional version.

- The Module based metrics offers the most functionality, by supporting both accumulation over multiple
    batches and automatic synchronization between multiple devices.

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

- Functional based metrics follows a simple input-output paradigme: a single batch is feed in and the metric
    is computed for only that

    .. testcode::

        import torch
        # import our library
        import torchmetrics

        # simulate a classification problem
        preds = torch.randn(10, 5).softmax(dim=-1)
        target = torch.randint(5, (10,))

        acc = torchmetrics.functional.accuracy(preds, target)


Implementing your own metric
****************************

Implementing your own metric is as easy as subclassing an :class:`~torch.nn.Module`. Simply, subclass :class:`~torchmetrics.Metric` and do the following:

1. Implement ``__init__`` where you call ``self.add_state`` for every internal state that is needed for the metrics computations
2. Implement ``update`` method, where all logic that is necessary for updating metric states go
3. Implement ``compute`` method, where the final metric computations happens
