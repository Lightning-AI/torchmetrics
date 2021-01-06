
############
Introduction
############

``torchmetrics`` is a Metrics API created for easy metric development and usage in
PyTorch and PyTorch Lightning. It is rigorously tested for all edge cases and includes a growing list of
common metric implementations.

The metrics API provides ``update()``, ``compute()``, ``reset()`` functions to the user. The metric base class inherits
``nn.Module`` which allows us to call ``metric(...)`` directly. The ``forward()`` method of the base ``Metric`` class
serves the dual purpose of calling ``update()`` on its input and simultaneously returning the value of the metric over the
provided input.

These metrics work with DDP in PyTorch and PyTorch Lightning by default. When ``.compute()`` is called in
distributed mode, the internal state of each metric is synced and reduced across each process, so that the
logic present in ``.compute()`` is applied to state information from all processes.

This metrics API is independent of PyTorch Lightning. Metrics can directly be used in PyTorch as shown in the example:

.. code-block:: python

    from torchmetrics.classification import Accuracy

    train_accuracy = metrics.Accuracy()
    valid_accuracy = metrics.Accuracy(compute_on_step=False)

    for epoch in range(epochs):
        for x, y in train_data:
            y_hat = model(x)

            # training step accuracy
            batch_acc = train_accuracy(y_hat, y)

        for x, y in valid_data:
            y_hat = model(x)
            valid_accuracy(y_hat, y)

    # total accuracy over all training batches
    total_train_accuracy = train_accuracy.compute()

    # total accuracy over all validation batches
    total_valid_accuracy = valid_accuracy.compute()

.. note::

    Metrics contain internal states that keep track of the data seen so far.
    Do not mix metric states across training, validation and testing.
    It is highly recommended to re-initialize the metric per mode as
    shown in the examples above.

.. note::

    Metric states are **not** added to the models ``state_dict`` by default.
    To change this, after initializing the metric, the method ``.persistent(mode)`` can
    be used to enable (``mode=True``) or disable (``mode=False``) this behaviour.

*********************
Implementing a Metric
*********************

To implement your custom metric, subclass the base ``Metric`` class and implement the following methods:

- ``__init__()``: Each state variable should be called using ``self.add_state(...)``.
- ``update()``: Any code needed to update the state given any inputs to the metric.
- ``compute()``: Computes a final value from the state of the metric.

All you need to do is call ``add_state`` correctly to implement a custom metric with DDP.
``reset()`` is called on metric state variables added using ``add_state()``.

To see how metric states are synchronized across distributed processes, refer to ``add_state()`` docs
from the base ``Metric`` class.

Example implementation:

.. testcode::

    from pytorch_lightning.metrics import Metric

    class MyAccuracy(Metric):
        def __init__(self, dist_sync_on_step=False):
            super().__init__(dist_sync_on_step=dist_sync_on_step)

            self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        def update(self, preds: torch.Tensor, target: torch.Tensor):
            preds, target = self._input_format(preds, target)
            assert preds.shape == target.shape

            self.correct += torch.sum(preds == target)
            self.total += target.numel()

        def compute(self):
            return self.correct.float() / self.total

Metrics support backpropagation, if all computations involved in the metric calculation
are differentiable. However, note that the cached state is detached from the computational
graph and cannot be backpropagated. Not doing this would mean storing the computational
graph for each update call, which can lead to out-of-memory errors.
In practise this means that:

.. code-block:: python

    metric = MyMetric()
    val = metric(pred, target) # this value can be backpropagated
    val = metric.compute() # this value cannot be backpropagated
