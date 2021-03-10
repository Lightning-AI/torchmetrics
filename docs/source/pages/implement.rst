.. _implement:

*********************
Implementing a Metric
*********************

To implement your own custom metric, subclass the base :class:`~torchmetrics.Metric` class and implement the following methods:

- ``__init__()``: Each state variable should be called using ``self.add_state(...)``.
- ``update()``: Any code needed to update the state given any inputs to the metric.
- ``compute()``: Computes a final value from the state of the metric.

All you need to do is call ``add_state`` correctly to implement a custom metric with DDP.
``reset()`` is called on metric state variables added using ``add_state()``.

To see how metric states are synchronized across distributed processes, refer to ``add_state()`` docs
from the base ``Metric`` class.

Example implementation:

.. testcode::

    from torchmetrics import Metric

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


Internal implementation details
-------------------------------

This section briefly describe how metrics work internally. We encourage looking at the source code for more info.
Internally, Lightning wraps the user defined ``update()`` and ``compute()`` method. We do this to automatically
synchronize and reduce metric states across multiple devices. More precisely, calling ``update()`` does the
following internally:

1. Clears computed cache
2. Calls user-defined ``update()``

Simiarly, calling ``compute()`` does the following internally

1. Syncs metric states between processes
2. Reduce gathered metric states
3. Calls the user defined ``compute()`` method on the gathered metric states
4. Cache computed result

From a user's standpoint this has one important side-effect: computed results are cached. This means that no
matter how many times ``compute`` is called after one and another, it will continue to return the same result.
The cache is first emptied on the next call to ``update``.

``forward`` serves the dual purpose of both returning the metric on the current data and updating the internal
metric state for accumulating over multiple batches. The ``forward()`` method achives this by combining calls
to ``update`` and ``compute`` in the following way (assuming metric is initialized with ``compute_on_step=True``):

1. Calls ``update()`` to update the global metric states (for accumulation over multiple batches)
2. Caches the global state
3. Calls ``reset()`` to clear global metric state
4. Calls ``update()`` to update local metric state
5. Calls ``compute()`` to calculate metric for current batch
6. Restores the global state

This procedure has the consequence of calling the user defined ``update`` **twice** during a single
forward call (one to update global statistics and one for getting the batch statistics).

Read more about logging in Lightning `here <https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#logging-from-a-lightningmodule>`_.


---------

.. autoclass:: torchmetrics.Metric
    :members:
