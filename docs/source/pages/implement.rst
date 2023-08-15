.. _implement:

.. testsetup:: *

    from typing import Optional, Sequence, Union
    from torch import Tensor

*********************
Implementing a Metric
*********************

To implement your own custom metric, subclass the base :class:`~torchmetrics.Metric` class and implement the following
methods:

- ``__init__()``: Each state variable should be called using ``self.add_state(...)``.
- ``update()``: Any code needed to update the state given any inputs to the metric.
- ``compute()``: Computes a final value from the state of the metric.

We provide the remaining interface, such as ``reset()`` that will make sure to correctly reset all metric
states that have been added using ``add_state``. You should therefore not implement ``reset()`` yourself.
Additionally, adding metric states with ``add_state`` will make sure that states are correctly synchronized
in distributed settings (DDP). To see how metric states are synchronized across distributed processes,
refer to ``add_state()`` docs from the base ``Metric`` class.

Example implementation:

.. testcode::

    from torchmetrics import Metric

    class MyAccuracy(Metric):
        def __init__(self):
            super().__init__()
            self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        def update(self, preds: Tensor, target: Tensor):
            preds, target = self._input_format(preds, target)
            assert preds.shape == target.shape

            self.correct += torch.sum(preds == target)
            self.total += target.numel()

        def compute(self):
            return self.correct.float() / self.total


Additionally you may want to set the class properties: `is_differentiable`, `higher_is_better` and
`full_state_update`. Note that none of them are strictly required for the metric to work.

.. testcode::

    from torchmetrics import Metric

    class MyMetric(Metric):
        # Set to True if the metric is differentiable else set to False
        is_differentiable: Optional[bool] = None

        # Set to True if the metric reaches it optimal value when the metric is maximized.
        # Set to False if it when the metric is minimized.
        higher_is_better: Optional[bool] = True

        # Set to True if the metric during 'update' requires access to the global metric
        # state for its calculations. If not, setting this to False indicates that all
        # batch states are independent and we will optimize the runtime of 'forward'
        full_state_update: bool = True

Finally, from torchmetrics v1.0.0 onwards, we also support plotting of metrics through the `.plot` method. By default
this method will raise `NotImplementedError` but can be implemented by the user to provide a custom plot for the metric.
For any metrics that returns a simple scalar tensor, or a dict of scalar tensors the internal `._plot` method can be
used, that provides the common plotting functionality for most metrics in torchmetrics.

.. testcode::

    from torchmetrics import Metric
    from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

    class MyMetric(Metric):
        ...

        def plot(
            self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
        ) -> _PLOT_OUT_TYPE:
            return self._plot(val, ax)

If the metric returns a more complex output, a custom implementation of the `plot` method is required. For more details
on the plotting API, see the this :ref:`page <plotting>` .

Internal implementation details
-------------------------------

This section briefly describes how metrics work internally. We encourage looking at the source code for more info.
Internally, TorchMetrics wraps the user defined ``update()`` and ``compute()`` method. We do this to automatically
synchronize and reduce metric states across multiple devices. More precisely, calling ``update()`` does the
following internally:

1. Clears computed cache.
2. Calls user-defined ``update()``.

Similarly, calling ``compute()`` does the following internally:

1. Syncs metric states between processes.
2. Reduce gathered metric states.
3. Calls the user defined ``compute()`` method on the gathered metric states.
4. Cache computed result.

From a user's standpoint this has one important side-effect: computed results are cached. This means that no
matter how many times ``compute`` is called after one and another, it will continue to return the same result.
The cache is first emptied on the next call to ``update``.

``forward`` serves the dual purpose of both returning the metric on the current data and updating the internal
metric state for accumulating over multiple batches. The ``forward()`` method achieves this by combining calls
to ``update``, ``compute`` and ``reset``. Depending on the class property ``full_state_update``, ``forward``
can behave in two ways:

1. If ``full_state_update`` is ``True`` it indicates that the metric during ``update`` requires access to the full
   metric state and we therefore need to do two calls to ``update`` to secure that the metric is calculated correctly

   1. Calls ``update()`` to update the global metric state (for accumulation over multiple batches)
   2. Caches the global state.
   3. Calls ``reset()`` to clear global metric state.
   4. Calls ``update()`` to update local metric state.
   5. Calls ``compute()`` to calculate metric for current batch.
   6. Restores the global state.

2. If ``full_state_update`` is ``False`` (default) the metric state of one batch is completly independent of the state of
   other batches, which means that we only need to call ``update`` once.

   1. Caches the global state.
   2. Calls ``reset`` the metric to its default state
   3. Calls ``update`` to update the state with local batch statistics
   4. Calls ``compute`` to calculate the metric for the current batch
   5. Reduce the global state and batch state into a single state that becomes the new global state

If implementing your own metric, we recommend trying out the metric with ``full_state_update`` class property set to
both ``True`` and ``False``. If the results are equal, then setting it to ``False`` will usually give the best performance.

---------

.. autoclass:: torchmetrics.Metric
    :noindex:
    :members:

Contributing your metric to TorchMetrics
----------------------------------------

Wanting to contribute the metric you have implemented? Great, we are always open to adding more metrics to ``torchmetrics``
as long as they serve a general purpose. However, to keep all our metrics consistent we request that the implementation
and tests gets formatted in the following way:

1. Start by reading our `contribution guidelines <https://torchmetrics.readthedocs.io/en/latest/generated/CONTRIBUTING.html>`_.
2. First implement the functional backend. This takes cares of all the logic that goes into the metric. The code should
   be put into a single file placed under ``torchmetrics/functional/"domain"/"new_metric".py`` where ``domain`` is the type of
   metric (classification, regression, nlp etc) and ``new_metric`` is the name of the metric. In this file, there should be the
   following three functions:

  1. ``_new_metric_update(...)``: everything that has to do with type/shape checking and all logic required before distributed syncing need to go here.
  2. ``_new_metric_compute(...)``: all remaining logic.
  3. ``new_metric(...)``: essentially wraps the ``_update`` and ``_compute`` private functions into one public function that
     makes up the functional interface for the metric.

  .. note::
     The `functional accuracy <https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/classification/accuracy.py>`_
     metric is a great example of this division of logic.

3. In a corresponding file placed in ``torchmetrics/"domain"/"new_metric".py`` create the module interface:

  1. Create a new module metric by subclassing ``torchmetrics.Metric``.
  2. In the ``__init__`` of the module call ``self.add_state`` for as many metric states are needed for the metric to
     proper accumulate metric statistics.
  3. The module interface should essentially call the private ``_new_metric_update(...)`` in its `update` method and similarly the
     ``_new_metric_compute(...)`` function in its ``compute``. No logic should really be implemented in the module interface.
     We do this to not have duplicate code to maintain.

   .. note::
     The module `Accuracy <https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/classification/accuracy.py>`_
     metric that corresponds to the above functional example showcases these steps.

4. Remember to add binding to the different relevant ``__init__`` files.

5. Testing is key to keeping ``torchmetrics`` trustworthy. This is why we have a very rigid testing protocol. This means
   that we in most cases require the metric to be tested against some other common framework (``sklearn``, ``scipy`` etc).

  1. Create a testing file in ``unittests/"domain"/test_"new_metric".py``. Only one file is needed as it is intended to test
     both the functional and module interface.
  2. In that file, start by defining a number of test inputs that your metric should be evaluated on.
  3. Create a testclass ``class NewMetric(MetricTester)`` that inherits from ``tests.helpers.testers.MetricTester``.
     This testclass should essentially implement the ``test_"new_metric"_class`` and ``test_"new_metric"_fn`` methods that
     respectively tests the module interface and the functional interface.
  4. The testclass should be parameterized (using ``@pytest.mark.parametrize``) by the different test inputs defined initially.
     Additionally, the ``test_"new_metric"_class`` method should also be parameterized with an ``ddp`` parameter such that it gets
     tested in a distributed setting. If your metric has additional parameters, then make sure to also parameterize these
     such that different combinations of inputs and parameters gets tested.
  5. (optional) If your metric raises any exception, please add tests that showcase this.

  .. note::
    The `test file for accuracy <https://github.com/Lightning-AI/torchmetrics/blob/master/tests/unittests/classification/test_accuracy.py>`_ metric
    shows how to implement such tests.

If you only can figure out part of the steps, do not fear to send a PR. We will much rather receive working
metrics that are not formatted exactly like our codebase, than not receiving any. Formatting can always be applied.
We will gladly guide and/or help implement the remaining :]
