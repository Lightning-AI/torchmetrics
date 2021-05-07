.. testsetup:: *

    import torch
    from pytorch_lightning.core.lightning import LightningModule

########
Overview
########

The ``torchmetrics`` is a Metrics API created for easy metric development and usage in
PyTorch and PyTorch Lightning. It is rigorously tested for all edge cases and includes a growing list of
common metric implementations.

The metrics API provides ``update()``, ``compute()``, ``reset()`` functions to the user. The metric :ref:`base class <references/modules:base class>` inherits
:class:`torch.nn.Module` which allows us to call ``metric(...)`` directly. The ``forward()`` method of the base ``Metric`` class
serves the dual purpose of calling ``update()`` on its input and simultaneously returning the value of the metric over the
provided input.

These metrics work with DDP in PyTorch and PyTorch Lightning by default. When ``.compute()`` is called in
distributed mode, the internal state of each metric is synced and reduced across each process, so that the
logic present in ``.compute()`` is applied to state information from all processes.

This metrics API is independent of PyTorch Lightning. Metrics can directly be used in PyTorch as shown in the example:

.. code-block:: python

    from torchmetrics.classification import Accuracy

    train_accuracy = Accuracy()
    valid_accuracy = Accuracy(compute_on_step=False)

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


*******************
Metrics and devices
*******************

Metrics are simple subclasses of :class:`~torch.nn.Module` and their metric states behave
similar to buffers and parameters of modules. This means that metrics states should
be moved to the same device as the input of the metric:

.. code-block:: python

    from torchmetrics import Accuracy

    target = torch.tensor([1, 1, 0, 0], device=torch.device("cuda", 0))
    preds = torch.tensor([0, 1, 0, 0], device=torch.device("cuda", 0))

    # Metric states are always initialized on cpu, and needs to be moved to
    # the correct device
    confmat = Accuracy(num_classes=2).to(torch.device("cuda", 0))
    out = confmat(preds, target)
    print(out.device) # cuda:0

However, when **properly defined** inside a :class:`~torch.nn.Module` or
:class:`~pytorch_lightning.core.lightning.LightningModule` the metric will be be automatically move
to the same device as the the module when using ``.to(device)``.  Being
**properly defined** means that the metric is correctly identified as a child module of the
model (check ``.children()`` attribute of the model). Therefore, metrics cannot be placed
in native python ``list`` and ``dict``, as they will not be correctly identified
as child modules. Instead of ``list`` use :class:`~torch.nn.ModuleList` and instead of
``dict`` use :class:`~torch.nn.ModuleDict`. Furthermore, when working with multiple metrics
the native `MetricCollection`_ module can also be used to wrap multiple metrics.

.. testcode::

    from torchmetrics import Accuracy, MetricCollection

    class MyModule(torch.nn.Module):
        def __init__(self):
            ...
            # valid ways metrics will be identified as child modules
            self.metric1 = Accuracy()
            self.metric2 = nn.ModuleList(Accuracy())
            self.metric3 = nn.ModuleDict({'accuracy': Accuracy()})
            self.metric4 = MetricCollection([Accuracy()]) # torchmetrics build-in collection class

        def forward(self, batch):
            data, target = batch
            preds = self(data)
            ...
            val1 = self.metric1(preds, target)
            val2 = self.metric2[0](preds, target)
            val3 = self.metric3['accuracy'](preds, target)
            val4 = self.metric4(preds, target)

Metrics in Dataparallel (DP) mode
=================================

When using metrics in `Dataparallel (DP) <https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel>`_
mode, one should be aware DP will both create and clean-up replicas of Metric objects during a single forward pass.
This has the consequence, that the metric state of the replicas will as default be destroyed before we can sync
them. It is therefore recommended, when using metrics in DP mode, to initialize them with ``dist_sync_on_step=True``
such that metric states are synchonized between the main process and the replicas before they are destroyed.

Metrics in Distributed Data Parallel (DDP) mode
===============================================

When using metrics in `Distributed Data Parallel (DPP) <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_
mode, one should be aware that DDP will add additional samples to your dataset if the size of your dataset is
not equally divisible by ``batch_size * num_processors``. The added samples will always be replicates of datapoints
already in your dataset. This is done to secure an equal load for all processes. However, this has the consequence
that the calculated metric value will be sligtly bias towards those replicated samples, leading to a wrong result.

During training and/or validation this may not be important, however it is highly recommended when evaluating
the test dataset to only run on a single gpu or use a `join <https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel.join>`_
context in conjunction with DDP to prevent this behaviour.

****************************
Metrics and 16-bit precision
****************************

Most metrics in our collection can be used with 16-bit precision (``torch.half``) tensors. However, we have found
the following limitations:

* In general ``pytorch`` had better support for 16-bit precision much earlier on GPU than CPU. Therefore, we
  recommend that anyone that want to use metrics with half precision on CPU, upgrade to atleast pytorch v1.6
  where support for operations such as addition, subtraction, multiplication ect. was added.
* Some metrics does not work at all in half precision on CPU. We have explicitly stated this in their docstring,
  but they are also listed below:

  - :ref:`references/modules:PSNR` and :ref:`references/functional:psnr [func]`
  - :ref:`references/modules:SSIM` and :ref:`references/functional:ssim [func]`

******************
Metric Arithmetics
******************

Metrics support most of python built-in operators for arithmetic, logic and bitwise operations.

For example for a metric that should return the sum of two different metrics, implementing a new metric is an
overhead that is not necessary. It can now be done with:

.. code-block:: python

    first_metric = MyFirstMetric()
    second_metric = MySecondMetric()

    new_metric = first_metric + second_metric

``new_metric.update(*args, **kwargs)`` now calls update of ``first_metric`` and ``second_metric``. It forwards
all positional arguments but forwards only the keyword arguments that are available in respective metric's update
declaration. Similarly ``new_metric.compute()`` now calls compute of ``first_metric`` and ``second_metric`` and
adds the results up. It is important to note that all implemented operations always returns a new metric object. This means
that the line ``first_metric == second_metric`` will not return a bool indicating if ``first_metric`` and ``second_metric``
is the same metric, but will return a new metric that checks if the ``first_metric.compute() == second_metric.compute()``.

This pattern is implemented for the following operators (with ``a`` being metrics and ``b`` being metrics, tensors, integer or floats):

* Addition (``a + b``)
* Bitwise AND (``a & b``)
* Equality (``a == b``)
* Floordivision (``a // b``)
* Greater Equal (``a >= b``)
* Greater (``a > b``)
* Less Equal (``a <= b``)
* Less (``a < b``)
* Matrix Multiplication (``a @ b``)
* Modulo (``a % b``)
* Multiplication (``a * b``)
* Inequality (``a != b``)
* Bitwise OR (``a | b``)
* Power (``a ** b``)
* Subtraction (``a - b``)
* True Division (``a / b``)
* Bitwise XOR (``a ^ b``)
* Absolute Value (``abs(a)``)
* Inversion (``~a``)
* Negative Value (``neg(a)``)
* Positive Value (``pos(a)``)
* Indexing (``a[0]``)

.. note::

    Some of these operations are only fully supported from Pytorch v1.4 and onwards, explicitly we found:
    ``add``, ``mul``, ``rmatmul``, ``rsub``, ``rmod``


****************
MetricCollection
****************

In many cases it is beneficial to evaluate the model output by multiple metrics.
In this case the ``MetricCollection`` class may come in handy. It accepts a sequence
of metrics and wraps theses into a single callable metric class, with the same
interface as any other metric.

Example:

.. testcode::

    from torchmetrics import MetricCollection, Accuracy, Precision, Recall
    target = torch.tensor([0, 2, 0, 2, 0, 1, 0, 2])
    preds = torch.tensor([2, 1, 2, 0, 1, 2, 2, 2])
    metric_collection = MetricCollection([
        Accuracy(),
        Precision(num_classes=3, average='macro'),
        Recall(num_classes=3, average='macro')
    ])
    print(metric_collection(preds, target))

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    {'Accuracy': tensor(0.1250),
     'Precision': tensor(0.0667),
     'Recall': tensor(0.1111)}

Similarly it can also reduce the amount of code required to log multiple metrics
inside your LightningModule

.. testcode::

    from torchmetrics import Accuracy, MetricCollection, Precision, Recall

    class MyModule(LightningModule):
        def __init__(self):
            metrics = MetricCollection([Accuracy(), Precision(), Recall()])
            self.train_metrics = metrics.clone(prefix='train_')
            self.valid_metrics = metrics.clone(prefix='val_')

        def training_step(self, batch, batch_idx):
            logits = self(x)
            # ...
            output = self.train_metrics(logits, y)
            # use log_dict instead of log
            # metrics are logged with keys: train_Accuracy, train_Precision and train_Recall
            self.log_dict(output)

        def validation_step(self, batch, batch_idx):
            logits = self(x)
            # ...
            output = self.valid_metrics(logits, y)
            # use log_dict instead of log
            # metrics are logged with keys: val_Accuracy, val_Precision and val_Recall
            self.log_dict(output)

.. note::

    `MetricCollection` as default assumes that all the metrics in the collection
    have the same call signature. If this is not the case, input that should be
    given to different metrics can given as keyword arguments to the collection.

.. autoclass:: torchmetrics.MetricCollection
    :noindex:


****************************
Module vs Functional Metrics
****************************

The functional metrics follow the simple paradigm input in, output out.
This means they don't provide any advanced mechanisms for syncing across DDP nodes or aggregation over batches.
They simply compute the metric value based on the given inputs.

Also, the integration within other parts of PyTorch Lightning will never be as tight as with the Module-based interface.
If you look for just computing the values, the functional metrics are the way to go.
However, if you are looking for the best integration and user experience, please consider also using the Module interface.


*****************************
Metrics and differentiability
*****************************

Metrics support backpropagation, if all computations involved in the metric calculation
are differentiable. All modular metrics have a property that determines if a metric is
differentible or not.

.. code-block:: python

    @property
    def is_differentiable(self) -> bool:
        return True/False

However, note that the cached state is detached from the computational
graph and cannot be backpropagated. Not doing this would mean storing the computational
graph for each update call, which can lead to out-of-memory errors.
In practise this means that:

.. code-block:: python

    metric = MyMetric()
    val = metric(pred, target) # this value can be backpropagated
    val = metric.compute() # this value cannot be backpropagated

A functional metric is differentiable if its corresponding modular metric is differentiable.
