
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

    class MyModule():
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
* Substraction (``a - b``)
* True Division (``a / b``)
* Bitwise XOR (``a ^ b``)
* Absolute Value (``abs(a)``)
* Inversion (``~a``)
* Negative Value (``neg(a)``)
* Positive Value (``pos(a)``)

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

.. code-block:: python

    def __init__(self):
        ...
        metrics = pl.metrics.MetricCollection(...)
        self.train_metrics = metrics.clone()
        self.valid_metrics = metrics.clone()

    def training_step(self, batch, batch_idx):
        logits = self(x)
        ...
        self.train_metrics(logits, y)
        # use log_dict instead of log
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False, prefix='train')

    def validation_step(self, batch, batch_idx):
        logits = self(x)
        ...
        self.valid_metrics(logits, y)
        # use log_dict instead of log
        self.log_dict(self.valid_metrics, on_step=True, on_epoch=True, prefix='val')

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
