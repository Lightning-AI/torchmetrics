###########
Quick Start
###########

TorchMetrics is a collection of 100+ PyTorch metrics implementations and an easy-to-use API to create custom metrics. It offers:

* A standardized interface to increase reproducibility
* Reduces Boilerplate
* Distributed-training compatible
* Rigorously tested
* Automatic accumulation over batches
* Automatic synchronization between multiple devices

You can use TorchMetrics in any PyTorch model, or within `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`_ to enjoy additional features:

* This means that your data will always be placed on the same device as your metrics.
* Native support for logging metrics in Lightning to reduce even more boilerplate.

Install
*******

You can install TorchMetrics using pip or conda:

.. code-block:: bash

    # Python Package Index (PyPI)
    pip install torchmetrics
    # Conda
    conda install -c conda-forge torchmetrics

Eventually if there is a missing PyTorch wheel for your OS or Python version you can simply compile `PyTorch from source <https://github.com/pytorch/pytorch>`_:

.. code-block:: bash

    # Optional if you do not need compile GPU support
    export USE_CUDA=0  # just to keep it simple
    # you can install the latest state from master
    pip install git+https://github.com/pytorch/pytorch.git
    # OR set a particular PyTorch release
    pip install git+https://github.com/pytorch/pytorch.git@<release-tag>
    # and finalize with installing TorchMetrics
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

    acc = torchmetrics.functional.accuracy(preds, target, task="multiclass", num_classes=5)

Module metrics
~~~~~~~~~~~~~~

Nearly all functional metrics have a corresponding class-based metric that calls it a functional counterpart underneath.
The class-based metrics are characterized by having one or more internal metrics states (similar to the parameters of
the PyTorch module) that allow them to offer additional functionalities:

* Accumulation of multiple batches
* Automatic synchronization between multiple devices
* Metric arithmetic

The code below shows how to use the class-based interface:

.. testcode::

    import torch
    # import our library
    import torchmetrics

    # initialize metric
    metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=5)

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

    # Resetting internal state such that metric ready for new data
    metric.reset()

.. testoutput::
   :hide:
   :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

    Accuracy on batch ...


Implementing your own metric
****************************

Implementing your own metric is as easy as subclassing a :class:`torch.nn.Module`. Simply, subclass :class:`~torchmetrics.Metric` and do the following:

1. Implement ``__init__`` where you call ``self.add_state`` for every internal state that is needed for the metrics computations
2. Implement ``update`` method, where all logic that is necessary for updating metric states go
3. Implement ``compute`` method, where the final metric computations happens

For practical examples and more info about implementing a metric, please see this :ref:`page <implement>`.


Development Environment
~~~~~~~~~~~~~~~~~~~~~~~

TorchMetrics provides a `Devcontainer <https://code.visualstudio.com/docs/remote/containers>`_ configuration for `Visual Studio Code <https://code.visualstudio.com/>`_ to use a `Docker container <https://www.docker.com/>`_ as a pre-configured development environment.
This avoids struggles setting up a development environment and makes them reproducible and consistent.
Please follow the `installation instructions <https://code.visualstudio.com/docs/remote/containers#_installation>`_ and make yourself familiar with the `container tutorials <https://code.visualstudio.com/docs/remote/containers-tutorial>`_ if you want to use them.
In order to use GPUs, you can enable them within the ``.devcontainer/devcontainer.json`` file.
