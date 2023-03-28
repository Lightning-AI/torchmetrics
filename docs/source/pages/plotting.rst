.. testsetup:: *

    import torch
    import matplotlib
    import matplotlib.pyplot as plt
    import torchmetrics

########
Plotting
########

.. note::
    The visualzation/plotting interface of Torchmetrics requires ``matplotlib`` to be installed. Install with either
    ``pip install matplotlib`` or ``pip install 'torchmetrics[visual]'``. If the ladder option is chosen the
    `Scienceplot package <https://github.com/garrettj403/SciencePlots>`_ is also installed and all plots in
    Torchmetrics will default to using that style.

Torchmetrics comes with build-in support for quick visualization of your metrics, by simply using the ``.plot`` method
that all modular metrics implement. This is method provides a consist interface

.. testcode:: python

    metric = AnyMetricYouLike()
    for _ in range(N):
        metric.update(preds[i], target[i])
    fig, ax = metric.plot()

``.plot`` will always return two objects: ``fig`` is an instance of :class:`~matplotlib.figure.Figure` which contains
figure level attributes and `ax` is an instance of :class:`~matplotlib.axes.Axes` that contains all the elements of the
plot. These two objects allow to change attributes of the plot after it is created. For example if you wanted to make
the fontsize of the x-axis a bit bigger and give the figure a nice title and finally save it on the above example, it
could be do like this:

.. testcode:: python

    ax.set_fontsize(fs=20)
    fig.set_title("This is a nice plot")
    fig.save_fig("my_awesome_plot.png")

If you want to include a Torchmetrics plot in a bigger figure that have subfigures and subaxises all ``.plot`` methods
support an optional `ax` argument where you can pass in the subaxises you want the plot to be inserted into:

.. testcode:: python

    # combine plotting of two metrics into one figure
    fig, ax = plt.subplots(nrows=1, ncols=2)
    metric1 = Metric1()
    metric2 = Metric2()
    for _ in range(N):
        metric1.update(preds[i], target[i])
        metric2.update(preds[i], target[i])
    metric1.plot(ax=ax[0])
    metric2.plot(ax=ax[1])

**********************
Plotting a single step
**********************

At the most basic level the ``.plot`` method can be used to plot the value from a single step. This can be done in two
ways:
* Either ``.plot`` method is called with no input, and internally ``metric.compute()`` is called and that value is
  plotted
* ``.plot`` is called on a single returned value by the metric, for example from ``metric.forward()``

In both cases it will generate a plot like this (Accuracy as an example):

.. testcode:: python

    metric = torchmetrics.Accuracy(task="binary")
    for _ in range(N):
        metric.update(torch.rand(10,), torch.randint(2, (10,)))
    fig, ax = metric.plot()

.. image:: binary_accuracy.png
   :height: 100px
   :width: 200 px
   :scale: 50 %
   :alt: binary accuracy plot
   :align: right

A single point plot is not that informative in itself, but if available we will try to include additional information
such as the lower and upper bounds the particular metric can take an if the metric should be minimized or maximized
to be optimal. This is true for all metrics that return a scalar tensor.
Some metrics returns multiple values (such as an tensor with multiple elements or an dict of scalar tensors), and in
that case calling ``.plot`` will return a figure similar to this:

.. testcode:: python

    metric = torchmetrics.Accuracy(task="multiclass", num_classes=3, average=None)
    for _ in range(N):
        metric.update(torch.randint(3, (10,)), torch.randint(3, (10,)))
    fig, ax = metric.plot()

.. image:: multiclass_accuracy.png
   :height: 100px
   :width: 200 px
   :scale: 50 %
   :alt: multiclass accuracy plot
   :align: right

Here, each element is assumed to be an independent metric and plotted as its own point for comparing. The above is true
for all metrics that returns a scalar tensor, but if the metric returns a tensor with multiple elements then the
``.plot`` method will return a specialized plot for that particular metric. Take for example the ``ConfusionMatrix``
metric:

.. testcode:: python

    metric = torchmetrics.ConfusionMatrix(num_classes=3)
    for _ in range(N):
        metric.update(torch.randint(3, (10,)), torch.randint(3, (10,)))
    fig, ax = metric.plot()

.. image:: confusionmatrix.png
   :height: 100px
   :width: 200 px
   :scale: 50 %
   :alt: confusionmatrix plot
   :align: right

If you prefer to use the functional interface of Torchmetrics, you can also plot the values returned by the functional.
However, you would still need to initialize the corresponding metric class to get the information about the metric:

.. testcode:: python

    plot_class = torchmetrics.Accuracy(task="multiclass", num_classes=3)
    value = torchmetrics.functional.accuracy(
        torch.randint(3, (10,)), torch.randint(3, (10,)), num_classes=3
    )
    fig, ax = plot_class.plot(value)

**********************
Plotting multi steps
**********************

In the above examples we have only plotted a single step, but it is also possible to plot multiple steps. This can be
done by provided a sequence of outputs from any metric, for example computed using ``metric.forward`` or
``metric.compute``. For example, if we wanted to plot the accuracy of a model over time, we could do it like this:

.. testcode:: python

    metric = torchmetrics.Accuracy(task="binary")
    values = [ ]
    for _ in range(num_steps)
        for _ in range(N):
            metric.update(torch.rand(10,), torch.randint(2, (10,)))
        values.append(metric.compute())  # save value
        metric.reset()
    fig, ax = metric.plot(metric.compute())

.. image:: multistep_accuracy.png
   :height: 100px
   :width: 200 px
   :scale: 50 %
   :alt: multistep accuracy plot
   :align: right

Do note that metrics that does not return simple scalar tensors, such as `ConfusionMatrix`, `ROC` that have specialized
visualzation does not support plotting multiple steps, out of the box and the user needs to manually plot the values
for each step.

********************************
Plotting a collection of metrics
********************************

``MetricCollection`` also supports `.plot` method and by default it works by just returinging a collection of plots for all
its members

Additionally, ``MetricCollection`` also implements the specialized `.plot_together` method that will combine all the
metrics into a single plot in one of two styles:
* `plot_type="lines"` will create a line plot similar
* `plot_type="radar"` will create a radar plot

The default output of ``MetricCollection.plot`` is a list of (fig, ax) pairs for each metric in the collection.

***************
Advance example
***************

In the following we are going to show how to use the ``.plot`` method to create a more advanced plot.
We are going to combine the functionality of several metrics using ``MetricCollection`` and plot them together. In
addition we are going to rely on ``MetricTracker`` to keep track of the metrics and plot them as they are updated.
