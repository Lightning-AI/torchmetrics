.. testsetup:: *

    import torch
    from pytorch_lightning.core.lightning import LightningModule

########
Plotting
########

.. note::
    The visualzation/plotting interface of Torchmetrics requires ``matplotlib`` to be installed. Install with either
    `pip install matplotlib` or `pip install 'torchmetrics[visual]'`. If the ladder option is chosen the [scienceplot]()
    package if also installed and all plots in Torchmetrics will default to using that style.

Torchmetrics comes with build-in support for quick visualization of your metrics, by simply using the `.plot` method
that all modular metrics implement. This is method provides a consist interface

```
metric = AnyMetricYouLike()
for _ in range(N):
    metric.update(preds[i], target[i])
fig, ax = metric.plot()
```

`.plot` will always return two objects: `fig` is an instance of `matplotlib.figure.Figure` which contains figure level
attributes is the figure object that contains the plot and `ax` is an instance of `matplotlib.axes.Axes` that contains
all the elements of the plot. So for example if I wanted to make the fontsize of the x-axis a bit bigger and give the
figure a nice title and finally save it, I could do it:

```
ax.set_fontsize(fs=20)
fig.set_title("This is a nice plot")
fig.save_fig("my_awesome_plot.png")
```

If you want to include a Torchmetrics plot in a bigger figure that have subfigures and subaxises all `.plot` methods
support an optional `ax` argument where you can pass in the subaxises you want the plot to be inserted into:

```
# combine plotting of two metrics into one figure
fig, ax = plt.subplots(nrows=1, ncols=2)
metric1 = Metric1()
metric2 = Metric2()
for _ in range(N):
    metric1.update(preds[i], target[i])
    metric2.update(preds[i], target[i])
metric1.plot(ax=ax[0])
metric2.plot(ax=ax[1])
```

**********************
Plotting a single step
**********************

At the most basic level the `.plot` method can be used to plot the value from a single step. This can be done in two
ways:
* Either `.plot` method is called with no input, and internally `metric.compute()` is called and that value is plotted
* `.plot` is called on a single returned value by the metric

In both cases it will generate a plot like this:


A single point plot is not that informative in itself, but if available we will try to include additional information
such as the lower and upper bounds the particular metric can take an if the metric should be minimized or maximized
to be optimal. This is true for all metrics that return a scalar tensor.
Some metrics returns multiple values (such as an tensor with multiple elements or an dict of scalar tensors), and in
that case calling `.plot` will return a figure similar to this:

Here, each element is assumed to be an independent metric and plotted as its own point for comparing.


Finally, some metrics

If you prefer to use the functional



**********************
Plotting a multi steps
**********************

Do note that metrics that does not return simple scalar tensors, such as `ConfusionMatrix`, `ROC` that have specialized
visualzation does not support plotting multiple steps.

********************************
Plotting a collection of metrics
********************************

``MetricCollection`` also supports `.plot` method and by default it works by just returinging a collection of plots for all
its members

Additionally, ``MetricCollection`` also implements the specialized `.plot_together` method that will combine all the
metrics into a single plot in one of two styles:
* `plot_type="lines"` will create a line plot similar
* `plot_type="radar"` will create a radar plot
