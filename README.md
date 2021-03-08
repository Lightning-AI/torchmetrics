# TorchMetrics

**Collection of metrics for easy evaluating machine learning models**

---

[![CI testing](https://github.com/PyTorchLightning/metrics/workflows/CI%20testing/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/torchmetrics/actions?query=workflow%3A%22CI+testing%22)
![Check Code formatting](https://github.com/PyTorchLightning/metrics/workflows/Check%20Code%20formatting/badge.svg?branch=master&event=push)
[![Build Status](https://dev.azure.com/PytorchLightning/Metrics/_apis/build/status/PyTorchLightning.metrics?branchName=master)](https://dev.azure.com/PytorchLightning/Metrics/_build/latest?definitionId=3&branchName=master)
[![codecov](https://codecov.io/gh/PyTorchLightning/metrics/branch/main/graph/badge.svg?token=NER6LPI3HS)](https://codecov.io/gh/PyTorchLightning/metrics)
[![Documentation Status](https://readthedocs.org/projects/torchmetrics/badge/?version=latest)](https://torchmetrics.readthedocs.io/en/latest/?badge=latest)

## Installation 

Pip / conda

```bash
pip install torchmetrics -U
conda install torchmetrics
```

Pip from source

```bash
# with git
pip install git+https://github.com/PytorchLightning/metrics.git@master

# OR from an archive
pip install https://github.com/PyTorchLightning/metrics/archive/master.zip
```

---

## What is Torchmetrics
Torchmetrics is a metrics API created for easy metric development and usage in both PyTorch and 
[PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/). It was originally a part of 
Pytorch Lightning, but got split off so users could take advantage of the large collection of metrics 
implemented without having to install Pytorch Lightning (eventhough we would love for you to try it out). 
We currently have around 25+ metrics implemented and we continuesly is adding more metrics, both within 
already covered domains (classification, regression ect.) but also new domains (object detection ect.). 
We make sure that all our metrics are rigorously tested such that you can trust them. 

## Build-in metrics

Similar to `torch.nn` most metrics comes both as class based version and simple functional version.

* The class based metrics offers the most functionality, by supporting both accumulation over multiple 
batches and automatic syncrenization between multiple devices.
  
``` python
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
```

  
* Functional based metrics follows a simple input-output paradigme: a single batch is feed in and the metric is computed 
for only that

``` python
import torch
# import our library
import torchmetrics

# simulate a classification problem
preds = torch.randn(10, 5).softmax(dim=-1)
target = torch.randint(5, (10,))

acc = torchmetrics.functional.accuracy(preds, target)
```

## Implementing your own metric
Implementing your own metric is as easy as subclassing an `torch.nn.Module`. Simply, subclass `torchmetrics.Metric` 
and do the following:

1. Implement `__init__` where you call `self.add_state`for every internal state that is needed for the metrics computations
2. Implement `update` method, where all logic that is nessesary for updating metric states go
3. Implement `compute` method, where the final metric computations happens

### Example: Root mean squared error
Root mean squared error is great example to showcase why many metric computations needs to be divided into 
two functions. It is defined as:

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=RMSE = \sqrt{ \frac{1}{N} \sum_{i=1}^N (\hat{y_i} - y_i)^2}">
</p>

To proper calculate RMSE, we need two metric states: `sum_squared_error` to keep track of the squared error 
between the target and the predictions and `n_observations` to know how many observations we have encountered.
```python
class RMSE(torchmetrics.Metric):
    def __init__(self)
	# dist_reduce_fx indicates the function that should be used to reduce 
	# state from multiple processes
        self.add_state("sum_squared_errors", torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_observations", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        # update metric states
        sum_squared_errors += torch.sum((preds - target) ** 2)
        n_observations += preds.numel()
       
    def compute(self):
        # compute final result
        return torch.sqrt(sum_squared_errors / n_observations)
```
Because `sqrt(a+b) != sqrt(a) + sqrt(b)` we cannot implement this metric as a simple mean of the RMSE 
score calculated per batch and instead needs to implement all logic that needs to happen before the 
square root in `update` and the remaining in `compute`.


## Contribute!
The lightning + torchmetric team is hard at work adding even more metrics. 
But we're looking for incredible contributors like you to submit new metrics
and improve existing ones!

Join our [Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A) 
to get help becoming a contributor!

## Community
For help or questions, join our huge community on [Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A)!

## Citations
We’re excited to continue the strong legacy of opensource software and have been inspired over the years by 
Caffee, Theano, Keras, PyTorch, torchbearer, ignite, sklearn and fast.ai. When/if a paper is written about this, 
we’ll be happy to cite these frameworks and the corresponding authors.

## License
Please observe the Apache 2.0 license that is listed in this repository. In addition
the Lightning framework is Patent Pending.
