<div align="center">

<img src="docs/source/_static/images/logo.png" width="400px">

**Collection of metrics for easy evaluating machine learning models**

---

<p align="center">
  <a href="https://www.pytorchlightning.ai/">Website</a> •
  <a href="#what-is-torchmetrics">What is Torchmetrics</a> •
  <a href="#installation">Installation</a> •
  <a href="https://torchmetrics.readthedocs.io/en/stable/">Docs</a> •
  <a href="#build-in-metrics">Build-in metrics</a> •
  <a href="#implementing-your-own-metric">Own metric</a> •
  <a href="#community">Community</a> •
  <a href="#license">License</a>
</p>

---

[![CI testing](https://github.com/PyTorchLightning/metrics/workflows/CI%20testing/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/torchmetrics/actions?query=workflow%3A%22CI+testing%22)
![Check Code formatting](https://github.com/PyTorchLightning/metrics/workflows/Check%20Code%20formatting/badge.svg?branch=master&event=push)
[![Build Status](https://dev.azure.com/PytorchLightning/Metrics/_apis/build/status/PyTorchLightning.metrics?branchName=master)](https://dev.azure.com/PytorchLightning/Metrics/_build/latest?definitionId=3&branchName=master)
[![codecov](https://codecov.io/gh/PyTorchLightning/metrics/branch/main/graph/badge.svg?token=NER6LPI3HS)](https://codecov.io/gh/PyTorchLightning/metrics)
[![Documentation Status](https://readthedocs.org/projects/torchmetrics/badge/?version=latest)](https://torchmetrics.readthedocs.io/en/latest/?badge=latest)


---

</div>


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
TorchMetrics is a collection of PyTorch metrics implementaions and an easy-to-use API to create custom metrics.
It is designed to be distrubuted-training compatible and offers:

* Automatic accumulation over batches
* Automatic synchronization between multiple devices

You can use TorchMetrics in any PyTorch model, or with in [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/) to enjoy additional features:

* Module metrics are automatically placed on the correct device when properly defined inside a LightningModule. This means that your data will always be placed on the same device as your metrics.
* Native support for logging metrics in Lightning using self.log inside your LightningModule. Lightning will log the metric based on on_step and on_epoch flags present in self.log(…). If on_epoch=True, the logger automatically logs the end of epoch metric value by calling .compute().
* The .reset() method of the metric will automatically be called and the end of an epoch.

## Build-in metrics
* Accuracy
* AveragePrecision
* AUC
* AUROC
* F1
* Hamming Distance
* ROC
* ExplainedVariance
* MeanSquaredError
* R2Score
* bleu_score
* embedding_similarity

And many more!

## Using functional metrics

Similar to `torch.nn`, most metrics have both a module-based and a functional version. The functional versions implement the basic operations required for computing each metric. They are simple python functions that as input take torch.tensors and return the corresponding metric as a torch.tensor.

``` python
import torch
# import our library
import torchmetrics

# simulate a classification problem
preds = torch.randn(10, 5).softmax(dim=-1)
target = torch.randint(5, (10,))

acc = torchmetrics.functional.accuracy(preds, target)
```

## Using Module metrics

Nearly all functional metrics have a corresponding module-based metric that calls it a functional counterpart underneath. The module-based metrics are characterized by having one or more internal metrics states (similar to the parameters of the PyTorch module) that allow them to offer additional functionalities:

* Accumulation of multiple batches
* Automatic synchronization between multiple devices
* Metric arithmetic
  
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
