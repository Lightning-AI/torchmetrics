<div align="center">

<img src="docs/source/_static/images/logo.png" width="400px">

**Machine learning metrics for distributed, scalable PyTorch models.**

---

<p align="center">
  <a href="#what-is-torchmetrics">What is Torchmetrics</a> •
  <a href="#implementing-your-own-metric">Implementing a metric</a> •
  <a href="#build-in-metrics">Built-in metrics</a> •
  <a href="https://torchmetrics.readthedocs.io/en/stable/">Docs</a> •
  <a href="#community">Community</a> •
  <a href="#license">License</a>
</p>

---

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchmetrics)](https://pypi.org/project/torchmetrics/)
[![PyPI Status](https://badge.fury.io/py/torchmetrics.svg)](https://badge.fury.io/py/torchmetrics)
[![PyPI Status](https://pepy.tech/badge/torchmetrics)](https://pepy.tech/project/torchmetrics)
[![Conda](https://img.shields.io/conda/v/conda-forge/torchmetrics?label=conda&color=success)](https://anaconda.org/conda-forge/torchmetrics)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/torchmetrics/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/PytorchLightning/metrics/blob/master/LICENSE)

[![CI testing - base](https://github.com/PyTorchLightning/metrics/actions/workflows/ci_test-base.yml/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/metrics/actions/workflows/ci_test-base.yml)
[![Build Status](https://dev.azure.com/PytorchLightning/Metrics/_apis/build/status/PyTorchLightning.metrics?branchName=master)](https://dev.azure.com/PytorchLightning/Metrics/_build/latest?definitionId=3&branchName=master)
[![codecov](https://codecov.io/gh/PyTorchLightning/metrics/branch/master/graph/badge.svg?token=NER6LPI3HS)](https://codecov.io/gh/PyTorchLightning/metrics)
[![Documentation Status](https://readthedocs.org/projects/torchmetrics/badge/?version=latest)](https://torchmetrics.readthedocs.io/en/latest/?badge=latest)

---

</div>


## Installation

Simple installation from PyPI
```bash
pip install torchmetrics -U
```

Or conda
```bash
conda install torchmetrics
```
---

## What is Torchmetrics
TorchMetrics is a collection of 25+ PyTorch metrics implementations and an easy-to-use API to create custom metrics. It offers:

* Optimized for distributed-training
* A standardized interface to increase reproducability
* Reduces Boilerplate
* Rigorously tested
* Automatic accumulation over batches
* Automatic synchronization between multiple devices

You can use TorchMetrics in any PyTorch model, or with in [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/) to enjoy additional features:

* Module metrics are automatically placed on the correct device.
* Native support for logging metrics in Lightning to reduce even more boilerplate.

## Implementing your own metric
Implementing your own metric is as easy as subclassing an [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html). Simply, subclass `torchmetrics.Metric`
and implement the following methods:

```python
class RMSE(torchmetrics.Metric):
    def __init__(self):
        # call `self.add_state`for every internal state that is needed for the metrics computations
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

## Built-in metrics

## Functional metrics

Similar to [`torch.nn`](https://pytorch.org/docs/stable/nn.html), most metrics have both a [module-based](https://pytorchlightning.github.io/metrics/references/modules.html) and a [functional](https://pytorchlightning.github.io/metrics/references/functional.html) version.
The functional versions are simple python functions that as input take [torch.tensors](https://pytorch.org/docs/stable/tensors.html) and return the corresponding metric as a [torch.tensor](https://pytorch.org/docs/stable/tensors.html).

``` python
import torch
# import our library
import torchmetrics

# simulate a classification problem
preds = torch.randn(10, 5).softmax(dim=-1)
target = torch.randint(5, (10,))

acc = torchmetrics.functional.accuracy(preds, target)
```

### Module metrics

The [module-based metrics](https://pytorchlightning.github.io/metrics/references/modules.html) contain internal metric states (similar to the parameters of the PyTorch module) that automate accumulation and synchronization across devices!

* Automatic accumulation over multiple batches
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

### Implemented metrics

* [Accuracy](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#accuracy)
* [AveragePrecision](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#averageprecision)
* [AUC](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#auc)
* [AUROC](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#auroc)
* [F1](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#f1) 
* [Hamming Distance](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#hamming-distance)
* [ROC](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#roc)
* [ExplainedVariance](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#explainedvariance)
* [MeanSquaredError](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#meansquarederror)
* [R2Score](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#r2score)
* [bleu_score](https://torchmetrics.readthedocs.io/en/latest/references/functional.html#bleu-score-func)
* [embedding_similarity](https://torchmetrics.readthedocs.io/en/latest/references/functional.html#embedding-similarity-func)

And many more!

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
