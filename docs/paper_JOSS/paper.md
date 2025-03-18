---
title: TorchMetrics - Measuring Reproducibility in PyTorch
tags:
  - python
  - deep learning
  - pytorch
authors:
  - name: Nicki Skafte Detlefsen
    affiliation: '1,2'
    orcid: 0000-0002-8133-682X
  - name: Jiri Borovec
    affiliation: '1'
    orcid: 0000-0001-7437-824X
  - name: Justus Schock
    affiliation: '1,3'
    orcid: 0000-0003-0512-3053
  - name: Ananya Harsh Jha
    affiliation: '1'
  - name: Teddy Koker
    affiliation: '1'
  - name: Luca Di Liello
    affiliation: '4'
  - name: Daniel Stancl
    affiliation: '5'
  - name: Changsheng Quan
    affiliation: '6'
  - name: Maxim Grechkin
    affiliation: '7'
  - name: William Falcon
    affiliation: '1,8'
affiliations:
  - name: Grid AI Labs
    index: 1
  - name: Technical University of Denmark
    index: 2
  - name: University Hospital Düsseldorf
    index: 3
  - name: University of Trento
    index: 4
  - name: Charles University
    index: 5
  - name: Zhejiang University
    index: 6
  - name: Independent Researcher
    index: 7
  - name: New York University
    index: 8
date: 08 Dec 2021
bibliography: paper.bib
---

# Summary

A main problem with reproducing machine learning publications is the variance of metric implementations across papers. A lack of standardization leads to different behavior in mechanisms such as checkpointing, learning rate schedulers or early stopping, that will influence the reported results. For example, a complex metric such as Fréchet inception distance (FID) for synthetic image quality evaluation [@fid] will differ based on the specific interpolation method used.

There have been a few attempts at tackling the reproducibility issues. Papers With Code [@papers_with_code] links research code with its corresponding paper. Similarly, arXiv [@arxiv] recently added a code and data section that links both official and community code to papers. However, these methods rely on the paper code to be made publicly accessible which is not always possible. Our approach is to provide the de-facto reference implementation for metrics. This approach enables proprietary work to still be comparable as long as they’ve used our reference implementations.

We introduce TorchMetrics, a general-purpose metrics package that covers a wide variety of tasks and domains used in the machine learning community. TorchMetrics provides standard classification and regression metrics; and domain-specific metrics for audio, computer vision, natural language processing, and information retrieval. Our process for adding a new metric is as follows, first we integrate a well-tested and established third-party library. Once we’ve verified the implementations and written tests for them, we re-implement them in native PyTorch [@pytorch] to enable hardware acceleration and remove any bottlenecks in inter-device transfer.

# Statement of need

Currently, there is no standard, widely-adopted metrics library for native PyTorch. Some native PyTorch libraries support domain-specific metrics such as Transformers [@transformers] for calculating NLP-specific metrics. However, no library exists that covers multiple domains. PyTorch users, therefore, often rely on non-PyTorch packages such as Scikit-learn [@scikit_learn] for computing even simple metrics such as accuracy, F1, or AUROC metrics.

However, while Scikit-learn is considered the gold standard for computing metrics in regression and classification, it relies on the core assumption that all predictions and targets are available simultaneously. This contradicts the typical workflow in a modern deep learning training/evaluation loop where data comes in batches. Therefore, the metric needs to be calculated in an online fashion. It is important to note that, in general, it is not possible to calculate a global metric as its average or sum of the metric calculated per batch.

TorchMetrics solves this problem by introducing stateful metrics that can calculate metric values on a stream of data alongside the classical functional and stateless metrics provided by other packages like Scikit-learn. We do this with an effortless `update` and `compute` interface, well known from packages such as Keras [@keras]. The `update` function takes in a batch of predictions and targets and updates the internal state. For example, for a metric such as accuracy, the internal states are simply the number of correctly classified samples and the total observed number of samples. When all batches have been passed to the `update` method, the `compute` method can get the accumulated accuracy over all the batches. In addition to `update` and `compute`, each metric also has a `forward` method (as any other `torch.nn.Module`) that can be used to both get the metric on the current batch of data and accumulate global state. This enables the user to get fine-grained info about the metric on the individual batch and the global metric of how well their model is doing.

```python
# Minimal example showcasing the TorchMetrics interface
import torch
from torch import tensor, Tensor
# base class all modular metrics inherit from
from torchmetrics import Metric

class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        # `self.add_state` defines the states of the metric
        #  that should be accumulated and will automatically
        #  be synchronized between devices
        self.add_state("correct", default=tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        # update takes `preds` and `target` and accumulate the current
        # stream of data into the global states for later
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self) -> Tensor:
        # compute takes the accumulated states
        # and returns the final metric value
        return self.correct / self.total
```

Another core feature of TorchMetrics is its ability to scale to multiple devices seamlessly. Modern deep learning models are often trained on hundreds of devices such as GPUs or TPUs (see @large_example1; @large_example2 for examples). This scale introduces the need to synchronize metrics across machines to get the correct value during training and evaluation. In distributed environments, TorchMetrics automatically accumulates across devices before reporting the calculated metric to the user.

In addition to stateful metrics (called modular metrics in TorchMetrics), we also support a functional interface that works similar to Scikit-learn. This interface provides simple Python functions that take as input PyTorch Tensors and return the corresponding metric as a PyTorch Tensor. These can be used when metrics are evaluated on single devices, and no accumulation is needed, making them very fast to compute.

TorchMetrics exhibits high test coverage on the various configurations, including all three major OS platforms (Linux, macOS, and Windows), and various Python, CUDA, and PyTorch versions. We test both minimum and latest package requirements for all combinations of OS and Python versions and include additional tests for each PyTorch version from 1.3 up to future development versions. On every pull request and merge to master, we run a full test suite. All standard tests run on CPU. In addition, we run all tests on a multi-GPU setting which reflects realistic Deep Learning workloads. For usability, we have auto-generated HTML documentation (hosted at [readthedocs](https://torchmetrics.readthedocs.io/en/stable/)) from the source code which updates in real-time with new merged pull requests.

TorchMetrics is released under the Apache 2.0 license. The source code is available at https://github.com/Lightning-AI/torchmetrics.

# Acknowledgement

The TorchMetrics team thanks Thomas Chaton, Ethan Harris, Carlos Mocholí, Sean Narenthiran, Adrian Wälchli, and Ananth Subramaniam for contributing ideas, participating in discussions on API design, and completing Pull Request reviews. We also thank all of our open-source contributors for reporting and resolving issues with this package. We are grateful to the PyTorch Lightning team for their ongoing and dedicated support of this project, and Grid.ai for providing computing resources and cloud credits needed to run our Continuous Integrations.

# References
