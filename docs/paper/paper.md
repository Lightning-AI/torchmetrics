---
title: 'TorchMetrics - Measuring Reproducibility in PyTorch'
tags:
  - python
  - deep learning
  - pytorch
authors:
  - Nicki Skafte Detlefsen
    affiliation: "1,2"
  - Jiri Borovec
    affiliation: "1"
  - Justus Schock
    affiliation: "1,3"
  - Ananya Harsh Jha
    affiliation: "1"
  - Teddy Koker
    affiliation: "1"
  - Luca Di Liello
    affiliation: "4"
  - Daniel Stancl
    affiliation: "5"
  - Changsheng Quan
    affiliation: "6"
  - William Falcon
    affiliation: "1,7"
affiliations:
 - name: Grid AI
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
 - name: New York University
   index: 7
date: 08 Dec 2021
bibliography: paper.bib
---

# Summary

Among recent machine learning publications, many cannot be reproduced \[@reproducibility\]. This is often due to different metrics implementations, which may lead to important training mechanisms like checkpointing, learning rate schedules, or early stopping to behave differently. For example, complex metrics like Fréchet inception distance (FID) for synthetic image quality evaluation \[@fid\] may differ widely based on the specific interpolation method used.

A metric\[^1\] hereby is an objective quantitative measure of the model’s performance. To tackle the problem of reproducibility, there are initiatives such as Papers With Code \[@papers_with_code\], which link the research code with the corresponding papers. However, not all authors provide their code openly accessible. Therefore, we aim to tackle the problem another way - We believe that unified implementations of the relevant metrics can help overcome reproducibility by providing a de-facto reference implementation.

\[^1\]: Not to be confused with mathematical definitions of metrics as per https://en.wikipedia.org/wiki/Metric_(mathematics) whose properties do not have to hold for the machine learning understanding of the same word.

TorchMetrics is a general-purpose metrics package covering a wide variety of tasks and domains. It includes standard classification and regression metrics and some domain-specific, for example, audio, computer vision, natural language processing, and information retrieval. In the case of particular metrics, we integrate with a well-tested and established third-party package as a starter, and later on, we re-implement them as PyTorch \[@pytorch\] native to keep minimal dependencies, make use of available hardware accelerators (mainly GPUs), avoid inter-device transfers whenever possible and maintain full control over each metric implementation.

# Statement of need

Currently, there exists no native package in PyTorch that implements commonly-used metrics within machine learning that is commonly adopted. Some native PyTorch libraries support domain-specific metrics such as Transformers \[@transformers\] for calculating NLP-specific metrics. However, no library exists that covers multiple domains. PyTorch users, therefore, often rely on non-PyTorch packages such as Scikit-learn \[@scikit_learn\] for computing even simple metrics such as accuracy, F1, or AUROC metrics.

However, while Scikit-learn is considered the gold standard for computing metrics in regression and classification, it relies on the core assumption that all predictions and targets are available simultaneously. This contradicts the typical workflow in a modern deep learning training/evaluation loop where data comes in batches. Therefore, the metric needs to be calculated in an online fashion. It is important to note that, in general, it is not possible to calculate a global metric as its average or sum of the metric calculated per batch.

TorchMetrics solves this problem by introducing stateful metrics that can calculate metric values on a stream of data alongside the classical functional and stateless metrics provided by other packages like Scikit-learn. We do this with an effortless `update` and `compute` interface, well known from packages such as Keras \[@keras\]. The `update` function takes in a batch of predictions and targets and updates the internal state. For example, for a metric such as accuracy, the internal states are simply the number of correctly classified samples and the total observed number of samples. When all batches have been passed to the `update` method, the `compute` method can get the accumulated accuracy over all the batches. In addition to `update` and `compute`, each metric also has a `forward` method (as any other `torch.nn.Module`) that can be used to both get the metric on the current batch of data and accumulate global state. This enables the user to get fine-grained info about the metric on the individual batch and the global metric of how well their model is doing.

```python
# Minimal example showcasing the Torchmetrics interface
import torch
from torchmetrics import Metric  # base class all modular metrics inherit from


class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        # self.add_state defines the states of the metric that should be accumulated
        # and will automatically be synchronized between devices
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # update takes preds and target and accumulate the current stream
        # of data into the global states for later
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self) -> torch.Tensor:
        # compute takes the accumulated states and returns the final metric
        return self.correct / self.total
```

Another core feature of TorchMetrics is its ability to scale to multiple devices seamlessly. Modern deep learning models are often trained on hundreds of devices GPUs or TPUs (see \[@large_example1\] \[@large_example2\] for examples), and the corresponding metrics calculated during training and evaluation, therefore, need to be synchronized to get the correct value. TorchMetrics completely takes care of this in the background, automatically detecting if a metric is being updated on multiple devices and accumulating the states from different devices before reporting the calculated metric to the user.

In addition to stateful metrics (called modular metrics in TorchMetrics), we also support a functional interface that works similar to Scikit-learn. They are simple python functions that, as input, take PyTorch Tensors and return the corresponding metric as a PyTorch Tensor. These can be used when metrics are evaluated on single devices, and no accumulation is needed, making them very fast to compute.

TorchMetrics exhibits high test coverage on the various configurations, including all three major OS platforms (Linux, macOS and Windows), Python, CUDA and PyTorch versions. We test both minimum and latest package requirements for all combinations of OS and Python versions and include additional tests for each PyTorch version from 1.3 up to future development versions. As the development cycle, we run a full test suite for each merge to master branch and a significant portion of these tests for each open PR by any contributor. All standard tests are running on CPU, but as Deep Learning is mainly focused on GPU computing, we are running all tests also in a multi-GPU setting in the cloud. For usability, we have auto-generated HTML documentation at https://torchmetrics.readthedocs.io/en/stable/ from the source code to keep it tied with all API changes.

TorchMetrics is released under the Apache 2.0 license, and the source code can be found at https://github.com/PytorchLightning/metrics.

# Acknowledgement

The TorchMetrics team thanks Thomas Chaton, Ethan Harris, Carlos Mocholí, Sean Naren and Adrian Wälchli for contributing ideas, discussions on API design, and contributing with Pull Request reviews. We also thank all our open-source contributors for reporting and resolving issues with this package. We are grateful to the PyTorch Lightning team for their ongoing and dedicated support of this project, and Grid.ai for providing computing resources and cloud credits needed to run our Continues Integrations.

# References
