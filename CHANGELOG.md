# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.2.0] - 2021-03-12


### Changed

- Decoupled PL dependency ([#13](https://github.com/PyTorchLightning/metrics/pull/13))
- Refactored functional - mimic the module-like structure: classification, regression, etc. ([#16](https://github.com/PyTorchLightning/metrics/pull/16))
- Refactored utilities -  split to topics/submodules ([#14](https://github.com/PyTorchLightning/metrics/pull/14))
- Refactored `MetricCollection` ([#19](https://github.com/PyTorchLightning/metrics/pull/19))

### Removed

- Removed deprecated metrics from PL base ([#12](https://github.com/PyTorchLightning/metrics/pull/12),
    [#15](https://github.com/PyTorchLightning/metrics/pull/15))



## [0.1.0] - 2021-02-22

- Added `Accuracy` metric now generalizes to Top-k accuracy for (multi-dimensional) multi-class inputs using the `top_k` parameter ([PL^4838](https://github.com/PyTorchLightning/pytorch-lightning/pull/4838))
- Added `Accuracy` metric now enables the computation of subset accuracy for multi-label or multi-dimensional multi-class inputs with the `subset_accuracy` parameter ([PL^4838](https://github.com/PyTorchLightning/pytorch-lightning/pull/4838))
- Added `HammingDistance` metric to compute the hamming distance (loss) ([PL^4838](https://github.com/PyTorchLightning/pytorch-lightning/pull/4838))
- Added `StatScores` metric to compute the number of true positives, false positives, true negatives and false negatives ([PL^4839](https://github.com/PyTorchLightning/pytorch-lightning/pull/4839))
- Added `R2Score` metric ([PL^5241](https://github.com/PyTorchLightning/pytorch-lightning/pull/5241))
- Added `MetricCollection` ([PL^4318](https://github.com/PyTorchLightning/pytorch-lightning/pull/4318))
- Added `.clone()` method to metrics ([PL^4318](https://github.com/PyTorchLightning/pytorch-lightning/pull/4318))
- Added `IoU` class interface ([PL^4704](https://github.com/PyTorchLightning/pytorch-lightning/pull/4704))
- The `Recall` and `Precision` metrics (and their functional counterparts `recall` and `precision`) can now be generalized to Recall@K and Precision@K with the use of `top_k` parameter ([PL^4842](https://github.com/PyTorchLightning/pytorch-lightning/pull/4842))
- Added compositional metrics ([PL^5464](https://github.com/PyTorchLightning/pytorch-lightning/pull/5464))
- Added AUC/AUROC class interface ([PL^5479](https://github.com/PyTorchLightning/pytorch-lightning/pull/5479))
- Added `QuantizationAwareTraining` callback ([PL^5706](https://github.com/PyTorchLightning/pytorch-lightning/pull/5706))
- Added `ConfusionMatrix` class interface ([PL^4348](https://github.com/PyTorchLightning/pytorch-lightning/pull/4348))
- Added multiclass AUROC metric ([PL^4236](https://github.com/PyTorchLightning/pytorch-lightning/pull/4236))
- Added `PrecisionRecallCurve, ROC, AveragePrecision` class metric ([PL^4549](https://github.com/PyTorchLightning/pytorch-lightning/pull/4549))
- Classification metrics overhaul ([PL^4837](https://github.com/PyTorchLightning/pytorch-lightning/pull/4837))
- Added `F1` class metric ([PL^4656](https://github.com/PyTorchLightning/pytorch-lightning/pull/4656))
- Added metrics aggregation in Horovod and fixed early stopping ([PL^3775](https://github.com/PyTorchLightning/pytorch-lightning/pull/3775))
- Added `persistent(mode)` method to metrics, to enable and disable metric states being added to `state_dict` ([PL^4482](https://github.com/PyTorchLightning/pytorch-lightning/pull/4482))
- Added unification of regression metrics ([PL^4166](https://github.com/PyTorchLightning/pytorch-lightning/pull/4166))
- Added persistent flag to `Metric.add_state` ([PL^4195](https://github.com/PyTorchLightning/pytorch-lightning/pull/4195))
- Added classification metrics ([PL^4043](https://github.com/PyTorchLightning/pytorch-lightning/pull/4043))
- Added new Metrics API. ([PL^3868](https://github.com/PyTorchLightning/pytorch-lightning/pull/3868), [PL^3921](https://github.com/PyTorchLightning/pytorch-lightning/pull/3921))
- Added EMB similarity ([PL^3349](https://github.com/PyTorchLightning/pytorch-lightning/pull/3349))
- Added SSIM metrics ([PL^2671](https://github.com/PyTorchLightning/pytorch-lightning/pull/2671))
- Added BLEU metrics ([PL^2535](https://github.com/PyTorchLightning/pytorch-lightning/pull/2535))