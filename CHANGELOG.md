# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.3.0] - 2021-04-DD

### Added

- Added `BootStrapper` to easely calculate confidence intervals for metrics ([#101](https://github.com/PyTorchLightning/metrics/pull/101))
- Added metrics for Information Retrieval ([(PL^5032)](https://github.com/PyTorchLightning/pytorch-lightning/pull/5032)):
    * Added `RetrievalMAP` ([#5032](https://github.com/PyTorchLightning/pytorch-lightning/pull/5032))
    * Added `RetrievalMRR` ([#119](https://github.com/PyTorchLightning/metrics/pull/119))
    * Added `RetrievalPrecision` ([#139](https://github.com/PyTorchLightning/metrics/pull/139))
    * Added `RetrievalRecall` ([#146](https://github.com/PyTorchLightning/metrics/pull/146))
    * Added `RetrievalNormalizedDCG` ([#160](https://github.com/PyTorchLightning/metrics/pull/160))
    * Added `RetrievalFallOut` ([#161](https://github.com/PyTorchLightning/metrics/pull/161))
- Added other metrics:
    * Added `CohenKappa` ([#69](https://github.com/PyTorchLightning/metrics/pull/69))
    * Added `MatthewsCorrcoef` ([#98](https://github.com/PyTorchLightning/metrics/pull/98))
    * Added `PearsonCorrcoef` ([#157](https://github.com/PyTorchLightning/metrics/pull/157))
    * Added `SpearmanCorrcoef` ([#158](https://github.com/PyTorchLightning/metrics/pull/158))
    * Added `Hinge` ([#120](https://github.com/PyTorchLightning/metrics/pull/120))
- Added Binned metrics  ([#128](https://github.com/PyTorchLightning/metrics/pull/128))
- Added `average='micro'` as an option in AUROC for multilabel problems ([#110](https://github.com/PyTorchLightning/metrics/pull/110))
- Added multilabel support to `ROC` metric ([#114](https://github.com/PyTorchLightning/metrics/pull/114))
- Added testing for `half` precision ([#77](https://github.com/PyTorchLightning/metrics/pull/77),
    [#135](https://github.com/PyTorchLightning/metrics/pull/135)
)
- Added `AverageMeter` for ad-hoc averages of values ([#138](https://github.com/PyTorchLightning/metrics/pull/138))
- Added `prefix` argument to `MetricCollection` ([#70](https://github.com/PyTorchLightning/metrics/pull/70))
- Added `__getitem__` as metric arithmetic operation ([#142](https://github.com/PyTorchLightning/metrics/pull/142))
- Added property `is_differentiable` to metrics and test for differentiability ([#154](https://github.com/PyTorchLightning/metrics/pull/154))
- Added support for `average`, `ignore_index` and `mdmc_average` in `Accuracy` metric ([#166](https://github.com/PyTorchLightning/metrics/pull/166))

### Changed

- Changed `ExplainedVariance` from storing all preds/targets to tracking 5 statistics ([#68](https://github.com/PyTorchLightning/metrics/pull/68))
- Changed behaviour of `confusionmatrix` for multilabel data to better match `multilabel_confusion_matrix` from sklearn ([#134](https://github.com/PyTorchLightning/metrics/pull/134))
- Updated FBeta arguments ([#111](https://github.com/PyTorchLightning/metrics/pull/111))
- Changed `reset` method to use `detach.clone()` instead of `deepcopy` when resetting to default ([#163](https://github.com/PyTorchLightning/metrics/pull/163))
- Metrics passed as dict to `MetricCollection` will now always be in deterministic order ([#173](https://github.com/PyTorchLightning/metrics/pull/173))
- Allowed `MetricCollection` pass metrics as arguments ([#176](https://github.com/PyTorchLightning/metrics/pull/176))

### Deprecated

- Rename argument `is_multiclass` -> `multiclass` ([#162](https://github.com/PyTorchLightning/metrics/pull/162))

### Removed

- Prune remaining deprecated ([#92](https://github.com/PyTorchLightning/metrics/pull/92))

### Fixed

- Fixed when `_stable_1d_sort` to work when n >= N ([PL^6177](https://github.com/PyTorchLightning/pytorch-lightning/pull/6177))
- Fixed `_computed` attribute not being correctly reset ([#147](https://github.com/PyTorchLightning/metrics/pull/147))
- Fixed to blau score ([#165](https://github.com/PyTorchLightning/metrics/pull/165))


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
