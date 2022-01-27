# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Note: we move fast, but still we preserve 0.1 version (one feature release) back compatibility.**


## [UnReleased] - 2022-MM-DD

### Added

- Added support for `MetricCollection` in `MetricTracker` ([#718](https://github.com/PyTorchLightning/metrics/pull/718))

### Changed

- Used `torch.bucketize` in calibration error when `torch>1.8` for faster computations ([#769](https://github.com/PyTorchLightning/metrics/pull/769))

### Deprecated


### Removed

- Removed deprecated functions, and warnings in Text ([#773](https://github.com/PyTorchLightning/metrics/pull/773))
  * `functional.wer`
  * `WER`


- Removed deprecated functions, and warnings in detection and pairwise ([#804](https://github.com/PyTorchLightning/metrics/pull/804))
  * `MAP`
  * `functional.pairwise.manhatten`


- Removed deprecated functions, and warnings in Audio ([#805](https://github.com/PyTorchLightning/metrics/pull/805))
  * `PESQ`
  * `PIT`
  * `SDR`
  * `SNR`
  * `STOI`
  * `functional.audio.pesq`
  * `functional.audio.pit`
  * `functional.audio.sdr`
  * `functional.audio.snr`
  * `functional.audio.stoi`
  * `functional.audio.si_sdr`
  * `functional.audio.si_snr`


### Fixed

- Fixed check for available modules ([#772](https://github.com/PyTorchLightning/metrics/pull/772))


- Fixed Matthews correlation coefficient when the denominator is 0 ([#781](https://github.com/PyTorchLightning/metrics/pull/781))


## [0.7.0] - 2022-01-17

### Added

- Added NLP metrics:
  - `MatchErrorRate` ([#619](https://github.com/PyTorchLightning/metrics/pull/619))
  - `WordInfoLost` and `WordInfoPreserved` ([#630](https://github.com/PyTorchLightning/metrics/pull/630))
  - `SQuAD` ([#623](https://github.com/PyTorchLightning/metrics/pull/623))
  - `CHRFScore` ([#641](https://github.com/PyTorchLightning/metrics/pull/641))
  - `TranslationEditRate` ([#646](https://github.com/PyTorchLightning/metrics/pull/646))
  - `ExtendedEditDistance` ([#668](https://github.com/PyTorchLightning/metrics/pull/668))
- Added `MultiScaleSSIM` into image metrics ([#679](https://github.com/PyTorchLightning/metrics/pull/679))
- Added Signal to Distortion Ratio (`SDR`) to audio package ([#565](https://github.com/PyTorchLightning/metrics/pull/565))
- Added `MinMaxMetric` to wrappers ([#556](https://github.com/PyTorchLightning/metrics/pull/556))
- Added `ignore_index` to retrieval metrics ([#676](https://github.com/PyTorchLightning/metrics/pull/676))
- Added support for multi references in `ROUGEScore` ([#680](https://github.com/PyTorchLightning/metrics/pull/680))
- Added a default VSCode devcontainer configuration ([#621](https://github.com/PyTorchLightning/metrics/pull/621))

### Changed

- Scalar metrics will now consistently have additional dimensions squeezed ([#622](https://github.com/PyTorchLightning/metrics/pull/622))
- Metrics having third party dependencies removed from global import ([#463](https://github.com/PyTorchLightning/metrics/pull/463))
- Untokenized for `BLEUScore` input stay consistent with all the other text metrics ([#640](https://github.com/PyTorchLightning/metrics/pull/640))
- Arguments reordered for `TER`, `BLEUScore`, `SacreBLEUScore`, `CHRFScore` now expect input order as predictions first and target second ([#696](https://github.com/PyTorchLightning/metrics/pull/696))
- Changed dtype of metric state from `torch.float` to `torch.long` in `ConfusionMatrix` to accommodate larger values ([#715](https://github.com/PyTorchLightning/metrics/pull/715))
- Unify `preds`, `target` input argument's naming across all text metrics ([#723](https://github.com/PyTorchLightning/metrics/pull/723), [#727](https://github.com/PyTorchLightning/metrics/pull/727))
  * `bert`, `bleu`, `chrf`, `sacre_bleu`, `wip`, `wil`, `cer`, `ter`, `wer`, `mer`, `rouge`, `squad`

### Deprecated

- Renamed IoU -> Jaccard Index ([#662](https://github.com/PyTorchLightning/metrics/pull/662))
- Renamed text WER metric ([#714](https://github.com/PyTorchLightning/metrics/pull/714))
  * `functional.wer` -> `functional.word_error_rate`
  * `WER` -> `WordErrorRate`
- Renamed correlation coefficient classes: ([#710](https://github.com/PyTorchLightning/metrics/pull/710))
  * `MatthewsCorrcoef` -> `MatthewsCorrCoef`
  * `PearsonCorrcoef` -> `PearsonCorrCoef`
  * `SpearmanCorrcoef` -> `SpearmanCorrCoef`
- Renamed audio STOI metric: ([#753](https://github.com/PyTorchLightning/metrics/pull/753), [#758](https://github.com/PyTorchLightning/metrics/pull/758))
  * `audio.STOI` to `audio.ShortTimeObjectiveIntelligibility`
  * `functional.audio.stoi` to `functional.audio.short_time_objective_intelligibility`
- Renamed audio PESQ metrics: ([#751](https://github.com/PyTorchLightning/metrics/pull/751))
  * `functional.audio.pesq` -> `functional.audio.perceptual_evaluation_speech_quality`
  * `audio.PESQ` -> `audio.PerceptualEvaluationSpeechQuality`
- Renamed audio SDR metrics: ([#711](https://github.com/PyTorchLightning/metrics/pull/711))
  * `functional.sdr` -> `functional.signal_distortion_ratio`
  * `functional.si_sdr` -> `functional.scale_invariant_signal_distortion_ratio`
  * `SDR` -> `SignalDistortionRatio`
  * `SI_SDR` -> `ScaleInvariantSignalDistortionRatio`
- Renamed audio SNR metrics: ([#712](https://github.com/PyTorchLightning/metrics/pull/712))
  * `functional.snr` -> `functional.signal_distortion_ratio`
  * `functional.si_snr` -> `functional.scale_invariant_signal_noise_ratio`
  * `SNR` -> `SignalNoiseRatio`
  * `SI_SNR` -> `ScaleInvariantSignalNoiseRatio`
- Renamed F-score metrics: ([#731](https://github.com/PyTorchLightning/metrics/pull/731), [#740](https://github.com/PyTorchLightning/metrics/pull/740))
  * `functional.f1` ->  `functional.f1_score`
  * `F1` ->  `F1Score`
  * `functional.fbeta` ->  `functional.fbeta_score`
  * `FBeta` ->  `FBetaScore`
- Renamed Hinge metric: ([#734](https://github.com/PyTorchLightning/metrics/pull/734))
  * `functional.hinge` ->  `functional.hinge_loss`
  * `Hinge` ->  `HingeLoss`
- Renamed image PSNR metrics ([#732](https://github.com/PyTorchLightning/metrics/pull/732))
  * `functional.psnr` -> `functional.peak_signal_noise_ratio`
  * `PSNR` -> `PeakSignalNoiseRatio`
- Renamed image PIT metric: ([#737](https://github.com/PyTorchLightning/metrics/pull/737))
  * `functional.pit` ->  `functional.permutation_invariant_training`
  * `PIT` ->  `PermutationInvariantTraining`
- Renamed image SSIM metric: ([#747](https://github.com/PyTorchLightning/metrics/pull/747))
  * `functional.ssim` ->  `functional.scale_invariant_signal_noise_ratio`
  * `SSIM` ->  `StructuralSimilarityIndexMeasure`
- Renamed detection `MAP` to `MeanAveragePrecision` metric ([#754](https://github.com/PyTorchLightning/metrics/pull/754))
- Renamed Fidelity & LPIPS image metric: ([#752](https://github.com/PyTorchLightning/metrics/pull/752))
  * `image.FID` ->  `image.FrechetInceptionDistance`
  * `image.KID` ->  `image.KernelInceptionDistance`
  * `image.LPIPS` ->  `image.LearnedPerceptualImagePatchSimilarity`

### Removed

- Removed `embedding_similarity` metric ([#638](https://github.com/PyTorchLightning/metrics/pull/638))
- Removed argument `concatenate_texts` from `wer` metric ([#638](https://github.com/PyTorchLightning/metrics/pull/638))
- Removed arguments `newline_sep` and `decimal_places` from `rouge` metric ([#638](https://github.com/PyTorchLightning/metrics/pull/638))

### Fixed

- Fixed MetricCollection kwargs filtering when no `kwargs` are present in update signature ([#707](https://github.com/PyTorchLightning/metrics/pull/707))


## [0.6.2] - 2021-12-15

### Fixed

- Fixed `torch.sort` currently does not support bool `dtype` on CUDA ([#665](https://github.com/PyTorchLightning/metrics/pull/665))
- Fixed mAP properly checks if ground truths are empty ([#684](https://github.com/PyTorchLightning/metrics/pull/684))
- Fixed initialization of tensors to be on correct device for `MAP` metric ([#673](https://github.com/PyTorchLightning/metrics/pull/673))


## [0.6.1] - 2021-12-06

### Changed

- Migrate MAP metrics from pycocotools to PyTorch ([#632](https://github.com/PyTorchLightning/metrics/pull/632))
- Use `torch.topk` instead of `torch.argsort` in retrieval precision for speedup ([#627](https://github.com/PyTorchLightning/metrics/pull/627))

### Fixed

- Fix empty predictions in MAP metric ([#594](https://github.com/PyTorchLightning/metrics/pull/594), [#610](https://github.com/PyTorchLightning/metrics/pull/610), [#624](https://github.com/PyTorchLightning/metrics/pull/624))
- Fix edge case of AUROC with `average=weighted` on GPU ([#606](https://github.com/PyTorchLightning/metrics/pull/606))
- Fixed `forward` in compositional metrics ([#645](https://github.com/PyTorchLightning/metrics/pull/645))


## [0.6.0] - 2021-10-28

### Added

- Added audio metrics:
  - Perceptual Evaluation of Speech Quality (PESQ) ([#353](https://github.com/PyTorchLightning/metrics/pull/353))
  - Short-Time Objective Intelligibility (STOI) ([#353](https://github.com/PyTorchLightning/metrics/pull/353))
- Added Information retrieval metrics:
  - `RetrievalRPrecision` ([#577](https://github.com/PyTorchLightning/metrics/pull/577))
  - `RetrievalHitRate` ([#576](https://github.com/PyTorchLightning/metrics/pull/576))
- Added NLP metrics:
  - `SacreBLEUScore` ([#546](https://github.com/PyTorchLightning/metrics/pull/546))
  - `CharErrorRate` ([#575](https://github.com/PyTorchLightning/metrics/pull/575))
- Added other metrics:
  - Tweedie Deviance Score ([#499](https://github.com/PyTorchLightning/metrics/pull/499))
  - Learned Perceptual Image Patch Similarity (LPIPS) ([#431](https://github.com/PyTorchLightning/metrics/pull/431))
- Added `MAP` (mean average precision) metric to new detection package ([#467](https://github.com/PyTorchLightning/metrics/pull/467))
- Added support for float targets in `nDCG` metric ([#437](https://github.com/PyTorchLightning/metrics/pull/437))
- Added `average` argument to `AveragePrecision` metric for reducing multi-label and multi-class problems ([#477](https://github.com/PyTorchLightning/metrics/pull/477))
- Added `MultioutputWrapper` ([#510](https://github.com/PyTorchLightning/metrics/pull/510))
- Added metric sweeping:
  - `higher_is_better` as constant attribute ([#544](https://github.com/PyTorchLightning/metrics/pull/544))
  - `higher_is_better` to rest of codebase ([#584](https://github.com/PyTorchLightning/metrics/pull/584))
- Added simple aggregation metrics: `SumMetric`, `MeanMetric`, `CatMetric`, `MinMetric`, `MaxMetric` ([#506](https://github.com/PyTorchLightning/metrics/pull/506))
- Added pairwise submodule with metrics ([#553](https://github.com/PyTorchLightning/metrics/pull/553))
  - `pairwise_cosine_similarity`
  - `pairwise_euclidean_distance`
  - `pairwise_linear_similarity`
  - `pairwise_manhatten_distance`

### Changed

- `AveragePrecision` will now as default output the `macro` average for multilabel and multiclass problems ([#477](https://github.com/PyTorchLightning/metrics/pull/477))
- `half`, `double`, `float` will no longer change the dtype of the metric states. Use `metric.set_dtype` instead ([#493](https://github.com/PyTorchLightning/metrics/pull/493))
- Renamed `AverageMeter` to `MeanMetric` ([#506](https://github.com/PyTorchLightning/metrics/pull/506))
- Changed `is_differentiable` from property to a constant attribute ([#551](https://github.com/PyTorchLightning/metrics/pull/551))
- `ROC` and `AUROC` will no longer throw an error when either the positive or negative class is missing. Instead return 0 score and give a warning

### Deprecated

- Deprecated  `functional.self_supervised.embedding_similarity` in favour of new pairwise submodule

### Removed

- Removed `dtype` property ([#493](https://github.com/PyTorchLightning/metrics/pull/493))

### Fixed

- Fixed bug in `F1` with `average='macro'` and `ignore_index!=None` ([#495](https://github.com/PyTorchLightning/metrics/pull/495))
- Fixed bug in `pit` by using the returned first result to initialize device and type ([#533](https://github.com/PyTorchLightning/metrics/pull/533))
- Fixed `SSIM` metric using too much memory ([#539](https://github.com/PyTorchLightning/metrics/pull/539))
- Fixed bug where `device` property was not properly update when metric was a child of a module (#542)

## [0.5.1] - 2021-08-30

### Added

- Added `device` and `dtype` properties ([#462](https://github.com/PyTorchLightning/metrics/pull/462))
- Added `TextTester` class for robustly testing text metrics ([#450](https://github.com/PyTorchLightning/metrics/pull/450))

### Changed

- Added support for float targets in `nDCG` metric ([#437](https://github.com/PyTorchLightning/metrics/pull/437))

### Removed

- Removed `rouge-score` as dependency for text package ([#443](https://github.com/PyTorchLightning/metrics/pull/443))
- Removed `jiwer` as dependency for text package ([#446](https://github.com/PyTorchLightning/metrics/pull/446))
- Removed `bert-score` as dependency for text package ([#473](https://github.com/PyTorchLightning/metrics/pull/473))

### Fixed

- Fixed ranking of samples in `SpearmanCorrCoef` metric ([#448](https://github.com/PyTorchLightning/metrics/pull/448))
- Fixed bug where compositional metrics where unable to sync because of type mismatch ([#454](https://github.com/PyTorchLightning/metrics/pull/454))
- Fixed metric hashing ([#478](https://github.com/PyTorchLightning/metrics/pull/478))
- Fixed `BootStrapper` metrics not working on GPU ([#462](https://github.com/PyTorchLightning/metrics/pull/462))
- Fixed the semantic ordering of kernel height and width in `SSIM` metric ([#474](https://github.com/PyTorchLightning/metrics/pull/474))


## [0.5.0] - 2021-08-09

### Added

- Added **Text-related (NLP) metrics**:
  - Word Error Rate (WER) ([#383](https://github.com/PyTorchLightning/metrics/pull/383))
  - ROUGE ([#399](https://github.com/PyTorchLightning/metrics/pull/399))
  - BERT score ([#424](https://github.com/PyTorchLightning/metrics/pull/424))
  - BLUE score ([#360](https://github.com/PyTorchLightning/metrics/pull/360))
- Added `MetricTracker` wrapper metric for keeping track of the same metric over multiple epochs ([#238](https://github.com/PyTorchLightning/metrics/pull/238))
- Added other metrics:
  - Symmetric Mean Absolute Percentage error (SMAPE) ([#375](https://github.com/PyTorchLightning/metrics/pull/375))
  - Calibration error ([#394](https://github.com/PyTorchLightning/metrics/pull/394))
  - Permutation Invariant Training (PIT) ([#384](https://github.com/PyTorchLightning/metrics/pull/384))
- Added support in `nDCG` metric for target with values larger than 1 ([#349](https://github.com/PyTorchLightning/metrics/pull/349))
- Added support for negative targets in `nDCG` metric ([#378](https://github.com/PyTorchLightning/metrics/pull/378))
- Added `None` as reduction option in `CosineSimilarity` metric ([#400](https://github.com/PyTorchLightning/metrics/pull/400))
- Allowed passing labels in (n_samples, n_classes) to `AveragePrecision` ([#386](https://github.com/PyTorchLightning/metrics/pull/386))

### Changed

- Moved `psnr` and `ssim` from `functional.regression.*` to `functional.image.*` ([#382](https://github.com/PyTorchLightning/metrics/pull/382))
- Moved `image_gradient` from `functional.image_gradients` to `functional.image.gradients` ([#381](https://github.com/PyTorchLightning/metrics/pull/381))
- Moved `R2Score` from `regression.r2score` to `regression.r2` ([#371](https://github.com/PyTorchLightning/metrics/pull/371))
- Pearson metric now only store 6 statistics instead of all predictions and targets ([#380](https://github.com/PyTorchLightning/metrics/pull/380))
- Use `torch.argmax` instead of `torch.topk` when `k=1` for better performance ([#419](https://github.com/PyTorchLightning/metrics/pull/419))
- Moved check for number of samples in R2 score to support single sample updating ([#426](https://github.com/PyTorchLightning/metrics/pull/426))

### Deprecated

- Rename `r2score` >> `r2_score` and `kldivergence` >> `kl_divergence` in `functional` ([#371](https://github.com/PyTorchLightning/metrics/pull/371))
- Moved `bleu_score` from `functional.nlp` to `functional.text.bleu` ([#360](https://github.com/PyTorchLightning/metrics/pull/360))

### Removed

- Removed restriction that `threshold` has to be in (0,1) range to support logit input (
    [#351](https://github.com/PyTorchLightning/metrics/pull/351)
    [#401](https://github.com/PyTorchLightning/metrics/pull/401))
- Removed restriction that `preds` could not be bigger than `num_classes` to support logit input ([#357](https://github.com/PyTorchLightning/metrics/pull/357))
- Removed module `regression.psnr` and `regression.ssim` ([#382](https://github.com/PyTorchLightning/metrics/pull/382)):
- Removed ([#379](https://github.com/PyTorchLightning/metrics/pull/379)):
    * function `functional.mean_relative_error`
    * `num_thresholds` argument in `BinnedPrecisionRecallCurve`

### Fixed

- Fixed bug where classification metrics with `average='macro'` would lead to wrong result if a class was missing ([#303](https://github.com/PyTorchLightning/metrics/pull/303))
- Fixed `weighted`, `multi-class` AUROC computation to allow for 0 observations of some class, as contribution to final AUROC is 0 ([#376](https://github.com/PyTorchLightning/metrics/pull/376))
- Fixed that `_forward_cache` and `_computed` attributes are also moved to the correct device if metric is moved ([#413](https://github.com/PyTorchLightning/metrics/pull/413))
- Fixed calculation in `IoU` metric when using `ignore_index` argument ([#328](https://github.com/PyTorchLightning/metrics/pull/328))


## [0.4.1] - 2021-07-05

### Changed

- Extend typing ([#330](https://github.com/PyTorchLightning/metrics/pull/330),
    [#332](https://github.com/PyTorchLightning/metrics/pull/332),
    [#333](https://github.com/PyTorchLightning/metrics/pull/333),
    [#335](https://github.com/PyTorchLightning/metrics/pull/335),
    [#314](https://github.com/PyTorchLightning/metrics/pull/314))

### Fixed

- Fixed DDP by `is_sync` logic to `Metric` ([#339](https://github.com/PyTorchLightning/metrics/pull/339))


## [0.4.0] - 2021-06-29

### Added

- Added **Image-related metrics**:
  - Fréchet inception distance (FID) ([#213](https://github.com/PyTorchLightning/metrics/pull/213))
  - Kernel Inception Distance (KID) ([#301](https://github.com/PyTorchLightning/metrics/pull/301))
  - Inception Score ([#299](https://github.com/PyTorchLightning/metrics/pull/299))
  - KL divergence ([#247](https://github.com/PyTorchLightning/metrics/pull/247))
- Added **Audio metrics**: SNR, SI_SDR, SI_SNR ([#292](https://github.com/PyTorchLightning/metrics/pull/292))
- Added other metrics:
  - Cosine Similarity ([#305](https://github.com/PyTorchLightning/metrics/pull/305))
  - Specificity ([#210](https://github.com/PyTorchLightning/metrics/pull/210))
  - Mean Absolute Percentage error (MAPE) ([#248](https://github.com/PyTorchLightning/metrics/pull/248))
- Added `add_metrics` method to `MetricCollection` for adding additional metrics after initialization ([#221](https://github.com/PyTorchLightning/metrics/pull/221))
- Added pre-gather reduction in the case of `dist_reduce_fx="cat"` to reduce communication cost ([#217](https://github.com/PyTorchLightning/metrics/pull/217))
- Added better error message for `AUROC` when `num_classes` is not provided for multiclass input ([#244](https://github.com/PyTorchLightning/metrics/pull/244))
- Added support for unnormalized scores (e.g. logits) in `Accuracy`, `Precision`, `Recall`, `FBeta`, `F1`, `StatScore`, `Hamming`, `ConfusionMatrix` metrics ([#200](https://github.com/PyTorchLightning/metrics/pull/200))
- Added `squared` argument to `MeanSquaredError` for computing `RMSE` ([#249](https://github.com/PyTorchLightning/metrics/pull/249))
- Added `is_differentiable` property to `ConfusionMatrix`, `F1`, `FBeta`, `Hamming`, `Hinge`, `IOU`, `MatthewsCorrcoef`, `Precision`, `Recall`, `PrecisionRecallCurve`, `ROC`, `StatScores` ([#253](https://github.com/PyTorchLightning/metrics/pull/253))
- Added `sync` and `sync_context` methods for manually controlling when metric states are synced ([#302](https://github.com/PyTorchLightning/metrics/pull/302))

### Changed

- Forward cache is reset when `reset` method is called ([#260](https://github.com/PyTorchLightning/metrics/pull/260))
- Improved per-class metric handling for imbalanced datasets for `precision`, `recall`, `precision_recall`, `fbeta`, `f1`, `accuracy`, and `specificity` ([#204](https://github.com/PyTorchLightning/metrics/pull/204))
- Decorated `torch.jit.unused` to `MetricCollection` forward ([#307](https://github.com/PyTorchLightning/metrics/pull/307))
- Renamed `thresholds` argument to binned metrics for manually controlling the thresholds ([#322](https://github.com/PyTorchLightning/metrics/pull/322))
- Extend typing ([#324](https://github.com/PyTorchLightning/metrics/pull/324),
    [#326](https://github.com/PyTorchLightning/metrics/pull/326),
    [#327](https://github.com/PyTorchLightning/metrics/pull/327))

### Deprecated

- Deprecated `functional.mean_relative_error`, use `functional.mean_absolute_percentage_error` ([#248](https://github.com/PyTorchLightning/metrics/pull/248))
- Deprecated `num_thresholds` argument in `BinnedPrecisionRecallCurve` ([#322](https://github.com/PyTorchLightning/metrics/pull/322))

### Removed

- Removed argument `is_multiclass` ([#319](https://github.com/PyTorchLightning/metrics/pull/319))

### Fixed

- AUC can also support more dimensional inputs when all but one dimension are of size 1 ([#242](https://github.com/PyTorchLightning/metrics/pull/242))
- Fixed `dtype` of modular metrics after reset has been called ([#243](https://github.com/PyTorchLightning/metrics/pull/243))
- Fixed calculation in `matthews_corrcoef` to correctly match formula ([#321](https://github.com/PyTorchLightning/metrics/pull/321))

## [0.3.2] - 2021-05-10

### Added

- Added `is_differentiable` property:
    * To `AUC`, `AUROC`, `CohenKappa` and `AveragePrecision` ([#178](https://github.com/PyTorchLightning/metrics/pull/178))
    * To `PearsonCorrCoef`, `SpearmanCorrcoef`, `R2Score` and `ExplainedVariance` ([#225](https://github.com/PyTorchLightning/metrics/pull/225))

### Changed

- `MetricCollection` should return metrics with prefix on `items()`, `keys()` ([#209](https://github.com/PyTorchLightning/metrics/pull/209))
- Calling `compute` before `update` will now give warning ([#164](https://github.com/PyTorchLightning/metrics/pull/164))

### Removed

- Removed `numpy` as direct dependency ([#212](https://github.com/PyTorchLightning/metrics/pull/212))

### Fixed

- Fixed auc calculation and add tests ([#197](https://github.com/PyTorchLightning/metrics/pull/197))
- Fixed loading persisted metric states using `load_state_dict()` ([#202](https://github.com/PyTorchLightning/metrics/pull/202))
- Fixed `PSNR` not working with `DDP` ([#214](https://github.com/PyTorchLightning/metrics/pull/214))
- Fixed metric calculation with unequal batch sizes ([#220](https://github.com/PyTorchLightning/metrics/pull/220))
- Fixed metric concatenation for list states for zero-dim input ([#229](https://github.com/PyTorchLightning/metrics/pull/229))
- Fixed numerical instability in `AUROC` metric for large input ([#230](https://github.com/PyTorchLightning/metrics/pull/230))

## [0.3.1] - 2021-04-21

- Cleaning remaining inconsistency and fix PL develop integration (
    [#191](https://github.com/PyTorchLightning/metrics/pull/191),
    [#192](https://github.com/PyTorchLightning/metrics/pull/192),
    [#193](https://github.com/PyTorchLightning/metrics/pull/193),
    [#194](https://github.com/PyTorchLightning/metrics/pull/194)
)


## [0.3.0] - 2021-04-20

### Added

- Added `BootStrapper` to easily calculate confidence intervals for metrics ([#101](https://github.com/PyTorchLightning/metrics/pull/101))
- Added Binned metrics  ([#128](https://github.com/PyTorchLightning/metrics/pull/128))
- Added metrics for Information Retrieval ([(PL^5032)](https://github.com/PyTorchLightning/pytorch-lightning/pull/5032)):
    * `RetrievalMAP` ([PL^5032](https://github.com/PyTorchLightning/pytorch-lightning/pull/5032))
    * `RetrievalMRR` ([#119](https://github.com/PyTorchLightning/metrics/pull/119))
    * `RetrievalPrecision` ([#139](https://github.com/PyTorchLightning/metrics/pull/139))
    * `RetrievalRecall` ([#146](https://github.com/PyTorchLightning/metrics/pull/146))
    * `RetrievalNormalizedDCG` ([#160](https://github.com/PyTorchLightning/metrics/pull/160))
    * `RetrievalFallOut` ([#161](https://github.com/PyTorchLightning/metrics/pull/161))
- Added other metrics:
    * `CohenKappa` ([#69](https://github.com/PyTorchLightning/metrics/pull/69))
    * `MatthewsCorrcoef` ([#98](https://github.com/PyTorchLightning/metrics/pull/98))
    * `PearsonCorrcoef` ([#157](https://github.com/PyTorchLightning/metrics/pull/157))
    * `SpearmanCorrcoef` ([#158](https://github.com/PyTorchLightning/metrics/pull/158))
    * `Hinge` ([#120](https://github.com/PyTorchLightning/metrics/pull/120))
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
- Added `postfix` arg to `MetricCollection` ([#188](https://github.com/PyTorchLightning/metrics/pull/188))

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

- Fixed when `_stable_1d_sort` to work when `n>=N` ([PL^6177](https://github.com/PyTorchLightning/pytorch-lightning/pull/6177))
- Fixed `_computed` attribute not being correctly reset ([#147](https://github.com/PyTorchLightning/metrics/pull/147))
- Fixed to Blau score ([#165](https://github.com/PyTorchLightning/metrics/pull/165))
- Fixed backwards compatibility for logging with older version of pytorch-lightning ([#182](https://github.com/PyTorchLightning/metrics/pull/182))


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
