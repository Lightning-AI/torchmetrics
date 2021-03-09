
####################
What is Torchmetrics
####################

Torchmetrics is a metrics API created for easy metric development and usage in both PyTorch and
`PyTorch Lightning <https://pytorch-lightning.readthedocs.io/en/stable/>`_. It was originally a part of
Pytorch Lightning, but got split off so all PyTorch users could take advantage of the large collection of metrics
implemented.
We currently have around 25+ metrics implemented and we continuesly is adding more metrics, both within
already covered domains (classification, regression ect.) but also new domains (object detection ect.).
We make sure that all our metrics are rigorously tested against other popular implemenetations.

Installation
============

.. code-block:: bash

    pip install torchmetrics

Available Metrics
=================


Class classification metrics
============================
.. currentmodule:: torchmetrics

.. autosummary::
    :toctree: generated
    :nosignatures:

    Accuracy
    AveragePrecision
    AUC
    AUROC
    ConfusionMatrix
    F1
    FBeta
    IoU
    HammingDistance
    Precision
    PrecisionRecallCurve
    Recall
    ROC
    StatScores

Class regression metrics
========================
.. currentmodule:: torchmetrics

.. autosummary::
    :toctree: generated
    :nosignatures:

    ExplainedVariance
    MeanAbsoluteError
    MeanSquaredError
    MeanSquaredLogError
    PSNR
    SSIM
    R2Score

Functional classification metrics
=================================
.. currentmodule:: torchmetrics.functional

.. autosummary::
    :toctree: generated
    :nosignatures:

    accuracy
    auc
    auroc
    average_precision
    confusion_matrix
    dice_score
    f1
    fbeta
    hamming_distance
    iou
    roc
    precision
    precision_recall
    precision_recall_curve
    recall
    stat_scores

Functional regression metrics
=============================

.. currentmodule:: torchmetrics.functional

.. autosummary::
    :toctree: generated
    :nosignatures:

    explained_variance
    image_gradients
    mean_absolute_error
    mean_squared_error
    mean_squared_log_error
    psnr
    ssim
    r2score

Functional domain metrics
=========================

.. currentmodule:: torchmetrics.functional

.. autosummary::
    :toctree: generated
    :nosignatures:

    bleu_score
    embedding_similarity
