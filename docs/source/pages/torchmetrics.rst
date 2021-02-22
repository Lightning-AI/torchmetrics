.. role:: hidden
    :class: hidden-section

torchmetrics
===================================

Classification
--------------------

For the purposes of classification metrics, inputs (predictions and targets) are split 
into these categories (``N`` stands for the batch size and ``C`` for number of classes):

.. csv-table:: \*dtype ``binary`` means integers that are either 0 or 1
    :header: "Type", "preds shape", "preds dtype", "target shape", "target dtype"
    :widths: 20, 10, 10, 10, 10

    "Binary", "(N,)", "``float``", "(N,)", "``binary``\*"
    "Multi-class", "(N,)", "``int``", "(N,)", "``int``"
    "Multi-class with probabilities", "(N, C)", "``float``", "(N,)", "``int``"
    "Multi-label", "(N, ...)", "``float``", "(N, ...)", "``binary``\*"
    "Multi-dimensional multi-class", "(N, ...)", "``int``", "(N, ...)", "``int``"
    "Multi-dimensional multi-class with probabilities", "(N, C, ...)", "``float``", "(N, ...)", "``int``"

.. note::
    All dimensions of size 1 (except ``N``) are "squeezed out" at the beginning, so 
    that, for example, a tensor of shape ``(N, 1)`` is treated as ``(N, )``.

When predictions or targets are integers, it is assumed that class labels start at 0, i.e. 
the possible class labels are 0, 1, 2, 3, etc. Below are some examples of different input types

.. testcode::

    # Binary inputs
    binary_preds  = torch.tensor([0.6, 0.1, 0.9])
    binary_target = torch.tensor([1, 0, 2])

    # Multi-class inputs
    mc_preds  = torch.tensor([0, 2, 1])
    mc_target = torch.tensor([0, 1, 2])

    # Multi-class inputs with probabilities
    mc_preds_probs  = torch.tensor([[0.8, 0.2, 0], [0.1, 0.2, 0.7], [0.3, 0.6, 0.1]])
    mc_target_probs = torch.tensor([0, 1, 2])

    # Multi-label inputs
    ml_preds  = torch.tensor([[0.2, 0.8, 0.9], [0.5, 0.6, 0.1], [0.3, 0.1, 0.1]])
    ml_target = torch.tensor([[0, 1, 1], [1, 0, 0], [0, 0, 0]])

In some rare cases, you might have inputs which appear to be (multi-dimensional) multi-class
but are actually binary/multi-label. For example, if both predictions and targets are 1d
binary tensors. Or it could be the other way around, you want to treat binary/multi-label
inputs as 2-class (multi-dimensional) multi-class inputs.

For these cases, the metrics where this distinction would make a difference, expose the
``is_multiclass`` argument.

.. currentmodule:: torchmetrics.classification

.. autosummary::
      :toctree: api 
      :nosignatures:
      :template: classtemplate.rst

      Accuracy
      AveragePrecision
      ConfusionMatrix
      F1
      FBeta
      Precision
      PrecisionRecallCurve
      Recall
      ROC


Regression
--------------------

.. currentmodule:: torchmetrics.regression

.. autosummary::
      :toctree: api 
      :nosignatures:
      :template: classtemplate.rst

      ExplainedVariance
      MeanAbsoluteError
      MeanSquaredError
      MeanSquaredLogError
      PSNR
      SSIM
