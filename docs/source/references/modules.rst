##############
Module metrics
##############

**********
Base class
**********

The base ``Metric`` class is an abstract base class that are used as the building block for all other Module
metrics.

.. autoclass:: torchmetrics.Metric
    :noindex:

**********************
Classification Metrics
**********************

Input types
~~~~~~~~~~~

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


Using the is_multiclass parameter
---------------------------------

In some cases, you might have inputs which appear to be (multi-dimensional) multi-class
but are actually binary/multi-label - for example, if both predictions and targets are
integer (binary) tensors. Or it could be the other way around, you want to treat
binary/multi-label inputs as 2-class (multi-dimensional) multi-class inputs.

For these cases, the metrics where this distinction would make a difference, expose the
``is_multiclass`` argument. Let's see how this is used on the example of
:class:`~torchmetrics.StatScores` metric.

First, let's consider the case with label predictions with 2 classes, which we want to
treat as binary.

.. testcode::

   from torchmetrics.functional import stat_scores

   # These inputs are supposed to be binary, but appear as multi-class
   preds  = torch.tensor([0, 1, 0])
   target = torch.tensor([1, 1, 0])

As you can see below, by default the inputs are treated
as multi-class. We can set ``is_multiclass=False`` to treat the inputs as binary -
which is the same as converting the predictions to float beforehand.

.. doctest::

    >>> stat_scores(preds, target, reduce='macro', num_classes=2)
    tensor([[1, 1, 1, 0, 1],
            [1, 0, 1, 1, 2]])
    >>> stat_scores(preds, target, reduce='macro', num_classes=1, is_multiclass=False)
    tensor([[1, 0, 1, 1, 2]])
    >>> stat_scores(preds.float(), target, reduce='macro', num_classes=1)
    tensor([[1, 0, 1, 1, 2]])

Next, consider the opposite example: inputs are binary (as predictions are probabilities),
but we would like to treat them as 2-class multi-class, to obtain the metric for both classes.

.. testcode::

   preds  = torch.tensor([0.2, 0.7, 0.3])
   target = torch.tensor([1, 1, 0])

In this case we can set ``is_multiclass=True``, to treat the inputs as multi-class.

.. doctest::

    >>> stat_scores(preds, target, reduce='macro', num_classes=1)
    tensor([[1, 0, 1, 1, 2]])
    >>> stat_scores(preds, target, reduce='macro', num_classes=2, is_multiclass=True)
    tensor([[1, 1, 1, 0, 1],
            [1, 0, 1, 1, 2]])

Accuracy
~~~~~~~~

.. autoclass:: torchmetrics.Accuracy
    :noindex:

AveragePrecision
~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.AveragePrecision
    :noindex:

AUC
~~~

.. autoclass:: torchmetrics.AUC
    :noindex:

AUROC
~~~~~

.. autoclass:: torchmetrics.AUROC
    :noindex:

CohenKappa
~~~~~~~~~~

.. autoclass:: torchmetrics.CohenKappa
    :noindex:

ConfusionMatrix
~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.ConfusionMatrix
    :noindex:

F1
~~

.. autoclass:: torchmetrics.F1
    :noindex:

FBeta
~~~~~

.. autoclass:: torchmetrics.FBeta
    :noindex:

HammingDistance
~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.HammingDistance
    :noindex:

Hinge
~~~~~

.. autoclass:: torchmetrics.Hinge
    :noindex:

IoU
~~~

.. autoclass:: torchmetrics.IoU
    :noindex:

MatthewsCorrcoef
~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.MatthewsCorrcoef
    :noindex:

Precision
~~~~~~~~~

.. autoclass:: torchmetrics.Precision
    :noindex:

PrecisionRecallCurve
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.PrecisionRecallCurve
    :noindex:

Recall
~~~~~~

.. autoclass:: torchmetrics.Recall
    :noindex:

ROC
~~~

.. autoclass:: torchmetrics.ROC
    :noindex:


StatScores
~~~~~~~~~~

.. autoclass:: torchmetrics.StatScores
    :noindex:


******************
Regression Metrics
******************

ExplainedVariance
~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.ExplainedVariance
    :noindex:


MeanAbsoluteError
~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.MeanAbsoluteError
    :noindex:


MeanSquaredError
~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.MeanSquaredError
    :noindex:


MeanSquaredLogError
~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.MeanSquaredLogError
    :noindex:


PSNR
~~~~

.. autoclass:: torchmetrics.PSNR
    :noindex:


SSIM
~~~~

.. autoclass:: torchmetrics.SSIM
    :noindex:


R2Score
~~~~~~~

.. autoclass:: torchmetrics.R2Score
    :noindex:


*********
Retrieval
*********

Input details
~~~~~~~~~~~~~

For the purposes of retrieval metrics, inputs (indexes, predictions and targets) must have the same size
(``N`` stands for the batch size) and the following types:

.. csv-table::
    :header: "indexes shape", "indexes dtype", "preds shape", "preds dtype", "target shape", "target dtype"
    :widths: 10, 10, 10, 10, 10, 10

    "``long``", "(N,...)", "``float``", "(N,...)", "``long`` or ``bool``", "(N,...)"

.. note::
    All dimensions are flattened at the beginning, so
    that, for example, a tensor of shape ``(N, M)`` is treated as ``(N * M, )``.

In Information Retrieval you have a query that is compared with a variable number of documents. For each pair ``(Q_i, D_j)``,
a score is computed that measures the relevance of document ``D`` w.r.t. query ``Q``. Documents are then sorted by score
and you hope that relevant documents are scored higher. ``target`` contains the labels for the documents (relevant or not).

Since a query may be compared with a variable number of documents, we use ``indexes`` to keep track of which scores belong to
the set of pairs ``(Q_i, D_j)`` having the same query ``Q_i``.

.. doctest::

    >>> from torchmetrics import RetrievalMAP
    >>> # functional version works on a single query at a time
    >>> from torchmetrics.functional import retrieval_average_precision

    >>> # the first query was compared with two documents, the second with three
    >>> indexes = torch.tensor([0, 0, 1, 1, 1])
    >>> preds = torch.tensor([0.8, -0.4, 1.0, 1.4, 0.0])
    >>> target = torch.tensor([0, 1, 0, 1, 1])

    >>> map = RetrievalMAP() # or some other retrieval metric
    >>> map(indexes, preds, target)
    tensor(0.6667)

    >>> # the previous instruction is roughly equivalent to
    >>> res = []
    >>> # iterate over indexes of first and second query
    >>> for idx in ([0, 1], [2, 3, 4]):
    ...     res.append(retrieval_average_precision(preds[idx], target[idx]))
    >>> torch.stack(res).mean()
    tensor(0.6667)


RetrievalMAP
~~~~~~~~~~~~

.. autoclass:: torchmetrics.RetrievalMAP
    :noindex:


RetrievalMRR
~~~~~~~~~~~~

.. autoclass:: torchmetrics.RetrievalMRR
    :noindex:


RetrievalPrecision
~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.RetrievalPrecision
    :noindex:
