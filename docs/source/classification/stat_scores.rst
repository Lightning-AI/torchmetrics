.. customcarditem::
   :header: Stat Scores
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

###########
Stat Scores
###########

Module Interface
________________

StatScores
^^^^^^^^^^

.. autoclass:: torchmetrics.StatScores
    :noindex:

BinaryStatScores
^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryStatScores
    :noindex:
    :exclude-members: update, compute

MulticlassStatScores
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassStatScores
    :noindex:
    :exclude-members: update, compute

MultilabelStatScores
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelStatScores
    :noindex:
    :exclude-members: update, compute

Functional Interface
____________________

stat_scores
^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.stat_scores
    :noindex:

binary_stat_scores
^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_stat_scores
    :noindex:

multiclass_stat_scores
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_stat_scores
    :noindex:

multilabel_stat_scores
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_stat_scores
    :noindex:
