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

.. autoclass:: torchmetrics.BinaryStatScores
    :noindex:
    :exclude-members: update, compute

MulticlassStatScores
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.MulticlassStatScores
    :noindex:
    :exclude-members: update, compute

MultilabelStatScores
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.MultilabelStatScores
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

.. autofunction:: torchmetrics.functional.binary_stat_scores
    :noindex:

multiclass_stat_scores
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.multiclass_stat_scores
    :noindex:

multilabel_stat_scores
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.multilabel_stat_scores
    :noindex:
