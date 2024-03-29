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

.. autoclass:: torchmetrics.StatScores
    :exclude-members: update, compute
    :special-members: __new__

BinaryStatScores
^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryStatScores
    :exclude-members: update, compute

MulticlassStatScores
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassStatScores
    :exclude-members: update, compute

MultilabelStatScores
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelStatScores
    :exclude-members: update, compute

Functional Interface
____________________

stat_scores
^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.stat_scores

binary_stat_scores
^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_stat_scores

multiclass_stat_scores
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_stat_scores

multilabel_stat_scores
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_stat_scores
