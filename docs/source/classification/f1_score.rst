.. customcarditem::
   :header: F-1 Score
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

#########
F-1 Score
#########

Module Interface
________________

.. autoclass:: torchmetrics.F1Score
    :exclude-members: update, compute
    :special-members: __new__

BinaryF1Score
^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryF1Score
    :exclude-members: update, compute

MulticlassF1Score
^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassF1Score
    :exclude-members: update, compute

MultilabelF1Score
^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelF1Score
    :exclude-members: update, compute

Functional Interface
____________________

f1_score
^^^^^^^^

.. autofunction:: torchmetrics.functional.f1_score

binary_f1_score
^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_f1_score

multiclass_f1_score
^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_f1_score

multilabel_f1_score
^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_f1_score
