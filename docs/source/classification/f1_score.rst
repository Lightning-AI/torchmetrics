.. customcarditem::
   :header: F-1 Score
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

#########
F-1 Score
#########

Module Interface
________________

F1Score
^^^^^^^

.. autoclass:: torchmetrics.F1Score
    :noindex:

BinaryF1Score
^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryF1Score
    :noindex:

MulticlassF1Score
^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassF1Score
    :noindex:

MultilabelF1Score
^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelF1Score
    :noindex:

Functional Interface
____________________

f1_score
^^^^^^^^

.. autofunction:: torchmetrics.functional.f1_score
    :noindex:

binary_f1_score
^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_f1_score
    :noindex:

multiclass_f1_score
^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_f1_score
    :noindex:

multilabel_f1_score
^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_f1_score
    :noindex:
