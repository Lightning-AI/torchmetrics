.. customcarditem::
   :header: Recall At Fixed Precision
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

#########################
Precision At Fixed Recall
#########################

Module Interface
________________

PrecisionAtFixedRecall
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.PrecisionAtFixedRecall
    :noindex:
    :exclude-members: update, compute
    :special-members: __new__

BinaryPrecisionAtFixedRecall
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryPrecisionAtFixedRecall
    :noindex:
    :exclude-members: update, compute

MulticlassPrecisionAtFixedRecall
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassPrecisionAtFixedRecall
    :noindex:
    :exclude-members: update, compute

MultilabelPrecisionAtFixedRecall
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelPrecisionAtFixedRecall
    :noindex:
    :exclude-members: update, compute

Functional Interface
____________________

binary_precision_at_fixed_recall
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_precision_at_fixed_recall
    :noindex:

multiclass_precision_at_fixed_recall
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_precision_at_fixed_recall
    :noindex:

multilabel_precision_at_fixed_recall
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_precision_at_fixed_recall
    :noindex:
