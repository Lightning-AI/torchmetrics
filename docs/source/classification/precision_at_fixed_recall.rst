.. customcarditem::
   :header: Recall At Fixed Precision
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

#########################
Precision At Fixed Recall
#########################

Module Interface
________________

.. autoclass:: torchmetrics.PrecisionAtFixedRecall
    :exclude-members: update, compute
    :special-members: __new__

BinaryPrecisionAtFixedRecall
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryPrecisionAtFixedRecall
    :exclude-members: update, compute

MulticlassPrecisionAtFixedRecall
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassPrecisionAtFixedRecall
    :exclude-members: update, compute

MultilabelPrecisionAtFixedRecall
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelPrecisionAtFixedRecall
    :exclude-members: update, compute

Functional Interface
____________________

binary_precision_at_fixed_recall
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_precision_at_fixed_recall

multiclass_precision_at_fixed_recall
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_precision_at_fixed_recall

multilabel_precision_at_fixed_recall
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_precision_at_fixed_recall
