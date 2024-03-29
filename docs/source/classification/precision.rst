.. customcarditem::
   :header: Precision
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

#########
Precision
#########

Module Interface
________________

.. autoclass:: torchmetrics.Precision
    :exclude-members: update, compute
    :special-members: __new__

BinaryPrecision
^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryPrecision
    :exclude-members: update, compute

MulticlassPrecision
^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassPrecision
    :exclude-members: update, compute

MultilabelPrecision
^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelPrecision
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.precision

binary_precision
^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_precision

multiclass_precision
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_precision

multilabel_precision
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_precision
