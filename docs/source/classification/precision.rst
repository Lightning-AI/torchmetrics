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

Precision
^^^^^^^^^

.. autoclass:: torchmetrics.Precision
    :noindex:
    :exclude-members: update, compute
    :special-members: __new__

BinaryPrecision
^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryPrecision
    :noindex:
    :exclude-members: update, compute

MulticlassPrecision
^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassPrecision
    :noindex:
    :exclude-members: update, compute

MultilabelPrecision
^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelPrecision
    :noindex:
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.precision
    :noindex:

binary_precision
^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_precision
    :noindex:

multiclass_precision
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_precision
    :noindex:

multilabel_precision
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_precision
    :noindex:
