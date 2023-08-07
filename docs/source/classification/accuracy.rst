.. customcarditem::
   :header: Accuracy
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

########
Accuracy
########

Module Interface
________________

.. autoclass:: torchmetrics.Accuracy
    :noindex:
    :exclude-members: update, compute
    :special-members: __new__

BinaryAccuracy
^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryAccuracy
    :noindex:
    :exclude-members: update, compute

MulticlassAccuracy
^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassAccuracy
    :noindex:
    :exclude-members: update, compute

MultilabelAccuracy
^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelAccuracy
    :noindex:
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.classification.accuracy
    :noindex:

binary_accuracy
^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_accuracy
    :noindex:

multiclass_accuracy
^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_accuracy
    :noindex:


multilabel_accuracy
^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_accuracy
    :noindex:
