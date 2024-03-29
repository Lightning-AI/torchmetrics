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
    :exclude-members: update, compute
    :special-members: __new__

BinaryAccuracy
^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryAccuracy
    :exclude-members: update, compute

MulticlassAccuracy
^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassAccuracy
    :exclude-members: update, compute

MultilabelAccuracy
^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelAccuracy
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.classification.accuracy

binary_accuracy
^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_accuracy

multiclass_accuracy
^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_accuracy

multilabel_accuracy
^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_accuracy
