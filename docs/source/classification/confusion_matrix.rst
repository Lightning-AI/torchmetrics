.. customcarditem::
   :header: Confusion Matrix
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

################
Confusion Matrix
################

Module Interface
________________

ConfusionMatrix
^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.ConfusionMatrix
    :noindex:

BinaryConfusionMatrix
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.BinaryConfusionMatrix
    :noindex:
    :exclude-members: update, compute

MulticlassConfusionMatrix
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.MulticlassConfusionMatrix
    :noindex:
    :exclude-members: update, compute

MultilabelConfusionMatrix
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.MultilabelConfusionMatrix
    :noindex:
    :exclude-members: update, compute

Functional Interface
____________________

confusion_matrix
^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.confusion_matrix
    :noindex:

binary_confusion_matrix
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.binary_confusion_matrix
    :noindex:

multiclass_confusion_matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.multiclass_confusion_matrix
    :noindex:

multilabel_confusion_matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.multilabel_confusion_matrix
    :noindex:
