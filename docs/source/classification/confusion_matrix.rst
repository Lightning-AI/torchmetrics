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

.. autoclass:: torchmetrics.classification.BinaryConfusionMatrix
    :noindex:

MulticlassConfusionMatrix
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassConfusionMatrix
    :noindex:

MultilabelConfusionMatrix
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelConfusionMatrix
    :noindex:

Functional Interface
____________________

confusion_matrix
^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.confusion_matrix
    :noindex:

binary_confusion_matrix
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_confusion_matrix
    :noindex:

multiclass_confusion_matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_confusion_matrix
    :noindex:

multilabel_confusion_matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_confusion_matrix
    :noindex:
