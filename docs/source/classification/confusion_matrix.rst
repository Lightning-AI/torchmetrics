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

.. autoclass:: torchmetrics.ConfusionMatrix
    :noindex:
    :exclude-members: update, compute
    :special-members: __new__

BinaryConfusionMatrix
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryConfusionMatrix
    :noindex:
    :exclude-members: update, compute

MulticlassConfusionMatrix
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassConfusionMatrix
    :noindex:
    :exclude-members: update, compute

MultilabelConfusionMatrix
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelConfusionMatrix
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
