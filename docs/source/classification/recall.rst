.. customcarditem::
   :header: Recall
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

######
Recall
######

Module Interface
________________

.. autoclass:: torchmetrics.Recall
    :exclude-members: update, compute
    :special-members: __new__

BinaryRecall
^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryRecall
    :exclude-members: update, compute

MulticlassRecall
^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassRecall
    :exclude-members: update, compute

MultilabelRecall
^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelRecall
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.recall

binary_recall
^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_recall

multiclass_recall
^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_recall

multilabel_recall
^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_recall
