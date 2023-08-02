.. customcarditem::
   :header: Recall
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

######
Recall
######

Module Interface
________________

Recall
^^^^^^

.. autoclass:: torchmetrics.Recall
    :noindex:
    :exclude-members: update, compute
    :special-members: __new__

BinaryRecall
^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryRecall
    :noindex:
    :exclude-members: update, compute

MulticlassRecall
^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassRecall
    :noindex:
    :exclude-members: update, compute

MultilabelRecall
^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelRecall
    :noindex:
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.recall
    :noindex:

binary_recall
^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_recall
    :noindex:

multiclass_recall
^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_recall
    :noindex:

multilabel_recall
^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_recall
    :noindex:
