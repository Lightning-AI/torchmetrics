.. customcarditem::
   :header: Receiver Operating Characteristic Curve (ROC)
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

###
ROC
###

Module Interface
________________

ROC
^^^

.. autoclass:: torchmetrics.ROC
    :noindex:
    :exclude-members: update, compute
    :special-members: __new__

BinaryROC
^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryROC
    :noindex:
    :exclude-members: update, compute

MulticlassROC
^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassROC
    :noindex:
    :exclude-members: update, compute

MultilabelROC
^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelROC
    :noindex:
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.roc
    :noindex:

binary_roc
^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_roc
    :noindex:

multiclass_roc
^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_roc
    :noindex:

multilabel_roc
^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_roc
    :noindex:
