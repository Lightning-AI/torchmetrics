.. customcarditem::
   :header: Receiver Operating Characteristic Curve (ROC)
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

###
ROC
###

Module Interface
________________

.. autoclass:: torchmetrics.ROC
    :exclude-members: update, compute
    :special-members: __new__

BinaryROC
^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryROC
    :exclude-members: update, compute

MulticlassROC
^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassROC
    :exclude-members: update, compute

MultilabelROC
^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelROC
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.roc

binary_roc
^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_roc

multiclass_roc
^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_roc

multilabel_roc
^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_roc
