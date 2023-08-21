.. customcarditem::
   :header: Area Under the Receiver Operating Characteristic Curve (AUROC)
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

#####
AUROC
#####

Module Interface
________________

.. autoclass:: torchmetrics.AUROC
    :exclude-members: update, compute
    :special-members: __new__

BinaryAUROC
^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryAUROC
    :exclude-members: update, compute

MulticlassAUROC
^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassAUROC
    :exclude-members: update, compute

MultilabelAUROC
^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelAUROC
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.auroc

binary_auroc
^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_auroc

multiclass_auroc
^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_auroc

multilabel_auroc
^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_auroc
