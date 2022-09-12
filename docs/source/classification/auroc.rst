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
    :noindex:

BinaryAUROC
^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryAUROC
    :noindex:

MulticlassAUROC
^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassAUROC
    :noindex:

MultilabelAUROC
^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelAUROC
    :noindex:

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.auroc
    :noindex:

binary_auroc
^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_auroc
    :noindex:

multiclass_auroc
^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_auroc
    :noindex:

multilabel_auroc
^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_auroc
    :noindex:
