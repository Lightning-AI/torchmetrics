.. customcarditem::
   :header: Log Area Receiver Operating Characteristic (LogAUC)
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

#######
Log AUC
#######

Module Interface
________________

.. autoclass:: torchmetrics.LogAUC
    :exclude-members: update, compute
    :special-members: __new__

BinaryLogAUC
^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryLogAUC
    :exclude-members: update, compute

MulticlassLogAUC
^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassLogAUC
    :exclude-members: update, compute

MultilabelLogAUC
^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelLogAUC
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.logauc

binary_logauc
^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_logauc

multiclass_logauc
^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_logauc

multilabel_logauc
^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_logauc
