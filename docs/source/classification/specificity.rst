.. customcarditem::
   :header: Specificity
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

###########
Specificity
###########

Module Interface
________________

.. autoclass:: torchmetrics.Specificity
    :exclude-members: update, compute
    :special-members: __new__

BinarySpecificity
^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinarySpecificity
    :exclude-members: update, compute

MulticlassSpecificity
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassSpecificity
    :exclude-members: update, compute

MultilabelSpecificity
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelSpecificity
    :exclude-members: update, compute


Functional Interface
____________________

.. autofunction:: torchmetrics.functional.specificity

binary_specificity
^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_specificity

multiclass_specificity
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_specificity

multilabel_specificity
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_specificity
