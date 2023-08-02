.. customcarditem::
   :header: Specificity
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

###########
Specificity
###########

Module Interface
________________

Specificity
^^^^^^^^^^^

.. autoclass:: torchmetrics.Specificity
    :noindex:
    :exclude-members: update, compute
    :special-members: __new__

BinarySpecificity
^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinarySpecificity
    :noindex:
    :exclude-members: update, compute

MulticlassSpecificity
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassSpecificity
    :noindex:
    :exclude-members: update, compute

MultilabelSpecificity
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelSpecificity
    :noindex:
    :exclude-members: update, compute


Functional Interface
____________________

.. autofunction:: torchmetrics.functional.specificity
    :noindex:

binary_specificity
^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_specificity
    :noindex:

multiclass_specificity
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_specificity
    :noindex:

multilabel_specificity
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_specificity
    :noindex:
