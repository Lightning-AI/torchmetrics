.. customcarditem::
   :header: Hinge Loss
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

##########
Hinge Loss
##########

Module Interface
________________

HingeLoss
^^^^^^^^^

.. autoclass:: torchmetrics.HingeLoss
    :noindex:
    :exclude-members: update, compute
    :special-members: __new__

BinaryHingeLoss
^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryHingeLoss
    :noindex:
    :exclude-members: update, compute

MulticlassHingeLoss
^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassHingeLoss
    :noindex:
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.hinge_loss
    :noindex:

binary_hinge_loss
^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_hinge_loss
    :noindex:

multiclass_hinge_loss
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_hinge_loss
    :noindex:
