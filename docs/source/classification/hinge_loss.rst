.. customcarditem::
   :header: Hinge Loss
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

##########
Hinge Loss
##########

Module Interface
________________

.. autoclass:: torchmetrics.HingeLoss
    :exclude-members: update, compute
    :special-members: __new__

BinaryHingeLoss
^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryHingeLoss
    :exclude-members: update, compute

MulticlassHingeLoss
^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassHingeLoss
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.hinge_loss

binary_hinge_loss
^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_hinge_loss

multiclass_hinge_loss
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_hinge_loss
