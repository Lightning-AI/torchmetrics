.. customcarditem::
   :header: Cohen Kappa
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

###########
Cohen Kappa
###########

Module Interface
________________

.. autoclass:: torchmetrics.CohenKappa
    :special-members: __new__

BinaryCohenKappa
^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryCohenKappa
    :exclude-members: update, compute

MulticlassCohenKappa
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassCohenKappa
^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.cohen_kappa

^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_cohen_kappa

multiclass_cohen_kappa

.. autofunction:: torchmetrics.functional.classification.multiclass_cohen_kappa
