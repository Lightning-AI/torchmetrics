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

CohenKappa
^^^^^^^^^^

.. autoclass:: torchmetrics.CohenKappa
    :noindex:

BinaryCohenKappa
^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryCohenKappa
    :noindex:
    :exclude-members: update, compute

MulticlassCohenKappa
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassCohenKappa
    :noindex:
    :exclude-members: update, compute

Functional Interface
____________________

cohen_kappa
^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.cohen_kappa
    :noindex:

binary_cohen_kappa
^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_cohen_kappa
    :noindex:

multiclass_cohen_kappa
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_cohen_kappa
    :noindex:
