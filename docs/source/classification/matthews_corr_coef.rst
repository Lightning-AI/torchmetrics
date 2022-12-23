.. customcarditem::
   :header: Matthews Correlation Coefficient
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

################################
Matthews Correlation Coefficient
################################

Module Interface
________________

MatthewsCorrCoef
^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.MatthewsCorrCoef
    :noindex:

BinaryMatthewsCorrCoef
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryMatthewsCorrCoef
    :noindex:
    :exclude-members: update, compute

MulticlassMatthewsCorrCoef
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassMatthewsCorrCoef
    :noindex:
    :exclude-members: update, compute

MultilabelMatthewsCorrCoef
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelMatthewsCorrCoef
    :noindex:
    :exclude-members: update, compute


Functional Interface
____________________

matthews_corrcoef
^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.matthews_corrcoef
    :noindex:

binary_matthews_corrcoef
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_matthews_corrcoef
    :noindex:

multiclass_matthews_corrcoef
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_matthews_corrcoef
    :noindex:

multilabel_matthews_corrcoef
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_matthews_corrcoef
    :noindex:
