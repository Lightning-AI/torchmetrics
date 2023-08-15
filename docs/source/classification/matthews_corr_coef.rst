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

.. autoclass:: torchmetrics.MatthewsCorrCoef
    :exclude-members: update, compute
    :special-members: __new__

BinaryMatthewsCorrCoef
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryMatthewsCorrCoef
    :exclude-members: update, compute

MulticlassMatthewsCorrCoef
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassMatthewsCorrCoef
    :exclude-members: update, compute

MultilabelMatthewsCorrCoef
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelMatthewsCorrCoef
    :exclude-members: update, compute


Functional Interface
____________________

matthews_corrcoef
^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.matthews_corrcoef

binary_matthews_corrcoef
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_matthews_corrcoef

multiclass_matthews_corrcoef
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_matthews_corrcoef

multilabel_matthews_corrcoef
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_matthews_corrcoef
