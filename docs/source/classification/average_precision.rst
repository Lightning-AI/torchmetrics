.. customcarditem::
   :header: Average Precision
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

#################
Average Precision
#################

Module Interface
________________

.. autoclass:: torchmetrics.AveragePrecision
    :exclude-members: update, compute
    :special-members: __new__

BinaryAveragePrecision
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryAveragePrecision
    :exclude-members: update, compute

MulticlassAveragePrecision
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassAveragePrecision
    :exclude-members: update, compute

MultilabelAveragePrecision
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelAveragePrecision
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.average_precision

binary_average_precision
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_average_precision

multiclass_average_precision
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_average_precision

multilabel_average_precision
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_average_precision
