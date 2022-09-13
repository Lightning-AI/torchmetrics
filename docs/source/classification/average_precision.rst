.. customcarditem::
   :header: Average Precision
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

#################
Average Precision
#################

Module Interface
________________

.. autoclass:: torchmetrics.AveragePrecision
    :noindex:

BinaryAveragePrecision
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryAveragePrecision
    :noindex:

MulticlassAveragePrecision
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassAveragePrecision
    :noindex:

MultilabelAveragePrecision
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelAveragePrecision
    :noindex:

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.average_precision
    :noindex:

binary_average_precision
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_average_precision
    :noindex:

multiclass_average_precision
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_average_precision
    :noindex:

multilabel_average_precision
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_average_precision
    :noindex:
