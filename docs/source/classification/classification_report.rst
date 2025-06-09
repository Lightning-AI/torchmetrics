.. customcarditem::
   :header: Classification Report
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

#######################
Classification Report
#######################

Module Interface
________________

.. autoclass:: torchmetrics.ClassificationReport
    :exclude-members: update, compute
    :special-members: __new__

BinaryClassificationReport
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryClassificationReport
    :exclude-members: update, compute

MulticlassClassificationReport
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassClassificationReport
    :exclude-members: update, compute

MultilabelClassificationReport
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelClassificationReport
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.classification.classification_report

binary_classification_report
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_classification_report

multiclass_classification_report
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_classification_report

multilabel_classification_report
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_classification_report
