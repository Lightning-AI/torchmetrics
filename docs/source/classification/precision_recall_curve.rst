.. customcarditem::
   :header: Precision Recall Curve
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

######################
Precision Recall Curve
######################

Module Interface
________________

.. autoclass:: torchmetrics.PrecisionRecallCurve
    :noindex:

BinaryPrecisionRecallCurve
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryPrecisionRecallCurve
    :noindex:

MulticlassPrecisionRecallCurve
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassPrecisionRecallCurve
    :noindex:

MultilabelPrecisionRecallCurve
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelPrecisionRecallCurve
    :noindex:

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.precision_recall_curve
    :noindex:

binary_precision_recall_curve
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_precision_recall_curve
    :noindex:

multiclass_precision_recall_curve
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_precision_recall_curve
    :noindex:

multilabel_precision_recall_curve
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_precision_recall_curve
    :noindex:
