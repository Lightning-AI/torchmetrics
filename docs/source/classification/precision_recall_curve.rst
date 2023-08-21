.. customcarditem::
   :header: Precision Recall Curve
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

######################
Precision Recall Curve
######################

Module Interface
________________

.. autoclass:: torchmetrics.PrecisionRecallCurve
    :exclude-members: update, compute
    :special-members: __new__

BinaryPrecisionRecallCurve
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryPrecisionRecallCurve
    :exclude-members: update, compute

MulticlassPrecisionRecallCurve
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassPrecisionRecallCurve
    :exclude-members: update, compute

MultilabelPrecisionRecallCurve
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelPrecisionRecallCurve
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.precision_recall_curve

binary_precision_recall_curve
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_precision_recall_curve

multiclass_precision_recall_curve
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_precision_recall_curve

multilabel_precision_recall_curve
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_precision_recall_curve
