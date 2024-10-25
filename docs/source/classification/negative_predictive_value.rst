.. customcarditem::
   :header: Negative Predictive Value
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

#########################
Negative Predictive Value
#########################

Module Interface
________________

.. autoclass:: torchmetrics.NegativePredictiveValue
    :exclude-members: update, compute
    :special-members: __new__

BinaryNegativePredictiveValue
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryNegativePredictiveValue
    :exclude-members: update, compute

MulticlassNegativePredictiveValue
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassNegativePredictiveValue
    :exclude-members: update, compute

MultilabelNegativePredictiveValue
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelNegativePredictiveValue
    :exclude-members: update, compute


Functional Interface
____________________

.. autofunction:: torchmetrics.functional.negative_predictive_value

binary_negative_predictive_value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_negative_predictive_value

multiclass_negative_predictive_value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_negative_predictive_value

multilabel_negative_predictive_value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_negative_predictive_value
