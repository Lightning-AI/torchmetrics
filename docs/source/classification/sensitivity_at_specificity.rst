.. customcarditem::
   :header: Sensitivity At Specificity
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

##########################
Sensitivity At Specificity
##########################

Module Interface
________________

.. autoclass:: torchmetrics.SensitivityAtSpecificity
    :exclude-members: update, compute
    :special-members: __new__

BinarySensitivityAtSpecificity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinarySensitivityAtSpecificity
    :exclude-members: update, compute

MulticlassSensitivityAtSpecificity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassSensitivityAtSpecificity
    :exclude-members: update, compute

MultilabelSensitivityAtSpecificity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelSensitivityAtSpecificity
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.classification.sensitivity_at_specificity

binary_sensitivity_at_specificity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_sensitivity_at_specificity

multiclass_sensitivity_at_specificity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_sensitivity_at_specificity

multilabel_sensitivity_at_specificity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_sensitivity_at_specificity
