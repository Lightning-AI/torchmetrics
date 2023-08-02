.. customcarditem::
   :header: Specificity At Sensitivity
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

##########################
Specificity At Sensitivity
##########################

Module Interface
________________

.. autoclass:: torchmetrics.SpecificityAtSensitivity
    :noindex:
    :exclude-members: update, compute
    :special-members: __new__

BinarySpecificityAtSensitivity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinarySpecificityAtSensitivity
    :noindex:
    :exclude-members: update, compute

MulticlassSpecificityAtSensitivity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassSpecificityAtSensitivity
    :noindex:
    :exclude-members: update, compute

MultilabelSpecificityAtSensitivity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelSpecificityAtSensitivity
    :noindex:
    :exclude-members: update, compute

Functional Interface
____________________

binary_specificity_at_sensitivity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_specificity_at_sensitivity
    :noindex:

multiclass_specificity_at_sensitivity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_specificity_at_sensitivity
    :noindex:

multilabel_specificity_at_sensitivity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_specificity_at_sensitivity
    :noindex:
