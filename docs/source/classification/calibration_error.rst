.. customcarditem::
   :header: Calibration Error
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

#################
Calibration Error
#################

Module Interface
________________

.. autoclass:: torchmetrics.CalibrationError
    :noindex:
    :exclude-members: update, compute
    :special-members: __new__

BinaryCalibrationError
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryCalibrationError
    :exclude-members: update, compute

MulticlassCalibrationError
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassCalibrationError
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.calibration_error

binary_calibration_error
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_calibration_error

multiclass_calibration_error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_calibration_error
