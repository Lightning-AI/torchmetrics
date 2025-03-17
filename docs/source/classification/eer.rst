.. customcarditem::
   :header: Expected Error Rate (EER)
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

#########################
Expected Error Rate (EER)
#########################

Module Interface
________________

.. autoclass:: torchmetrics.classification.EER
    :exclude-members: update, compute
    :special-members: __new__

BinaryEER
^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryEER
    :exclude-members: update, compute

MulticlassEER
^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassEER
    :exclude-members: update, compute

MultilabelEER
^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelEER
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.eer

binary_eer
^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_eer

multiclass_eer
^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_eer

multilabel_eer
^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_eer
