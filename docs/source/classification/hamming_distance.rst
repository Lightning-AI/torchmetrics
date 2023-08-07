.. customcarditem::
   :header: Hamming Distance
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

################
Hamming Distance
################

Module Interface
________________

.. autoclass:: torchmetrics.HammingDistance
    :noindex:
    :exclude-members: update, compute
    :special-members: __new__

BinaryHammingDistance
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryHammingDistance
    :noindex:
    :exclude-members: update, compute

MulticlassHammingDistance
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassHammingDistance
    :noindex:
    :exclude-members: update, compute

MultilabelHammingDistance
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelHammingDistance
    :noindex:
    :exclude-members: update, compute

Functional Interface
____________________

hamming_distance
^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.hamming_distance
    :noindex:

binary_hamming_distance
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_hamming_distance
    :noindex:

multiclass_hamming_distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_hamming_distance
    :noindex:

multilabel_hamming_distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_hamming_distance
    :noindex:
