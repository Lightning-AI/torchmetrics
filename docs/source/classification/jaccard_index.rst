.. customcarditem::
   :header: Jaccard Index
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

#############
Jaccard Index
#############

Module Interface
________________

.. autoclass:: torchmetrics.JaccardIndex
    :exclude-members: update, compute
    :special-members: __new__

BinaryJaccardIndex
^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryJaccardIndex
    :exclude-members: update, compute

MulticlassJaccardIndex
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassJaccardIndex
    :exclude-members: update, compute

MultilabelJaccardIndex
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelJaccardIndex
    :exclude-members: update, compute


Functional Interface
____________________

jaccard_index
^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.jaccard_index

binary_jaccard_index
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_jaccard_index

multiclass_jaccard_index
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_jaccard_index

multilabel_jaccard_index
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_jaccard_index
