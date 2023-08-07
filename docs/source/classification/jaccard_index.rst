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
    :noindex:
    :exclude-members: update, compute
    :special-members: __new__

BinaryJaccardIndex
^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryJaccardIndex
    :noindex:
    :exclude-members: update, compute

MulticlassJaccardIndex
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassJaccardIndex
    :noindex:
    :exclude-members: update, compute

MultilabelJaccardIndex
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelJaccardIndex
    :noindex:
    :exclude-members: update, compute


Functional Interface
____________________

jaccard_index
^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.jaccard_index
    :noindex:

binary_jaccard_index
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_jaccard_index
    :noindex:

multiclass_jaccard_index
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_jaccard_index
    :noindex:

multilabel_jaccard_index
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_jaccard_index
    :noindex:
