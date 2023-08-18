.. customcarditem::
   :header: Exact Match
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

###########
Exact Match
###########

Module Interface
________________

.. autoclass:: torchmetrics.ExactMatch
    :exclude-members: update, compute
    :special-members: __new__

MulticlassExactMatch
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassExactMatch
    :exclude-members: update, compute

MultilabelExactMatch
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelExactMatch
    :exclude-members: update, compute


Functional Interface
____________________

exact_match
^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.exact_match

multiclass_exact_match
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_exact_match

multilabel_exact_match
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_exact_match
