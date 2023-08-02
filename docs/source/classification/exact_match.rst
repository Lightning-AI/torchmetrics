.. customcarditem::
   :header: Exact Match
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

###########
Exact Match
###########

Module Interface
________________

ExactMatch
^^^^^^^^^^

.. autoclass:: torchmetrics.ExactMatch
    :noindex:
    :exclude-members: update, compute
    :special-members: __new__

MulticlassExactMatch
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassExactMatch
    :noindex:
    :exclude-members: update, compute

MultilabelExactMatch
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelExactMatch
    :noindex:
    :exclude-members: update, compute


Functional Interface
____________________

exact_match
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_exact_match
    :noindex:

multiclass_exact_match
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_exact_match
    :noindex:

multilabel_exact_match
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_exact_match
    :noindex:
