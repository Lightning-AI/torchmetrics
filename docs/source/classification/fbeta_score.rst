.. customcarditem::
   :header: F-Beta Score
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

############
F-Beta Score
############

Module Interface
________________

FBetaScore
^^^^^^^^^^

.. autoclass:: torchmetrics.FBetaScore
    :noindex:
    :exclude-members: update, compute
    :special-members: __new__

BinaryFBetaScore
^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryFBetaScore
    :noindex:
    :exclude-members: update, compute

MulticlassFBetaScore
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassFBetaScore
    :noindex:
    :exclude-members: update, compute

MultilabelFBetaScore
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelFBetaScore
    :noindex:
    :exclude-members: update, compute

Functional Interface
____________________

fbeta_score
^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.fbeta_score
    :noindex:

binary_fbeta_score
^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_fbeta_score
    :noindex:

multiclass_fbeta_score
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_fbeta_score
    :noindex:

multilabel_fbeta_score
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_fbeta_score
    :noindex:
