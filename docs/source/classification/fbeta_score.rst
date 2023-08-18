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

.. autoclass:: torchmetrics.FBetaScore
    :exclude-members: update, compute
    :special-members: __new__

BinaryFBetaScore
^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryFBetaScore
    :exclude-members: update, compute

MulticlassFBetaScore
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MulticlassFBetaScore
    :exclude-members: update, compute

MultilabelFBetaScore
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.MultilabelFBetaScore
    :exclude-members: update, compute

Functional Interface
____________________

fbeta_score
^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.fbeta_score

binary_fbeta_score
^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_fbeta_score

multiclass_fbeta_score
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multiclass_fbeta_score

multilabel_fbeta_score
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.multilabel_fbeta_score
