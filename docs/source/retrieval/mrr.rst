.. customcarditem::
   :header: Retrieval Mean Reciprocal Rank (MRR)
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/text_classification.svg
   :tags: Retrieval

.. include:: ../links.rst

####################################
Retrieval Mean Reciprocal Rank (MRR)
####################################

Module Interface
________________

.. autoclass:: torchmetrics.retrieval.RetrievalMRR
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.retrieval.retrieval_reciprocal_rank
