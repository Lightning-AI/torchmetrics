.. customcarditem::
   :header: BERT Score
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/summarization.svg
   :tags: Text

.. include:: ../links.rst

##########
BERT Score
##########

Module Interface
________________

.. autoclass:: torchmetrics.text.bert.BERTScore
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.text.bert.bert_score
