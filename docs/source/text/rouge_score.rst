.. customcarditem::
   :header: ROUGE Score
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/summarization.svg
   :tags: Text

.. include:: ../links.rst

###########
ROUGE Score
###########

Module Interface
________________

.. autoclass:: torchmetrics.text.rouge.ROUGEScore
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.text.rouge.rouge_score
