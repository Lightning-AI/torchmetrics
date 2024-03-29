.. customcarditem::
   :header: BLEU Score
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/summarization.svg
   :tags: Text

.. include:: ../links.rst

##########
BLEU Score
##########

Module Interface
________________

.. autoclass:: torchmetrics.text.BLEUScore
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.text.bleu_score
