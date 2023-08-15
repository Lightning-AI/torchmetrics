.. customcarditem::
   :header: Sacre BLEU Score
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/summarization.svg
   :tags: Text

.. include:: ../links.rst

################
Sacre BLEU Score
################

Module Interface
________________

.. autoclass:: torchmetrics.text.SacreBLEUScore
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.text.sacre_bleu_score
