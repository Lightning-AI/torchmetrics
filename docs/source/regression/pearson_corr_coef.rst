.. customcarditem::
   :header: Pearson Correlation Coefficient
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Regression

.. include:: ../links.rst

###################
Pearson Corr. Coef.
###################

Module Interface
________________

Constant inputs have undefined correlation and return ``nan`` with a warning.
A sparse but nonconstant signal remains valid, including long sequences with only a few nonzero observations.

.. autoclass:: torchmetrics.PearsonCorrCoef
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.pearson_corrcoef
