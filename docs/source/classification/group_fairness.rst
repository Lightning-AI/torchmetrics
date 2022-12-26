.. customcarditem::
   :header: Group Fairness
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

##############
Group Fairness
##############

Module Interface
________________

BinaryFairness
^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.group_fairness.BinaryFairness
    :noindex:
    :exclude-members: update, compute

BinaryGroupStatRates
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.group_fairness.BinaryGroupStatRates
    :noindex:
    :exclude-members: update, compute

Functional Interface
____________________

binary_fairness
^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_fairness
    :noindex:

demographic_parity
^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.demographic_parity
    :noindex:

equal_opportunity
^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.equal_opportunity
    :noindex:

binary_groups_stat_rates
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_groups_stat_rates
    :noindex:
