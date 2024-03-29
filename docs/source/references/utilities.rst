.. role:: hidden
    :class: hidden-section

######################
torchmetrics.utilities
######################

In the following is listed public utility functions that may be beneficial to use in your own code. These functions are
not part of the public API and may change at any time.

***************************
torchmetrics.utilities.data
***************************

The `data` utilities are used to help with data manipulation, such as converting labels in classification from one
format to another.

select_topk
~~~~~~~~~~~

.. autofunction:: torchmetrics.utilities.data.select_topk

to_categorical
~~~~~~~~~~~~~~

.. autofunction:: torchmetrics.utilities.data.to_categorical

to_onehot
~~~~~~~~~

.. autofunction:: torchmetrics.utilities.data.to_onehot

dim_zero_cat
~~~~~~~~~~~~

.. autofunction:: torchmetrics.utilities.data.dim_zero_cat

dim_zero_max
~~~~~~~~~~~~

.. autofunction:: torchmetrics.utilities.data.dim_zero_max

dim_zero_mean
~~~~~~~~~~~~~

.. autofunction:: torchmetrics.utilities.data.dim_zero_mean

dim_zero_min
~~~~~~~~~~~~

.. autofunction:: torchmetrics.utilities.data.dim_zero_min

dim_zero_sum
~~~~~~~~~~~~

.. autofunction:: torchmetrics.utilities.data.dim_zero_sum

**********************************
torchmetrics.utilities.distributed
**********************************

The `distributed` utilities are used to help with synchronization of metrics across multiple processes.

gather_all_tensors
~~~~~~~~~~~~~~~~~~

.. autofunction:: torchmetrics.utilities.distributed.gather_all_tensors
    :noindex:

*********************************
torchmetrics.utilities.exceptions
*********************************

TorchMetricsUserError
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.utilities.exceptions.TorchMetricsUserError

TorchMetricsUserWarning
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.utilities.exceptions.TorchMetricsUserWarning
