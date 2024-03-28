.. TorchMetrics documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TorchMetrics
=======================

.. raw:: html

   <div class="row" style='font-size: 14px'>
      <div class='col-md-12'>

TorchMetrics is a collection of 100+ PyTorch metrics implementations and an easy-to-use API to create custom metrics. It offers:

* A standardized interface to increase reproducibility
* Reduces Boilerplate
* Distributed-training compatible
* Rigorously tested
* Automatic accumulation over batches
* Automatic synchronization between multiple devices

You can use TorchMetrics in any PyTorch model, or within `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`_ to enjoy the following additional benefits:

* Your data will always be placed on the same device as your metrics
* You can log :class:`~torchmetrics.Metric` objects directly in Lightning to reduce even more boilerplate

.. raw:: html

      </div>
   </div>

.. raw:: html

   <hr class="docutils" style="margin: 50px 0 50px 0">


Install TorchMetrics
--------------------

.. raw:: html

   <div class="row" style='font-size: 14px'>
      <div class='col-md-6'>

For pip users

.. code-block:: bash

    pip install torchmetrics

.. raw:: html

      </div>
      <div class='col-md-6'>

Or directly from conda

.. code-block:: bash

    conda install -c conda-forge torchmetrics

.. raw:: html

      </div>
   </div>

.. raw:: html

   <hr class="docutils" style="margin: 50px 0 50px 0">

.. raw:: html

    <div class="tutorials-callout-container">
        <div class="row">

.. Add callout items below this line

.. customcalloutitem::
   :description: Use this quickstart guide to learn key concepts.
   :header: New to TorchMetrics?
   :button_link:  pages/quickstart.html


.. customcalloutitem::
   :description: Easily use TorchMetrics in your PyTorch Lightning code.
   :header: TorchMetrics with PyTorch Lightning
   :button_link: pages/lightning.html


.. customcalloutitem::
   :description: View the full list of metrics and filter by task and data type.
   :header: Metrics
   :button_link: all-metrics.html


.. customcalloutitem::
   :description: A detailed overview of the TorchMetrics API and concepts.
   :header: Overview
   :button_link: pages/overview.html


.. customcalloutitem::
   :description: Learn how to implement a custom metric with TorchMetrics.
   :header: Custom Metrics
   :button_link: pages/implement.html


.. customcalloutitem::
   :description: Detailed descriptions of each API package.
   :header: API Reference
   :button_link: references/metric.html


.. raw:: html

        </div>
    </div>

.. End of callout item section

.. raw:: html

   <div style="display:none">

.. toctree::
   :maxdepth: 2
   :name: guide
   :caption: User Guide

   pages/quickstart
   all-metrics
   pages/overview
   pages/plotting
   pages/implement
   pages/lightning

.. toctree::
   :maxdepth: 2
   :name: aggregation
   :caption: Aggregation
   :glob:

   aggregation/*

.. toctree::
   :maxdepth: 2
   :name: audio
   :caption: Audio
   :glob:

   audio/*

.. toctree::
   :maxdepth: 2
   :name: classification
   :caption: Classification
   :glob:

   classification/*

.. toctree::
   :maxdepth: 2
   :name: clustering
   :caption: Clustering
   :glob:

   clustering/*

.. toctree::
   :maxdepth: 2
   :name: detection
   :caption: Detection
   :glob:

   detection/*

.. toctree::
   :maxdepth: 2
   :name: image
   :caption: Image
   :glob:

   image/*

.. toctree::
   :maxdepth: 2
   :name: multimodal
   :caption: Multimodal
   :glob:

   multimodal/*

.. toctree::
   :maxdepth: 2
   :name: nominal
   :caption: Nominal
   :glob:

   nominal/*

.. toctree::
   :maxdepth: 2
   :name: pairwise
   :caption: Pairwise
   :glob:

   pairwise/*

.. toctree::
   :maxdepth: 2
   :name: regression
   :caption: Regression
   :glob:

   regression/*

.. toctree::
   :maxdepth: 2
   :name: retrieval
   :caption: Retrieval
   :glob:

   retrieval/*

.. toctree::
   :maxdepth: 2
   :name: text
   :caption: Text
   :glob:

   text/*

.. toctree::
   :maxdepth: 2
   :name: wrappers
   :caption: Wrappers
   :glob:

   wrappers/*

.. toctree::
   :maxdepth: 3
   :name: metrics
   :caption: API Reference

   references/metric
   references/utilities

.. toctree::
   :maxdepth: 1
   :name: community
   :caption: Community

   governance
   generated/CODE_OF_CONDUCT.md
   generated/CONTRIBUTING.md
   generated/CHANGELOG.md

.. raw:: html

   </div>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
