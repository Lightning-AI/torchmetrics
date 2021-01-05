# TorchMetrics

This is starter project which shall simplify initial steps for each new PL project...

[![CI testing](https://github.com/PyTorchLightning/metrics/workflows/CI%20testing/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/torchmetrics/actions?query=workflow%3A%22CI+testing%22)
![Check Code formatting](https://github.com/PyTorchLightning/metrics/workflows/Check%20Code%20formatting/badge.svg?branch=master&event=push)
[![Documentation Status](https://readthedocs.org/projects/metrics/badge/?version=latest)](https://metrics.readthedocs.io/en/latest/?badge=latest)

\* the Read-The-Docs is failing as this one leads to the public domain which requires the repo to be public too

## To be Done

You still need to enable some external integrations such as:
 - in GH setting lock the master breach - no direct push without PR
 - init Read-The-Docs (add this new project)
 - specify license in `LICENSE` file and package init

## Tests / Docs notes

* We are using [Napoleon style](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) and we shall use static types...
* It is nice to se [doctest](https://docs.python.org/3/library/doctest.html) as they are also generated as examples in documentation
* For wider and edge cases testing use [pytest parametrization](https://docs.pytest.org/en/stable/parametrize.html) :]

