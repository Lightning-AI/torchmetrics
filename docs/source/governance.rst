.. _governance:

TorchMetrics Governance
#######################

This document describes governance processes we follow in developing TorchMetrics.

Persons of Interest
*******************

Leads
-----
- Nicki Skafte (`skaftenicki <https://github.com/SkafteNicki>`_)
- Jirka Borovec (`Borda <https://github.com/Borda>`_)


Core Maintainers
----------------
-


Alumni
------
- Teddy Koler (`ydcjeff <https://github.com/>`_)


Releases
********

We release a new minor version (e.g., 1.5.0) every three months and bugfix releases every week.
The minor versions contain new features, API changes, deprecations, removals, potential backward-incompatible
changes and also all previous bugfixes included in any bugfix release. With every release, we publish a changelog
where we list additions, removals, changed functionality and fixes.

Project Management and Decision Making
**************************************

The decision what goes into a release is governed by the :ref:`staff contributors and leaders <governance>` of
Lightning development. Whenever possible, discussion happens publicly on GitHub and includes the whole community.
When a consensus is reached, staff and core contributors assign milestones and labels to the issue and/or pull request and start tracking the development. It is possible that priorities change over time.

Commits to the project are exclusively to be added by pull requests on GitHub and anyone in the community is welcome to review them.
However, reviews submitted by
`code owners <https://github.com/PyTorchLightning/metrics/blob/master/.github/CODEOWNERS>`_
have higher weight and it is necessary to get the approval of code owners before a pull request can be merged.
Additional requirements may apply case by case.

API Evolution
*************

Lightning's development is driven by research and best practices in a rapidly developing field of AI and machine
learning. Change is inevitable and when it happens, the Lightning team is committed to minimizing user friction and
maximizing ease of transition from one version to the next. We take backward compatibility and reproducibility very
seriously.

For API removal, renaming or other forms of backward-incompatible changes, the procedure is:

#. A deprecation process is initiated at version X, producing warning messages at runtime and in the documentation.
#. Calls to the deprecated API remain unchanged in their function during the deprecation phase.
#. One minor versions in the future at version X+2 the breaking change takes effect.

The "X+1" rule is a recommendation and not a strict requirement. Longer deprecation cycles may apply for some cases.
