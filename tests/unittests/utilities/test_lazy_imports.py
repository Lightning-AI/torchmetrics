# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Optional heavy dependencies must not be imported just because they happen to be installed.

`import torchmetrics` is on the critical path for downstream libraries — Lightning imports it merely to compare
versions — so eagerly pulling in plotting or signal-processing stacks costs every user, including those who never
touch the features that need them. See https://github.com/Lightning-AI/torchmetrics/issues/3457.

"""

import subprocess
import sys

import pytest

from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _SCIPI_AVAILABLE

# checked in a subprocess: the test session itself has already imported large parts of the world
_PROBE = (
    "import sys, torchmetrics; "
    "print(','.join(m for m in ('scipy.signal', 'matplotlib', 'matplotlib.pyplot') if m in sys.modules))"
)


def _modules_after_importing_torchmetrics() -> set:
    """Return the probed optional modules that a bare ``import torchmetrics`` pulled in."""
    # S603 is suppressed below: the interpreter path and the probe are fixed literals, no external input is involved
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, check=True
    )
    return {name for name in out.stdout.strip().split(",") if name}


@pytest.mark.skipif(not _SCIPI_AVAILABLE, reason="test only meaningful when scipy is installed")
def test_importing_torchmetrics_does_not_import_scipy_signal():
    """`scipy.signal` was only pulled in by the SRMRpy back-compat patch, which no longer runs at import.

    Asserted on `scipy.signal` rather than `scipy` on purpose: when `transformers` is installed,
    `functional/text/bert.py` imports it at module level and that pulls in `scipy.sparse`. That is a separate eager
    import from the one this test guards, so asserting on the `scipy` root would make this test fail for an unrelated
    reason.

    """
    loaded = _modules_after_importing_torchmetrics()
    assert "scipy.signal" not in loaded, f"`import torchmetrics` pulled in {sorted(loaded)}"


@pytest.mark.skipif(not _MATPLOTLIB_AVAILABLE, reason="test only meaningful when matplotlib is installed")
def test_importing_torchmetrics_does_not_import_matplotlib():
    """`matplotlib` is only needed by ``.plot()``, which most users never call."""
    loaded = _modules_after_importing_torchmetrics()
    assert "matplotlib" not in loaded, f"`import torchmetrics` pulled in {sorted(loaded)}"
    assert "matplotlib.pyplot" not in loaded


@pytest.mark.skipif(not _MATPLOTLIB_AVAILABLE, reason="requires matplotlib")
def test_plot_type_aliases_are_defined_at_runtime():
    """The aliases are annotations on ~140 metrics, so they must still resolve without matplotlib loaded."""
    from torchmetrics.utilities.plot import _AX_TYPE, _CMAP_TYPE, _PLOT_OUT_TYPE

    assert _AX_TYPE is not None
    assert _CMAP_TYPE is not None
    assert _PLOT_OUT_TYPE is not None


@pytest.mark.skipif(not _MATPLOTLIB_AVAILABLE, reason="requires matplotlib")
def test_is_axes_discriminates_real_axes():
    """``_AX_TYPE`` is ``object`` at runtime, so ``isinstance(x, _AX_TYPE)`` would match anything.

    Several call sites branch on whether an argument is a real ``Axes`` — ``trim_axs``, ``MetricCollection.plot`` and
    ``MultitaskWrapper.plot``. They must use ``_is_axes``, otherwise the checks silently pass for every input:
    validation stops raising and ``trim_axs`` returns the untrimmed array.

    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from torchmetrics.utilities.plot import _AX_TYPE, _is_axes

    _, ax = plt.subplots()
    _, axs = plt.subplots(nrows=2, ncols=2)

    assert _is_axes(ax)
    assert not _is_axes(axs)  # an ndarray of axes, not an Axes
    assert not _is_axes("not an axis")
    assert not _is_axes(None)

    # the alias itself must not be used for this: every object is an instance of `object`
    assert isinstance("not an axis", _AX_TYPE)

    plt.close("all")


def test_style_change_works_as_context_manager_and_decorator():
    """``style_change`` is applied as a decorator at import time, so it must not need matplotlib to be constructed."""
    from torchmetrics.utilities.plot import _style, style_change

    with style_change(_style):
        pass

    @style_change(_style)
    def _fn() -> str:
        return "ok"

    # a context manager used as a decorator must survive being called more than once
    assert _fn() == "ok"
    assert _fn() == "ok"
