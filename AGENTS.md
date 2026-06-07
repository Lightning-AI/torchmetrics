# AGENTS.md - TorchMetrics

AI coding agent guide. For contribution process, coding style, and PR workflow see
[CONTRIBUTING.md](.github/CONTRIBUTING.md).

______________________________________________________________________

## Essential commands

```bash
# Install (editable + all dev deps)
pip install -e . -r requirements/_devel.txt

# Download sample data required by audio/image/detection tests
make data

# Lint and format (ruff, docformatter, codespell -- via pre-commit)
pre-commit run --all-files
# Type check (mypy is NOT in pre-commit -- run separately)
mypy src/torchmetrics/

# Tests -- two separate steps (src layout)
cd src && python -m pytest torchmetrics          # doctests inside package
cd tests && python -m pytest unittests -v        # unit tests

# Single domain (run from repo root)
pytest tests/unittests/classification/ -v

# DDP tests
USE_PYTEST_POOL="1" pytest -m DDP tests/

# Docs
make docs    # output: docs/build/html/index.html
```

______________________________________________________________________

## Source layout

```
src/torchmetrics/
  metric.py            # Metric base class
  collections.py       # MetricCollection
  aggregation.py       # MeanMetric, SumMetric, MinMetric, MaxMetric, CatMetric
  wrappers/            # BootStrapper, ClasswiseWrapper, Running, MetricTracker, ...
  utilities/           # imports.py, data.py, distributed.py, checks.py, plot.py
  <domain>/            # audio, classification, clustering, detection, image,
                       # multimodal, nominal, regression, retrieval,
                       # segmentation, shape, text, video
    __init__.py
    <metric>.py        # module-based Metric subclass
  functional/
    <domain>/          # pure-function counterparts; also includes pairwise/
                       # (pairwise has functional API only, no module package)

tests/
  _cache-references/   # cachier-backed reference output cache
  unittests/
    _helpers/testers.py   # MetricTester -- base for all metric tests
    text/_helpers.py      # TextTester -- specialised base for text domain tests
    conftest.py           # BATCH_SIZE=32, NUM_BATCHES=8, NUM_CLASSES=5, NUM_PROCESSES=2
    <domain>/             # mirrors src/torchmetrics/<domain>/; pairwise/ present here too
```

______________________________________________________________________

## Metric base class contract

Every metric subclasses `torchmetrics.Metric` (which subclasses `torch.nn.Module`).

```python
class MyMetric(Metric):
    is_differentiable: bool = False  # class-level, not instance
    higher_is_better: bool = True
    full_state_update: bool = False  # set True only if update needs full history

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Register ALL persistent state here -- nowhere else
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        # Accumulate into self.* state -- no return value
        self.correct += (preds == target).sum()
        self.total += target.numel()

    def compute(self) -> Tensor:
        # Reduce accumulated state to scalar/tensor -- no side effects
        return self.correct.float() / self.total
```

Avoid overriding `reset()` or `forward()` for ordinary metrics -- the base class owns
them. Wrappers (`wrappers/`) may override both; rare metrics with non-default reset
logic may override `reset()` (e.g. `image/fid.py`).

`dist_reduce_fx` options: `"sum"`, `"mean"`, `"cat"`, `"min"`, `"max"`, or a callable.
List states (append-style) use `dist_reduce_fx="cat"`.

______________________________________________________________________

## Adding a new metric -- checklist

1. **Functional helpers** -- `src/torchmetrics/functional/<domain>/<metric>.py`
   (most metrics; skip for complex stateful metrics like FID/KID that are module-only)

   - `_<metric>_update(...)` -> returns intermediate tensors (tp, fp, ...)
   - `_<metric>_compute(...)` or `_<metric>_reduce(...)` -> returns final value
   - Input validation in `_<metric>_arg_validation(...)` / `_<metric>_tensor_validation(...)`

2. **Module class** -- `src/torchmetrics/<domain>/<metric>.py`

   - Subclass `Metric`; call functional helpers from `update()` and `compute()`

3. **Classification task variants** -- classification metrics use task-specific subclasses.
   Most implement all three: `BinaryXxx`, `MulticlassXxx`, `MultilabelXxx`.
   Some support only a subset when a task is semantically inapplicable (e.g. `CohenKappa`
   has no multilabel variant; `ExactMatch` has no binary variant). A public wrapper class
   (e.g. `Accuracy`) inherits `_ClassificationTaskWrapper` and overrides `__new__` to
   dispatch on `task: Literal["binary","multiclass","multilabel"]`.

4. **Exports** -- add to all that apply:

   - `src/torchmetrics/<domain>/__init__.py`
   - `src/torchmetrics/functional/<domain>/__init__.py`
   - `src/torchmetrics/__init__.py` (top-level, for metrics exposed at package root)

5. **Optional dependencies** -- gate with `RequirementCache` from
   `torchmetrics.utilities.imports`; add `__doctest_skip__` at module level:

   ```python
   if not _MATPLOTLIB_AVAILABLE:
       __doctest_skip__ = ["MyMetric.plot"]
   ```

6. **Docs page** -- `docs/source/<domain>/<metric>.rst` following the domain pattern

7. **Tests** -- `tests/unittests/<domain>/test_<metric>.py`:

   - Use `MetricTester` as base class for all domains
   - **Exception**: text metrics must subclass `TextTester` from
     `tests/unittests/text/_helpers.py` -- it adds string/DDP handling

______________________________________________________________________

## Testing with MetricTester

`MetricTester` (in `tests/unittests/_helpers/testers.py`) compares metric output
against a reference implementation (scikit-learn/scipy), checks pickling, and optionally
runs DDP synchronization. Run tests from the `tests/` directory; import accordingly:

```python
# run as: cd tests && python -m pytest unittests/...
import pytest
from unittests._helpers.testers import MetricTester

NUM_BATCHES = 8  # or import from unittests (re-exported by tests/unittests/__init__.py)
BATCH_SIZE = 32


class TestMyMetric(MetricTester):
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    @pytest.mark.parametrize("preds, target", [...])
    def test_my_metric_class(self, preds, target, ddp):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MyMetric,
            reference_metric=sklearn_reference,
        )
```

DDP mode only runs when `USE_PYTEST_POOL=1` is set; the `pytest.mark.DDP` marker gates
it. Reference outputs are cached by `cachier` in `tests/_cache-references/` -- delete
that dir to force recomputation.

______________________________________________________________________

## Optional dependency pattern

```python
# src/torchmetrics/utilities/imports.py -- add RequirementCache entry
from lightning_utilities.core.imports import RequirementCache

_SCIPY_AVAILABLE = RequirementCache("scipy")

# In metric file -- guard import and doctest skip
from torchmetrics.utilities.imports import _SCIPY_AVAILABLE

if not _SCIPY_AVAILABLE:
    __doctest_skip__ = ["MyMetric", "MyMetric.compute"]
```

______________________________________________________________________

## Docstring format

Google-style, enforced by Napoleon + docformatter. Target style for new code:

```python
def update(self, preds: Tensor, target: Tensor) -> None:
    """One-line summary.

    Args:
        preds: shape ``(N, C)`` predictions.
        target: shape ``(N,)`` ground truth.

    Raises:
        ValueError: If shapes mismatch.

    Example:
        >>> metric = MyMetric()
        >>> metric.update(preds, target)
    """
```

Key rules: `Returns:` (plural, not `Return:`), line length 120, f-strings everywhere
except `logging.*` calls (use `%`-style there). Existing code uses `Optional[float]`
from `typing`; new code may use `float | None` (Python 3.10+).

______________________________________________________________________

## Common mistakes to avoid

| Mistake                                        | Correct pattern                                         |
| ---------------------------------------------- | ------------------------------------------------------- |
| Override `reset()` on plain metric             | Only wrappers and rare special cases do this            |
| Override `forward()`                           | Don't -- base class calls `update()` + `compute()`      |
| Call `add_state()` outside `__init__`          | Only in `__init__`                                      |
| Return value from `update()`                   | `update()` returns `None`; mutate state in-place        |
| `Return:` in docstring                         | Use `Returns:`                                          |
| Export only in one `__init__.py`               | Export in domain + functional + top-level as applicable |
| Use `MetricTester` for text metrics            | Text domain uses `TextTester` from `text/_helpers.py`   |
| Add functional helpers for module-only metrics | FID/KID-style metrics intentionally skip this step      |
| Hardcode task triple when subset applies       | Check existing metrics -- some omit inapplicable tasks  |
| Run tests from repo root without `cd tests`    | Import paths assume `tests/` as working directory       |
| `import *`                                     | Explicit imports only                                   |
