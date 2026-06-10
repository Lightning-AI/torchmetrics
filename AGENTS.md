# AGENTS.md — TorchMetrics

AI coding agent guide. Contribution process, coding style, PR workflow → [CONTRIBUTING.md](.github/CONTRIBUTING.md).

______________________________________________________________________

## Commands

```bash
# Install
pip install -e . -r requirements/_devel.txt

# Sample data (audio/image/detection tests)
make data

# Lint/format via pre-commit (ruff, docformatter, codespell)
pre-commit run --all-files
# Type check — not in pre-commit, run separately
mypy src/torchmetrics/

# Tests
python -m pytest src/torchmetrics     # doctests inside package
python -m pytest tests/ -v            # unit tests
pytest tests/unittests/classification/ -v   # single domain
USE_PYTEST_POOL="1" pytest -m DDP tests/    # DDP tests

# Docs → docs/build/html/index.html
make docs
```

______________________________________________________________________

## Source layout

```
src/torchmetrics/
├── metric.py            # Metric base class
├── collections.py       # MetricCollection
├── aggregation.py       # MeanMetric, SumMetric, MinMetric, MaxMetric, CatMetric
├── wrappers/            # BootStrapper, ClasswiseWrapper, Running, MetricTracker, …
├── utilities/           # imports.py, data.py, distributed.py, checks.py, plot.py
├── functional/
│   └── <domain>/        # pure-function counterparts; pairwise/ has functional API only
└── <domain>/            # audio, classification, clustering, detection, image,
                         # multimodal, nominal, regression, retrieval,
                         # segmentation, shape, text, video
    ├── __init__.py
    └── <metric>.py      # module-based Metric subclass

tests/
├── _cache-references/   # cachier-backed ref output cache (env: PYTEST_REFERENCE_CACHE)
└── unittests/
    ├── _helpers/testers.py   # MetricTester — base for all metric tests
    ├── text/_helpers.py      # TextTester — text domain specialisation
    ├── conftest.py           # BATCH_SIZE=32, NUM_BATCHES=2*NUM_PROCESSES, NUM_CLASSES=5, NUM_PROCESSES=2
    └── <domain>/             # mirrors src/torchmetrics/<domain>/; pairwise/ present too
```

______________________________________________________________________

## Metric base class contract

Every metric subclasses `torchmetrics.Metric` (→ `torch.nn.Module`).

```python
class MyMetric(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False  # True only if update needs full history

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ALL persistent state registered here — nowhere else
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        # Accumulate into self.* state — no return value
        self.correct += (preds == target).sum()
        self.total += target.numel()

    def compute(self) -> Tensor:
        # Reduce accumulated state — no side effects
        return self.correct.float() / self.total
```

- Never override `reset()` or `forward()` on plain metrics — base class owns them
- Wrappers (`wrappers/`) may override both; rare metrics may override `reset()` (e.g. `image/fid.py`)
- `dist_reduce_fx` options: `"sum"`, `"mean"`, `"cat"`, `"min"`, `"max"`, or callable
- List (append-style) states use `dist_reduce_fx="cat"`

______________________________________________________________________

## Adding a new metric

1. **Functional helpers** — `src/torchmetrics/functional/<domain>/<metric>.py`
   - `_<metric>_update(...)` → intermediate tensors (tp, fp, …)
   - `_<metric>_compute(...)` / `_<metric>_reduce(...)` → final value
   - `_<metric>_arg_validation(...)` / `_<metric>_tensor_validation(...)` for input checks
   - Skip for module-only metrics (FID/KID style — intentionally no functional layer)

2. **Module class** — `src/torchmetrics/<domain>/<metric>.py`
   - Subclass `Metric`; call functional helpers from `update()` and `compute()`

3. **Classification task variants** — implement `BinaryXxx`, `MulticlassXxx`, `MultilabelXxx`
   unless a task is semantically inapplicable (e.g. `CohenKappa` has no multilabel;
   `ExactMatch` has no binary). Public wrapper (e.g. `Accuracy`) inherits
   `_ClassificationTaskWrapper`, overrides `__new__` to dispatch on
   `task: Literal["binary","multiclass","multilabel"]`.

4. **Exports** — add to all applicable:
   - `src/torchmetrics/<domain>/__init__.py`
   - `src/torchmetrics/functional/<domain>/__init__.py`
   - `src/torchmetrics/__init__.py` (top-level)

5. **Optional dependencies** — gate with `RequirementCache`; add `__doctest_skip__`:
   ```python
   if not _MATPLOTLIB_AVAILABLE:
       __doctest_skip__ = ["MyMetric.plot"]
   ```
   See [Optional dependency pattern](#optional-dependency-pattern).

6. **Docs page** — `docs/source/<domain>/<metric>.rst`

7. **Tests** — `tests/unittests/<domain>/test_<metric>.py`
   - All domains: subclass `MetricTester`
   - Text domain only: subclass `TextTester` from `tests/unittests/text/_helpers.py`

______________________________________________________________________

## Testing with MetricTester

`MetricTester` compares against a reference (scikit-learn/scipy), checks pickling, optionally runs DDP sync.

```python
# run as: cd tests && python -m pytest unittests/...
from unittests import BATCH_SIZE, NUM_BATCHES
from unittests._helpers.testers import MetricTester

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

Two equivalent invocation styles:
- `cd tests && python -m pytest unittests/...`
- `pytest tests/...` from repo root

DDP runs only when `USE_PYTEST_POOL=1`; `pytest.mark.DDP` gates the parametrize value.
Reference cache at `tests/_cache-references/` (or `$PYTEST_REFERENCE_CACHE`) — delete to force recompute.

______________________________________________________________________

## Optional dependency pattern

```python
# src/torchmetrics/utilities/imports.py — add entry
_SCIPY_AVAILABLE = RequirementCache("scipy")

# In metric file
from torchmetrics.utilities.imports import _SCIPY_AVAILABLE

if not _SCIPY_AVAILABLE:
    __doctest_skip__ = ["MyMetric", "MyMetric.compute"]
```

______________________________________________________________________

## Docstring format

Google style enforced by Napoleon + docformatter — see [CONTRIBUTING.md](.github/CONTRIBUTING.md).
Agent-critical rules (commonly missed):

- `Returns:` plural — not `Return:`
- Line length 120
- f-strings everywhere except `logging.*` calls (use `%`-style there)
- `Optional[float]` from `typing` — keep consistent with existing code

______________________________________________________________________

## Common pitfalls

- Override `reset()`/`forward()` on plain metric → only wrappers and rare special cases
- Call `add_state()` outside `__init__` → register state only in `__init__`
- Return value from `update()` → returns `None`; mutate state in-place
- `import *` → explicit imports only
- Export in one `__init__.py` only → export in domain + functional + top-level
- Use `MetricTester` for text metrics → text domain needs `TextTester` from `text/_helpers.py`
- Add functional helpers for module-only metrics → FID/KID intentionally skips that step
- Hardcode all 3 task variants → check existing metrics; some omit inapplicable tasks
- `Return:` in docstring → use `Returns:`
- Run `python -m pytest unittests/...` from repo root → `cd tests` first, or use `pytest tests/...`
