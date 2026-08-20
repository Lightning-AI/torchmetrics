# Contributing

Welcome to the Torchmetrics community! We're building largest collection of native pytorch metrics, with the
goal of reducing boilerplate and increasing reproducibility.

## Contribution Types

We are always looking for help implementing new features or fixing bugs.

### Bug Fixes:

1. If you find a bug please submit a github issue.

   - Make sure the title explains the issue.
   - Describe your setup, what you are trying to do, expected vs. actual behaviour. Please add configs and code samples.
   - Add details on how to reproduce the issue - a minimal test case is always best, colab is also great.
     Note, that the sample code shall be minimal and if needed with publicly available data.

2. Try to fix it or recommend a solution. We highly recommend to use test-driven approach:

   - Convert your minimal code example to a unit/integration test with assert on expected results.
   - Start by debugging the issue... You can run just this particular test in your IDE and draft a fix.
   - Verify that your test case fails on the master branch and only passes with the fix applied.

3. Submit a PR!

_**Note**, even if you do not find the solution, sending a PR with a test covering the issue is a valid contribution and we can
help you or finish it with you :\]_

### New Features:

1. Submit a github issue - describe what is the motivation of such feature (adding the use case or an example is helpful).

2. Let's discuss to determine the feature scope.

3. Submit a PR! We recommend test driven approach to adding new features as well:

   - Write a test for the functionality you want to add.
   - Write the functional code until the test passes.

4. Add/update the relevant tests!

- See the [implementing a metric guide](https://lightning.ai/docs/torchmetrics/stable/pages/implement.html) for a step-by-step walkthrough of adding a new metric

#### Metric implementation checklist

New metrics follow a consistent structure across the codebase:

1. **Functional helpers** in `src/torchmetrics/functional/<domain>/<metric>.py` — pure functions named
   `_<metric>_update(...)` and `_<metric>_compute(...)` / `_<metric>_reduce(...)`.
2. **Module class** in `src/torchmetrics/<domain>/<metric>.py` — subclass `Metric`, register state via
   `self.add_state(...)` in `__init__`, delegate to the functional helpers in `update()` and `compute()`.
3. **Classification metrics** exist as three task-specific variants (`BinaryXxx`, `MulticlassXxx`,
   `MultilabelXxx`) plus a public wrapper class (e.g. `Accuracy`) that accepts a `task` argument and
   dispatches to the right variant.
4. **Exports** — add to both `src/torchmetrics/<domain>/__init__.py` and
   `src/torchmetrics/functional/<domain>/__init__.py`.
5. **Optional dependencies** — if the metric requires an optional package, gate it with the
   `RequirementCache` flags from `torchmetrics.utilities.imports` and add a module-level
   `__doctest_requires__` mapping so doctests are skipped when the dependency is absent.
6. **Tests** in `tests/unittests/<domain>/test_<metric>.py` using `MetricTester` as the base class.

### Test cases:

Want to keep Torchmetrics healthy? Love seeing those green tests? So do we! How do we keep it that way?
We write tests! We value tests contribution even more than new features. One of the core values of torchmetrics
is that our users can trust our metric implementation. We can only guarantee this if our metrics are well tested.

______________________________________________________________________

## Guidelines

### Developments scripts

To build the documentation locally, simply execute the following commands from project root (only for Unix):

- `make clean` cleans repo from temp/generated files
- `make docs` builds documentation under `docs/build/html` (open `docs/build/html/index.html` in a browser)
- `make test` runs all project's tests with coverage

By default all make commands will use `python`/`pip` but if you are using [uv](https://docs.astral.sh/uv/)
as your dependency manager, you can instead run `make USE_UV=1 clean` etc. to use `uv` under the hood.

### Original code

All added or edited code shall be the own original work of the particular contributor.
If you use some third-party implementation, all such blocks/functions/modules shall be properly referred and if
possible also agreed by code's author. For example - `This code is inspired from http://...`.
In case you adding new dependencies, make sure that they are compatible with the actual Torchmetrics license
(ie. dependencies should be _at least_ as permissive as the Torchmetrics license).

### Coding Style

1. Use f-strings for output formation (except logging when we stay with lazy `logging.info("Hello %s!", name)`.
2. You can use `pre-commit` to make sure your code style is correct.

### Documentation

We are using Sphinx with Napoleon extension.
Moreover, we set Google style to follow with type convention.

- [Napoleon formatting with Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- [ReStructured Text (reST)](https://docs.pylonsproject.org/projects/docs-style-guide/)
- [Paragraph-level markup](https://www.sphinx-doc.org/en/1.5/markup/para.html)

See following short example of a sample function taking one position string and optional

```python
def my_func(param_a: int, param_b: float | None = None) -> str:
    """Sample function.

    Args:
        param_a: first parameter
        param_b: second parameter

    Returns:
        sum of both numbers

    Example:
        Sample doctest example...
        >>> my_func(1, 2)
        3

    .. hint:: If you want to add something.
    """
    p = param_b if param_b else 0
    return str(param_a + p)
```

When updating the docs make sure to build them first locally and visually inspect the HTML files in a browser for
formatting errors. A missing blank line or a wrong indent can break the layout. Run `make docs` and open
`docs/build/html/index.html`.

Notes:

- You need to have LaTeX installed for rendering math equations. You can for example install TeXLive by doing one of the following:
  - on Ubuntu (Linux) run `apt-get install texlive` or otherwise follow the instructions on the TeXLive website
  - use the [RTD docker image](https://hub.docker.com/r/readthedocs/build)
- Python 3.10 or higher is required

When you send a PR the continuous integration will run tests and build the docs.

### Testing

**Local:** Testing your work locally will help you speed up the process since it allows you to focus on particular (failing) test-cases.
To setup a local development environment, install the package in editable mode with all development dependencies:

```bash
pip install -e . -r requirements/_devel.txt
# or with uv:
make USE_UV=1 env
```

The source lives in `src/torchmetrics/` and unit tests in `tests/unittests/`, mirroring the domain structure.

You can run the full test suite via:

```bash
make test
# or natively (two separate steps matching the src layout):
(cd src && python -m pytest torchmetrics)          # doctests inside the package
(cd tests && python -m pytest unittests -v)        # unit tests

# run tests for a specific domain only
pytest tests/unittests/classification/ -v

# run DDP tests (requires process pool)
USE_PYTEST_POOL="1" pytest -m DDP tests/
```

Note: if your computer does not have multi-GPU nor TPU, DDP and GPU tests are skipped automatically.

All metric test classes use `MetricTester` from `tests/unittests/_helpers/testers.py`, which runs each metric
in both single-process and DDP modes and compares results against a reference implementation (usually scikit-learn).
Use it as the base for any new metric test.

**GitHub Actions:** For convenience, you can also use your own GHActions building which will be triggered with each commit.
This is useful if you do not test against all required dependency versions.
