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
help you or finish it with you :]_

### New Features:

1. Submit a github issue - describe what is the motivation of such feature (adding the use case or an example is helpful).
2. Let's discuss to determine the feature scope.
3. Submit a PR! We recommend test driven approach to adding new features as well:

   - Write a test for the functionality you want to add.
   - Write the functional code until the test passes.

4. Add/update the relevant tests!

- [This PR](https://github.com/PyTorchLightning/pytorch-lightning/pull/5241) is a good example for adding a new metric

### Test cases:

Want to keep Torchmetrics healthy? Love seeing those green tests? So do we! How to we keep it that way? 
We write tests! We value tests contribution even more than new features. One of the core values of torchmetrics
is that our users can trust our metric implementation. We can only garantee this if our metrics are well tested.

---

## Guidelines

### Implementing a new metric
Want to contribute a new metric. Great, but to keep all our metrics consistent please format in the following way:

1. First implement the functional backend. This takes cares of all logic that does into the metric. The code should
to put into single file placed under `torchmetrics/functional/"domain"/"new_metric".py` where `domain` is the type of
metric (classification, regression, nlp ect) and `new_metric` is the name of the metric. In this file should be the
following three functions:
    1. `_new_metric_update(...)`: everything that has to do with type/shape checking and all logic required before
  distributed syncing need to go here.
    2. `_new_metric_compute(...)`: all remaining logic
    3. `new_metric(...)`: essentially wraps the `_update` and `_compute` private functions into one public function that
  makes up the functional interface for the metric.

    The [accuracy](https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/functional/classification/accuracy.py)
metric is a great example of this division of logic. 

2. In a corresponding file placed in `torchmetrics/"domain"/"new_metric".py` create the module interface:
    1. Create a new module metric by subclassing `torchmetrics.Metric`
    2. In the `__init__` of the module call `self.add_state` for as many metric states are needed for the metric to
  proper accumulate metric statistics
    3. The module interface should essentially call the private `_new_metric_update(...)` in its `update` method and simiarly the 
  `_new_metric_compute(...)` function in its `compute`. No logic should really be implemented in the module interface.
  We do this to not have duplicate code to maintain.

    The module interface [Accuracy](https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/classification/accuracy.py)
that correspond to the above functional example showcases these steps

3. Remember to add binding to the different relevant `__init__` files

4. Testing is key to keeping torchmetrics trustworty. This is why we have a very rigid testing protocol. This means
that we in most cases require the metric to be tested against some other commen framework (`sklearn`, `scipy` ect).
    1. Create a testing file in `tests/"domain"/test_"new_metric".py`. Only one file is needed as it is intended to test
  both the functional and module interface
    2. In that file, start by defining a number of test inputs that your metric should be evaluated on 
    3. Create a testclass `class NewMetric(MetricTester)` that inherits from `tests.helpers.testers.MetricTester`.
  This testclass should essentially implement the `test_"new_metric"_class` and `test_"new_metric"_fn` methods that
  respectively tests the module interface and the functional interface.
    4. The testclass should be parametrized (using `@pytest.mark.parametrize`) by the different test inputs defined initiallly.
  Additionally, the `test_"new_metric"_class` method should also be parametrized with an `ddp` parameter such that it gets
  tested in a distributed setting. If your metric has additionally parameters, then make sure to also parametrize these 
  such that different combinations of input and parameters gets tested.
    5. (optional) Ff your metrics raises any exceptions, please add tests that showcases this

    The test file for [accuracy](https://github.com/PyTorchLightning/metrics/blob/master/tests/classification/test_accuracy.py) metric
shows how to implement such tests. 

If you only can figure out part of the steps, do not fear to send a PR. We will much rather receive working
metrics that are not formatted exactly like our codebase, than not receiving any. Formatting can always be applied. 
We will gladly guide and/or help implement the remaining :]

### Developments scripts
To build the documentation locally, simply execute the following commands from project root (only for Unix):
- `make clean` cleans repo from temp/generated files
- `make docs` builds documentation under _docs/build/html_
- `make test` runs all project's tests with coverage

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
from typing import Optional

def my_func(param_a: int, param_b: Optional[float] = None) -> str:
    """Sample function.

    Args:
        param_a: first parameter
        param_b: second parameter

    Return:
        sum of both numbers

    Example:
        Sample doctest example...
        >>> my_func(1, 2)
        3

    .. note:: If you want to add something.
    """
    p = param_b if param_b else 0
    return str(param_a + p)
```

When updating the docs make sure to build them first locally and visually inspect the html files (in the browser) for
formatting errors. In certain cases, a missing blank line or a wrong indent can lead to a broken layout.
Run these commands

```bash
pip install -r requirements/docs.txt
cd docs
make html
```

and open `docs/build/html/index.html` in your browser.

Notes:

- You need to have LaTeX installed for rendering math equations. You can for example install TeXLive by doing one of the following:
  - on Ubuntu (Linux) run `apt-get install texlive` or otherwise follow the instructions on the TeXLive website
  - use the [RTD docker image](https://hub.docker.com/r/readthedocs/build)
- with PL used class meta you need to use python 3.7 or higher

When you send a PR the continuous integration will run tests and build the docs. You can access a preview of the html pages in the
_Artifacts_ tab in CircleCI when you click on the task named _ci/circleci: Build-Docs_ at the bottom of the PR page.

### Testing

**Local:** Testing your work locally will help you speed up the process since it allows you to focus on particular (failing) test-cases.
To setup a local development environment, install both local and test dependencies:

```bash
python -m pip install ".[dev, examples]"
python -m pip install pre-commit
```

You can run the full test-case in your terminal via this make script:

```bash
make test
```

Note: if your computer does not have multi-GPU nor TPU these tests are skipped.

**GitHub Actions:** For convenience, you can also use your own GHActions building which will be triggered with each commit.
This is useful if you do not test against all required dependency versions.
