# Running tests locally

To run the tests locally, you need to have the full development environment set up. This can be setup by running
the following command in the root directory of the project:

```bash
pip install . -r requirements/_devel.txt
```

Then for Windows users, to execute the tests (unit tests and integration tests) run the following command (will only
run non-DDP tests):

```bash
pytest tests/
```

For Linux/Mac users you will need to provide the `-m` argument to indicate if `ddp` tests should also be executed:

```bash
USE_PYTEST_POOL="1" pytest -m DDP tests/  # to run only DDP tests
pytest -m "not DDP" tests/  # to run all tests except DDP tests
```

Some tests depends on real data, and will not run if the data is not available. To pull this data locally, run the
following commands:

```bash
cd tests/
S3_DATA=https://pl-public-data.s3.amazonaws.com/metrics/data.zip  # data location
pip install -q "urllib3>1.0"
python -c "from urllib.request import urlretrieve ; urlretrieve('$S3_DATA', 'data.zip')"  # fetch data
unzip -o data.zip  # unzip data
ls -l _data/*  # list data
```

## Simply Make

Alternatively, for Unix with `make` installed, simply running `make test` from the root of the project will install
all requirements and run the full test suit.

## Test particular domain

To run only unittests, point the command only to the `tests/unittests` directory. Similarly, to only run a subset of the
unittests, like all tests related to the regression domain, run the following command:

```bash
pytest tests/unittests/regression/
```
