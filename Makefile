.PHONY: clean test env data uv_env

export TOKENIZERS_PARALLELISM=false
export FREEZE_REQUIREMENTS=1
# assume you have installed need packages
export SPHINX_MOCK_REQUIREMENTS=1
export SPHINX_FETCH_ASSETS=0
export SPHINX_PIN_RELEASE_VERSIONS=1

USE_UV ?= 0
HAS_UV := $(shell command -v uv >/dev/null 2>&1 && echo 1 || echo 0)

ifeq ($(USE_UV),1)
  ifeq ($(HAS_UV),1)
    PIP_CMD := uv pip install
    RUN_CMD := uv run --no-project
	UV_ENV_CMD := uv venv --no-project --allow-existing
  else
    $(error "USE_UV=1 but uv not found in PATH")
  endif
else
  PIP_CMD := pip install
  RUN_CMD := python
  UV_ENV_CMD := @true  # no-op if uv not used
endif

uv_env:
	$(UV_ENV_CMD)

clean:
	# clean all temp runs
	rm -rf $(shell find . -name "mlruns")
	rm -rf _ckpt_*
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf tests/.pytest_cache
	rm -rf ./docs/build
	rm -rf ./docs/source/generated
	rm -rf ./docs/source/*/generated
	rm -rf ./docs/source/api
	rm -rf ./docs/source/gallery
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf src/*.egg-info

test: clean env data uv_env
	# run tests with coverage
	cd src && $(RUN_CMD) -m pytest torchmetrics
	cd tests && $(RUN_CMD) -m pytest unittests -v --cov=torchmetrics
	cd tests && $(RUN_CMD) -m coverage report

env:
	$(PIP_CMD) -e . -U -r requirements/_devel.txt

data: uv_env
	$(PIP_CMD) -q wget
	$(RUN_CMD) -m wget https://pl-public-data.s3.amazonaws.com/metrics/data.zip
	unzip -o data.zip -d ./tests
