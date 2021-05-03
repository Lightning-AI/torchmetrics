.PHONY: test clean docs env

# assume you have installed need packages
export SPHINX_MOCK_REQUIREMENTS=1

clean:
	# clean all temp runs
	rm -rf $(shell find . -name "mlruns")
	rm -rf _ckpt_*
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf ./docs/build
	rm -rf ./docs/source/generated
	rm -rf ./docs/source/*/generated
	rm -rf ./docs/source/api

test: clean env

	# run tests with coverage
	python -m pytest torchmetrics tests -v --cov=torchmetrics
	python -m coverage report

docs: clean
	pip install --quiet -r requirements/docs.txt
	python -m sphinx -b html -W --keep-going docs/source docs/build

env:
	pip install -r requirements.txt
	pip install -r requirements/test.txt
