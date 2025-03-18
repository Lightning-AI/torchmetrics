.PHONY: clean test get-sphinx-template docs live-docs env data

export TOKENIZERS_PARALLELISM=false
export FREEZE_REQUIREMENTS=1
# assume you have installed need packages
export SPHINX_MOCK_REQUIREMENTS=1
export SPHINX_FETCH_ASSETS=0
export SPHINX_PIN_RELEASE_VERSIONS=1

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

test: clean env data
	# run tests with coverage
	cd src && python -m pytest torchmetrics
	cd tests && python -m pytest unittests -v --cov=torchmetrics
	cd tests && python -m coverage report

get-sphinx-template:
	pip install -q awscli
	aws s3 sync --no-sign-request s3://sphinx-packages/ dist/
	pip install lai-sphinx-theme -q -U -f dist/

docs: clean get-sphinx-template
	pip install -e . --quiet -r requirements/_docs.txt
	# apt-get install -y texlive-latex-extra dvipng texlive-pictures texlive-fonts-recommended cm-super
	cd docs && make html --debug --jobs $(nproc) SPHINXOPTS="-W --keep-going"

live-docs: get-sphinx-template
	pip install -e . --quiet -r requirements/_docs.txt
	cd docs && make livehtml --jobs $(nproc)

env:
	pip install -e . -U -r requirements/_devel.txt

data:
	pip install -q wget
	python -m wget https://pl-public-data.s3.amazonaws.com/metrics/data.zip
	unzip -o data.zip -d ./tests
