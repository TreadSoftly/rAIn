# projects/argos/Makefile — convenience targets for local dev
.PHONY: build install test fmt lint run clean

build:
	- python -m pip uninstall -y opencv-python-headless opencv-contrib-python-headless
	python -m pip install -U pip setuptools wheel
	python -m pip install -c constraints.txt -e ".[dev]" opencv-python
	python bootstrap.py --ensure --yes --reinstall

install:
	python -m pip install -c constraints.txt -e ".[dev]" opencv-python

test:
	py -m pytest -q

fmt:
	black panoptes tests
	isort panoptes tests

lint:
	ruff check panoptes tests

run:
	target tests/raw/assets.jpg hm
