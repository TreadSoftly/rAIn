.PHONY: install test run fmt

install:
	python -m pip install -e .[dev]

test:
	pytest -q

fmt:
	black dronevision tests
	isort dronevision tests

run:
	# example - edit to taste
	target tests/assets/bus.jpg
