# Helper targets for local development - keep it *very* lightweight
.PHONY: install test fmt lint run clean

# --------------------------------------------------------------------------- #
#  Install the project in editable-mode (with dev extras)
# --------------------------------------------------------------------------- #
install:
	python -m pip install -e .[dev]

# --------------------------------------------------------------------------- #
#  Run the pytest suite quietly
# --------------------------------------------------------------------------- #
test:
	pytest -q

# --------------------------------------------------------------------------- #
#  Format + import-sort everything
# --------------------------------------------------------------------------- #
fmt:
	black dronevision tests
	isort dronevision tests

# --------------------------------------------------------------------------- #
#  Basic static-analysis / style checks (expand as desired)
# --------------------------------------------------------------------------- #
lint:
	ruff check dronevision tests

# --------------------------------------------------------------------------- #
#  Quick manual sanity-run (edit the target image to taste)
# --------------------------------------------------------------------------- #
run:
	# Example CLI invocation - feel free to customise
	target tests/assets/shibuya.jpg hm --alpha 0.5
