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
	py -m pytest -q

# --------------------------------------------------------------------------- #
#  Format + import-sort everything
# --------------------------------------------------------------------------- #
fmt:
	black panoptes tests
	isort panoptes tests

# --------------------------------------------------------------------------- #
#  Basic static-analysis / style checks (expand as desired)
# --------------------------------------------------------------------------- #
lint:
	ruff check panoptes tests

# --------------------------------------------------------------------------- #
#  Quick manual sanity-run (edit the target image to taste)
# --------------------------------------------------------------------------- #
run:
	# Example CLI invocations - feel free to customise:
	#   target tests/assets/assets.jpg hm
	#   target tests/assets/assets.jpg pse
	#   target tests/assets/assets.jpg clf --topk 3 --annotate
	#   target tests/assets/assets.jpg obb
	target tests/assets/assets.jpg hm
