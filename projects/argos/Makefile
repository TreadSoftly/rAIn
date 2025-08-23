# Helper targets for local development - keep it *very* lightweight
.PHONY: build install test fmt lint run clean

# --------------------------------------------------------------------------- #
#  Build everything with one command (editable + GUI OpenCV + bootstrap)
#  Run this from projects/argos/ (Windows PowerShell/cmd/bash all OK)
# --------------------------------------------------------------------------- #
build:
	# 1) Remove any headless OpenCV that would disable HighGUI windows
	-python -m pip uninstall -y opencv-python-headless opencv-contrib-python-headless
	# 2) Ensure pip tooling is up to date
	python -m pip install -U pip setuptools wheel
	# 3) Install Argos in editable mode with dev extras, honoring constraints;
	#    also explicitly install GUI-capable OpenCV (no headless).
	python -m pip install -c constraints.txt -e ".[dev]" opencv-python
	# 4) Bootstrap ensures venv, installs editable inside it, and checks models
	python bootstrap.py --ensure --yes --reinstall

# --------------------------------------------------------------------------- #
#  Install the project in editable-mode (with dev extras) in the *current* env
# --------------------------------------------------------------------------- #
install:
	python -m pip install -c constraints.txt -e ".[dev]" opencv-python

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
	#   target tests/raw/assets.jpg hm
	#   target tests/raw/assets.jpg pse
	#   target tests/raw/assets.jpg clf --topk 3 --annotate
	#   target tests/raw/assets.jpg obb
	target tests/raw/assets.jpg hm
