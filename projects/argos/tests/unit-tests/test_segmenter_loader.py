# projects/argos/tests/unit-tests/test_segmenter_loader.py
from panoptes.model_registry import load_segmenter  # type: ignore


def test_segmenter_present():
    """
    Guard against regressions: a missing segmentation checkpoint would silently
    degrade heat-maps to red rectangles.
    """
    assert load_segmenter() is not None, (
        "No segmentation weights were loaded - heat-maps will not show masks!"
    )
