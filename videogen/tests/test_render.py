"""CPU golden test: the wireframe renderer must reproduce the reference frames exactly."""
import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest

from scenariocontrol_video.render import _frame_inputs, render_frame

DATA = Path(__file__).parent / "data"


@pytest.mark.parametrize("name", ["frame_0000", "frame_0001"])  # own rig / canonical rig
def test_matches_golden(name):
    with open(DATA / f"{name}.pkl", "rb") as f:
        img = render_frame(*_frame_inputs(pickle.load(f)))
    golden = cv2.cvtColor(cv2.imread(str(DATA / f"{name}_golden.png")), cv2.COLOR_BGR2RGB)
    assert img.dtype == np.uint8 and img.shape == golden.shape == (720, 1280, 3)
    assert (img != golden).any(axis=-1).sum() == 0
