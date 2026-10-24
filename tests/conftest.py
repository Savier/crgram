from pathlib import Path

import cv2
import numpy as np
import pytest

SAMPLES = Path(__file__).parent.parent / "samples"


def load(name: str) -> np.ndarray:
    return cv2.imread(str(SAMPLES / name))


@pytest.fixture
def screen_01() -> Path:
    return SAMPLES / "screen_01.png"
