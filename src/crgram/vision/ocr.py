"""Fallback OCR through the tesseract CLI (optional dependency)."""

import shutil
import subprocess
from functools import cache

import cv2
import numpy as np

from crgram.model import ALPHABET

PUNCT = ',.!?:;-—…"«»'


@cache
def available() -> bool:
    return shutil.which("tesseract") is not None


def _prepare(gray: np.ndarray, scale: float) -> bytes:
    img = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    pad = max(10, img.shape[0] // 4)
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    return cv2.imencode(".png", img)[1].tobytes()


def _run(gray: np.ndarray, psm: int, whitelist: str, scale: float = 1.0) -> str:
    if not available():
        return ""
    proc = subprocess.run(
        [
            "tesseract",
            "stdin",
            "stdout",
            "-l",
            "rus",
            "--psm",
            str(psm),
            "-c",
            f"tessedit_char_whitelist={whitelist}",
        ],
        input=_prepare(gray, scale),
        capture_output=True,
        timeout=10,
        check=False,
    )
    return proc.stdout.decode(errors="replace").strip()


def read_number(gray: np.ndarray) -> int | None:
    # psm 7 (text line) sometimes returns nothing for a lone digit; psm 8 (single word) catches it.
    for psm in (7, 8):
        text = _run(gray, psm, "0123456789")
        if text.isdigit():
            return int(text)
    return None


def read_char(gray: np.ndarray) -> str | None:
    """Majority vote over several page segmentation modes: each one misses some letters."""
    votes: dict[str, int] = {}
    for psm, scale in ((10, 1.0), (8, 1.0), (8, 2.0), (13, 1.0)):
        text = _run(gray, psm, ALPHABET + PUNCT, scale=scale).replace(" ", "")
        if len(text) == 1:
            votes[text] = votes.get(text, 0) + 1
    return max(votes, key=votes.__getitem__) if votes else None


def read_word(gray: np.ndarray) -> str:
    return _run(gray, 7, ALPHABET + PUNCT).replace(" ", "")
