"""Glyph recognition by template matching against a bank of learned samples.

The game uses one font, so a handful of labelled samples per symbol is enough.
Unknown symbols fall back to OCR (see `ocr.py`); confirmed results are added
to the bank so the next scan recognizes them directly.
"""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from crgram.vision.layout import Glyph

SIZE = 32


def normalize(mask: np.ndarray) -> np.ndarray:
    """Scale a tight glyph mask into a SIZExSIZE float image, preserving aspect ratio."""
    h, w = mask.shape
    k = (SIZE - 2) / max(h, w)
    nh, nw = max(1, round(h * k)), max(1, round(w * k))
    small = cv2.resize(mask.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((SIZE, SIZE), np.float32)
    y, x = (SIZE - nh) // 2, (SIZE - nw) // 2
    out[y : y + nh, x : x + nw] = small
    return cv2.GaussianBlur(out, (3, 3), 0)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d else 0.0


@dataclass
class Match:
    label: str
    score: float
    margin: float  # lead over the best sample with another label


class GlyphBank:
    """Labelled glyph samples: read-only seed samples shipped with the package plus
    samples learned by the user, persisted in a single .npz file."""

    def __init__(
        self,
        path: Path,
        seed: Path | None = None,
        *,
        min_score: float,
        min_margin: float = 0.0,
        clear_lead: tuple[float, float] | None = None,
    ) -> None:
        """A match is accepted when score >= min_score and margin >= min_margin, or,
        with `clear_lead=(score, margin)`, when a lower score leads the others clearly."""
        self.path = path
        self.min_score, self.min_margin, self.clear_lead = min_score, min_margin, clear_lead
        self.labels: list[str] = []
        self.images: list[np.ndarray] = []
        self.feats: list[tuple[float, float]] = []  # (rel_h, rel_cy)
        if seed is not None and seed.exists() and seed != path:
            self._load(seed)
        self._n_seed = len(self.labels)
        if path.exists():
            self._load(path)

    def _load(self, path: Path) -> None:
        with np.load(path) as data:
            self.labels += [str(x) for x in data["labels"]]
            self.images += list(data["images"])
            self.feats += [tuple(f) for f in data["feats"]]  # type: ignore[misc]

    def __len__(self) -> int:
        return len(self.labels)

    def match(self, glyph: Glyph) -> Match | None:
        if not self.labels:
            return None
        img = normalize(glyph.mask)
        best: dict[str, float] = {}
        for i, (label, tmpl) in enumerate(zip(self.labels, self.images, strict=True)):
            # Shape only: glyph size and position relative to the plate differ between game themes.
            score = _ncc(img, tmpl)
            if i >= self._n_seed:
                score += 0.02  # user corrections win over shipped samples
            best[label] = max(best.get(label, -1.0), score)
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        label, score = ranked[0]
        return Match(label, score, score - ranked[1][1] if len(ranked) > 1 else 1.0)

    def classify(self, glyph: Glyph) -> str | None:
        m = self.match(glyph)
        if m is None:
            return None
        if m.score >= self.min_score and m.margin >= self.min_margin:
            return m.label
        if self.clear_lead and m.score >= self.clear_lead[0] and m.margin >= self.clear_lead[1]:
            return m.label
        return None

    def add(self, label: str, glyph: Glyph, *, save: bool = True) -> None:
        img = normalize(glyph.mask)
        # Drop learned look-alikes with another label (a correction), skip duplicates.
        for i in range(len(self.labels) - 1, self._n_seed - 1, -1):
            if _ncc(img, self.images[i]) > 0.97 and self.labels[i] != label:
                del self.labels[i], self.images[i], self.feats[i]
        for lab, tmpl in zip(self.labels, self.images, strict=True):
            if lab == label and _ncc(img, tmpl) > 0.97:
                return
        self.labels.append(label)
        self.images.append(img)
        self.feats.append((glyph.rel_h, glyph.rel_cy))
        if save:
            self.save()

    def save(self) -> None:
        n = self._n_seed
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".npz")
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(
                f,
                labels=np.array(self.labels[n:], dtype=str),
                images=np.array(self.images[n:], np.float32).reshape(-1, SIZE, SIZE),
                feats=np.array(self.feats[n:], np.float32).reshape(-1, 2),
            )
        os.replace(tmp, self.path)
