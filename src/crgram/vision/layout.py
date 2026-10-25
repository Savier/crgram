"""Geometry of the game board: plates (words), slots (cells) and glyph crops.

No recognition happens here — only segmentation. All size thresholds are
relative to the typical plate height so the code does not depend on screen
resolution, and colours are judged relative to white plates, so the board
background colour does not matter.
"""

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from crgram.model import BBox

type Gray = np.ndarray  # HxW uint8
type Mask = np.ndarray  # HxW bool

WHITE_MIN = 245  # min channel value of plate background
DARK_MAX = 128  # luminance of ink (letters, digits, punctuation)
PUNCT_MAX_H = 0.2  # symbols lower than this (relative to plate) are punctuation


@dataclass
class Glyph:
    """A single dark symbol, cropped tight, with position relative to its plate."""

    mask: Mask
    gray: Gray  # same region with a margin, for fallback OCR
    rel_h: float  # glyph height / plate height
    rel_cy: float  # glyph centre y from plate top / plate height
    bbox: BBox


@dataclass
class Slot:
    kind: Literal["box", "glyph", "lock"]
    bbox: BBox
    selected: bool = False
    # Cut by the screen edge: the cell runs into the plate border, part of it is not visible
    clipped: bool = False
    glyphs: list[Glyph] = field(default_factory=list)  # letter / punctuation (top band)
    digits: list[Glyph] = field(default_factory=list)  # code digits (bottom band), left to right
    number_gray: Gray | None = None


@dataclass
class Plate:
    bbox: BBox
    hint: bool
    slots: list[Slot] = field(default_factory=list)
    hint_gray: Gray | None = None
    row: int = 0
    baseline: int = 0  # y of the letters' bottom, to tell "." from "'"


@dataclass
class Layout:
    plates: list[Plate]
    plate_h: int
    width: int
    height: int


def _components(mask: Mask) -> list[tuple[int, int, int, int, int]]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [tuple(int(v) for v in s) for s in stats[1:n]]  # type: ignore[misc]


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) ranges of consecutive True values."""
    padded = np.concatenate([[False], flags.astype(bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def _is_card(gray: Gray, x: int, y: int, w: int, h: int) -> bool:
    """Bottom letter cards: white face inside a dark outline inside a thin white rim."""
    cy = y + h // 2
    xs = [x - d for d in range(1, 16) if x - d >= 0]
    if len(xs) < 15:
        return False
    profile = gray[cy, xs]
    dark = profile < 90
    return bool(dark[:8].any() and (profile[dark.argmax() :] > 230).any())


def _is_dotted_frame(gray: Gray, x: int, y: int, w: int, h: int) -> bool:
    """Hint words are drawn inside a dotted rectangle."""
    best = 0
    for dy in range(4, max(5, h // 5)):
        line = gray[y + dy, x + w // 10 : x + w - w // 10] < 200
        best = max(best, len(_runs(line)))
    return best >= 12


def _glyph(ink: Mask, gray: Gray, ox: int, oy: int, plate: BBox) -> Glyph | None:
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = max(4, plate.h // 20)
    gy0, gx0 = max(oy + y0 - m, 0), max(ox + x0 - m, 0)
    return Glyph(
        mask=ink[y0:y1, x0:x1].copy(),
        gray=gray[gy0 : oy + y1 + m, gx0 : ox + x1 + m].copy(),
        rel_h=(y1 - y0) / plate.h,
        rel_cy=(oy + (y0 + y1) / 2 - plate.y) / plate.h,
        bbox=BBox(x=int(ox + x0), y=int(oy + y0), w=int(x1 - x0), h=int(y1 - y0)),
    )


def _split_glyphs(ink: Mask, gray: Gray, ox: int, oy: int, plate: BBox, *, gap: int) -> list[Glyph]:
    """Split ink into symbols by empty columns (gaps narrower than `gap` are kept inside one symbol)."""
    cols = _runs(ink.any(0))
    merged: list[list[int]] = []
    for a, b in cols:
        if merged and a - merged[-1][1] < gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        if g := _glyph(ink[:, a:b], gray, ox + a, oy, plate):
            out.append(g)
    return out


def _is_lock(nonwhite: Mask, ink: Mask, ph: int) -> bool:
    """A lock is a light padlock with two dots above; a dot turns dark when a neighbour
    is guessed, so a little ink is allowed, but much less than any letter."""
    body = (nonwhite & ~ink).sum()
    return bool(body > 0.02 * ph * ph and ink.sum() < 0.01 * ph * ph)


def _segment_plate(rgb: np.ndarray, gray: Gray, plate: Plate) -> None:
    b = plate.bbox
    ph = b.h
    top0, top1 = b.y + int(0.02 * ph), b.y + int(0.77 * ph)
    bot0, bot1 = b.y + int(0.74 * ph), b.y + b.h - int(0.02 * ph)
    edge = max(3, int(0.04 * ph))

    top_rgb = rgb[top0:top1, b.x : b.x + b.w].astype(np.int16)
    top_gray = gray[top0:top1, b.x : b.x + b.w]
    nonwhite = top_rgb.min(-1) < 235
    ink = top_gray < DARK_MAX

    # A run touching the plate border is a cell cut by the screen edge (plates have white margins).
    runs = [(a, e) for a, e in _runs(nonwhite.any(0)) if e - a >= 0.12 * ph or edge <= a < e <= b.w - edge]
    # Letters like "Ы" consist of separated strokes: merge close ink-only runs.
    gap = int(0.07 * ph)
    merged: list[list[int]] = []
    for a, e in runs:
        if merged and a - merged[-1][1] < gap and ink[:, a:e].any() and ink[:, merged[-1][0] : merged[-1][1]].any():
            merged[-1][1] = e
        else:
            merged.append([a, e])

    for a, e in merged:
        w = e - a
        clipped = a < edge or e > b.w - edge
        col = nonwhite[:, a:e]
        rows = np.flatnonzero(col.any(1))
        r0, r1 = rows.min(), rows.max() + 1
        fill = col[r0:r1].mean()
        bbox = BBox(x=b.x + a, y=top0 + int(r0), w=w, h=int(r1 - r0))
        if clipped:
            # Too narrow for the width tests: a box is filled solid, a lock is a light outline.
            if fill > 0.7:
                slot = Slot("box", bbox)
            elif _is_lock(col, ink[:, a:e], ph):
                slot = Slot("lock", bbox)
            else:
                slot = Slot("glyph", bbox, glyphs=_split_glyphs(ink[:, a:e], gray, b.x + a, top0, b, gap=gap))
            slot.clipped = True
        elif w >= 0.35 * ph and fill > 0.7:
            px = top_rgb[r0:r1, a:e].reshape(-1, 3)
            hsv = cv2.cvtColor(px.astype(np.uint8)[None], cv2.COLOR_RGB2HSV)[0]
            selected = float(np.median(hsv[:, 1])) > 120
            slot = Slot("box", bbox, selected=selected)
        elif w >= 0.12 * ph and _is_lock(nonwhite[:, a:e], ink[:, a:e], ph):
            slot = Slot("lock", bbox)
        elif ink[:, a:e].sum() > 0.0005 * ph * ph:
            slot = Slot(
                "glyph",
                bbox,
                glyphs=_split_glyphs(ink[:, a:e], gray, b.x + a, top0, b, gap=gap),
            )
        elif w >= 0.12 * ph:
            slot = Slot("lock", bbox)
        else:  # thin light overlay, e.g. the game's scrollbar
            continue
        plate.slots.append(slot)

    boxes = [s.bbox.w for s in plate.slots if s.kind == "box" and not s.clipped]
    cell_w = int(np.median(boxes)) if boxes else ph // 2

    letters = [g.bbox.y + g.bbox.h for s in plate.slots for g in s.glyphs if g.rel_h >= PUNCT_MAX_H]
    plate.baseline = int(np.median(letters)) if letters else b.y + ph // 2

    # Code digits below each slot
    half = int(0.30 * ph)
    for slot in plate.slots:
        if slot.kind == "lock" or (slot.glyphs and max(g.rel_h for g in slot.glyphs) < PUNCT_MAX_H):
            continue
        x0, x1 = max(slot.bbox.cx - half, b.x + edge), min(slot.bbox.cx + half, b.x + b.w - edge)
        if slot.clipped:
            x0, x1 = max(slot.bbox.x - edge, b.x), min(slot.bbox.x + slot.bbox.w + edge, b.x + b.w)
        _read_digits(gray, slot, b, (x0, x1), (bot0, bot1))
        if slot.clipped and not _code_visible(slot, plate, cell_w):
            slot.digits, slot.number_gray = [], None


def _code_visible(slot: Slot, plate: Plate, cell_w: int) -> bool:
    """Whether the code under a clipped cell is read whole: no digit touches the cut and the
    digits are centred where the full cell would be (a digit cut away entirely shifts them)."""
    if not slot.digits:
        return False
    b, s = plate.bbox, slot.bbox
    m = max(2, b.h // 50)
    left, right = min(g.bbox.x for g in slot.digits), max(g.bbox.x + g.bbox.w for g in slot.digits)
    if left <= b.x + m or right >= b.x + b.w - m:
        return False
    full_cx = s.x + s.w - cell_w / 2 if s.x - b.x < b.x + b.w - (s.x + s.w) else s.x + cell_w / 2
    return abs((left + right) / 2 - full_cx) < 0.06 * b.h


def _underline_row(rgb: np.ndarray, gray: Gray, b: BBox) -> int | None:
    """Row (absolute y) of the cell underlines if the plate uses the underlined style."""
    ph = b.h
    y0, y1 = b.y + int(0.4 * ph), b.y + int(0.75 * ph)
    area = rgb[y0:y1, b.x : b.x + b.w].astype(np.int16)
    # Underlines are light, greyish and not ink.
    line = (area.min(-1) < 235) & (gray[y0:y1, b.x : b.x + b.w] >= 150) & (np.ptp(area, -1) < 40)
    counts = line.sum(1)
    if counts.max() < 0.25 * b.w:
        return None
    strong = counts >= 0.5 * counts.max()
    if strong.sum() > 0.05 * ph:  # a filled box, not a thin line
        return None
    y = y0 + int(np.flatnonzero(strong).mean())
    # A real underline has white paper right above it; anti-aliased letter bottoms have ink there.
    cols = line[y - y0]
    d = max(3, int(0.04 * ph))
    above = rgb[y - d, b.x : b.x + b.w][cols].min(-1)
    if (above >= 235).mean() < 0.8:
        return None
    return y


def _segment_underlined(rgb: np.ndarray, gray: Gray, plate: Plate, line_y: int) -> None:
    """Plate style where each cell is an underline with the letter above and the code below."""
    b = plate.bbox
    ph = b.h
    edge = max(3, int(0.04 * ph))
    x0, x1 = b.x + edge, b.x + b.w - edge
    row = rgb[line_y - 1 : line_y + 2, x0:x1].astype(np.int16)
    green = _is_green(row)
    lined = (row.min(-1) < 235) & (gray[line_y - 1 : line_y + 2, x0:x1] >= 150)
    cell_cols = (lined | green).any(0)

    top0, top1 = b.y + edge, line_y - max(2, ph // 50)
    ink = gray[top0:top1, x0:x1] < DARK_MAX
    bot0, bot1 = line_y + max(2, ph // 50), b.y + b.h - edge
    slots: list[Slot] = []
    covered = np.zeros(x1 - x0, bool)
    for a, e in _runs(cell_cols):
        if e - a < 0.1 * ph:
            continue
        covered[a:e] = True
        cell_ink = ink[:, a:e]
        sel = bool(green[:, a:e].mean() > 0.5)
        if cell_ink.sum() > 0.0005 * ph * ph and not sel:
            rows = np.flatnonzero(cell_ink.any(1))
            bbox = BBox(x=x0 + a, y=top0 + int(rows.min()), w=e - a, h=int(rows.max() - rows.min() + 1))
            slot = Slot("glyph", bbox, glyphs=_split_glyphs(cell_ink, gray, x0 + a, top0, b, gap=int(0.07 * ph)))
        else:
            slot = Slot("box", BBox(x=x0 + a, y=top0, w=e - a, h=line_y - top0), selected=sel)
        _read_digits(gray, slot, b, (x0 + a - edge, x0 + e + edge), (bot0, bot1))
        slots.append(slot)

    # Punctuation sits outside the underlined cells.
    pink = gray[top0:bot0, x0:x1] < DARK_MAX
    for a, e in _runs(pink.any(0) & ~covered):
        if g := _glyph(pink[:, a:e], gray, x0 + a, top0, b):
            slots.append(Slot("glyph", g.bbox, glyphs=[g]))
    plate.slots = sorted(slots, key=lambda s: s.bbox.x)
    plate.baseline = line_y


def _is_green(px: np.ndarray) -> Mask:
    """Selected cell colour: saturated green."""
    r, g, bl = px[..., 0].astype(int), px[..., 1].astype(int), px[..., 2].astype(int)
    return (g > 150) & (g - r > 60) & (g - bl > 60)


def _read_digits(gray: Gray, slot: Slot, plate: BBox, cols: tuple[int, int], rows: tuple[int, int]) -> None:
    """Code number under a slot, searched in the given absolute column/row ranges."""
    ph = plate.h
    (x0, x1), (y0, y1) = cols, rows
    x0, x1 = max(x0, plate.x), min(x1, plate.x + plate.w)
    band = gray[y0:y1, x0:x1] < DARK_MAX + 20  # digits are a bit lighter than letters
    if band.sum() < 0.0005 * ph * ph:
        return
    digits = []
    for x, y, w, h, _area in _components(band):
        if h < 0.08 * ph or h > 0.25 * ph:  # noise, punctuation tail, selected cell outline
            continue
        if g := _glyph(band[y : y + h, x : x + w], gray, x0 + x, y0 + y, plate):
            digits.append(g)
    if digits:
        slot.digits = sorted(digits, key=lambda g: g.bbox.x)
        m = int(0.05 * ph)
        slot.number_gray = gray[y0 - m : y1, x0:x1].copy()


def _segment_hint(gray: Gray, plate: Plate) -> None:
    b = plate.bbox
    inset = int(0.1 * b.h)
    plate.hint_gray = gray[b.y + inset : b.y + b.h - inset, b.x + inset : b.x + b.w - inset].copy()


def _drop_keyboard(cands: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """On-screen keyboard: rows of many equal white keys. Word plates differ in width."""
    keys: set[int] = set()
    for i, (_, y, w, h) in enumerate(cands):
        same = [
            j for j, (_, y2, w2, h2) in enumerate(cands) if abs(y2 - y) <= 3 and abs(w2 - w) <= 3 and abs(h2 - h) <= 3
        ]
        if len(same) >= 7:
            keys.update(same)
            keys.add(i)
    return [c for i, c in enumerate(cands) if i not in keys]


def _drop_scrollbar(white: np.ndarray, max_len: int) -> None:
    """The game's scrollbar is a white bar through all rows: it would glue their plates into one.
    No plate is nearly as tall, so white column runs longer than `max_len` are erased."""
    for x in range(white.shape[1]):
        for a, e in _runs(white[:, x]):
            if e - a > max_len:
                white[a:e, x] = 0


def _assign_rows(plates: list[Plate]) -> None:
    plates.sort(key=lambda p: p.bbox.cy)
    row, last = -1, -(10**9)
    for p in plates:
        if p.bbox.cy - last > p.bbox.h * 0.5:
            row += 1
            last = p.bbox.cy
        p.row = row
    plates.sort(key=lambda p: (p.row, p.bbox.x))


def analyze(image_bgr: np.ndarray) -> Layout:
    H, W = image_bgr.shape[:2]
    rgb = image_bgr[..., ::-1]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    white = (rgb.min(-1) >= WHITE_MIN).astype(np.uint8)
    # Opening removes thin white rims (cards, icons) while keeping plate margins.
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    _drop_scrollbar(white, W // 4)

    min_side = W * 0.04
    cands = [
        (x, y, w, h)
        for x, y, w, h, area in _components(white.astype(bool))
        if w >= min_side and h >= min_side and not _is_card(gray, x, y, w, h)
    ]
    cands = _drop_keyboard(cands)
    if not cands:
        return Layout([], 0, W, H)

    # Typical plate height: the most common height (with tolerance) among candidates.
    hs = np.array([h for *_, h in cands])
    votes = [(np.abs(hs - h) <= max(3, h * 0.03)).sum() for h in hs]
    plate_h = int(np.median(hs[np.abs(hs - hs[int(np.argmax(votes))]) <= 3]))

    plates: list[Plate] = []
    for x, y, w, h in cands:
        bbox = BBox(x=x, y=y, w=w, h=h)
        if abs(h - plate_h) <= max(3, plate_h * 0.04):
            plate = Plate(bbox, hint=False)
            if (line_y := _underline_row(rgb, gray, bbox)) is not None:
                _segment_underlined(rgb, gray, plate, line_y)
            else:
                _segment_plate(rgb, gray, plate)
            if plate.slots:
                plates.append(plate)
        elif 0.55 * plate_h <= h < plate_h and _is_dotted_frame(gray, x, y, w, h):
            plate = Plate(bbox, hint=True)
            _segment_hint(gray, plate)
            plates.append(plate)

    _assign_rows(plates)
    return Layout(plates, plate_h, W, H)
