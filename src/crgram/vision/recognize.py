"""Turn a screenshot into a `Board`: layout segmentation + glyph recognition."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from crgram.model import ALPHABET, Board, Cell, CellKind, Word
from crgram.vision import ocr
from crgram.vision.glyphs import GlyphBank
from crgram.vision.layout import PUNCT_MAX_H, Glyph, Slot, _components, analyze

MAX_CODE = len(ALPHABET)
SEED_DIR = Path(__file__).parent.parent / "data"

type CellRef = tuple[int, int]  # (word index, cell index)


@dataclass
class Scan:
    board: Board
    # Letter glyph of every recognized LETTER/PUNCT cell, used to learn corrections.
    glyphs: dict[CellRef, Glyph] = field(default_factory=dict)


def _punct_by_shape(g: Glyph, baseline: int) -> str:
    h, w = g.mask.shape
    plate_h = h / g.rel_h
    if w > 1.5 * h:
        # A dash is a standalone wide bar, a hyphen inside a word is short.
        return "—" if w / plate_h > 0.25 else "-"
    if g.bbox.y + g.bbox.h < baseline - 0.15 * plate_h:
        return "'"
    return "," if h > 1.4 * w else "."


def _stacked_punct(g: Glyph) -> str | None:
    """Tall punctuation (`;` `:` `!`): a narrow glyph of two marks one above the other.
    It is as high as a letter, but no letter is made of two stacked parts this narrow."""
    h, w = g.mask.shape
    if w > 0.12 * h / g.rel_h:
        return None
    parts = sorted(_components(g.mask), key=lambda c: c[1])
    if len(parts) != 2 or parts[0][1] + parts[0][3] > parts[1][1]:
        return None
    (_, _, tw, th, _), (_, _, bw, bh, _) = parts
    if th > 1.4 * tw:
        return "!"
    return ";" if bh > 1.4 * bw else ":"


class Recognizer:
    def __init__(self, data_dir: Path) -> None:
        glyphs = data_dir / "glyphs"
        # Letters are large: the same glyph scores ~1.0, look-alikes (Б/В, Е/Ь) stay below 0.9 with
        # a small lead; a pixel shift can drop a true match to ~0.9, but then it leads by far.
        self.letters = GlyphBank(
            glyphs / "letters.npz", seed=SEED_DIR / "letters.npz", min_score=0.93, clear_lead=(0.85, 0.15)
        )
        # Digits are small and noisier, but the nearest sample is reliable when it leads clearly.
        self.digits = GlyphBank(glyphs / "digits.npz", seed=SEED_DIR / "digits.npz", min_score=0.7, min_margin=0.03)

    def recognize(self, image_bgr: np.ndarray) -> Scan:
        layout = analyze(image_bgr)
        scan = Scan(Board(words=[], width=layout.width, height=layout.height))
        digits_learned = False
        for plate in layout.plates:
            if plate.hint:
                text = ocr.read_word(plate.hint_gray) if plate.hint_gray is not None else ""
                cells = [
                    Cell(kind=CellKind.LETTER if ch in ALPHABET else CellKind.PUNCT, letter=ch, bbox=plate.bbox)
                    for ch in text
                ]
                scan.board.words.append(Word(cells=cells, bbox=plate.bbox, row=plate.row, hint=True))
                continue
            wi = len(scan.board.words)
            word = Word(cells=[], bbox=plate.bbox, row=plate.row)
            for slot in plate.slots:
                cell, glyph, learned = self._cell(slot, plate.baseline)
                digits_learned |= learned
                if glyph is not None:
                    scan.glyphs[(wi, len(word.cells))] = glyph
                word.cells.append(cell)
            scan.board.words.append(word)
        if digits_learned:
            self.digits.save()
        return scan

    def _cell(self, slot: Slot, baseline: int) -> tuple[Cell, Glyph | None, bool]:
        code, learned = self._code(slot)
        clipped = slot.clipped
        match slot.kind:
            case "box" if code is None:
                # A cell cut by the screen edge with its code unreadable: like a lock, the code is unknown.
                return Cell(kind=CellKind.LOCKED, clipped=clipped, bbox=slot.bbox), None, learned
            case "box":
                return (
                    Cell(
                        kind=CellKind.EMPTY,
                        code=code,
                        selected=slot.selected,
                        clipped=clipped,
                        bbox=slot.bbox,
                    ),
                    None,
                    learned,
                )
            case "lock":
                return Cell(kind=CellKind.LOCKED, clipped=clipped, bbox=slot.bbox), None, learned
        glyph = max(slot.glyphs, key=lambda g: g.mask.size)
        punct = _punct_by_shape(glyph, baseline) if glyph.rel_h < PUNCT_MAX_H else _stacked_punct(glyph)
        if punct:
            return (
                Cell(kind=CellKind.PUNCT, letter=punct, clipped=clipped, bbox=slot.bbox),
                glyph,
                learned,
            )
        letter, verified = self.letters.classify(glyph), not clipped
        if letter is None:
            letter, verified = ocr.read_char(glyph.gray), False
        kind = CellKind.LETTER if letter is None or letter in ALPHABET else CellKind.PUNCT
        return (
            Cell(kind=kind, letter=letter, code=code, verified=verified, clipped=clipped, bbox=slot.bbox),
            glyph,
            learned,
        )

    def _code(self, slot: Slot) -> tuple[int | None, bool]:
        """Read the code number under a slot; returns (code, whether digit bank was extended)."""
        if not slot.digits:
            return None, False
        labels = [self.digits.classify(g) for g in slot.digits]
        if all(labels):
            text = "".join(labels)  # type: ignore[arg-type]
            return (int(text) if 0 < int(text) <= MAX_CODE else None), False
        if slot.number_gray is None or (code := ocr.read_number(slot.number_gray)) is None:
            return None, False
        text = str(code)
        learned = False
        if len(text) == len(slot.digits) and 0 < code <= MAX_CODE:
            for ch, g in zip(text, slot.digits, strict=True):
                self.digits.add(ch, g, save=False)
            learned = True
        return (code if 0 < code <= MAX_CODE else None), learned

    def learn_letter(self, scan: Scan, ref: CellRef, letter: str) -> None:
        """Remember a user-confirmed letter for a recognized glyph and update the board."""
        glyph = scan.glyphs.get(ref)
        if glyph is None:
            return
        self.letters.add(letter, glyph)
        wi, ci = ref
        cell = scan.board.words[wi].cells[ci]
        cell.letter, cell.verified = letter, True
        cell.kind = CellKind.LETTER if letter in ALPHABET else CellKind.PUNCT
