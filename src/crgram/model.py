"""Board model: what was recognized on the game screen."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

ALPHABET = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"


class CellKind(StrEnum):
    EMPTY = "empty"  # letter not yet guessed, code is shown
    LETTER = "letter"  # letter is open (code may be hidden when all its cells are open)
    LOCKED = "locked"  # blocked until neighbours are guessed; letter and code unknown
    PUNCT = "punct"  # punctuation mark attached to the word


class BBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2


class Cell(BaseModel):
    kind: CellKind
    letter: str | None = None
    code: int | None = None
    selected: bool = False
    # False when the letter came from a fallback OCR guess and was not confirmed
    verified: bool = True
    # Position on the screenshot; absent for boards loaded from a file
    bbox: BBox | None = None


class Word(BaseModel):
    cells: list[Cell]
    bbox: BBox | None = None
    row: int = 0
    # Whole word given as a hint by the game (no codes)
    hint: bool = False

    @property
    def text(self) -> str:
        return "".join(c.letter or "_" for c in self.cells)


class Board(BaseModel):
    """Words in reading order; `Word.row` groups them into lines as on the screen."""

    words: list[Word]
    width: int = 0
    height: int = 0

    def cells(self) -> list[tuple[int, int, Cell]]:
        return [(wi, ci, c) for wi, w in enumerate(self.words) for ci, c in enumerate(w.cells)]

    def rows(self) -> list[list[int]]:
        """Word indices grouped by row."""
        out: dict[int, list[int]] = {}
        for wi, w in enumerate(self.words):
            out.setdefault(w.row, []).append(wi)
        return [ws for _, ws in sorted(out.items())]
