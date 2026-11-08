"""Board file: the hand-editable format between recognition and solving.

A TOML document whose `board` key holds the puzzle as text, one line per
screen row, words separated by `|`, cells by spaces:

    _16     empty cell with code 16          *_16   the cell selected in the game
    Т20     open letter with its code        Д      open letter, code hidden
    Т?20    letter recognized by OCR, not confirmed
    _?      open symbol that was not recognized at all (type the right one in the TUI)
    #       locked cell                      , . — ! ?   punctuation
    [ОБЩЕСТВЕННЫМ]   a whole word given as a hint

Optional tables keep the solver state:

    [guesses]       letter for a code:  16 = "Ж"
    [locks]         letter guessed for a locked cell, keyed "word.cell"
                    (both counted from 1 in reading order, punctuation included): "3.2" = "А"
"""

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from crgram.model import ALPHABET, Board, Cell, CellKind, Word

LOCK = "#"
SELECTED = "*"
UNVERIFIED = "?"
WORD_SEP = "|"

_CELL = re.compile(rf"^(\{SELECTED})?([{ALPHABET}_])(\?)?(\d+)?$")


class BoardFormatError(ValueError):
    pass


@dataclass
class BoardFile:
    board: Board
    guesses: dict[int, str] = field(default_factory=dict)
    source: str | None = None
    lock_guesses: dict[tuple[int, int], str] = field(default_factory=dict)  # (word, cell), 0-based
    # Errors skipped by a lenient load: the rest of the file is still usable
    problems: list[str] = field(default_factory=list)


# --- writing -------------------------------------------------------------------


def format_cell(c: Cell) -> str:
    match c.kind:
        case CellKind.LOCKED:
            return LOCK
        case CellKind.PUNCT:
            return c.letter or "?"
    letter = c.letter if c.kind is CellKind.LETTER and c.letter else "_"
    return (
        (SELECTED if c.selected else "")
        + letter
        + ("" if c.verified else UNVERIFIED)
        + ("" if c.code is None else str(c.code))
    )


def format_word(w: Word) -> str:
    if w.hint:
        return f"[{w.text}]"
    return " ".join(format_cell(c) for c in w.cells)


def format_board(board: Board) -> str:
    return "\n".join(f" {WORD_SEP} ".join(format_word(board.words[wi]) for wi in row) for row in board.rows())


def _toml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def dumps(bf: BoardFile) -> str:
    out = ["# crgram board, see `crgram.boardfile` for the format"]
    if bf.source:
        out.append(f"source = {_toml_str(bf.source)}")
    out += ["", 'board = """', format_board(bf.board), '"""']
    if bf.guesses:
        out += ["", "[guesses]"]
        out += [f"{code} = {_toml_str(letter)}" for code, letter in sorted(bf.guesses.items())]
    if bf.lock_guesses:
        out += ["", "[locks]"]
        out += [f'"{wi + 1}.{ci + 1}" = {_toml_str(letter)}' for (wi, ci), letter in sorted(bf.lock_guesses.items())]
    return "\n".join(out) + "\n"


def save(path: Path, bf: BoardFile) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(dumps(bf), encoding="utf-8")
    tmp.replace(path)


# --- reading -------------------------------------------------------------------


def parse_cell(token: str) -> Cell:
    if token in {LOCK, "🔒"}:
        return Cell(kind=CellKind.LOCKED)
    m = _CELL.match(token.upper())
    if m is None:
        if not any(ch.isalnum() for ch in token):
            return Cell(kind=CellKind.PUNCT, letter=token)
        raise BoardFormatError(f"непонятная клетка {token!r}")
    selected, letter, unverified, code = m.groups()
    if letter == "_" and code is None and unverified:
        return Cell(kind=CellKind.LETTER, verified=False)
    if letter == "_" and code is None:
        raise BoardFormatError(f"у пустой клетки {token!r} нет кода")
    return Cell(
        kind=CellKind.EMPTY if letter == "_" else CellKind.LETTER,
        letter=None if letter == "_" else letter,
        code=int(code) if code else None,
        selected=bool(selected),
        verified=not unverified,
    )


def parse_word(text: str, row: int, problems: list[str] | None = None) -> Word:
    """With `problems`, a bad cell is kept as its raw text (shown as punctuation) and reported there."""
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        cells = [
            Cell(kind=CellKind.LETTER if ch in ALPHABET else CellKind.PUNCT, letter=ch)
            for ch in text[1:-1].upper()
            if not ch.isspace()
        ]
        return Word(cells=cells, row=row, hint=True)
    cells = []
    for token in text.split():
        try:
            cells.append(parse_cell(token))
        except BoardFormatError as e:
            if problems is None:
                raise
            problems.append(str(e))
            cells.append(Cell(kind=CellKind.PUNCT, letter=token))
    return Word(cells=cells, row=row)


def parse_board(text: str, problems: list[str] | None = None) -> Board:
    words: list[Word] = []
    row = 0
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        found: list[str] | None = None if problems is None else []
        try:
            words += [parse_word(part, row, found) for part in line.split(WORD_SEP) if part.strip()]
        except BoardFormatError as e:
            raise BoardFormatError(f"строка {lineno}: {e}") from None
        if problems is not None and found:
            problems += [f"строка {lineno}: {p}" for p in found]
        row += 1
    return Board(words=words)


def loads(text: str, *, lenient: bool = False) -> BoardFile:
    """Strict by default; `lenient` skips bad cells, guesses and locks, listing them in `problems`."""
    problems: list[str] = []

    def bad(message: str) -> None:
        if not lenient:
            raise BoardFormatError(message)
        problems.append(message)

    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise BoardFormatError(f"TOML: {e}") from e
    if not isinstance(data.get("board"), str):
        raise BoardFormatError("нет строки `board`")
    guesses = {}
    for code, letter in data.get("guesses", {}).items():
        if not code.isdigit() or not isinstance(letter, str) or letter.upper() not in ALPHABET:
            bad(f"догадка {code} = {letter!r}")
            continue
        guesses[int(code)] = letter.upper()
    board = parse_board(data["board"], problems if lenient else None)
    locks = {}
    for key, letter in data.get("locks", {}).items():
        wi, _, ci = key.partition(".")
        ok = wi.isdigit() and ci.isdigit() and isinstance(letter, str) and letter.upper() in ALPHABET
        if ok and 0 < int(wi) <= len(board.words) and 0 < int(ci) <= len(cells := board.words[int(wi) - 1].cells):
            ok = cells[int(ci) - 1].kind is CellKind.LOCKED
        else:
            ok = False
        if not ok:
            bad(f"замок {key} = {letter!r}: нет такой клетки-замка или не буква")
            continue
        locks[(int(wi) - 1, int(ci) - 1)] = letter.upper()
    return BoardFile(board, guesses, data.get("source"), locks, problems)


def load(path: Path, *, lenient: bool = False) -> BoardFile:
    return loads(path.read_text(encoding="utf-8"), lenient=lenient)
