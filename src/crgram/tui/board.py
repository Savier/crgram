"""Board widget: renders the puzzle as letter/code pairs and handles cell navigation."""

from dataclasses import dataclass
from typing import ClassVar

from rich.text import Text
from textual.binding import Binding, BindingType
from textual.events import Key
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget

from crgram.model import ALPHABET, CellKind
from crgram.solver import Solver
from crgram.tui import style as st

type CellRef = tuple[int, int]

CELL_W = 3
WORD_GAP = 3
# Typing on a Latin layout: map keys to the ЙЦУКЕН positions.
_LATIN = "qwertyuiop[]asdfghjkl;'zxcvbnm,.`"
_CYR = "йцукенгшщзхъфывапролджэячсмитьбюё"
LATIN_TO_CYR = str.maketrans(_LATIN + _LATIN.upper(), _CYR + _CYR.upper())


@dataclass
class _Pos:
    line: int  # board row index
    col: int  # character column of the cell start


class BoardView(Widget, can_focus=True):
    """Shows the board; arrows move the cursor, letters assign the cursor cell's code,
    digits (then Enter) set the cell's code, `#` turns the cell into a lock."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("left", "move(-1)", "←", show=False),
        Binding("right", "move(1)", "→", show=False),
        Binding("up", "move_line(-1)", "↑", show=False),
        Binding("down", "move_line(1)", "↓", show=False),
        Binding("tab", "next_empty", "След. пустая"),
        Binding("backspace,delete", "erase", "Стереть"),
        Binding("enter", "confirm", "Подтвердить"),
        Binding("escape", "cancel_code", "Отмена ввода кода", show=False),
    ]

    cursor: reactive[CellRef | None] = reactive(None)

    class Typed(Message):
        def __init__(self, ref: CellRef, letter: str) -> None:
            self.ref, self.letter = ref, letter
            super().__init__()

    class Erased(Message):
        def __init__(self, ref: CellRef) -> None:
            self.ref = ref
            super().__init__()

    class CursorMoved(Message):
        pass

    class CodeEntered(Message):
        """Code typed for a cell; None turns the cell into a lock."""

        def __init__(self, ref: CellRef, code: int | None) -> None:
            self.ref, self.code = ref, code
            super().__init__()

    class Confirmed(Message):
        def __init__(self, ref: CellRef) -> None:
            self.ref = ref
            super().__init__()

    def __init__(self, solver: Solver, **kwargs) -> None:
        super().__init__(**kwargs)
        self.solver = solver
        self._pos: dict[CellRef, _Pos] = {}
        self._order: list[CellRef] = []
        self.code_buffer = ""  # digits typed for the cursor cell, applied by Enter

    # --- layout -------------------------------------------------------------

    def _rows(self) -> list[list[int]]:
        return self.solver.board.rows()

    def _relayout(self) -> None:
        self._pos.clear()
        self._order.clear()
        for line, row in enumerate(self._rows()):
            col = 0
            for wi in row:
                word = self.solver.board.words[wi]
                for ci, cell in enumerate(word.cells):
                    width = 1 if cell.kind is CellKind.PUNCT else CELL_W
                    if not word.hint and cell.kind is not CellKind.PUNCT:
                        self._pos[(wi, ci)] = _Pos(line, col)
                        self._order.append((wi, ci))
                    col += width
                col += WORD_GAP

    def board_changed(self) -> None:
        self._relayout()
        if self.cursor not in self._pos:
            selected = [r for r in self._order if self._cell(r).selected]
            self.cursor = (
                selected[0] if selected else (self._first_empty() or (self._order[0] if self._order else None))
            )
        self.refresh(layout=True)

    def _cell(self, ref: CellRef):
        wi, ci = ref
        return self.solver.board.words[wi].cells[ci]

    def _is_open(self, ref: CellRef) -> bool:
        cell = self._cell(ref)
        return cell.kind is CellKind.EMPTY and cell.code is not None and self.solver.letter_for(cell.code)[0] is None

    def _first_empty(self, after: CellRef | None = None) -> CellRef | None:
        start = self._order.index(after) + 1 if after in self._pos else 0
        n = len(self._order)
        for i in range(n):
            ref = self._order[(start + i) % n]
            if self._is_open(ref):
                return ref
        return None

    # --- rendering ----------------------------------------------------------

    def get_content_width(self, container, viewport) -> int:
        self._relayout()
        widths = [0]
        for row in self._rows():
            w = 0
            for wi in row:
                word = self.solver.board.words[wi]
                w += sum(1 if c.kind is CellKind.PUNCT else CELL_W for c in word.cells) + WORD_GAP
            widths.append(w)
        return max(widths)

    def get_content_height(self, container, viewport, width: int) -> int:
        return max(1, len(self._rows()) * 3)

    def render(self) -> Text:
        self._relayout()
        s = self.solver
        mapping = s.mapping()
        known = s.known()
        conflicts = s.conflicts()
        cur = self.cursor
        cur_code = self._cell(cur).code if cur in self._pos else None
        out = Text()
        for row in self._rows():
            top, bottom = Text(), Text()
            for wi in row:
                word = s.board.words[wi]
                for ci, c in enumerate(word.cells):
                    if c.kind is CellKind.PUNCT:
                        top.append(c.letter or "?", "bold")
                        bottom.append(" ")
                        continue
                    if word.hint:
                        top.append(f"{c.letter:^{CELL_W}}", st.HINT)
                        bottom.append(" " * CELL_W)
                        continue
                    letter, style = "·", st.EMPTY
                    match c.kind:
                        case CellKind.LOCKED if (wi, ci) in s.cell_guesses:
                            letter, style = s.cell_guesses[(wi, ci)], st.LOCK_GUESS
                        case CellKind.LOCKED:
                            letter, style = "▒", st.LOCKED
                        case CellKind.LETTER:
                            letter = c.letter or "?"
                            style = st.OPEN if c.verified else st.UNVERIFIED
                        case CellKind.EMPTY if c.code in mapping:
                            letter = mapping[c.code]
                            style = st.PLACEABLE if c.code in known else st.GUESS
                    if c.code is not None and c.code in conflicts:
                        style += " " + st.CONFLICT
                    elif c.code is not None and c.code == cur_code and style != st.PLACEABLE:
                        style += " " + st.SAME_CODE
                    code = "" if c.code is None else str(c.code)
                    code_style = st.SELECTED_CODE if c.selected else st.CODE
                    if (wi, ci) == cur:
                        # Brackets keep the letter's own colour visible under the cursor.
                        top.append("[", st.CURSOR)
                        top.append(letter, style)
                        top.append("]", st.CURSOR)
                        code_style = st.CURSOR
                        if self.code_buffer:
                            code, code_style = self.code_buffer + "▏", st.CODE_INPUT
                    else:
                        top.append(f"{letter:^{CELL_W}}", style)
                    bottom.append(f"{code:^{CELL_W}}", code_style)
                top.append(" " * WORD_GAP)
                bottom.append(" " * WORD_GAP)
            out.append_text(top)
            out.append("\n")
            out.append_text(bottom)
            out.append("\n\n")
        out.rstrip()
        return out

    # --- navigation & input -------------------------------------------------

    def watch_cursor(self) -> None:
        self.code_buffer = ""
        self.refresh()
        self.post_message(self.CursorMoved())

    def action_move(self, delta: int) -> None:
        if not self._order:
            return
        i = self._order.index(self.cursor) if self.cursor in self._pos else 0
        self.cursor = self._order[(i + delta) % len(self._order)]

    def action_move_line(self, delta: int) -> None:
        if self.cursor not in self._pos:
            return
        here = self._pos[self.cursor]
        lines = sorted({p.line for p in self._pos.values()})
        idx = lines.index(here.line) + delta
        if not 0 <= idx < len(lines):
            return
        target = lines[idx]
        self.cursor = min(
            (r for r, p in self._pos.items() if p.line == target),
            key=lambda r: abs(self._pos[r].col - here.col),
        )

    def action_next_empty(self) -> None:
        if ref := self._first_empty(self.cursor):
            self.cursor = ref

    def action_erase(self) -> None:
        if self.code_buffer:
            self._set_buffer(self.code_buffer[:-1])
        elif self.cursor in self._pos:
            self.post_message(self.Erased(self.cursor))

    def action_confirm(self) -> None:
        if self.cursor not in self._pos:
            return
        if self.code_buffer:
            code = int(self.code_buffer)
            self._set_buffer("")
            if code > 0:
                self.post_message(self.CodeEntered(self.cursor, code))
        else:
            self.post_message(self.Confirmed(self.cursor))

    def action_cancel_code(self) -> None:
        self._set_buffer("")

    def _set_buffer(self, value: str) -> None:
        self.code_buffer = value
        self.refresh()
        self.post_message(self.CursorMoved())

    def on_key(self, event: Key) -> None:
        if self.cursor not in self._pos:
            return
        raw = event.character or ""
        if raw.isdigit() and len(raw) == 1:
            event.stop()
            if len(self.code_buffer) < 2:
                self._set_buffer(self.code_buffer + raw)
            return
        if raw == "#":
            event.stop()
            self._set_buffer("")
            self.post_message(self.CodeEntered(self.cursor, None))
            return
        ch = raw.translate(LATIN_TO_CYR).upper()
        if len(ch) == 1 and ch in ALPHABET:
            event.stop()
            self._set_buffer("")
            self.post_message(self.Typed(self.cursor, ch))

    def advance(self) -> None:
        """Move to the next cell that still needs a letter."""
        if ref := self._first_empty(self.cursor):
            self.cursor = ref
