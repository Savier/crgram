"""Interactive solving state: code -> letter assignments on top of the recognized board."""

from dataclasses import dataclass, field

from crgram.model import ALPHABET, Board, CellKind

type CellRef = tuple[int, int]  # (word index, cell index)


@dataclass
class _Snapshot:
    board: Board
    guesses: dict[int, str]
    cell_guesses: dict[CellRef, str]


@dataclass
class Solver:
    board: Board
    guesses: dict[int, str] = field(default_factory=dict)
    # Letters guessed for locked cells, whose code is unknown: they apply to that cell only.
    cell_guesses: dict[CellRef, str] = field(default_factory=dict)
    _history: list[_Snapshot] = field(default_factory=list)

    def _remember(self) -> None:
        self._history.append(_Snapshot(self.board.model_copy(deep=True), dict(self.guesses), dict(self.cell_guesses)))

    def known(self) -> dict[int, str]:
        """Code -> letter as revealed on the board."""
        out: dict[int, str] = {}
        for _, _, c in self.board.cells():
            if c.kind is CellKind.LETTER and c.code is not None and c.letter:
                out[c.code] = c.letter
        return out

    def codes(self) -> list[int]:
        return sorted({c.code for _, _, c in self.board.cells() if c.code is not None})

    def frequency(self) -> list[tuple[int, int]]:
        """(code, number of cells with it) for every code on the board, most frequent first."""
        counts: dict[int, int] = {}
        for _, _, c in self.board.cells():
            if c.code is not None:
                counts[c.code] = counts.get(c.code, 0) + 1
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))

    def empty_count(self, code: int) -> int:
        """How many cells with this code are still empty in the game."""
        return sum(1 for _, _, c in self.board.cells() if c.kind is CellKind.EMPTY and c.code == code)

    def placeable(self) -> dict[int, tuple[str, int]]:
        """Codes already revealed by an open letter that still have empty cells:
        code -> (letter, number of empty cells). These can be filled in the game right away."""
        out = {}
        for code, letter in sorted(self.known().items()):
            if n := self.empty_count(code):
                out[code] = (letter, n)
        return out

    def unknown_codes(self) -> list[int]:
        m = self.mapping()
        return [code for code in self.codes() if code not in m]

    def letter_for(self, code: int) -> tuple[str | None, str | None]:
        """(letter, source) where source is "board", "guess" or None."""
        if (letter := self.known().get(code)) is not None:
            return letter, "board"
        if (letter := self.guesses.get(code)) is not None:
            return letter, "guess"
        return None, None

    def mapping(self) -> dict[int, str]:
        return self.guesses | self.known()

    def completed_letters(self) -> set[str]:
        """Letters open on the board without a code: all their cells are already revealed."""
        return {
            c.letter
            for w in self.board.words
            if not w.hint
            for c in w.cells
            if c.kind is CellKind.LETTER and c.code is None and c.letter
        }

    def used_letters(self) -> set[str]:
        return set(self.mapping().values()) | self.completed_letters()

    def free_letters(self) -> list[str]:
        used = self.used_letters()
        return [ch for ch in ALPHABET if ch not in used]

    def conflicts(self) -> set[int]:
        """Codes whose letter clashes with another code or with a completed letter."""
        by_letter: dict[str, list[int]] = {}
        for code, letter in self.mapping().items():
            by_letter.setdefault(letter, []).append(code)
        done = self.completed_letters()
        bad: set[int] = set()
        for letter, codes in by_letter.items():
            if len(codes) > 1 or letter in done:
                bad.update(codes)
        return bad

    def assign(self, code: int, letter: str | None) -> None:
        if self.guesses.get(code) == letter or code in self.known():
            return
        self._remember()
        if letter is None:
            self.guesses.pop(code, None)
        else:
            self.guesses[code] = letter

    def assign_cell(self, ref: CellRef, letter: str | None) -> None:
        """Guess a letter for a locked cell (its code is unknown)."""
        if self.cell_guesses.get(ref) == letter:
            return
        self._remember()
        if letter is None:
            self.cell_guesses.pop(ref, None)
        else:
            self.cell_guesses[ref] = letter

    def set_code(self, ref: CellRef, code: int | None) -> None:
        """Correct the code of a cell. A locked cell with a code becomes an ordinary empty one,
        `None` turns a cell back into a lock. A letter guessed for the lock moves to the code."""
        wi, ci = ref
        cell = self.board.words[wi].cells[ci]
        if cell.kind is CellKind.PUNCT or (cell.code == code and cell.kind is not CellKind.LOCKED):
            return
        self._remember()
        if code is None:
            cell.kind, cell.code, cell.letter, cell.verified = CellKind.LOCKED, None, None, True
            return
        cell.code = code
        if cell.kind is CellKind.LOCKED:
            cell.kind = CellKind.EMPTY
            letter = self.cell_guesses.pop(ref, None)
            if letter and code not in self.mapping():
                self.guesses[code] = letter

    def replace_guesses(self, guesses: dict[int, str], cell_guesses: dict[CellRef, str] | None = None) -> None:
        known = self.known()
        new = {code: ch for code, ch in guesses.items() if code not in known}
        new_cells = self._locked_only(cell_guesses or {})
        if new != self.guesses or new_cells != self.cell_guesses:
            self._remember()
            self.guesses, self.cell_guesses = new, new_cells

    def clear(self) -> None:
        if self.guesses or self.cell_guesses:
            self._remember()
            self.guesses, self.cell_guesses = {}, {}

    def undo(self) -> bool:
        if not self._history:
            return False
        snap = self._history.pop()
        self.board, self.guesses, self.cell_guesses = snap.board, snap.guesses, snap.cell_guesses
        return True

    def _locked_only(self, cell_guesses: dict[CellRef, str]) -> dict[CellRef, str]:
        out = {}
        for (wi, ci), letter in cell_guesses.items():
            words = self.board.words
            if wi < len(words) and ci < len(words[wi].cells) and words[wi].cells[ci].kind is CellKind.LOCKED:
                out[(wi, ci)] = letter
        return out

    def update_board(self, board: Board) -> None:
        """Replace the board after a rescan, dropping guesses the game has confirmed.
        A guess for a lock the game has opened moves to the revealed code."""
        old = self.cell_guesses
        self.board = board
        known = self.known()
        self.guesses = {code: ch for code, ch in self.guesses.items() if code not in known}
        self.cell_guesses = self._locked_only(old)
        for (wi, ci), letter in old.items():
            if (wi, ci) in self.cell_guesses or wi >= len(board.words) or ci >= len(board.words[wi].cells):
                continue
            cell = board.words[wi].cells[ci]
            if cell.kind is CellKind.EMPTY and cell.code is not None and cell.code not in self.mapping():
                self.guesses[cell.code] = letter

    def pattern(self, word_index: int) -> str:
        """Word with known/guessed letters and `.` for unknown ones (for dictionary lookup)."""
        word = self.board.words[word_index]
        m = self.mapping()
        out = []
        for ci, c in enumerate(word.cells):
            if c.kind is CellKind.PUNCT:
                continue
            if c.kind is CellKind.LOCKED:
                letter = self.cell_guesses.get((word_index, ci))
            else:
                letter = c.letter if c.kind is CellKind.LETTER else m.get(c.code) if c.code else None
            out.append(letter or ".")
        return "".join(out)
