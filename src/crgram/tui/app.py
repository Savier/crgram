"""Textual application: board + side panel with the code table and free letters."""

import time
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.timer import Timer
from textual.widgets import Footer, Header, Static

from crgram import boardfile
from crgram.boardfile import BoardFile, BoardFormatError
from crgram.feeds import Feed, FileFeed
from crgram.model import ALPHABET, Board, Cell, CellKind
from crgram.solver import Solver
from crgram.sources import SourceError
from crgram.tui import style as st
from crgram.tui.board import BoardView
from crgram.vision.recognize import CellRef

FREQUENCY_TOP = 10  # rows in the frequency section
FREQUENCY_BAR = 20  # width of the longest bar

SECTIONS = [
    ("summary", "Поле"),
    ("placeable", "Можно поставить"),
    ("codes", "Коды"),
    ("frequency", "Частота"),
    ("letters", "Буквы"),
    ("cell", "Клетка"),
    ("legend", "Обозначения"),
]


class CrgramApp(App[None]):
    TITLE = "crgram"
    CSS_PATH = "app.tcss"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+r,f5", "rescan", "Обновить"),
        Binding("f6", "toggle_watch", "Авто-обновление"),
        Binding("ctrl+z", "undo", "Отменить"),
        Binding("ctrl+l", "clear", "Сбросить догадки"),
        Binding("ctrl+s", "save", "Сохранить"),
        Binding("ctrl+q", "quit", "Выход"),
    ]

    def __init__(self, feed: Feed, save_path: Path, watch: float = 0) -> None:
        super().__init__()
        self.feed = feed
        self.save_path = save_path
        self.solver = Solver(Board(words=[]))
        self.watch_interval = watch or 2.0
        self._watch_timer: Timer | None = None
        self._loading = False
        self._watch_on = bool(watch)

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with ScrollableContainer(id="board-box"):
                yield BoardView(self.solver, id="board")
            with Vertical(id="side"):
                for wid, title in SECTIONS:
                    section = Static(id=wid, classes="section")
                    section.border_title = title
                    yield section
        yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = self.feed.name
        self._update("legend", self._legend())
        self.query_one(BoardView).focus()
        self.action_rescan()
        if self._watch_on:
            self._start_watch()

    # --- loading ------------------------------------------------------------

    def action_rescan(self) -> None:
        if not self._loading:
            self._loading = True
            self._set_status("загрузка…")
            self._load_worker()

    @work(thread=True, exclusive=True, group="load")
    def _load_worker(self) -> None:
        t0 = time.perf_counter()
        try:
            bf = self.feed.load()
        except (SourceError, BoardFormatError) as e:
            self.call_from_thread(self._load_failed, str(e))
            return
        self.call_from_thread(self._apply, bf, time.perf_counter() - t0)

    def _load_failed(self, message: str) -> None:
        self._loading = False
        self._set_status(f"[red]ошибка:[/] {message}")

    def _apply(self, bf: BoardFile, elapsed: float) -> None:
        self._loading = False
        self.solver.update_board(bf.board)
        if bf.guesses or bf.lock_guesses:
            self.solver.replace_guesses(bf.guesses, bf.lock_guesses)
        self.query_one(BoardView).board_changed()
        unverified = sum(1 for _, _, c in bf.board.cells() if not c.verified)
        msg = f"слов: {len(bf.board.words)}, {elapsed:.2f} с"
        if unverified:
            msg += f" · [yellow]непроверенных букв: {unverified}[/] (Enter — верно, или введите правильную)"
        if clipped := [wi + 1 for wi, w in enumerate(bf.board.words) if any(c.clipped for c in w.cells)]:
            words = ", ".join(map(str, clipped))
            msg += f" · [yellow]обрезаны краем экрана слова {words}[/] (проверьте крайние клетки)"
        if not bf.board.words:
            msg = "[red]поле не найдено[/]"
        self._set_status(msg)
        self._refresh_side()

    def action_toggle_watch(self) -> None:
        if self._watch_timer:
            self._watch_timer.stop()
            self._watch_timer = None
            self.notify("Авто-обновление выключено")
        else:
            self._start_watch()
            self.notify(f"Авто-обновление каждые {self.watch_interval:g} с")

    def _start_watch(self) -> None:
        self._watch_timer = self.set_interval(self.watch_interval, self._watch_tick)

    def _watch_tick(self) -> None:
        if self.feed.changed():
            self.action_rescan()

    def action_save(self) -> None:
        s = self.solver
        bf = BoardFile(s.board, dict(s.guesses), source=self.feed.name, lock_guesses=dict(s.cell_guesses))
        try:
            boardfile.save(self.save_path, bf)
        except OSError as e:
            self.notify(f"Не удалось сохранить: {e}", severity="error")
            return
        if isinstance(self.feed, FileFeed) and self.feed.path == self.save_path:
            self.feed.mark_current()  # our own write must not trigger a reload
        self.notify(f"Сохранено в {self.save_path}")

    # --- editing ------------------------------------------------------------

    def _cell(self, ref: CellRef) -> Cell:
        wi, ci = ref
        return self.solver.board.words[wi].cells[ci]

    def _set_letter(self, ref: CellRef, letter: str) -> None:
        """Fix a recognized letter on the board and teach the recognizer when possible."""
        cell = self._cell(ref)
        learned = self.feed.learn(ref, letter)
        cell.letter, cell.verified = letter, True
        self.notify(f"Буква «{letter}»" + (" запомнена" if learned else " исправлена"))

    def on_board_view_typed(self, msg: BoardView.Typed) -> None:
        board = self.query_one(BoardView)
        cell = self._cell(msg.ref)
        match cell.kind:
            case CellKind.LETTER:
                self._set_letter(msg.ref, msg.letter)
            case CellKind.EMPTY if cell.code in self.solver.known():
                letter = self.solver.known()[cell.code]  # type: ignore[index]
                self.notify(f"Код {cell.code} уже известен: «{letter}» — поставьте её в игре")
                board.advance()
            case CellKind.EMPTY if cell.code is not None:
                code = cell.code
                other = [c for c, ch in self.solver.mapping().items() if ch == msg.letter and c != code]
                self.solver.assign(code, msg.letter)
                if other or msg.letter in self.solver.completed_letters():
                    where = f"коду {other[0]}" if other else "уже открытой букве"
                    self.notify(f"«{msg.letter}» уже соответствует {where}", severity="warning")
                else:
                    board.advance()
            case CellKind.LOCKED:
                # Code unknown: the letter goes to this cell only.
                self.solver.assign_cell(msg.ref, msg.letter)
                board.advance()
        self._changed()

    def on_board_view_erased(self, msg: BoardView.Erased) -> None:
        cell = self._cell(msg.ref)
        if cell.kind is CellKind.LOCKED:
            self.solver.assign_cell(msg.ref, None)
        elif cell.code is not None:
            self.solver.assign(cell.code, None)
        self._changed()

    def on_board_view_code_entered(self, msg: BoardView.CodeEntered) -> None:
        was_locked = self._cell(msg.ref).kind is CellKind.LOCKED
        self.solver.set_code(msg.ref, msg.code)
        if msg.code is None:
            self.notify("Клетка помечена замком")
        elif was_locked:
            self.notify(f"Замок снят, код {msg.code}")
        else:
            self.notify(f"Код клетки исправлен на {msg.code}")
        self.query_one(BoardView).board_changed()
        self._changed()

    def on_board_view_confirmed(self, msg: BoardView.Confirmed) -> None:
        cell = self._cell(msg.ref)
        if cell.kind is CellKind.LETTER and not cell.verified and cell.letter:
            self._set_letter(msg.ref, cell.letter)
            self._changed()

    def on_board_view_cursor_moved(self) -> None:
        self._refresh_side()

    def action_undo(self) -> None:
        if self.solver.undo():
            self._changed()

    def action_clear(self) -> None:
        self.solver.clear()
        self._changed()

    def _changed(self) -> None:
        self.query_one(BoardView).refresh()
        self._refresh_side()

    # --- side panel ---------------------------------------------------------

    def _set_status(self, markup: str) -> None:
        self.query_one("#status", Static).update(markup)

    def _refresh_side(self) -> None:
        s = self.solver
        cells = [c for _, _, c in s.board.cells()]
        known, mapping, conflicts = s.known(), s.mapping(), s.conflicts()
        placeable = s.placeable()
        board = self.query_one(BoardView)
        cur = s.board.words[board.cursor[0]].cells[board.cursor[1]] if board.cursor else None

        summary = Text()
        codes = s.codes()
        summary.append(f"кодов {len(codes)}: ", st.CODE)
        summary.append(f"известно {len(known)}", st.OPEN)
        summary.append(" · ", st.CODE)
        summary.append(f"догадок {len(s.guesses)}", st.GUESS)
        summary.append(" · ", st.CODE)
        summary.append(f"нет {len(s.unknown_codes())}", st.EMPTY)
        empty = sum(1 for c in cells if c.kind is CellKind.EMPTY)
        locked = sum(1 for c in cells if c.kind is CellKind.LOCKED)
        summary.append(f"\nпустых клеток {empty} · замков {locked}", st.CODE)
        self._update("summary", summary)

        place = Text()
        for i, (code, (letter, n)) in enumerate(placeable.items()):
            place.append(f" {letter} ", st.PLACEABLE)
            place.append(f" код {code}, клеток {n}", st.CODE)
            place.append("\n" if i < len(placeable) - 1 else "")
        self._update("placeable", place if placeable else Text("нет", st.EMPTY))

        def code_style(code: int) -> str:
            if code in conflicts:
                style = "bold red"
            elif code in placeable:
                style = st.PLACEABLE
            elif code in known:
                style = st.OPEN
            elif code in mapping:
                style = st.GUESS
            else:
                style = st.EMPTY
            if cur is not None and cur.code == code:
                style += " reverse"
            return style

        table = Text()
        for i, code in enumerate(codes):
            letter = mapping.get(code)
            style = code_style(code)
            table.append(f"{code:>2}", st.CODE)
            table.append(" ")
            table.append(letter or "·", style)
            table.append("\n" if i % 5 == 4 else "  ")
        table.rstrip()
        self._update("codes", table if codes else Text("—", st.EMPTY))

        freq = s.frequency()
        top = freq[0][1] if freq else 0
        frequency = Text()
        for code, n in freq[:FREQUENCY_TOP]:
            frequency.append(f"{code:>2} ", st.CODE)
            frequency.append(mapping.get(code, "·"), code_style(code))
            frequency.append(f" {n:>2} ", "bold")
            frequency.append("█" * max(1, round(n * FREQUENCY_BAR / top)), st.CODE)
            frequency.append("\n")
        frequency.rstrip()
        self._update("frequency", frequency if freq else Text("—", st.EMPTY))

        open_letters = set(known.values()) | s.completed_letters()
        guessed = set(s.guesses.values())
        letters = Text()
        for ch in ALPHABET:
            if ch in open_letters:
                letters.append(ch, "grey35 strike")
            elif ch in guessed:
                letters.append(ch, st.GUESS)
            else:
                letters.append(ch, "bold")
            letters.append(" ")
        self._update("letters", letters)

        info = Text()
        if cur is not None and board.cursor is not None:
            if cur.code is not None:
                letter, src = s.letter_for(cur.code)
                n = s.empty_count(cur.code)
                info.append(f"код {cur.code}: ", st.CODE)
                if src == "board":
                    info.append(letter or "", st.PLACEABLE if n else st.OPEN)
                    info.append(" — известна", st.CODE)
                elif src == "guess":
                    info.append(letter or "", st.GUESS)
                    info.append(" — догадка", st.CODE)
                else:
                    info.append("неизвестна", st.EMPTY)
                info.append(f", пустых клеток {n}\n", st.CODE)
            elif cur.kind is CellKind.LOCKED:
                info.append("замок", st.LOCKED)
                if letter := s.cell_guesses.get(board.cursor):
                    info.append(", буква ", st.CODE)
                    info.append(letter, st.LOCK_GUESS)
                info.append("\nбуква — только в эту клетку,\nцифры+Enter — снять замок с кодом\n", st.CODE)
            else:
                info.append("код скрыт: все такие буквы открыты\n", st.CODE)
            if board.code_buffer:
                info.append("новый код: ", st.CODE)
                info.append(f" {board.code_buffer}▏", st.CODE_INPUT)
                info.append("  Enter / Esc\n", st.CODE)
            pattern = s.pattern(board.cursor[0])
            info.append("слово: ", st.CODE)
            info.append(pattern, "bold")
            info.append(f" ({len(pattern)})", st.CODE)
        self._update("cell", info)

    def _update(self, section: str, content: Text) -> None:
        self.query_one(f"#{section}", Static).update(content)

    def _legend(self) -> Text:
        legend = Text()
        for sample, style, label in [
            (" А ", st.OPEN, "открыта в игре"),
            (" А ", st.PLACEABLE, "можно поставить"),
            (" А ", st.GUESS, "ваша догадка"),
            (" А ", st.UNVERIFIED, "не уверен (Enter)"),
            (" · ", st.EMPTY, "неизвестна"),
            (" ▒ ", st.LOCKED, "замок (# — поставить)"),
            (" А ", st.LOCK_GUESS, "буква в замке"),
            (" А ", "bold " + st.CONFLICT, "конфликт"),
        ]:
            legend.append(sample, style)
            legend.append(f" {label}\n", st.CODE)
        legend.rstrip()
        return legend
