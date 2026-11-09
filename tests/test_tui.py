import asyncio

import pytest

from crgram import boardfile
from crgram.feeds import FileFeed, LiveFeed
from crgram.sources import ImageSource
from crgram.tui import style as st
from crgram.tui.app import CrgramApp
from crgram.tui.board import BoardView
from crgram.vision import ocr
from crgram.vision.recognize import Recognizer


@pytest.mark.skipif(not ocr.available(), reason="tesseract is needed for the hint word")
def test_type_letters(screen_01, tmp_path):
    async def scenario() -> None:
        app = CrgramApp(LiveFeed(ImageSource(screen_01), Recognizer(tmp_path)), tmp_path / "board.toml")
        async with app.run_test(size=(140, 40)) as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            board = app.query_one(BoardView)
            assert board.cursor == (0, 0)  # the cell selected in the game
            await pilot.press("ж")
            assert app.solver.guesses == {16: "Ж"}
            assert board.cursor == (0, 2)  # skipped the lock, next empty cell
            await pilot.press("f")  # Latin key on the ЙЦУКЕН position of "А"
            assert app.solver.guesses == {16: "Ж", 21: "А"}
            board.cursor = (0, 0)
            await pilot.press("backspace")
            assert app.solver.guesses == {21: "А"}
            await pilot.press("ctrl+z")
            assert app.solver.guesses == {16: "Ж", 21: "А"}
            assert app.solver.placeable() == {3: ("К", 1), 15: ("У", 1), 20: ("Т", 3), 26: ("З", 1)}
            board.cursor = (2, 0)
            await pilot.press("б")  # code 21 is guessed, typing overrides the guess everywhere
            assert app.solver.guesses[21] == "Б"

    asyncio.run(scenario())


def test_file_feed_edit_and_save(tmp_path):
    path = tmp_path / "b.toml"
    path.write_text('board = """\n*_16 Г?21 | _21\n"""\n', encoding="utf-8")

    async def scenario() -> None:
        app = CrgramApp(FileFeed(path), path)
        async with app.run_test(size=(140, 40)) as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            board = app.query_one(BoardView)
            await pilot.press("о")
            assert app.solver.guesses == {16: "О"}
            board.cursor = (0, 1)
            await pilot.press("д")  # fix the misrecognized letter
            await pilot.press("ctrl+s")
            await pilot.pause()

    asyncio.run(scenario())
    saved = boardfile.load(path)
    assert saved.guesses == {16: "О"}
    assert boardfile.format_board(saved.board) == "*_16 Д21 | _21"


def test_lock_editing(tmp_path):
    path = tmp_path / "b.toml"
    path.write_text('board = """\n_1 # _2\n"""\n', encoding="utf-8")

    async def scenario() -> None:
        app = CrgramApp(FileFeed(path), path)
        async with app.run_test(size=(140, 40)) as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            board = app.query_one(BoardView)
            board.cursor = (0, 1)
            await pilot.press("ш")
            assert app.solver.cell_guesses == {(0, 1): "Ш"}
            board.cursor = (0, 1)
            await pilot.press("1", "7", "enter")
            cell = app.solver.board.words[0].cells[1]
            assert cell.code == 17 and app.solver.guesses == {17: "Ш"}
            board.cursor = (0, 0)
            await pilot.press("#")
            await pilot.press("tab")  # the only lock now: the cursor stays on it
            assert board.cursor == (0, 0)
            await pilot.press("ctrl+s")
            await pilot.pause()

    asyncio.run(scenario())
    assert boardfile.format_board(boardfile.load(path).board) == "# _17 _2"


def test_tab_jumps_between_locks(tmp_path):
    path = tmp_path / "b.toml"
    path.write_text('board = """\n*_1 # _2 | Т3 # , | #\n"""\n', encoding="utf-8")

    async def scenario() -> None:
        app = CrgramApp(FileFeed(path), path)
        async with app.run_test(size=(140, 40)) as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            board = app.query_one(BoardView)
            visited = []
            for _ in range(4):
                await pilot.press("tab")
                visited.append(board.cursor)
            assert visited == [(0, 1), (1, 1), (2, 0), (0, 1)]  # punctuation skipped, wraps around
            back = []
            for _ in range(3):
                await pilot.press("shift+tab")
                back.append(board.cursor)
            assert back == [(2, 0), (1, 1), (0, 1)]  # backwards, wrapping to the last lock

    asyncio.run(scenario())


def test_typing_open_letter_finds_it(tmp_path):
    path = tmp_path / "b.toml"
    path.write_text('board = """\n*_1 Т2 _3 | _2 Т | _4\n"""\n', encoding="utf-8")

    async def scenario() -> None:
        app = CrgramApp(FileFeed(path), path)
        async with app.run_test(size=(140, 40)) as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            board = app.query_one(BoardView)
            assert board.cursor == (0, 0)
            await pilot.press("т")  # "Т" is open: not a guess, a search
            assert app.solver.guesses == {}
            assert board.highlight == "Т" and board.cursor == (0, 1)
            await pilot.press("т", "т", "т")  # (1, 0) is empty with known code 2, then wraps around
            assert board.cursor == (0, 1)
            await pilot.press("right")
            assert board.highlight is None
            await pilot.press("о")  # a new letter is still a guess
            assert app.solver.guesses == {3: "О"} and board.highlight is None
            board.cursor = (2, 0)
            await pilot.press("о")  # "О" is guessed for code 3: a search, code 4 stays unknown
            assert app.solver.guesses == {3: "О"} and board.highlight == "О" and board.cursor == (0, 2)

    asyncio.run(scenario())


def test_open_letter_search_undo_and_russian_shortcuts(tmp_path):
    path = tmp_path / "b.toml"
    path.write_text('board = """\nР1 О2 Г?3 | _4 Р1\n"""\n[guesses]\n4 = "А"\n', encoding="utf-8")

    async def scenario() -> None:
        app = CrgramApp(FileFeed(path), path)
        async with app.run_test(size=(140, 40)) as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            board = app.query_one(BoardView)
            cells = app.solver.board.words[0].cells
            board.cursor = (0, 1)
            await pilot.press("р")  # "Р" is on the board: a search, the open "О" stays
            assert cells[1].letter == "О" and board.highlight == "Р" and board.cursor == (1, 1)
            board.cursor = (0, 2)
            await pilot.press("ф")  # an unconfirmed letter is corrected
            assert cells[2].letter == "Ф" and cells[2].verified
            await pilot.press("ctrl+я")  # undo on the Russian layout: the letter comes back...
            assert app.solver.board.words[0].cells[2].letter == "Г"
            assert app.solver.guesses == {4: "А"}  # ...and the guesses loaded with the board stay
            await pilot.press("ctrl+ы")
            await pilot.pause()

    asyncio.run(scenario())
    assert path.read_text(encoding="utf-8").startswith("# crgram board")  # saved by Ctrl+S on the Russian layout


def test_search_does_not_highlight_hint_words(tmp_path):
    path = tmp_path / "b.toml"
    path.write_text('board = """\n*_1 Т2 | [ТОТ]\n"""\n', encoding="utf-8")

    async def scenario() -> None:
        app = CrgramApp(FileFeed(path), path)
        async with app.run_test(size=(140, 40)) as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            board = app.query_one(BoardView)
            await pilot.press("т")
            assert board.highlight == "Т"
            rendered = board.render()
            found = [
                rendered.plain[sp.start : sp.end] for sp in rendered.spans if st.FOUND.split()[-1] in str(sp.style)
            ]
            assert found == ["Т"]  # only the board cell, not the letters of the hint word

    asyncio.run(scenario())


def test_accept_green_letters(tmp_path):
    path = tmp_path / "b.toml"
    path.write_text('board = """\n*_1 Т2 _2 | _2\n"""\n', encoding="utf-8")

    async def scenario() -> None:
        app = CrgramApp(FileFeed(path), path)
        async with app.run_test(size=(140, 40)) as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.solver.placeable() == {2: ("Т", 2)}
            await pilot.press("ctrl+a")
            assert app.solver.placeable() == {}
            await pilot.press("ctrl+s")
            await pilot.pause()

    asyncio.run(scenario())
    assert boardfile.format_board(boardfile.load(path).board) == "*_1 Т2 Т2 | Т2"


def test_fixing_confident_letter_needs_second_press(tmp_path):
    path = tmp_path / "b.toml"
    path.write_text('board = """\n*_1 Е2 | _3\n"""\n', encoding="utf-8")

    async def scenario() -> None:
        app = CrgramApp(FileFeed(path), path)
        async with app.run_test(size=(140, 40)) as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            board = app.query_one(BoardView)
            cell = app.solver.board.words[0].cells[1]
            board.cursor = (0, 1)
            await pilot.press("ф")  # a stray key: nothing changes yet
            assert cell.letter == "Е"
            await pilot.press("ш", "ф")  # another key in between cancels the pending fix
            assert cell.letter == "Е"
            await pilot.press("ф")
            assert cell.letter == "Ф"

    asyncio.run(scenario())
