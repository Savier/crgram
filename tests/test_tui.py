import asyncio

import pytest

from crgram import boardfile
from crgram.feeds import FileFeed, LiveFeed
from crgram.sources import ImageSource
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
            await pilot.press("ctrl+s")
            await pilot.pause()

    asyncio.run(scenario())
    assert boardfile.format_board(boardfile.load(path).board) == "# _17 _2"
