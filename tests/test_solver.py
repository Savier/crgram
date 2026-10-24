from crgram.model import Board, Cell, CellKind, Word
from crgram.solver import Solver


def board() -> Board:
    cells = [
        Cell(kind=CellKind.EMPTY, code=1),
        Cell(kind=CellKind.LETTER, letter="Т", code=2),
        Cell(kind=CellKind.EMPTY, code=3),
        Cell(kind=CellKind.LOCKED),
        Cell(kind=CellKind.LETTER, letter="К"),  # all "К" open: code hidden
        Cell(kind=CellKind.PUNCT, letter=","),
    ]
    return Board(words=[Word(cells=cells)])


def test_assign_undo_pattern():
    s = Solver(board())
    s.assign(1, "О")
    assert s.pattern(0) == "ОТ..К"
    s.assign(1, None)
    assert s.pattern(0) == ".Т..К"
    assert s.undo() and s.guesses == {1: "О"}
    s.assign(2, "А")  # board letters cannot be overridden
    assert s.letter_for(2) == ("Т", "board")


def test_conflicts_and_free_letters():
    s = Solver(board())
    s.assign(1, "Т")
    s.assign(3, "К")
    assert s.conflicts() == {1, 2, 3}
    assert "Т" not in s.free_letters() and "К" not in s.free_letters()


def test_update_board_drops_confirmed_guesses():
    s = Solver(board())
    s.assign(1, "О")
    s.assign(3, "Н")
    b = board()
    b.words[0].cells[0] = Cell(kind=CellKind.LETTER, letter="О", code=1)
    s.update_board(b)
    assert s.guesses == {3: "Н"}


def test_placeable_codes():
    b = board()
    b.words[0].cells.append(Cell(kind=CellKind.EMPTY, code=2))  # one more "Т" still empty
    s = Solver(b)
    assert s.placeable() == {2: ("Т", 1)}
    assert s.unknown_codes() == [1, 3]
    s.assign(1, "О")
    assert s.unknown_codes() == [3]


def test_lock_guess_and_unlock():
    s = Solver(board())
    s.assign_cell((0, 3), "Ш")
    assert s.pattern(0) == ".Т.ШК"
    s.set_code((0, 3), 7)  # the lock had code 7: its letter becomes a code guess
    cell = s.board.words[0].cells[3]
    assert (cell.kind, cell.code) == (CellKind.EMPTY, 7)
    assert s.guesses == {7: "Ш"} and s.cell_guesses == {}
    assert s.undo()
    assert s.board.words[0].cells[3].kind is CellKind.LOCKED and s.cell_guesses == {(0, 3): "Ш"}
    s.set_code((0, 2), None)  # mark a cell as locked
    assert s.board.words[0].cells[2].kind is CellKind.LOCKED


def test_lock_opened_by_game_keeps_guess():
    s = Solver(board())
    s.assign_cell((0, 3), "Ш")
    b = board()
    b.words[0].cells[3] = Cell(kind=CellKind.EMPTY, code=7)
    s.update_board(b)
    assert s.guesses == {7: "Ш"} and s.cell_guesses == {}
