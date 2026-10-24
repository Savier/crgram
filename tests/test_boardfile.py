import pytest
from conftest import SAMPLES, load

from crgram import boardfile
from crgram.boardfile import BoardFile, BoardFormatError
from crgram.model import CellKind
from crgram.vision import ocr
from crgram.vision.recognize import Recognizer

TEXT = """
# hand-written board
board = \"\"\"
*_16 # _21 , | _3 т20 _32
Г?21 Д | [общественным.] | —
\"\"\"

[guesses]
16 = "ж"
"""


def test_parse():
    bf = boardfile.loads(TEXT)
    assert bf.guesses == {16: "Ж"}
    assert [w.row for w in bf.board.words] == [0, 0, 1, 1, 1]
    first = bf.board.words[0].cells
    assert [c.kind for c in first] == [CellKind.EMPTY, CellKind.LOCKED, CellKind.EMPTY, CellKind.PUNCT]
    assert first[0].selected and first[0].code == 16
    t = bf.board.words[1].cells[1]
    assert (t.kind, t.letter, t.code) == (CellKind.LETTER, "Т", 20)
    g, d = bf.board.words[2].cells
    assert (g.letter, g.code, g.verified) == ("Г", 21, False)
    assert (d.letter, d.code) == ("Д", None)
    hint = bf.board.words[3]
    assert hint.hint and hint.text == "ОБЩЕСТВЕННЫМ." and hint.cells[-1].kind is CellKind.PUNCT


def test_roundtrip_text():
    bf = boardfile.loads(TEXT)
    again = boardfile.loads(boardfile.dumps(bf))
    assert again.board == bf.board and again.guesses == bf.guesses


@pytest.mark.parametrize("bad", ["_ | _3", "Q12", "_3 ab"])
def test_errors_point_to_line(bad):
    with pytest.raises(BoardFormatError, match="строка 2"):
        boardfile.loads(f'board = """\n_1\n{bad}\n"""')


@pytest.mark.skipif(not ocr.available(), reason="tesseract is needed for hint words")
@pytest.mark.parametrize("sample", sorted(p.name for p in SAMPLES.glob("screen_*.png")))
def test_roundtrip_recognized(sample, tmp_path):
    board = Recognizer(tmp_path).recognize(load(sample)).board
    parsed = boardfile.loads(boardfile.dumps(BoardFile(board))).board
    assert _essence(parsed) == _essence(board)


def _essence(board):
    """Everything except screen coordinates."""
    return [(w.row, w.hint, [(c.kind, c.letter, c.code, c.selected, c.verified) for c in w.cells]) for w in board.words]


def test_lock_guesses_roundtrip():
    bf = boardfile.loads('board = """\n_1 # _2\n"""\n[locks]\n"1.2" = "ш"\n')
    assert bf.lock_guesses == {(0, 1): "Ш"}
    assert boardfile.loads(boardfile.dumps(bf)).lock_guesses == {(0, 1): "Ш"}
    with pytest.raises(BoardFormatError, match=r"замок 1\.1"):
        boardfile.loads('board = """\n_1 # _2\n"""\n[locks]\n"1.1" = "ш"\n')
