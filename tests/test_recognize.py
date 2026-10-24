import cv2
import pytest
from conftest import load

from crgram.boardfile import format_board
from crgram.model import CellKind
from crgram.vision import ocr
from crgram.vision.layout import analyze
from crgram.vision.recognize import Recognizer

CODES = [
    [16, None, 21, None],
    [3, 20, 32],
    [21, 30, 32],
    [26, 13, None, 18, None],
    [15, 16, 6, 2, 6, 18, 1],
    [21, 30, 32],
    [26, 6],
    [9, 21, 17, 20, None, 32, 17, 20, 29],
    [1],
    [13, 21, 15, 17, 20, 6, 13, 13, 32, 21],
    [16, 13, 1, 25, 6, 13, 1, 21],
    [3],
    [None] * 12,  # hint word
    [None, 21, None, 6, None, None],
]


def test_layout(screen_01):
    layout = analyze(cv2.imread(str(screen_01)))
    assert len(layout.plates) == 14
    assert [p.row for p in layout.plates] == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4]
    assert [p.hint for p in layout.plates].count(True) == 1
    kinds = [s.kind for p in layout.plates for s in p.slots]
    assert kinds.count("box") == 48
    assert kinds.count("lock") == 5


@pytest.mark.skipif(not ocr.available(), reason="tesseract is needed to bootstrap digits")
def test_recognize(screen_01, tmp_path):
    rec = Recognizer(tmp_path)
    for _ in range(2):  # second pass uses the learned digit bank
        board = rec.recognize(cv2.imread(str(screen_01))).board
        assert [[c.code for c in w.cells] for w in board.words] == CODES
    assert board.words[12].hint and board.words[12].text == "ОБЩЕСТВЕННЫМ"
    assert [c.letter for w in board.words for c in w.cells if c.kind is CellKind.PUNCT] == [",", ",", "."]
    selected = [(wi, ci) for wi, ci, c in board.cells() if c.selected]
    assert selected == [(0, 0)]
    letters = {c.code: c.letter for _, _, c in board.cells() if c.kind is CellKind.LETTER and c.code}
    assert letters == {20: "Т", 26: "З", 15: "У", 25: "М", 3: "К"}
    assert board.words[13].cells[0].letter == "Д"
    assert all(c.verified for _, _, c in board.cells())


@pytest.mark.skipif(not ocr.available(), reason="tesseract is needed for the hint word")
def test_recognize_mostly_solved(tmp_path):
    board = Recognizer(tmp_path).recognize(load("screen_02.png")).board
    assert " ".join(w.text for w in board.words) == (
        "ВСЕ, КТО ЕГО ЗН_Л, УВАЖАЛИ ЕГО ЗА ЧЕСТНОСТЬ И НЕУСТАННОЕ ВНИМАНИЕ К ОБЩЕСТВЕННЫМ ДЕЛАМ."
    )
    assert all(c.verified for _, _, c in board.cells())
    # codes are shown only under some of the open letters
    assert {c.code for _, _, c in board.cells() if c.code} == {6}
    assert [(wi, ci) for wi, ci, c in board.cells() if c.selected] == [(3, 2)]


@pytest.mark.skipif(not ocr.available(), reason="tesseract is needed to bootstrap digits")
def test_learn_letter(screen_01, tmp_path):
    rec = Recognizer(tmp_path)
    scan = rec.recognize(cv2.imread(str(screen_01)))
    rec.learn_letter(scan, (13, 0), "Б")  # a user correction overrides the shipped sample
    cell = Recognizer(tmp_path).recognize(cv2.imread(str(screen_01))).board.words[13].cells[0]
    assert (cell.letter, cell.verified) == ("Б", True)


def test_recognize_new_level(tmp_path):
    board = Recognizer(tmp_path).recognize(load("screen_03.png")).board
    assert format_board(board).splitlines() == [
        "Н25 О1 | *_8 _20 _17 _1 Л3 _29 _25 _20 | _12 _1 Ф # _17",
        "_1 _12 _3 _20 _4 _20 _3 _20 | _25 Е15 _24 _20 # _17 _16 _4 _25 # _31",
        "_9 _31 _1 _31 , | _29 | _18 | _25 _15 # З24 _10 _1 _4 # _19 | _15 Е15",
        "_31 _9 _26 _15 # _27 # _1 | _18 _1 _24 _17 _20 _6 _27 _20 _3 _1 ,",
        "П _1 _4 _4 _15 _17 _26 _29 _18 _20 _16 | _15 Е15 .",
    ]


def test_recognize_lookalike_letters(tmp_path):
    """Б/В and Я must not be confused; codes are shown only under some open letters."""
    board = Recognizer(tmp_path).recognize(load("screen_04.png")).board
    assert " ".join(w.text for w in board.words) == (
        "НО КАРОЛИНА БОФОР ОБЛАДАЛА НЕЗАУРЯДНЫМ УМОМ, И В НЕВЗГОДАХ ЕЕ МУЖЕСТВ_ ВОЗРАСТАЛО, ПОДДЕРЖИВАЯ ЕЕ."
    )
    assert all(c.verified for _, _, c in board.cells())
    assert {c.code for _, _, c in board.cells() if c.code} == {1}
    assert [c.code for _, _, c in board.cells() if c.selected] == [1]


@pytest.mark.skipif(not ocr.available(), reason="tesseract is needed for the hint words")
def test_recognize_hints_and_dash(tmp_path):
    board = Recognizer(tmp_path).recognize(load("screen_05.png")).board
    assert format_board(board).splitlines() == [
        "_21 _32 Ж Н21 _30 _32 | _31 _16 # _18 _11 | _29 _16 _9 Е32 *_2 _11 | _11",
        "_12 # _30 Б24 # _16 | _15 _9 _5 _16 , | _8 | _28 _15 _24 _2 _30 _29",
        "[УДОВОЛЬСТВИЕМ]",
        "Г1 _31 _6 _28 _32 В3 _19 _32 _1 _15 | Н21 _16 | _29 _32 _21 _6 , | —",
        "_29 # _11 | _23 _32 _2 _3 _30 _32",
        "[ВОСПОМИНАНИЯ.]",
    ]
    assert board.words[-1].cells[-1].kind is CellKind.PUNCT


@pytest.mark.skipif(not ocr.available(), reason="tesseract is needed for the hint words")
def test_recognize_shifted_glyph(tmp_path):
    """A letter shifted by a pixel scores lower, but still leads the look-alikes clearly."""
    board = Recognizer(tmp_path).recognize(load("screen_06.png")).board
    assert " ".join(w.text for w in board.words) == (
        "НЕЖНЫЕ ЛАСКИ МАТЕРИ И УЛЫБКА ОТЦА, С ДОБРЫМ УДОВОЛЬСТВИЕМ ГЛЯДЕВШЕГО НА _ЕНЯ, — МОИ ПЕРВЫЕ ВОСПОМИНАНИЯ."
    )
    assert all(c.verified for _, _, c in board.cells())
    assert {c.code for _, _, c in board.cells() if c.code} == {29}


def test_recognize_underlined_theme(tmp_path):
    """Another theme: cells are underlines, codes stay under open letters, keyboard at the bottom."""
    board = Recognizer(tmp_path).recognize(load("screen_07.png")).board
    assert format_board(board).splitlines() == [
        "*_32 _23 _8 _29 | _7 _5 _25 _24 _12 _29 | _25 _7 _24 _20 _29 _32 _29 | _5 Д8 _10",
        "_21 _24 _8 _1 _32 _18 _1 | _8 _1 | _22 _1 _18 _1 , | _25 _24 _25 | _10 З21 Н14 _24 _32 _29 ,",
        "Ч17 _22 _1 | _9 _22 _10 | _30 _24 _28 _1 _22 _10 | _8 _5 _32 _24 _23 _22",
        "_25 _30 _1 Ш _5 _17 _14 Ы15 _5 | М12 _29 _25 _30 _1 _28 _15 .",
    ]


def test_recognize_lock_with_dark_dot(tmp_path):
    """A lock with one dot already dark (one neighbour guessed) is still a lock, not an apostrophe."""
    board = Recognizer(tmp_path).recognize(load("screen_08.png")).board
    assert format_board(board).splitlines() == [
        "_32 _23 _8 _23 _29 О7 _5 _25 , | _24 О7 _12 _20 _10 | _5 _12 # _1",
        "О7 _8 А23 _18 И22 _29 | _14 _7 _10 | _18 _23 _8 _17 _14 , | _22 ,",
        "*_9 _18 # _30 # _28 | О7 _1 | _18 _23 Д _7 С5 # _22 , | _28",
        "_5 _7 _7 _15 Щ _22 _29 | О7 # _19 У17 | _7 | _5 В12 _7 Е21 _14",
        "_7 _1 _32 _18 _20 _1 _22 _22 .",
    ]
