"""Command line entry point."""

import argparse
import os
import sys
from pathlib import Path

from crgram.boardfile import BoardFile, BoardFormatError
from crgram.sources import AdbSource, ImageSource, Source, SourceError

COMMANDS = {"tui", "scan", "check", "shot"}


def default_data_dir() -> Path:
    if env := os.environ.get("CRGRAM_DATA_DIR"):
        return Path(env)
    from platformdirs import user_data_dir

    return Path(user_data_dir("crgram"))


def _parser() -> argparse.ArgumentParser:
    src = argparse.ArgumentParser(add_help=False)
    g = src.add_argument_group("источник экрана (по умолчанию — adb)")
    g.add_argument("-i", "--image", type=Path, help="готовый скриншот вместо adb")
    g.add_argument("-s", "--serial", help="серийный номер adb-устройства")
    g.add_argument("-d", "--display", type=int, help="номер дисплея (0, 1, …); по умолчанию первый активный")
    src.add_argument("--data-dir", type=Path, default=None, help="каталог для выученных символов")

    p = argparse.ArgumentParser(
        prog="crgram",
        description="Помощник для Cryptogram: распознаёт поле и помогает решать.",
    )
    sub = p.add_subparsers(dest="cmd")

    tui = sub.add_parser("tui", parents=[src], help="интерактивный решатель (по умолчанию)")
    tui.add_argument("board", nargs="?", type=Path, help="файл поля (.toml) вместо распознавания экрана")
    tui.add_argument("-o", "--output", type=Path, help="куда сохранять по Ctrl+S (иначе файл поля или board.toml)")
    tui.add_argument("-w", "--watch", type=float, default=0, metavar="SEC", help="обновлять каждые SEC секунд")

    scan = sub.add_parser("scan", parents=[src], help="распознать поле в файл поля")
    scan.add_argument("-o", "--output", type=Path, help="записать в файл вместо вывода на экран")
    scan.add_argument("--json", action="store_true", help="вывести модель поля в JSON (с координатами)")

    check = sub.add_parser("check", help="проверить файл поля после ручной правки")
    check.add_argument("board", type=Path)
    check.add_argument("--fix", action="store_true", help="переписать файл в каноническом виде")

    shot = sub.add_parser("shot", parents=[src], help="сохранить скриншот с устройства")
    shot.add_argument("output", type=Path)
    return p


def _source(args: argparse.Namespace) -> Source:
    if args.image:
        return ImageSource(args.image)
    return AdbSource(serial=args.serial, display=args.display)


def _fail(message: str) -> int:
    print(f"ошибка: {message}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS | {"-h", "--help"}:
        argv = ["tui", *argv]
    args = _parser().parse_args(argv)

    match args.cmd:
        case "tui":
            from crgram.feeds import FileFeed, LiveFeed
            from crgram.tui.app import CrgramApp
            from crgram.vision.recognize import Recognizer

            if args.board and args.image:
                return _fail("укажите либо файл поля, либо --image")
            if args.board:
                feed = FileFeed(args.board)
            else:
                feed = LiveFeed(_source(args), Recognizer(args.data_dir or default_data_dir()))
            save_path = args.output or args.board or Path("board.toml")
            CrgramApp(feed, save_path, watch=args.watch).run()

        case "scan":
            from crgram import boardfile
            from crgram.vision.recognize import Recognizer

            source = _source(args)
            try:
                scan = Recognizer(args.data_dir or default_data_dir()).recognize(source.capture())
            except SourceError as e:
                return _fail(str(e))
            if args.json:
                text = scan.board.model_dump_json(indent=2) + "\n"
            else:
                text = boardfile.dumps(BoardFile(scan.board, source=source.name))
            if args.output:
                args.output.write_text(text, encoding="utf-8")
                print(f"{args.output}: слов {len(scan.board.words)}", file=sys.stderr)
            else:
                sys.stdout.write(text)

        case "check":
            from crgram import boardfile
            from crgram.solver import Solver

            try:
                bf = boardfile.load(args.board)
            except (OSError, BoardFormatError) as e:
                return _fail(str(e))
            solver = Solver(bf.board, dict(bf.guesses))
            print(boardfile.format_board(bf.board))
            print(f"\nслов {len(bf.board.words)}, кодов {len(solver.codes())}, известно {len(solver.known())}")
            if bad := solver.conflicts():
                print(f"конфликты по кодам: {sorted(bad)}")
            if args.fix:
                boardfile.save(args.board, bf)

        case "shot":
            import cv2

            try:
                image = _source(args).capture()
            except SourceError as e:
                return _fail(str(e))
            cv2.imwrite(str(args.output), image)
            print(f"{args.output} ({image.shape[1]}x{image.shape[0]})")
    return 0
