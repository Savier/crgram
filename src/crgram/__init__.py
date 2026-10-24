"""crgram — interactive helper for the Cryptogram puzzle game."""


def main() -> int:
    from crgram.cli import main as cli_main

    return cli_main()
