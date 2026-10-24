"""Shared colours of the board and the side panel."""

OPEN = "bold"  # letter open in the game
PLACEABLE = "bold black on green3"  # empty in the game, but its code is known: can be filled now
GUESS = "bold cyan"  # user's guess
UNVERIFIED = "bold yellow underline"  # recognized by fallback OCR, not confirmed
HINT = "italic grey62"  # whole word given by the game
EMPTY = "grey35"
LOCKED = "magenta"
LOCK_GUESS = "bold magenta underline"  # letter guessed for a lock, applies to that cell only
CODE_INPUT = "bold black on bright_yellow"
CONFLICT = "on red"
SAME_CODE = "on grey23"
CODE = "grey50"
SELECTED_CODE = "bold green"
CURSOR = "bold bright_yellow"
