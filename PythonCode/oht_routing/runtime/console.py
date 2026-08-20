"""Small terminal-aware helpers for readable contextual runtime output."""

from __future__ import annotations

import os
import sys
import textwrap


RESET = "\033[0m"
BOLD = "1"
DIM = "2"
CYAN = "96"
BLUE = "94"
GREEN = "92"
YELLOW = "93"
WHITE = "97"


def color_enabled(stream=None) -> bool:
    """Return whether ANSI styling should be emitted for this output stream."""
    stream = stream or sys.stdout
    if "NO_COLOR" in os.environ:
        return False
    if os.environ.get("FORCE_COLOR", "").lower() not in {"", "0", "false"}:
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


def styled(value, *codes, stream=None) -> str:
    """Apply ANSI codes when output is an interactive color terminal."""
    text = str(value)
    if not color_enabled(stream):
        return text
    return f"\033[{';'.join(codes)}m{text}{RESET}"


def _styled_value(value) -> str:
    if value is True:
        return styled(value, BOLD, GREEN)
    if value is False:
        return styled(value, BOLD, YELLOW)
    if value is None:
        return styled("None", DIM)
    return styled(value, WHITE)


def print_header(name) -> None:
    print(styled(f"[{name}]", BOLD, CYAN))


def print_entries(entries, *, indent=4) -> None:
    """Print aligned key/value entries with terminal-aware value coloring."""
    width = max(len(label) for label, _ in entries)
    padding = " " * indent
    for label, value in entries:
        key = styled(f"{label:<{width}}", DIM)
        separator = styled(":", DIM)
        print(f"{padding}{key} {separator} {_styled_value(value)}")


def print_section(title, entries) -> None:
    """Print a blue section title followed by aligned entries."""
    print(f"  {styled(title, BOLD, BLUE)}:")
    print_entries(entries)


def print_wrapped_items(items, *, width=100, indent=4) -> None:
    """Wrap a long comma-separated item list without counting ANSI codes."""
    padding = " " * indent
    lines = textwrap.wrap(
        ", ".join(items),
        width=max(1, width - indent),
        break_long_words=False,
        break_on_hyphens=False,
    )
    for line in lines:
        print(f"{padding}{styled(line, DIM)}")
