"""Compatibility entry point for the contextual TD7 v2 runtime."""

from __future__ import annotations

import sys

from main_contextual import main as run_contextual


REMOVED_FLAGS = {"--region"}


def main():
    removed = sorted(REMOVED_FLAGS.intersection(sys.argv[1:]))
    if removed:
        raise SystemExit(
            f"{', '.join(removed)} was removed with the legacy token-TD7 "
            "runtime; use main_contextual.py and its contextual options."
        )
    run_contextual()


if __name__ == "__main__":
    main()
