"""Single semantic release version for contextual TD7 artifacts."""

import re


CONTEXTUAL_VERSION = "v10.8.0"
_SEMVER_PATTERN = re.compile(r"^v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def parse_contextual_version(version: str) -> tuple[int, int, int]:
    match = _SEMVER_PATTERN.fullmatch(str(version))
    if match is None:
        raise ValueError(
            f"invalid contextual version {version!r}; expected vMAJOR.MINOR.PATCH"
        )
    return tuple(int(part) for part in match.groups())


def is_compatible_contextual_version(
    saved_version: str,
    runtime_version: str = CONTEXTUAL_VERSION,
) -> bool:
    """Accept older artifacts within the current compatible major release."""
    try:
        saved = parse_contextual_version(saved_version)
        runtime = parse_contextual_version(runtime_version)
    except ValueError:
        return False
    return saved[0] == runtime[0] and saved <= runtime


__all__ = (
    "CONTEXTUAL_VERSION",
    "is_compatible_contextual_version",
    "parse_contextual_version",
)
