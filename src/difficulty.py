"""
src/difficulty.py
-----------------
Quiz difficulty levels. The API, prompts and question bank all use the English values
(easy | medium | hard). The old Turkish values (kolay | orta | zor) are still accepted as
input so older clients keep working, but they are normalized to English immediately.
"""

from enum import Enum
from typing import Annotated, Any

from pydantic import BeforeValidator


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


DEFAULT_DIFFICULTY = Difficulty.MEDIUM

_LEGACY_ALIASES = {
    "kolay": Difficulty.EASY,
    "orta": Difficulty.MEDIUM,
    "zor": Difficulty.HARD,
}


def _coerce(value: Any) -> Any:
    """Accepts enum members, English values and legacy Turkish aliases (case-insensitive)."""
    if isinstance(value, Difficulty):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _LEGACY_ALIASES:
            return _LEGACY_ALIASES[key]
        return key
    return value


def normalize_difficulty(value: Any = None) -> str:
    """Returns the canonical English value ("easy" | "medium" | "hard"). None means the default."""
    if value is None or value == "":
        return DEFAULT_DIFFICULTY.value
    try:
        return Difficulty(_coerce(value)).value
    except ValueError:
        raise ValueError(f"Invalid difficulty: {value!r}. Use one of: easy, medium, hard.") from None


# Type for FastAPI parameters and Pydantic fields: OpenAPI shows only easy | medium | hard,
# legacy Turkish values are converted before validation.
DifficultyParam = Annotated[Difficulty, BeforeValidator(_coerce)]
