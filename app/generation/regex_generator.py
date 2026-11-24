"""Generate strings that match a practical subset of regular expressions.

Supported: literals, escaped literals, ``\\d \\w \\s``, character classes
(``[A-Z0-9_]``, negation not supported), ``.``, groups with alternation
``(a|b)``, quantifiers ``? * + {n} {n,m}`` and anchors ``^ $``. Unsupported
syntax raises :class:`SchemaError` at schema-parse time, never mid-job.
"""

from __future__ import annotations

import random
import re
import string

from app.core.exceptions import SchemaError

_CLASS_SHORTCUTS = {
    "d": string.digits,
    "w": string.ascii_letters + string.digits + "_",
    "s": " ",
}
_UNBOUNDED_MAX = 6

Node = tuple  # ("lit", chars) | ("alt", [seq, seq]) with quantifier attached


class RegexGenerator:
    """Pre-compiled generator for one pattern.

    Args:
        pattern: The regular expression to satisfy.

    Raises:
        SchemaError: If the pattern uses unsupported syntax.
    """

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern
        self._pos = 0
        try:
            self._tree = self._parse_sequence()
        except (IndexError, ValueError) as exc:
            raise SchemaError(f"Unsupported regex pattern {pattern!r}: {exc}") from exc
        if self._pos != len(pattern):
            raise SchemaError(f"Unsupported regex pattern {pattern!r} near position {self._pos}")
        self._compiled = re.compile(pattern)

    def cardinality(self) -> int:
        """Upper bound on the number of distinct strings the pattern can produce.

        Used to check that a ``unique`` regex field can hold the requested number
        of records without constant collisions.
        """
        return self._count(self._tree)

    def _count(self, sequence: list[Node]) -> int:
        total = 1
        for kind, payload, low, high in sequence:
            unit = len(payload) if kind == "set" else sum(self._count(branch) for branch in payload)
            total *= sum(unit ** k for k in range(low, high + 1))
        return total

    def generate(self, rng: random.Random) -> str:
        """Return a random string matching the pattern.

        Args:
            rng: Random source.

        Returns:
            A matching string.

        Raises:
            SchemaError: If generated output fails to match (pattern semantics
                outside the supported subset).
        """
        value = self._emit(self._tree, rng)
        if not self._compiled.fullmatch(value):
            raise SchemaError(f"Regex generator could not satisfy pattern {self.pattern!r}")
        return value

    # -- parsing -----------------------------------------------------------------
    def _peek(self) -> str | None:
        return self.pattern[self._pos] if self._pos < len(self.pattern) else None

    def _parse_sequence(self) -> list[Node]:
        items: list[Node] = []
        while (char := self._peek()) is not None and char not in "|)":
            atom = self._parse_atom()
            if atom is None:
                continue
            items.append(self._parse_quantifier(atom))
        return items

    def _parse_atom(self) -> Node | None:
        char = self.pattern[self._pos]
        self._pos += 1
        if char in "^$":
            return None
        if char == "\\":
            esc = self.pattern[self._pos]
            self._pos += 1
            return ("set", _CLASS_SHORTCUTS.get(esc, esc))
        if char == ".":
            return ("set", string.ascii_letters + string.digits)
        if char == "[":
            return ("set", self._parse_class())
        if char == "(":
            if self.pattern.startswith("?:", self._pos):
                self._pos += 2
            branches = [self._parse_sequence()]
            while self._peek() == "|":
                self._pos += 1
                branches.append(self._parse_sequence())
            if self._peek() != ")":
                raise ValueError("unterminated group")
            self._pos += 1
            return ("alt", branches)
        if char in "*+?{":
            raise ValueError(f"dangling quantifier {char!r}")
        return ("set", char)

    def _parse_class(self) -> str:
        if self._peek() == "^":
            raise ValueError("negated character classes are not supported")
        chars: list[str] = []
        while (char := self.pattern[self._pos]) != "]":
            self._pos += 1
            if char == "\\":
                esc = self.pattern[self._pos]
                self._pos += 1
                chars.extend(_CLASS_SHORTCUTS.get(esc, esc))
            elif self._peek() == "-" and self.pattern[self._pos + 1] != "]":
                end = self.pattern[self._pos + 1]
                self._pos += 2
                chars.extend(chr(c) for c in range(ord(char), ord(end) + 1))
            else:
                chars.append(char)
        self._pos += 1
        if not chars:
            raise ValueError("empty character class")
        return "".join(dict.fromkeys(chars))

    def _parse_quantifier(self, atom: Node) -> Node:
        char = self._peek()
        low, high = 1, 1
        if char == "?":
            low, high = 0, 1
        elif char == "*":
            low, high = 0, _UNBOUNDED_MAX
        elif char == "+":
            low, high = 1, _UNBOUNDED_MAX
        elif char == "{":
            end = self.pattern.index("}", self._pos)
            body = self.pattern[self._pos + 1:end]
            parts = body.split(",")
            low = int(parts[0])
            high = low if len(parts) == 1 else (int(parts[1]) if parts[1] else low + _UNBOUNDED_MAX)
            self._pos = end
        else:
            return (*atom, 1, 1)
        self._pos += 1
        return (*atom, low, high)

    # -- emitting ----------------------------------------------------------------
    def _emit(self, sequence: list[Node], rng: random.Random) -> str:
        out: list[str] = []
        for kind, payload, low, high in sequence:
            for _ in range(rng.randint(low, high)):
                if kind == "set":
                    out.append(rng.choice(payload))
                else:
                    out.append(self._emit(rng.choice(payload), rng))
        return "".join(out)
