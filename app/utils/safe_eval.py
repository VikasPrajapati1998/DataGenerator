"""Safe evaluation of business-rule expressions written in a Python subset.

Rules such as ``"hire_date > birth_date"`` or
``"salary >= 400000 if experience_years > 10 else True"`` are parsed with
:mod:`ast` and only whitelisted node types are executed. Attribute access,
calls to arbitrary functions, imports, lambdas and comprehensions are
rejected, so user-supplied rules cannot run code.
"""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

from app.core.exceptions import SchemaError

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CMP_OPS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_, ast.IsNot: operator.is_not,
}
_UNARY_OPS = {ast.Not: operator.not_, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _to_date(value: Any) -> date | None:
    if value is None or isinstance(value, date):
        return value
    text = str(value)
    try:
        return datetime.fromisoformat(text).date() if "T" in text or " " in text else date.fromisoformat(text)
    except ValueError:
        return None


def _years_between(start: Any, end: Any) -> float | None:
    first, second = _to_date(start), _to_date(end)
    if first is None or second is None:
        return None
    return (second - first).days / 365.2425


def _days_between(start: Any, end: Any) -> int | None:
    first, second = _to_date(start), _to_date(end)
    if first is None or second is None:
        return None
    return (second - first).days


def _add_days(start: Any, days: Any) -> str | None:
    first = _to_date(start)
    if first is None or days is None:
        return None
    from datetime import timedelta  # pylint: disable=import-outside-toplevel

    return (first + timedelta(days=int(round(float(days))))).isoformat()


def _date_part(part: str):  # noqa: ANN202
    def get(value: Any) -> int | None:
        day = _to_date(value)
        if day is None:
            return None
        return {"year": day.year, "month": day.month, "day": day.day, "week": day.isocalendar()[1],
                "weekday": day.isoweekday()}[part]
    return get


_FUNCTIONS: dict[str, Any] = {
    "len": len, "abs": abs, "min": min, "max": max, "round": round,
    "lower": lambda s: str(s).lower(), "upper": lambda s: str(s).upper(),
    "startswith": lambda s, p: str(s).startswith(str(p)),
    "endswith": lambda s, p: str(s).endswith(str(p)),
    "contains": lambda s, p: str(p).lower() in str(s).lower(),
    "matches": lambda s, p: re.fullmatch(str(p), str(s)) is not None,
    "date": _to_date, "years_between": _years_between, "days_between": _days_between,
    "today": date.today, "is_null": lambda v: v is None or v == "",
    "add_days": _add_days, "year": _date_part("year"), "month": _date_part("month"), "day": _date_part("day"),
    "iso_week": _date_part("week"), "weekday": _date_part("weekday"),
    "int": lambda v: int(float(v)), "float": float, "str": str, "concat": lambda *parts: "".join(map(str, parts)),
}


class RuleExpression:
    """A pre-validated, reusable business-rule expression.

    Args:
        expression: Expression in the supported Python subset.

    Raises:
        SchemaError: If the expression is syntactically invalid or uses a
            disallowed construct.
    """

    def __init__(self, expression: str) -> None:
        self.expression = expression
        try:
            self._tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise SchemaError(f"Invalid rule expression {expression!r}: {exc.msg}") from exc
        self._check(self._tree)
        self.names = sorted({n.id for n in ast.walk(self._tree) if isinstance(n, ast.Name)} - set(_FUNCTIONS)
                            - {"True", "False", "None"})

    def _check(self, node: ast.AST) -> None:
        allowed = (
            ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.BinOp, ast.UnaryOp, ast.Compare,
            ast.IfExp, ast.Name, ast.Load, ast.Constant, ast.Call, ast.Tuple, ast.List, ast.Set,
            *_BIN_OPS, *_CMP_OPS, *_UNARY_OPS,
        )
        for child in ast.walk(node):
            if not isinstance(child, allowed):
                raise SchemaError(
                    f"Rule {self.expression!r} uses unsupported syntax: {type(child).__name__}"
                )
            if isinstance(child, ast.Call):
                if not isinstance(child.func, ast.Name) or child.func.id not in _FUNCTIONS:
                    raise SchemaError(f"Rule {self.expression!r} calls a function that is not allowed")
                if child.keywords:
                    raise SchemaError(f"Rule {self.expression!r} may not use keyword arguments")

    def evaluate(self, record: Mapping[str, Any]) -> bool:
        """Evaluate the rule against ``record``.

        Dates stored as ISO strings are compared as dates automatically when
        both sides of a comparison look like dates.

        Args:
            record: Field values by name.

        Returns:
            ``True`` when the rule holds. A rule that cannot be evaluated
            because of missing/null operands is treated as failed.
        """
        try:
            return bool(self._eval(self._tree.body, record))
        except (TypeError, ValueError, ZeroDivisionError, KeyError):
            return False

    def compute(self, record: Mapping[str, Any]) -> Any:
        """Evaluate the expression as a value (for ``formula`` fields).

        Returns:
            The value, or ``None`` when an operand is missing or null.
        """
        try:
            return self._eval(self._tree.body, record)
        except (TypeError, ValueError, ZeroDivisionError, KeyError, OverflowError):
            return None

    def _eval(self, node: ast.AST, record: Mapping[str, Any]) -> Any:  # noqa: PLR0911
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in record:
                return record[node.id]
            if node.id in {"True", "False", "None"}:
                return {"True": True, "False": False, "None": None}[node.id]
            raise KeyError(node.id)
        if isinstance(node, ast.BoolOp):
            values = (self._eval(v, record) for v in node.values)
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.UnaryOp):
            return _UNARY_OPS[type(node.op)](self._eval(node.operand, record))
        if isinstance(node, ast.BinOp):
            return _BIN_OPS[type(node.op)](self._eval(node.left, record), self._eval(node.right, record))
        if isinstance(node, ast.IfExp):
            branch = node.body if self._eval(node.test, record) else node.orelse
            return self._eval(branch, record)
        if isinstance(node, ast.Compare):
            left = self._eval(node.left, record)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval(comparator, record)
                lhs, rhs = _coerce_pair(left, right)
                if not _CMP_OPS[type(op)](lhs, rhs):
                    return False
                left = right
            return True
        if isinstance(node, ast.Call):
            func = _FUNCTIONS[node.func.id]  # type: ignore[attr-defined]
            return func(*(self._eval(arg, record) for arg in node.args))
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return [self._eval(elt, record) for elt in node.elts]
        raise ValueError(f"Unsupported node {type(node).__name__}")


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _coerce_pair(left: Any, right: Any) -> tuple[Any, Any]:
    """Make ISO-date strings and numeric strings comparable."""
    if isinstance(left, str) and isinstance(right, str) and _ISO_DATE.match(left) and _ISO_DATE.match(right):
        return _to_date(left), _to_date(right)
    if isinstance(left, date) and isinstance(right, str):
        return left, _to_date(right)
    if isinstance(right, date) and isinstance(left, str):
        return _to_date(left), right
    return left, right
