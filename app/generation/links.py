"""Links between tables and exact-count assignment, both deterministic in the global record index.

Multi-table runs generate parent tables first. The columns that child tables need from a parent (keys, inherited
fields, filter columns) are written to a *key store*: ``<name>.jsonl`` with one JSON row per parent record plus
``<name>.offsets`` (int64 byte offsets, ``rows + 1`` entries) for random access by row number. Child workers map
both files read-only, so any number of threads and processes can read them.

Which parent row a child record gets depends only on the run seed, the link and the child's global index, never on
batch sizes, core counts or regeneration attempts. All columns taken through one link come from the same parent
row, so composite keys (``po_no`` + ``line_no``) always align.
"""

from __future__ import annotations

import json
import mmap
import threading
from array import array
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from app.core.exceptions import SchemaError
from app.utils.hashing import normalize_value

_MASK64 = (1 << 64) - 1
LINK_MODES = ("random", "cover", "unique")


def mix64(*parts: int) -> int:
    """Combine integers into one well-mixed 64-bit value (splitmix64 finaliser)."""
    value = 0x9E3779B97F4A7C15
    for part in parts:
        value = (value ^ (int(part) & _MASK64)) * 0xBF58476D1CE4E5B9 & _MASK64
        value ^= value >> 31
        value = value * 0x94D049BB133111EB & _MASK64
        value ^= value >> 29
    return value


def text_key(text: str) -> int:
    """A process-stable integer for a name (used to key permutations and link choices)."""
    value = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        value = (value ^ byte) * 0x100000001B3 & _MASK64
    return value


class Permutation:
    """A seeded bijection of ``range(n)`` (4-round Feistel network with cycle walking).

    Args:
        n: Domain size.
        key: Seed of the permutation.
    """

    def __init__(self, n: int, key: int) -> None:
        self.n = n
        bits = max(2, (max(n, 1) - 1).bit_length())
        bits += bits % 2
        self.half = bits // 2
        self.mask = (1 << self.half) - 1
        self.keys = [mix64(key, round_no) for round_no in range(4)]

    def __call__(self, index: int) -> int:
        if self.n <= 1:
            return 0
        value = index % self.n
        while True:
            left, right = value >> self.half, value & self.mask
            for key in self.keys:
                left, right = right, left ^ (mix64(right, key) & self.mask)
            value = (left << self.half) | right
            if value < self.n:
                return value


def resolve_counts(name: str, allowed: list[Any], counts: list[int | None] | None, weights: list[float] | None,
                   total: int) -> list[int]:
    """Turn ``counts`` (one may be ``None`` = the rest) or exact ``weights`` into counts summing to ``total``.

    Raises:
        SchemaError: If the counts cannot add up to ``total``.
    """
    if counts is not None:
        fixed = sum(c for c in counts if c is not None)
        if None in counts:
            rest = total - fixed
            if rest < 0:
                raise SchemaError(f"Field {name!r}: counts add up to {fixed:,}, more than the {total:,} records")
            return [rest if c is None else c for c in counts]
        if fixed != total:
            raise SchemaError(f"Field {name!r}: counts add up to {fixed:,} but the table has {total:,} records; "
                              "set one count to null to take the rest")
        return list(counts)
    shares = weights or [1.0] * len(allowed)
    scale = total / sum(shares)
    raw = [w * scale for w in shares]
    result = [int(r) for r in raw]
    for position in sorted(range(len(raw)), key=lambda p: raw[p] - result[p], reverse=True)[:total - sum(result)]:
        result[position] += 1
    return result


class ExactAssigner:
    """Assigns enum values so each value occurs exactly its count, spread evenly over the dataset."""

    def __init__(self, values: list[Any], counts: list[int], key: int) -> None:
        self.values = values
        self.total = sum(counts)
        self.bounds: list[int] = []
        running = 0
        for count in counts:
            running += count
            self.bounds.append(running)
        self.perm = Permutation(self.total, key)

    def value(self, index: int) -> Any:
        """Value of the record at global ``index``."""
        position = self.perm(index % max(self.total, 1))
        return self.values[min(bisect_right(self.bounds, position), len(self.values) - 1)]


def match_key(values: Iterable[Any]) -> str:
    """Normalised key of the values a child matches its parent on."""
    return json.dumps([normalize_value(v) for v in values], ensure_ascii=False)


def row_matches(row: Mapping[str, Any], where: Mapping[str, Any] | None) -> bool:
    """Whether a parent row passes a ``where`` filter (``{column: value or [values]}``)."""
    for column, wanted in (where or {}).items():
        options = wanted if isinstance(wanted, list) else [wanted]
        if normalize_value(row.get(column)) not in {normalize_value(v) for v in options}:
            return False
    return True


def write_int64(path: Path, values: Iterable[int]) -> None:
    """Write integers as a little-endian int64 file."""
    data = array("q", values)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        data.tofile(handle)


class _Mapped:
    """A read-only memory map of one file (closed explicitly so Windows can delete the job folder)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.size = path.stat().st_size
        self._handle = path.open("rb")
        self.map = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ) if self.size else None
        self.ints = memoryview(self.map).cast("q") if self.map is not None and path.suffix != ".jsonl" else None

    def close(self) -> None:
        if self.ints is not None:
            self.ints.release()
            self.ints = None
        if self.map is not None:
            self.map.close()
            self.map = None
        self._handle.close()


class LinkResolver:
    """Gives each child record its parent rows (and position within its parent).

    Args:
        context: The ``links`` context the multi-table runner puts in the request: ``links`` (name -> store,
            rows, mode, eligible file, groups), ``per_parent`` (link name and the cumulative counts file).
        seed: Seed of the table.
        total: Records of the whole table.
    """

    def __init__(self, context: Mapping[str, Any], seed: int, total: int) -> None:
        self.links: dict[str, dict[str, Any]] = dict(context.get("links") or {})
        self.per_parent: dict[str, Any] | None = context.get("per_parent")
        self.seed = seed
        self.total = max(total, 1)
        self._maps: dict[str, _Mapped] = {}
        self._groups: dict[str, dict[str, tuple[int, int]]] = {}
        self._perms: dict[str, Permutation] = {}
        self._lock = threading.Lock()
        self._local = threading.local()

    def _map(self, path: str) -> _Mapped:
        mapped = self._maps.get(path)
        if mapped is None:
            with self._lock:
                mapped = self._maps.get(path)
                if mapped is None:
                    mapped = self._maps[path] = _Mapped(Path(path))
        return mapped

    def _link(self, name: str) -> dict[str, Any]:
        link = self.links.get(name)
        if link is None:
            raise SchemaError(f"Link {name!r} is not defined for this table (links are set in a multi-table "
                              "schema: \"links\": {name: {\"table\": ...}})")
        return link

    def _perm(self, name: str, size: int) -> Permutation:
        key = f"{name}:{size}"
        perm = self._perms.get(key)
        if perm is None:
            perm = self._perms[key] = Permutation(size, mix64(self.seed, text_key(name)))
        return perm

    def _candidates(self, name: str, link: dict[str, Any], values: Mapping[str, Any]) -> tuple[Any, int, int]:
        """Return (int array, start, length) of the parent rows this record may link to."""
        if not link.get("eligible"):
            return None, 0, int(link["rows"])
        ints = self._map(link["eligible"]).ints
        if not link.get("match"):
            return ints, 0, len(ints) if ints is not None else 0
        groups = self._groups.get(name)
        if groups is None:
            raw = json.loads(Path(link["groups"]).read_text(encoding="utf-8"))
            groups = self._groups[name] = {k: (int(v[0]), int(v[1])) for k, v in raw.items()}
        key = match_key(values.get(child) for child in link["match"])
        start, length = groups.get(key, (0, 0))
        if not length:
            shown = {child: values.get(child) for child in link["match"]}
            raise SchemaError(f"Link {name!r}: no row of table {link['table']!r} matches {shown}")
        return ints, start, length

    def row(self, name: str, index: int, values: Mapping[str, Any]) -> int:
        """Parent row number of the record at global ``index`` through link ``name``."""
        if self.per_parent and name == self.per_parent["link"]:
            return self._parent_of(index)[0]
        link = self._link(name)
        ints, start, length = self._candidates(name, link, values)
        if length <= 0:
            raise SchemaError(f"Link {name!r}: table {link['table']!r} has no rows to link to")
        mode = link.get("mode", "random")
        position = index % self.total
        if mode == "unique":
            pick = self._perm(name, max(length, self.total))(position) % length
        elif mode == "cover":
            spread = self._perm(name, self.total)(position)
            pick = spread if spread < length else mix64(self.seed, text_key(name), index) % length
        else:
            pick = mix64(self.seed, text_key(name), index) % length
        return int(ints[start + pick]) if ints is not None else pick

    def _parent_of(self, index: int) -> tuple[int, int]:
        starts = self._map(self.per_parent["counts"]).ints  # type: ignore[index]
        if starts is None or len(starts) < 2:
            raise SchemaError("The parent table has no rows")
        position = index % max(starts[len(starts) - 1], 1)
        row = bisect_right(starts, position) - 1
        return row, position - starts[row]

    def child_position(self, index: int) -> int:
        """Zero-based position of the record among the children of its parent."""
        if not self.per_parent:
            raise SchemaError("child_index needs a table with \"per_parent\"")
        return self._parent_of(index)[1]

    def parent_row(self, name: str, row: int) -> dict[str, Any]:
        """Stored columns of parent row ``row`` of link ``name``."""
        store = self.per_parent["store"] if self.per_parent and name == self.per_parent["link"] \
            else self._link(name)["store"]
        offsets = self._map(f"{store}.offsets").ints
        data = self._map(f"{store}.jsonl").map
        if offsets is None or data is None or row + 1 >= len(offsets):
            raise SchemaError(f"Link {name!r}: parent row {row} does not exist")
        return json.loads(data[offsets[row]:offsets[row + 1]])

    def value(self, name: str, column: str, index: int, values: Mapping[str, Any]) -> Any:
        """Column ``column`` of the parent row linked to the record at ``index`` (one lookup per record and link)."""
        cache = getattr(self._local, "cache", None)
        if cache is None:
            cache = self._local.cache = {}
        match = (self.links.get(name) or {}).get("match") or ()
        tag = (index, *(normalize_value(values.get(child)) for child in match))
        hit = cache.get(name)
        if hit is None or hit[0] != tag:
            hit = (tag, self.parent_row(name, self.row(name, index, values)))
            cache[name] = hit
        return hit[1].get(column)

    def close(self) -> None:
        """Release every memory map."""
        with self._lock:
            for mapped in self._maps.values():
                mapped.close()
            self._maps.clear()
