"""Curated city/state reference data for geographically consistent records.

Faker generates cities and states independently, which produces impossible
combinations (e.g. "Pune, Kerala"). This table keeps them consistent. Weights
roughly follow metropolitan population so category frequencies look natural.
"""

from __future__ import annotations

from functools import lru_cache

# (city, state, relative weight)
_CITIES: dict[str, tuple[tuple[str, str, float], ...]] = {
    "en_IN": (
        ("Mumbai", "Maharashtra", 20.7), ("Delhi", "Delhi", 19.0), ("Bengaluru", "Karnataka", 13.2),
        ("Hyderabad", "Telangana", 10.5), ("Ahmedabad", "Gujarat", 8.5), ("Chennai", "Tamil Nadu", 11.5),
        ("Kolkata", "West Bengal", 15.1), ("Pune", "Maharashtra", 7.4), ("Jaipur", "Rajasthan", 4.1),
        ("Surat", "Gujarat", 7.8), ("Lucknow", "Uttar Pradesh", 3.9), ("Kanpur", "Uttar Pradesh", 3.2),
        ("Nagpur", "Maharashtra", 2.9), ("Indore", "Madhya Pradesh", 3.2), ("Bhopal", "Madhya Pradesh", 2.4),
        ("Thane", "Maharashtra", 2.5), ("Visakhapatnam", "Andhra Pradesh", 2.3), ("Patna", "Bihar", 2.5),
        ("Vadodara", "Gujarat", 2.2), ("Ludhiana", "Punjab", 1.9), ("Agra", "Uttar Pradesh", 1.8),
        ("Nashik", "Maharashtra", 1.9), ("Coimbatore", "Tamil Nadu", 2.8), ("Kochi", "Kerala", 2.4),
        ("Thiruvananthapuram", "Kerala", 1.8), ("Chandigarh", "Chandigarh", 1.2), ("Guwahati", "Assam", 1.1),
        ("Bhubaneswar", "Odisha", 1.0), ("Mysuru", "Karnataka", 1.1), ("Madurai", "Tamil Nadu", 1.6),
        ("Noida", "Uttar Pradesh", 1.0), ("Gurugram", "Haryana", 1.2), ("Dehradun", "Uttarakhand", 0.8),
        ("Ranchi", "Jharkhand", 1.2), ("Raipur", "Chhattisgarh", 1.1), ("Vijayawada", "Andhra Pradesh", 1.5),
        ("Mangaluru", "Karnataka", 0.7), ("Jodhpur", "Rajasthan", 1.1), ("Amritsar", "Punjab", 1.2),
        ("Goa", "Goa", 0.6),
    ),
    "en_US": (
        ("New York", "New York", 19.5), ("Los Angeles", "California", 12.9), ("Chicago", "Illinois", 9.4),
        ("Houston", "Texas", 7.1), ("Dallas", "Texas", 7.6), ("Philadelphia", "Pennsylvania", 6.2),
        ("Miami", "Florida", 6.1), ("Atlanta", "Georgia", 6.1), ("Washington", "District of Columbia", 6.3),
        ("Boston", "Massachusetts", 4.9), ("Phoenix", "Arizona", 4.9), ("San Francisco", "California", 4.7),
        ("Seattle", "Washington", 4.0), ("Detroit", "Michigan", 4.3), ("Minneapolis", "Minnesota", 3.7),
        ("San Diego", "California", 3.3), ("Tampa", "Florida", 3.2), ("Denver", "Colorado", 3.0),
        ("Baltimore", "Maryland", 2.8), ("St. Louis", "Missouri", 2.8), ("Orlando", "Florida", 2.7),
        ("Charlotte", "North Carolina", 2.7), ("San Antonio", "Texas", 2.6), ("Portland", "Oregon", 2.5),
        ("Austin", "Texas", 2.4), ("Pittsburgh", "Pennsylvania", 2.4), ("Las Vegas", "Nevada", 2.3),
        ("Columbus", "Ohio", 2.1), ("Nashville", "Tennessee", 2.0), ("Salt Lake City", "Utah", 1.3),
    ),
    "en_GB": (
        ("London", "England", 14.0), ("Manchester", "England", 2.8), ("Birmingham", "England", 2.9),
        ("Leeds", "England", 1.9), ("Glasgow", "Scotland", 1.7), ("Liverpool", "England", 1.5),
        ("Bristol", "England", 1.0), ("Edinburgh", "Scotland", 0.9), ("Cardiff", "Wales", 0.5),
        ("Belfast", "Northern Ireland", 0.6), ("Sheffield", "England", 1.4), ("Newcastle", "England", 1.1),
        ("Nottingham", "England", 0.9), ("Southampton", "England", 0.9), ("Aberdeen", "Scotland", 0.3),
    ),
}


@lru_cache(maxsize=None)
def supported_geo_locales() -> tuple[str, ...]:
    """Return locales with curated city/state data."""
    return tuple(sorted(_CITIES))


def cities(locale: str) -> tuple[tuple[str, str, float], ...]:
    """Return ``(city, state, weight)`` tuples for ``locale``.

    Args:
        locale: Faker locale such as ``en_IN``.

    Returns:
        The curated tuples.

    Raises:
        KeyError: If the locale has no curated data.
    """
    return _CITIES[locale]


@lru_cache(maxsize=None)
def state_of(locale: str) -> dict[str, str]:
    """Return a ``city -> state`` mapping for ``locale``."""
    return {city: state for city, state, _ in _CITIES[locale]}


@lru_cache(maxsize=None)
def states(locale: str) -> dict[str, float]:
    """Return ``state -> total weight`` for ``locale``."""
    totals: dict[str, float] = {}
    for _, state, weight in _CITIES[locale]:
        totals[state] = totals.get(state, 0.0) + weight
    return totals
