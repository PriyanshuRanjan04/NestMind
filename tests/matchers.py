"""
Entity matching utilities used by both tests and evaluation scripts.

Rules (from fixtures/README.md):
  - tickers:       case-folded, exchange suffix optional
  - topics/sectors: case-folded, substring match per element
  - amount/rate:   within ±5%
  - period_years:  exact
  - currency:      ISO 4217 exact
  - index, action, goal, frequency, horizon, time_period: exact
"""
from __future__ import annotations

from typing import Any


def _normalise_ticker(t: str) -> str:
    """Strip exchange suffix and lowercase for comparison."""
    return t.split(".")[0].upper()


def _ticker_match(expected: str, actual_list: list[str]) -> bool:
    """Check if *expected* ticker is present in *actual_list* (suffix-tolerant)."""
    exp_norm = _normalise_ticker(expected)
    return any(_normalise_ticker(a) == exp_norm for a in actual_list)


def _within_pct(expected: float, actual: float, pct: float = 5.0) -> bool:
    if expected == 0:
        return abs(actual) < 1e-9
    return abs(actual - expected) / abs(expected) * 100 <= pct


def entity_subset_match(
    expected_entities: dict[str, Any],
    actual_entities: dict[str, Any],
) -> tuple[bool, list[str]]:
    """
    Returns (passed: bool, failures: list[str]).
    *actual_entities* may be a superset of *expected_entities*.
    """
    failures: list[str] = []

    for field, expected_value in expected_entities.items():
        actual_value = actual_entities.get(field)

        if field == "tickers":
            if actual_value is None:
                failures.append(f"tickers: expected {expected_value}, got None")
                continue
            for exp_ticker in expected_value:
                if not _ticker_match(exp_ticker, actual_value):
                    failures.append(
                        f"tickers: expected '{exp_ticker}' not found in {actual_value}"
                    )

        elif field in ("topics", "sectors"):
            if actual_value is None:
                failures.append(f"{field}: expected {expected_value}, got None")
                continue
            for exp_item in expected_value:
                exp_norm = exp_item.lower()
                found = any(exp_norm in a.lower() for a in actual_value)
                if not found:
                    failures.append(
                        f"{field}: '{exp_item}' not found in {actual_value}"
                    )

        elif field in ("amount", "rate"):
            if actual_value is None:
                failures.append(f"{field}: expected {expected_value}, got None")
                continue
            if not _within_pct(float(expected_value), float(actual_value)):
                failures.append(
                    f"{field}: expected {expected_value} ±5%, got {actual_value}"
                )

        elif field == "period_years":
            if actual_value is None:
                failures.append(f"period_years: expected {expected_value}, got None")
                continue
            if int(actual_value) != int(expected_value):
                failures.append(
                    f"period_years: expected {expected_value}, got {actual_value}"
                )

        else:
            # Exact match for: currency, index, action, goal, frequency,
            #                  horizon, time_period
            if actual_value is None:
                failures.append(f"{field}: expected '{expected_value}', got None")
                continue
            if str(actual_value).strip() != str(expected_value).strip():
                failures.append(
                    f"{field}: expected '{expected_value}', got '{actual_value}'"
                )

    return (len(failures) == 0), failures
