from __future__ import annotations

import pytest

from kapitalkompass.data.rates import CACHE, curve_at, load_curves, tenor_years


def test_tenor_years() -> None:
    assert tenor_years("3M") == 0.25
    assert tenor_years("10Y") == 10.0


@pytest.mark.skipif(not CACHE.exists(), reason="rate cache not fetched")
def test_cache_schema_and_curve() -> None:
    df = load_curves()
    assert df.height > 1000
    c = curve_at()
    assert c.height == 11
    assert c["tenor_years"].is_sorted()
