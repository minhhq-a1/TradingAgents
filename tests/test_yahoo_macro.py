import pandas as pd

from tradingagents.dataflows.vendors.yahoo import macro


def test_known_alias_reports_latest(monkeypatch):
    frame = pd.DataFrame({"Date": pd.to_datetime(["2026-10-19", "2026-10-20"]), "Close": [4.0, 4.5]})
    monkeypatch.setattr(macro, "load_ohlcv", lambda *a, **k: frame)
    out = macro.get_macro_data("10y_treasury", "2026-10-20", 30)
    assert "**Latest:** 4.500 (2026-10-20)" in out and "^TNX" in out


def test_unsupported_alias_says_so():
    assert "not available without FRED" in macro.get_macro_data("cpi", "2026-10-20")
