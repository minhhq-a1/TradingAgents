"""Keyless macro fallback for FRED: market-traded rates, dollar and volatility from Yahoo.

Covers only what trades (Treasury yields, DXY, VIX); data released on a schedule
(CPI, unemployment, GDP) needs FRED.
"""
from datetime import datetime, timedelta

from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv

MACRO_SYMBOLS = {
    "13w_treasury": "^IRX",
    "5y_treasury": "^FVX",
    "10y_treasury": "^TNX",
    "30y_treasury": "^TYX",
    "dollar_index": "DX-Y.NYB",
    "vix": "^VIX",
}
DEFAULT_LOOKBACK_DAYS = 365
MAX_ROWS = 40


def get_macro_data(indicator: str, as_of_date: str, look_back_days: int | None = None) -> str:
    """Same contract as the FRED vendor: a markdown report for one macro series."""
    key = indicator.strip().lower().replace(" ", "_").replace("-", "_")
    symbol = MACRO_SYMBOLS.get(key)
    if symbol is None:
        return (
            f"Yahoo macro: '{indicator}' is not available without FRED. Supported: "
            f"{', '.join(MACRO_SYMBOLS)}."
        )
    start = (datetime.strptime(as_of_date, "%Y-%m-%d")
             - timedelta(days=look_back_days or DEFAULT_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    data = load_ohlcv(symbol, as_of_date, fill_gaps=False)
    points = [(str(d.date()), c) for d, c in zip(data["Date"], data["Close"], strict=True)
              if str(d.date()) >= start and c == c]  # c == c drops NaN
    if not points:
        return f"Yahoo macro: no observations for {symbol} between {start} and {as_of_date}."
    (first_date, first), (last_date, last) = points[0], points[-1]
    pct = f" ({(last - first) / first * 100:+.2f}%)" if first else ""
    note = f"\n_(showing the most recent {MAX_ROWS} of {len(points)} observations)_\n" if len(points) > MAX_ROWS else ""
    table = "\n".join(f"| {d} | {v:.3f} |" for d, v in points[-MAX_ROWS:])
    return (
        f"## Yahoo Finance: {indicator} ({symbol})\n"
        f"- Window: {start} to {as_of_date}\n\n"
        f"**Latest:** {last:.3f} ({last_date}) | "
        f"**Change from {first_date} to {last_date}:** {last - first:+.3f}{pct}, from {first:.3f}\n"
        f"{note}\n| Date | Close |\n| --- | --- |\n{table}\n"
    )
