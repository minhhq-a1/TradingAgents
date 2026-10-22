"""Backtest of the book in ``trend.py``: 100% hold, 100% SMA200 and the core/trend mix.

Run on any machine with network access (one public Bitstamp endpoint, paged):

    python -m tradingagents.trend_backtest [--start 2014-01-01] [--core 0.5] [--csv prices.csv]

``--csv`` takes a file with Date, Open and Close columns instead of downloading. It uses the same
``trend.simulate`` as the paper book, so the two cannot disagree.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

import pandas as pd
import requests

from tradingagents import trend

BITSTAMP = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"


def bitstamp(start: str) -> pd.DataFrame:
    """Completed UTC daily bars from ``start``, 1000 per request."""
    cursor = int(datetime.fromisoformat(start).replace(tzinfo=UTC).timestamp())
    today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    rows = []
    while cursor < today:
        response = requests.get(BITSTAMP, params={"step": 86400, "limit": 1000, "start": cursor}, timeout=30)
        response.raise_for_status()
        page = response.json()["data"]["ohlc"]
        if not page:
            break
        rows += page
        cursor = int(page[-1]["timestamp"]) + 86400
    frame = pd.DataFrame(rows).drop_duplicates("timestamp")
    frame["Date"] = pd.to_datetime(frame["timestamp"].astype(int), unit="s").dt.normalize()
    frame = frame[frame["Date"] < pd.Timestamp(today, unit="s")]  # the open day is partial
    return frame.set_index("Date")[["open", "close"]].astype(float).rename(columns={"open": "Open", "close": "Close"})


def report(frame: pd.DataFrame, weight: float) -> str:
    book = trend.simulate(frame, weight)
    state = trend.states(frame["Close"])
    flips = int((state != state.shift()).sum() - 1)
    lines = [f"BTC daily {book.index[0].date()} to {book.index[-1].date()}, fee {trend.FEE:.1%} per side, "
             f"signal at close, fill at next open, {flips} state changes",
             f"{'':<22}{'CAGR':>9}{'MaxDD':>9}{'Sharpe':>8}{'Total':>10}"]
    for name, column in (("100% hold", "hold"), ("100% SMA200", "trend"), (f"{weight:.0%} core + {1 - weight:.0%} SMA200", "mix")):
        m = trend.metrics(book[column])
        lines.append(f"{name:<22}{m['cagr']:>9.1%}{m['max_drawdown']:>9.1%}{m['sharpe']:>8.2f}{m['total']:>10.0%}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--core", type=float, default=trend.core_weight())
    parser.add_argument("--csv")
    args = parser.parse_args()
    frame = (pd.read_csv(args.csv, parse_dates=["Date"]).set_index("Date")[["Open", "Close"]]
             if args.csv else bitstamp(args.start))
    print(report(frame, args.core))


if __name__ == "__main__":
    main()
