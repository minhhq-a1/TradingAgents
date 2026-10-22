"""Backtest of the book in ``trend.py``: 100% hold, 100% SMA200 and the core/trend mix.

Run on any machine with network access (one public Bitstamp endpoint, paged):

    python -m tradingagents.trend_backtest [--start 2014-01-01] [--core 0.5] [--csv prices.csv]

``--csv`` takes a file with Date, Open and Close columns instead of downloading. It uses the same
``trend.simulate`` as the paper book, so the two cannot disagree.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

import pandas as pd
import requests

from tradingagents import trend

BITSTAMP = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"


def bitstamp(start: str) -> pd.DataFrame:
    """Completed UTC daily bars, 1000 per request, from the SMA warm-up before ``start`` so that
    ``start`` itself is the first day the rule can be evaluated."""
    cursor = int((datetime.fromisoformat(start).replace(tzinfo=UTC) - timedelta(days=trend.WINDOW - 1)).timestamp())
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


def report(frame: pd.DataFrame, weight: float, start: str | None = None) -> str:
    """Three separate strategies over the days from ``start`` (default: the first day with a full SMA200)."""
    state = trend.states(frame["Close"])
    if start:
        state = state.loc[pd.Timestamp(start):]
    book = trend.simulate(frame.loc[state.index[0]:], state, weight)
    flips = int((state != state.shift()).sum() - 1)
    lines = [f"BTC daily, evaluated {book.index[0].date()} to {book.index[-1].date()}, fee {trend.FEE:.1%} per side, "
             f"cash on the first day, signal at close and fill at the next open, {flips} state changes",
             f"{'':<46}{'CAGR':>9}{'MaxDD':>9}{'Sharpe':>8}{'Total':>10}"]
    if start and book.index[0] != pd.Timestamp(start):
        lines.insert(1, f"note: no full SMA200 window at {start}; the first day evaluated is {book.index[0].date()}")
    for name, column in (("100% hold (buy-and-hold)", "hold"), ("100% SMA200 / cash (the validated rule)", "trend"),
                         (f"{weight:.0%} core + {1 - weight:.0%} SMA200 (added strategy)", "mix")):
        m = trend.metrics(book[column])
        lines.append(f"{name:<46}{m['cagr']:>9.1%}{m['max_drawdown']:>9.1%}{m['sharpe']:>8.2f}{m['total']:>10.0%}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", help="first evaluated day; default 2014-01-01, or the first full SMA200 day with --csv")
    parser.add_argument("--core", type=float, default=trend.core_weight())
    parser.add_argument("--csv")
    args = parser.parse_args()
    start = args.start or (None if args.csv else "2014-01-01")
    frame = (pd.read_csv(args.csv, parse_dates=["Date"]).set_index("Date")[["Open", "Close"]]
             if args.csv else bitstamp(start))
    print(report(frame, args.core, start))


if __name__ == "__main__":
    main()
