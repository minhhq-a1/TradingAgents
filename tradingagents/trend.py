"""BTC trend rule, paper only: UTC close above its 200-day SMA means hold BTC, otherwise cash.

The rule is fixed, with nothing to tune. ``check`` reads the last *closed* UTC daily bar
(Binance first, Yahoo as the fallback), compares the state with the one saved in the data
volume, and sends a message only when it flipped. The decision is made here from prices; any
text written around it can never change it. No order is ever placed.

The book is split: ``TREND_CORE_WEIGHT`` (default 0.5) of it is held in BTC permanently and the
rest follows the rule. ``simulate`` is the one model of that book, used by ``/paper`` and by
``scripts/core_trend_backtest.py``: a signal read at a close fills at the next open, 0.1% fee per
side, no rebalancing between the two sleeves.

State is saved before a message goes out and cleared once it is delivered, so a restart can
neither skip an alert nor repeat a delivered one. The ledger keeps the detection time and the
send time apart: the backtest fills at the next 00:00 UTC open, which nobody who reads a message
afterwards can get.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from tradingagents.dataflows.vendors.binance_klines import fetch_daily_klines
from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv

logger = logging.getLogger(__name__)

WINDOW = 200
FEE = 0.001  # per side, as in the validation
WATCH = ("BTCUSDT",)  # the only symbol the rule was validated on
RUN_AT = (0, 5)  # UTC hour, minute: the daily bar is final a few minutes after midnight
RETRY_SECONDS, RETRIES = 600, 12  # a late bar or a failed fetch is retried for about two hours
DIR = os.path.join(os.path.expanduser("~"), ".tradingagents", "trend")

HOLD, CASH = "HOLD", "CASH"
WHIPSAW = (
    "Whipsaw warning: in the 2014-2026 Bitstamp validation, 29 of 44 closed trades lost money and "
    "lasted 14 days or less. The rule helps by sitting out long bear markets (max drawdown -67% "
    "against -83% for buy-and-hold), not by being right on each signal."
)


class StaleData(Exception):
    """The newest closed bar is older than yesterday (UTC), so the signal is not for today."""


def core_weight() -> float:
    """Share of the book held in BTC permanently (env ``TREND_CORE_WEIGHT``, default 0.5)."""
    weight = float(os.environ.get("TREND_CORE_WEIGHT", "0.5"))
    if not 0 <= weight <= 1:
        raise ValueError(f"TREND_CORE_WEIGHT must be between 0 and 1, got {weight}")
    return weight


def bars(symbol: str, today: datetime | None = None) -> pd.DataFrame:
    """Open and Close of the completed UTC days, oldest first, indexed by date."""
    today = (today or datetime.now(UTC)).date()
    frame = fetch_daily_klines(symbol, (today - timedelta(days=900)).isoformat(), today.isoformat())
    if frame is None:  # Binance unreachable: the fallback, with the open day cut off
        frame = load_ohlcv(symbol, (today - timedelta(days=1)).isoformat()).set_index("Date")
        frame = frame[frame.index.normalize().date < today]
    out = frame[["Open", "Close"]].astype(float)
    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    return out


def states(close: pd.Series) -> pd.Series:
    """HOLD or CASH for each day with a full SMA200; earlier days are left out."""
    sma = close.rolling(WINDOW).mean()
    return pd.Series(HOLD, index=close.index).where(close > sma, CASH)[sma.notna()]


def simulate(frame: pd.DataFrame, weight: float = 0.5, fee: float = FEE) -> pd.DataFrame:
    """Equity of the hold sleeve, the trend sleeve and their mix, each 1.0 at the first signal day.

    The signal read at a close fills at the next open. A day's return is close to close while the
    position carries over, open to close on the day it is bought, and previous close to open on
    the day it is sold; each trade costs ``fee``. The sleeves are never rebalanced.
    """
    state = states(frame["Close"])
    frame = frame.loc[state.index[0]:]
    held = (state == HOLD).astype(float)
    during = held.shift(fill_value=held.iloc[0])  # the signal of the previous close, in force all day
    before = during.shift(fill_value=held.iloc[0])  # what was carried in from the previous close
    prev, open_, close = frame["Close"].shift(), frame["Open"], frame["Close"]
    gross = np.select(
        [(before == 1) & (during == 1), (before == 1) & (during == 0), (before == 0) & (during == 1)],
        [close / prev - 1, open_ / prev - 1, close / open_ - 1], 0.0)
    ret = pd.Series(gross, index=frame.index)
    ret = ((1 + ret) * (1 - fee * (before != during))) - 1
    ret.iloc[0] = 0.0
    trend = (1 + ret).cumprod() * (1 - fee * held.iloc[0])  # the opening position is bought at the first close
    hold = close / close.iloc[0] * (1 - fee)
    return pd.DataFrame({"hold": hold, "trend": trend, "mix": weight * hold + (1 - weight) * trend})


def metrics(equity: pd.Series) -> dict:
    daily = equity.pct_change().dropna()
    years = len(daily) / 365
    return {"cagr": equity.iloc[-1] ** (1 / years) - 1, "max_drawdown": (equity / equity.cummax() - 1).min(),
            "sharpe": daily.mean() / daily.std() * np.sqrt(365), "total": equity.iloc[-1] - 1}


def latest(symbol: str, today: datetime | None = None, frame: pd.DataFrame | None = None) -> dict:
    frame = frame if frame is not None else bars(symbol, today)
    close = frame["Close"]
    state = states(close)
    if state.empty:
        raise StaleData(f"{symbol}: fewer than {WINDOW} closed daily bars")
    day = state.index[-1]
    if day.date() < ((today or datetime.now(UTC)).date() - timedelta(days=1)):
        raise StaleData(f"{symbol}: newest closed bar is {day.date()}")
    run = state[state != state.iloc[-1]]  # the streak began the day after the last opposite state
    since = state.index[state.index.get_loc(run.index[-1]) + 1] if len(run) else state.index[0]
    return {"symbol": symbol, "state": state.iloc[-1], "date": day.date().isoformat(), "close": float(close.iloc[-1]),
            "sma": float(close.rolling(WINDOW).mean().iloc[-1]), "since": since.date().isoformat()}


def _load() -> dict:
    try:
        with open(os.path.join(DIR, "state.json"), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _save(data: dict) -> None:
    os.makedirs(DIR, exist_ok=True)
    path = os.path.join(DIR, "state.json")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(path + ".tmp", path)  # a crash mid-write leaves the old state, not half a file


def _log(event: str, **fields) -> None:
    os.makedirs(DIR, exist_ok=True)
    with open(os.path.join(DIR, "log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": event, "at": datetime.now(UTC).isoformat(timespec="seconds"), **fields}) + "\n")


def _distance(info: dict) -> str:
    return f"UTC close {info['date']}: {info['close']:,.0f} vs SMA200 {info['sma']:,.0f} ({info['close'] / info['sma'] - 1:+.1%})."


def _split() -> str:
    weight = core_weight()
    return f"Book split: {weight:.0%} stays in BTC permanently, {1 - weight:.0%} follows this rule."


def alert_text(info: dict, detected_at: str) -> str:
    entering = info["state"] == HOLD
    return "\n".join([
        f"{'🟢' if entering else '🔴'} {info['symbol']} trend turned {'UP: rule says HOLD BTC' if entering else 'DOWN: rule says CASH'}",
        _distance(info),
        "Rule: UTC close above SMA200 = hold BTC, otherwise cash. Paper signal only, no order was placed.",
        _split(),
        f"Detected {detected_at[11:16]} UTC. The backtest fills at the 00:00 UTC open after this close; you cannot "
        "get that price, so a real fill will differ.",
        WHIPSAW,
    ])


def status_text(symbol: str = "BTCUSDT") -> str:
    if symbol.upper() not in WATCH:
        return f"The trend rule was only validated on {', '.join(WATCH)}."
    try:
        info = latest(symbol.upper())
    except Exception as exc:  # report, keep serving
        return f"Trend data unavailable: {exc}"
    now = "HOLD BTC" if info["state"] == HOLD else "CASH"
    return "\n".join([
        f"{info['symbol']} trend (paper): {now} since {info['since']}",
        _distance(info),
        "Rule: UTC close above SMA200 = hold BTC, otherwise cash. Fixed, no parameters.",
        _split(),
        WHIPSAW,
    ])


def paper_text(symbol: str = "BTCUSDT", today: datetime | None = None) -> str:
    """The paper book since the first check: ledger of flips and equity against buy-and-hold."""
    entry = _load().get(symbol)
    if entry is None:
        return "No paper book yet: the first daily check has not run."
    try:
        frame = bars(symbol, today)
        info = latest(symbol, today, frame)
    except Exception as exc:
        return f"Paper data unavailable: {exc}"
    start = pd.Timestamp(entry["start"]["date"])
    weight = core_weight()
    book = simulate(frame, weight).loc[start:]
    book = book / book.iloc[0]  # shortcut: the opening purchase fee is left out; fees are 0.1% on a book that runs months
    mix, hold = book["mix"].iloc[-1] * 100 - 100, book["hold"].iloc[-1] * 100 - 100
    in_btc = weight + (1 - weight) * (info["state"] == HOLD)
    lines = [
        f"{symbol} paper book since {entry['start']['date']} ({len(book) - 1} days, no real orders)",
        f"Now: {in_btc:.0%} in BTC ({weight:.0%} core + trend sleeve {info['state']}). {_distance(info)}",
        f"Equity: mix {mix:+.1f}% | buy-and-hold {hold:+.1f}% | trend only {book['trend'].iloc[-1] * 100 - 100:+.1f}% (0.1% fee per side, fills at next open)",
        "Ledger (signal close -> next open fill, detected / sent UTC):",
    ]
    for row in entry["ledger"][-5:]:
        fill = "pending" if row["fill_open"] is None else f"{row['fill_open']:,.0f}"
        sent = row["sent_at"][5:16].replace("T", " ") if row["sent_at"] else "not yet"
        detected = row["detected_at"][5:16].replace("T", " ")
        lines.append(f"  {row['signal_date']} {row['state']}: close {row['signal_close']:,.0f} -> open {fill} "
                     f"({row['fill_date']}), detected {detected}, sent {sent}")
    if not entry["ledger"]:
        lines.append("  no flips yet")
    lines.append("Reference only: " + WHIPSAW)
    return "\n".join(lines)


def check(symbol: str, notify, today: datetime | None = None) -> bool:
    """Compare today's state with the saved one and alert on a flip. True when an alert went out."""
    frame = bars(symbol, today)
    info = latest(symbol, today, frame)
    data = _load()
    saved = data.get(symbol)
    if saved is None:  # first run: record where we are, alert on nothing
        data[symbol] = {"state": info["state"], "checked": info["date"], "pending": None, "ledger": [],
                        "start": {"date": info["date"], "close": info["close"], "state": info["state"]}}
        _save(data)
        _log("baseline", **info)
        return False

    for row in saved["ledger"]:  # the fill price is known once the fill day's bar has closed
        if row["fill_open"] is None and pd.Timestamp(row["fill_date"]) in frame.index:
            row["fill_open"] = float(frame.loc[pd.Timestamp(row["fill_date"]), "Open"])
    if info["state"] != saved["state"]:  # a newer flip replaces an undelivered older one
        detected = datetime.now(UTC).isoformat(timespec="seconds")
        fill_date = (pd.Timestamp(info["date"]) + timedelta(days=1)).date().isoformat()
        saved["ledger"].append({"signal_date": info["date"], "state": info["state"], "signal_close": info["close"],
                                "sma200": info["sma"], "fill_date": fill_date, "fill_open": None,
                                "detected_at": detected, "sent_at": None})
        saved.update(state=info["state"], pending={"info": info, "detected_at": detected, "row": len(saved["ledger"]) - 1})
        _save(data)  # before sending: a crash now resends once, it cannot lose the flip
        _log("detected", detected_at=detected, **info)
    saved["checked"] = info["date"]
    pending = saved["pending"]
    if not pending:
        _save(data)
        return False
    try:
        notify(alert_text(pending["info"], pending["detected_at"]))
    except Exception as exc:  # stays pending, tried again at the next check
        _log("send_failed", error=repr(exc), **pending["info"])
        _save(data)
        raise
    sent_at = datetime.now(UTC).isoformat(timespec="seconds")
    saved["ledger"][pending["row"]]["sent_at"] = sent_at
    saved["pending"] = None
    _save(data)
    _log("sent", detected_at=pending["detected_at"], sent_at=sent_at, **pending["info"])
    return True


def _seconds_to_run(now: datetime) -> float:
    run = now.replace(hour=RUN_AT[0], minute=RUN_AT[1], second=0, microsecond=0)
    return ((run if run > now else run + timedelta(days=1)) - now).total_seconds()


def watch(notify) -> None:
    """Check at start-up (catch-up), then every day just after 00:00 UTC. Never returns."""
    while True:
        for _ in range(RETRIES):
            try:
                for symbol in WATCH:
                    check(symbol, notify)
                break
            except Exception as exc:  # a late bar or a network error: try again shortly
                logger.warning("trend check failed: %r", exc)
                time.sleep(RETRY_SECONDS)
        time.sleep(_seconds_to_run(datetime.now(UTC)))
