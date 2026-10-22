"""BTC trend rule, paper only: UTC close above its 200-day SMA means hold BTC, otherwise cash.

The rule is fixed, with nothing to tune. ``check`` reads the last *closed* UTC daily bars
(Binance first, Yahoo as the fallback), turns each new bar into a state, and messages Telegram
only when the state flips. The decision is made here from prices; any text written around it can
never change it. No order is ever placed.

Two things are kept apart. The *rule* is what the SMA200 says on every closed day. The *operated*
book is what this bot actually saw: a day processed late, after the bot was down, is a day nobody
could act on, so the operated book holds its position through an outage and catches up when the
bot is back. ``/paper`` shows both, from the bars stored when each day was processed, so a later
data revision or a changed ``TREND_CORE_WEIGHT`` cannot rewrite the past.

The book starts as cash at the close of its first day and every signal, the first included,
fills at the next open: 0.1% fee per side, ``TREND_CORE_WEIGHT`` (default 0.5) of the capital in
a BTC sleeve that is never sold, the rest in a sleeve that follows the rule, no rebalancing.
``simulate`` is that one model, shared with ``tradingagents.trend_backtest``.

State is saved before a message goes out and each chat's delivery is recorded on its own, so a
restart can neither lose a flip nor repeat a delivered one, and one failing chat blocks no other.
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
REVISION = 0.005  # a stored close that a later fetch moves by more than this is a data event
DIR = os.path.join(os.path.expanduser("~"), ".tradingagents", "trend")

HOLD, CASH = "HOLD", "CASH"
WHIPSAW = (
    "Whipsaw warning: in the 2014-2026 Bitstamp validation, 29 of 44 closed trades lost money and "
    "lasted 14 days or less. The rule helps by sitting out long bear markets (max drawdown -67% "
    "against -83% for buy-and-hold), not by being right on each signal."
)


class StaleData(Exception):
    """The newest closed bar is older than yesterday (UTC), so the signal is not for today."""


class DataGap(Exception):
    """A calendar day is missing from the bars the signal needs, so no signal is made."""


class SourceUnverified(Exception):
    """The bars come from another source than the book's and do not reproduce what it stored: no signal."""


class SendFailed(Exception):
    """At least one chat did not get a message; the others did, and only this one is retried."""


def core_weight() -> float:
    """Share of the book held in BTC permanently (env ``TREND_CORE_WEIGHT``, default 0.5)."""
    weight = float(os.environ.get("TREND_CORE_WEIGHT", "0.5"))
    if not 0 <= weight <= 1:
        raise ValueError(f"TREND_CORE_WEIGHT must be between 0 and 1, got {weight}")
    return weight


def bars(symbol: str, today: datetime | None = None) -> tuple[pd.DataFrame, str]:
    """Open and Close of the completed UTC days, oldest first, indexed by date, and where they came from."""
    today = (today or datetime.now(UTC)).date()
    frame = fetch_daily_klines(symbol, (today - timedelta(days=900)).isoformat(), today.isoformat())
    source = "binance"
    if frame is None:  # Binance unreachable: the fallback, with the open day cut off and no gaps filled in
        source = "yahoo"
        frame = load_ohlcv(symbol, (today - timedelta(days=1)).isoformat(), fill_gaps=False).set_index("Date")
        frame = frame[frame.index.normalize().date < today]
    out = frame[["Open", "Close"]].astype(float)
    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    return out, source


def require_continuous(frame: pd.DataFrame, extra: int = 0) -> None:
    """Every calendar day of the SMA window (plus ``extra`` newer days) must be present."""
    last = frame.index[-1]
    expected = pd.date_range(end=last, periods=WINDOW + extra, freq="D")
    missing = expected.difference(frame.index)
    if len(missing):
        raise DataGap(f"{len(missing)} day(s) missing from the last {len(expected)}, first {missing[0].date()}")


def states(close: pd.Series) -> pd.Series:
    """HOLD or CASH for each day with a full SMA200; earlier days are left out."""
    sma = close.rolling(WINDOW).mean()
    return pd.Series(HOLD, index=close.index).where(close > sma, CASH)[sma.notna()]


def _sleeve(frame: pd.DataFrame, held: pd.Series, fee: float) -> pd.Series:
    """Equity of one sleeve: cash on the first day, a signal read at a close fills at the next open."""
    during = held.shift(fill_value=0.0)  # the previous close's signal, in force all day
    before = during.shift(fill_value=0.0)  # what was carried in from the previous close
    prev, open_, close = frame["Close"].shift(), frame["Open"], frame["Close"]
    gross = np.select(
        [(before == 1) & (during == 1), (before == 1) & (during == 0), (before == 0) & (during == 1)],
        [close / prev - 1, open_ / prev - 1, close / open_ - 1], 0.0)
    ret = ((1 + pd.Series(gross, index=frame.index)) * (1 - fee * (before != during))) - 1
    ret.iloc[0] = 0.0
    return (1 + ret).cumprod()


def simulate(frame: pd.DataFrame, state: pd.Series, weight: float = 0.5, fee: float = FEE) -> pd.DataFrame:
    """Equity of the hold sleeve, the trend sleeve and their mix, each 1.0 (all cash) on the first day.

    ``state`` is the HOLD/CASH decided at each close, on the same days as ``frame``. A day's return
    is close to close while the position carries over, open to close on the day it is bought, and
    previous close to open on the day it is sold; each trade costs ``fee``.
    """
    assert frame.index.equals(state.index), "frame and state must cover the same days"
    hold = _sleeve(frame, pd.Series(1.0, index=frame.index), fee)
    trend = _sleeve(frame, (state == HOLD).astype(float), fee)
    return pd.DataFrame({"hold": hold, "trend": trend, "mix": weight * hold + (1 - weight) * trend})


def metrics(equity: pd.Series) -> dict:
    daily = equity.pct_change().dropna()
    years = len(daily) / 365
    return {"cagr": equity.iloc[-1] ** (1 / years) - 1, "max_drawdown": (equity / equity.cummax() - 1).min(),
            "sharpe": daily.mean() / daily.std() * np.sqrt(365), "total": equity.iloc[-1] - 1}


def latest(symbol: str, today: datetime | None = None, frame: pd.DataFrame | None = None) -> dict:
    frame = frame if frame is not None else bars(symbol, today)[0]
    close = frame["Close"]
    state = states(close)
    if state.empty:
        raise StaleData(f"{symbol}: fewer than {WINDOW} closed daily bars")
    day = state.index[-1]
    if day.date() < ((today or datetime.now(UTC)).date() - timedelta(days=1)):
        raise StaleData(f"{symbol}: newest closed bar is {day.date()}")
    require_continuous(frame)
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


def _split(weight: float) -> str:
    return (f"Starting split: {weight:.0%} core BTC (never sold), {1 - weight:.0%} follows this rule; "
            "the two parts are not rebalanced.")


def _head(state: str) -> str:
    return f"{'🟢' if state == HOLD else '🔴'} BTCUSDT trend turned {'UP: rule says HOLD BTC' if state == HOLD else 'DOWN: rule says CASH'}"


def alert_text(info: dict, detected_at: str, weight: float | None = None) -> str:
    return "\n".join([
        f"{_head(info['state'])}",
        _distance(info),
        "Rule: UTC close above SMA200 = hold BTC, otherwise cash. Paper signal only, no order was placed.",
        _split(core_weight() if weight is None else weight),
        f"Detected {detected_at[11:16]} UTC. The backtest fills at the 00:00 UTC open after this close; you cannot "
        "get that price, so a real fill will differ.",
        WHIPSAW,
    ])


def late_text(flips: list[dict], info: dict, outage: dict, detected_at: str, weight: float) -> str:
    """One message for signal changes found after the bot was offline: nobody could act on them on time."""
    changes = [f"  {f['signal_date']}: {f['state']} (close {f['signal_close']:,.0f})" for f in flips]
    return "\n".join([
        f"⚠️ BTCUSDT trend check was offline from {outage['from']} to {outage['to']} ({outage['days']} days); "
        f"{len(flips)} signal change(s) were found late, at {detected_at[11:16]} UTC, and could not be acted on in time:",
        *changes,
        f"Now: {'HOLD BTC' if info['state'] == HOLD else 'CASH'}. {_distance(info)}",
        "The paper book shows two lines: the rule (every change filled at its next open) and as operated "
        "(position held through the outage, moved to the current state when the bot was back).",
        _split(weight),
        WHIPSAW,
    ])


def status_text(symbol: str = "BTCUSDT") -> str:
    if symbol.upper() not in WATCH:
        return f"The trend rule was only validated on {', '.join(WATCH)}."
    try:
        info = latest(symbol.upper())
        weight = (_load().get(symbol.upper()) or {}).get("core", core_weight())
    except Exception as exc:  # report, keep serving
        return f"Trend data unavailable: {exc}"
    now = "HOLD BTC" if info["state"] == HOLD else "CASH"
    return "\n".join([
        f"{info['symbol']} trend (paper): {now} since {info['since']}",
        _distance(info),
        "Rule: UTC close above SMA200 = hold BTC, otherwise cash. Fixed, no parameters.",
        _split(weight),
        WHIPSAW,
    ])


def _book(entry: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """The stored days as bars, the rule's state on each, and the state the operated book held."""
    days = pd.DataFrame(entry["days"]).set_index("date")
    days.index = pd.DatetimeIndex(days.index)
    held = []
    for state, live in zip(days["state"], days["live"], strict=True):
        held.append(state if live or not held else held[-1])  # a day processed late is not acted on
    return days[["Open", "Close"]].astype(float), days["state"], pd.Series(held, index=days.index)


def paper_text(symbol: str = "BTCUSDT") -> str:
    """The paper book since the first check, from the stored days: no fetch, no revision, no env."""
    entry = _load().get(symbol)
    if entry is None:
        return "No paper book yet: the first daily check has not run."
    frame, rule, operated = _book(entry)
    weight = entry["core"]
    if len(frame) < 2:
        notes = "".join(f"\n⚠ Data check: {event}" for event in entry["data_events"][-2:])
        return f"{symbol} paper book opened {entry['start']['date']} as cash; no day has closed since, so no equity yet.{notes}"
    ideal, ops = simulate(frame, rule, weight), simulate(frame, operated, weight)

    def pct(equity: pd.Series) -> str:
        return f"{equity.iloc[-1] * 100 - 100:+.1f}%"

    in_btc = operated.shift(fill_value=CASH).iloc[-1] == HOLD  # filled: yesterday's signal, bought at today's open
    core, sleeve = weight * ops["hold"].iloc[-1], (1 - weight) * ops["trend"].iloc[-1] * in_btc
    total = ops["mix"].iloc[-1]
    pending = ""
    if operated.iloc[-1] != (HOLD if in_btc else CASH):
        pending = f" Pending at the next open: trend sleeve to {operated.iloc[-1]} (signal of {frame.index[-1].date()})."
    lines = [
        f"{symbol} paper book since {entry['start']['date']} ({len(frame) - 1} days, no real orders), "
        f"core {weight:.0%} fixed at the start",
        f"Now, filled through {frame.index[-1].date()}: {(core + sleeve) / total:.1%} of the book in BTC "
        f"(core {core / total:.1%} + trend sleeve {sleeve / total:.1%}), the rest cash.{pending} "
        "Sleeves start as cash and fill at the next open, 0.1% fee per side, no rebalancing.",
        f"Equity, rule (every change filled): mix {pct(ideal['mix'])} | 100% SMA200 {pct(ideal['trend'])} | buy-and-hold {pct(ideal['hold'])}",
    ]
    if entry["gaps"]:
        lines.append(f"Equity, as operated: mix {pct(ops['mix'])} | 100% SMA200 {pct(ops['trend'])} "
                     f"(⚠ {sum(g['days'] for g in entry['gaps'])} offline day(s), position held through them)")
    for event in entry["data_events"][-2:]:
        lines.append(f"⚠ Data check: {event}")
    lines.append("Ledger (signal close -> next open fill; detected / sent UTC):")
    for row in entry["ledger"][-5:]:
        fill = "pending" if row["fill_open"] is None else f"{row['fill_open']:,.0f}"
        sent = min(row["sent"].values())[5:16].replace("T", " ") if row["sent"] else "not yet"
        lines.append(f"  {row['signal_date']} {row['state']}{' (late)' if row['late'] else ''}: close {row['signal_close']:,.0f} "
                     f"-> open {fill} ({row['fill_date']}), detected {row['detected_at'][5:16].replace('T', ' ')}, sent {sent}")
    if not entry["ledger"]:
        lines.append("  no flips yet")
    lines.append("Reference only: " + WHIPSAW)
    return "\n".join(lines)


def _day(date: pd.Timestamp, frame: pd.DataFrame, state: pd.Series, sma: pd.Series, source: str, live: bool) -> dict:
    return {"date": date.date().isoformat(), "Open": float(frame.loc[date, "Open"]), "Close": float(frame.loc[date, "Close"]),
            "sma": float(sma[date]), "state": state[date], "source": source, "live": live}


def _data_events(entry: dict, frame: pd.DataFrame, source: str) -> None:
    """A stored close that a later fetch moved is logged and shown, never acted on."""
    notes = []
    for day in entry["days"]:
        stamp = pd.Timestamp(day["date"])
        if stamp in frame.index and abs(frame.loc[stamp, "Close"] / day["Close"] - 1) > REVISION and day["date"] not in entry["revised"]:
            entry["revised"].append(day["date"])
            notes.append(f"close of {day['date']} is now {frame.loc[stamp, 'Close']:,.0f}, stored {day['Close']:,.0f} (kept)")
    for note in notes:
        entry["data_events"].append(note)
        _log("data_event", note=note, source=source)


def _verify_source(entry: dict, frame: pd.DataFrame, source: str, state: pd.Series, sma: pd.Series, new) -> str:
    """Another source is used only if it reproduces the last stored day. Returns why not, or ''."""
    last = entry["days"][-1]
    stamp = pd.Timestamp(last["date"])
    if stamp not in frame.index or pd.isna(sma.get(stamp)):
        return f"{last['date']} is not in the {source} bars"
    close, level = float(frame.loc[stamp, "Close"]), float(sma[stamp])
    if abs(close / last["Close"] - 1) > REVISION or abs(level / last["sma"] - 1) > REVISION or state[stamp] != last["state"]:
        return (f"{source} has {last['date']} at close {close:,.1f}, SMA200 {level:,.1f}, {state[stamp]}; "
                f"the book stored {last['Close']:,.1f}, {last['sma']:,.1f}, {last['state']}")
    current = entry["state"]
    for date in new:  # a flip inside the gap between the two sources' SMAs may exist in one of them only
        if state[date] != current and abs(frame.loc[date, "Close"] / sma[date] - 1) <= REVISION:
            return f"{source} flips to {state[date]} on {date.date()} within {REVISION:.1%} of the SMA200"
        current = state[date]
    return ""


def check(symbol: str, send, chats, today: datetime | None = None) -> bool:
    """Process every closed day since the last check and alert on flips.

    ``send(chat, text)`` delivers to one chat. True when an alert was newly delivered to every chat.
    """
    now = today or datetime.now(UTC)
    frame, source = bars(symbol, now)
    close = frame["Close"]
    state = states(close)
    sma = close.rolling(WINDOW).mean()
    if state.empty:
        raise StaleData(f"{symbol}: fewer than {WINDOW} closed daily bars")
    newest = state.index[-1]
    if newest.date() < now.date() - timedelta(days=1):
        raise StaleData(f"{symbol}: newest closed bar is {newest.date()}")
    data = _load()
    entry = data.get(symbol)
    if entry is None:  # first run: the book opens as cash at this close, and nothing is alerted
        require_continuous(frame)
        data[symbol] = entry = {
            "core": core_weight(), "source": source, "state": state[newest], "checked": newest.date().isoformat(),
            "start": {"date": newest.date().isoformat(), "close": float(close[newest])},
            "days": [_day(newest, frame, state, sma, source, True)], "ledger": [], "gaps": [], "data_events": [],
            "revised": [], "pending": []}
        _save(data)
        _log("baseline", source=source, date=newest.date().isoformat())
        return False

    new = state.index[state.index > pd.Timestamp(entry["checked"])]
    require_continuous(frame, len(new))
    if source != entry["source"]:
        why = _verify_source(entry, frame, source, state, sma, new)
        note = f"source changed from {entry['source']} to {source}" + (f" and does not match: {why}" if why else " (verified)")
        if note not in entry["data_events"][-1:]:
            entry["data_events"].append(note)
            _log("data_event", note=note, source=source)
        if why:  # no signal from numbers the book cannot reproduce; the stored source returning ends this
            _save(data)
            raise SourceUnverified(note)
        entry["source"] = source
    _data_events(entry, frame, source)
    flips = []
    for date in new:
        live = date == newest  # only the newest bar is processed on time; the rest were missed while offline
        entry["days"].append(_day(date, frame, state, sma, source, live))
        for row in entry["ledger"]:
            if row["fill_open"] is None and row["fill_date"] == date.date().isoformat():
                row["fill_open"] = float(frame.loc[date, "Open"])
        if state[date] != entry["state"]:
            entry["state"] = state[date]
            entry["ledger"].append({"signal_date": date.date().isoformat(), "state": state[date], "signal_close": float(close[date]),
                                    "sma200": float(sma[date]), "fill_date": (date + timedelta(days=1)).date().isoformat(),
                                    "fill_open": None, "detected_at": now.isoformat(timespec="seconds"),
                                    "late": not live, "sent": {}})
            flips.append(len(entry["ledger"]) - 1)
        entry["checked"] = date.date().isoformat()
    outage = None
    if len(new) > 1:
        outage = {"from": new[0].date().isoformat(), "to": new[-2].date().isoformat(), "days": len(new) - 1}
        entry["gaps"].append(outage)
        _log("outage", **outage)
    if flips:
        rows = [entry["ledger"][i] for i in flips]
        info = latest(symbol, now, frame)
        detected = now.isoformat(timespec="seconds")
        text = (late_text(rows, info, outage, detected, entry["core"]) if outage
                else alert_text(info, detected, entry["core"]))
        entry["pending"].append({"text": text, "rows": flips, "sent_to": []})  # an undelivered older alert stays queued
        _log("detected", detected_at=detected, flips=[r["signal_date"] for r in rows], late=bool(outage))
    _save(data)  # before sending: a crash now resends to the chats not yet reached, it cannot lose the flip
    return _deliver(data, entry, send, chats)


def _deliver(data: dict, entry: dict, send, chats) -> bool:
    """Send every queued alert, oldest first, to each chat that has not had it; a chat that fails one waits
    with the rest of its queue, so it never gets a later alert before an earlier one."""
    if not entry["pending"]:
        return False
    failed = []
    for pending in entry["pending"]:
        for chat in chats:
            if chat in pending["sent_to"] or chat in failed:
                continue
            try:
                send(chat, pending["text"])
            except Exception as exc:  # the other chats still get theirs; only this one is retried
                failed.append(chat)
                _log("send_failed", chat=chat, error=repr(exc))
                continue
            stamp = datetime.now(UTC).isoformat(timespec="seconds")
            pending["sent_to"].append(chat)
            for i in pending["rows"]:
                entry["ledger"][i]["sent"][chat] = stamp
            _save(data)
            _log("sent", chat=chat, sent_at=stamp, rows=pending["rows"])
    entry["pending"] = [p for p in entry["pending"] if not set(chats) <= set(p["sent_to"])]
    _save(data)
    if failed:
        raise SendFailed(f"not delivered to {', '.join(failed)}")
    return True


def _seconds_to_run(now: datetime) -> float:
    run = now.replace(hour=RUN_AT[0], minute=RUN_AT[1], second=0, microsecond=0)
    return ((run if run > now else run + timedelta(days=1)) - now).total_seconds()


def watch(send, chats) -> None:
    """Check at start-up (catch-up), then every day just after 00:00 UTC. Never returns."""
    while True:
        error = None
        for _ in range(RETRIES):
            try:
                for symbol in WATCH:
                    check(symbol, send, chats)
                error = None
                break
            except Exception as exc:  # a late bar, a data gap or a network error: try again shortly
                error = exc
                logger.warning("trend check failed: %r", exc)
                time.sleep(RETRY_SECONDS)
        if error is not None:  # silence would read as "no signal": say that today's check did not happen
            _log("check_failed", error=repr(error))
            for chat in chats:
                try:
                    send(chat, f"⚠️ BTCUSDT trend check failed after {RETRIES} tries ({error}). No signal was made today; "
                               "the next check is tomorrow.")
                except Exception as exc:
                    logger.warning("could not report the failed check to %s: %r", chat, exc)
        time.sleep(_seconds_to_run(datetime.now(UTC)))
