"""BTC trend rule, paper only: UTC close above its 200-day SMA means hold BTC, otherwise cash.

A second, shadow book runs the same way next to it: UTC close above the Ichimoku cloud (9/26/52)
means hold BTC. It has its own state file, alerts and ops rows, so it can never change the SMA200
book; it is there to see more signals live (research of 2026-10-24), not to replace the rule.

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

Every message is Telegram HTML: bold heads, numbers in ``<pre>`` blocks, dynamic text escaped.
Each one is written in English and Vietnamese; a chat gets the language it chose with /lang.

State is saved before a message goes out and each chat's delivery is recorded on its own, so a
restart can neither lose a flip nor repeat a delivered one, and one failing chat blocks no other.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from html import escape as _html

import numpy as np
import pandas as pd
import requests

from tradingagents.dataflows.symbols import crypto_base
from tradingagents.dataflows.vendors.binance_klines import HOSTS, fetch_daily_klines
from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv

logger = logging.getLogger(__name__)

WINDOW = 200
FEE = 0.001  # per side, as in the validation
SMA200, ICHIMOKU = "sma200", "ichimoku"
BOOKS = {"BTCUSDT": SMA200, "BTCUSDT-ICHIMOKU": ICHIMOKU}  # book id -> rule; the validated SMA200 book first
RUN_AT = (0, 5)  # UTC hour, minute: the daily bar is final a few minutes after midnight
DEADMAN_AT = (0, 30)  # UTC: no completed run by now means the job is down, and a chat is told
RETRY_SECONDS, RETRIES = 600, 12  # a late bar or a failed fetch is retried for about two hours
REVISION = 0.005  # a stored close that a later fetch moves by more than this is a data event
DIR = os.path.join(os.path.expanduser("~"), ".tradingagents", "trend")

HOLD, CASH = "HOLD", "CASH"
WHIPSAW = (
    "<i>⚠️ Whipsaw: 29 of 44 closed trades (2014-2026) lost money and lasted 14 days or less. The edge is "
    "sitting out bear markets (max drawdown -67% vs -83% buy-and-hold), not each signal.</i>"
)
WHIPSAW_ICHIMOKU = (
    "<i>⚠️ Shadow book, not the rule: about 7 trades a year; 51 of 84 closed trades (2015-2026) lost money and lasted "
    "14 days or less. Its edge was a smaller drawdown than SMA200 (-51% vs -69%); it lagged in strong bull years.</i>"
)
WHIPSAW_VI = (
    "<i>⚠️ Nhiễu (whipsaw): 29/44 lệnh đã đóng (2014-2026) bị lỗ và chỉ kéo dài tối đa 14 ngày. Lợi thế nằm ở việc "
    "đứng ngoài thị trường giá xuống (sụt giảm tối đa -67% so với -83% nếu mua và giữ), không nằm ở từng tín hiệu.</i>"
)
WHIPSAW_ICHIMOKU_VI = (
    "<i>⚠️ Sổ phụ, không phải quy tắc chính: khoảng 7 lệnh/năm; 51/84 lệnh đã đóng (2015-2026) bị lỗ và chỉ kéo dài "
    "tối đa 14 ngày. Lợi thế là sụt giảm nhỏ hơn SMA200 (-51% so với -69%); kém hơn trong các năm tăng mạnh.</i>"
)
LANGS = ("en", "vi")


class StaleData(Exception):
    """The newest closed bar is older than yesterday (UTC), so the signal is not for today."""


class DataGap(Exception):
    """A calendar day is missing from the bars the signal needs, so no signal is made."""


class SourceUnverified(Exception):
    """The bars come from another source than the book's and do not reproduce what it stored: no signal."""


class SendFailed(Exception):
    """At least one chat did not get a message; the others did, and only this one is retried."""


def _t(lang: str, en: str, vi: str) -> str:
    """The English or the Vietnamese text of a message."""
    return vi if lang == "vi" else en


def _days(n: int, lang: str = "en") -> str:
    return _t(lang, f"{n} day{'s' * (n != 1)}", f"{n} ngày")


def _state(state: str, lang: str = "en") -> str:
    return state if lang != "vi" else ("GIỮ BTC" if state == HOLD else "TIỀN MẶT")


def chat_lang(chat) -> str:
    """The language a chat chose with /lang; English until it chooses."""
    try:
        with open(os.path.join(DIR, "lang.json"), encoding="utf-8") as f:
            return json.load(f).get(str(chat), "en")
    except (FileNotFoundError, ValueError):
        return "en"


def set_chat_lang(chat, lang: str) -> None:
    assert lang in LANGS, lang
    path = os.path.join(DIR, "lang.json")
    try:
        with open(path, encoding="utf-8") as f:
            langs = json.load(f)
    except (FileNotFoundError, ValueError):
        langs = {}
    langs[str(chat)] = lang
    os.makedirs(DIR, exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(langs, f)
    os.replace(path + ".tmp", path)


def book(name: str) -> str:
    """The book id a chat command names: ``ichimoku`` is the shadow BTC book, anything else a symbol."""
    name = name.upper()
    return "BTCUSDT-ICHIMOKU" if name in ("ICHIMOKU", "ICHI", "BTCUSDT-ICHIMOKU") else name


def _pair(book_id: str) -> str:
    return book_id.split("-")[0]


def _rule(book_id: str) -> str:
    return BOOKS.get(book_id, SMA200)


def _title(book_id: str, lang: str = "en") -> str:
    if _rule(book_id) == ICHIMOKU:
        return f"{_pair(book_id)} Ichimoku {_t(lang, '(shadow)', '(sổ phụ)')}"
    return _t(lang, f"{_pair(book_id)} trend", f"{_pair(book_id)} xu hướng")


def _name(book_id: str, lang: str = "en") -> str:
    """How /paper and /status head a book: the SMA200 book keeps its plain symbol."""
    return _title(book_id, lang) if _rule(book_id) == ICHIMOKU else book_id


def _short(book_id: str) -> str:
    return "Ichimoku" if _rule(book_id) == ICHIMOKU else "SMA200"


def _level_name(book_id: str, lang: str = "en") -> str:
    return _t(lang, "Cloud top", "Đỉnh mây") if _rule(book_id) == ICHIMOKU else "SMA200"


def _whipsaw(book_id: str, lang: str = "en") -> str:
    if _rule(book_id) == ICHIMOKU:
        return _t(lang, WHIPSAW_ICHIMOKU, WHIPSAW_ICHIMOKU_VI)
    return _t(lang, WHIPSAW, WHIPSAW_VI)


def core_weight() -> float:
    """Share of the book held in BTC permanently (env ``TREND_CORE_WEIGHT``, default 0.5)."""
    weight = float(os.environ.get("TREND_CORE_WEIGHT", "0.5"))
    if not 0 <= weight <= 1:
        raise ValueError(f"TREND_CORE_WEIGHT must be between 0 and 1, got {weight}")
    return weight


def bars(symbol: str, today: datetime | None = None) -> tuple[pd.DataFrame, str]:
    """Open, High, Low and Close of the completed UTC days, oldest first, indexed by date, and where they came from."""
    today = (today or datetime.now(UTC)).date()
    frame = fetch_daily_klines(symbol, (today - timedelta(days=900)).isoformat(), today.isoformat())
    source = "binance"
    if frame is None:  # Binance unreachable: the fallback, with the open day cut off and no gaps filled in
        source = "yahoo"
        frame = load_ohlcv(symbol, (today - timedelta(days=1)).isoformat(), fill_gaps=False).set_index("Date")
        frame = frame[frame.index.normalize().date < today]
    out = frame[["Open", "High", "Low", "Close"]].astype(float)
    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    return out, source


def require_continuous(frame: pd.DataFrame, extra: int = 0) -> None:
    """Every calendar day of the SMA window (plus ``extra`` newer days) must be present."""
    last = frame.index[-1]
    expected = pd.date_range(end=last, periods=WINDOW + extra, freq="D")
    missing = expected.difference(frame.index)
    if len(missing):
        raise DataGap(f"{_days(len(missing))} missing from the last {len(expected)}, first {missing[0].date()}")


def states(close: pd.Series) -> pd.Series:
    """HOLD or CASH for each day with a full SMA200; earlier days are left out."""
    sma = close.rolling(WINDOW).mean()
    return pd.Series(HOLD, index=close.index).where(close > sma, CASH)[sma.notna()]


def cloud_top(frame: pd.DataFrame) -> pd.Series:
    """Top of the Ichimoku cloud (9/26/52, TradingView defaults) under each day. The spans are drawn 26 bars
    ahead, so a day's cloud is built from highs and lows of 25 days earlier and never repaints; the Chikou
    line, which looks ahead, is not used."""
    def mid(n):
        return (frame["High"].rolling(n).max() + frame["Low"].rolling(n).min()) / 2
    return pd.concat([((mid(9) + mid(26)) / 2).shift(25), mid(52).shift(25)], axis=1).max(axis=1, skipna=False)


def level(frame: pd.DataFrame, rule: str = SMA200) -> pd.Series:
    """The line the close is compared with: the SMA200, or the cloud top."""
    return cloud_top(frame) if rule == ICHIMOKU else frame["Close"].rolling(WINDOW).mean()


def rule_states(frame: pd.DataFrame, rule: str = SMA200) -> pd.Series:
    """HOLD or CASH for each day the rule's line exists; for SMA200 this is ``states``."""
    line = level(frame, rule)
    return pd.Series(HOLD, index=frame.index).where(frame["Close"] > line, CASH)[line.notna()]


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
    frame = frame if frame is not None else bars(_pair(symbol), today)[0]
    close = frame["Close"]
    state = rule_states(frame, _rule(symbol))
    if state.empty:
        raise StaleData(f"{symbol}: fewer than {WINDOW} closed daily bars")
    day = state.index[-1]
    if day.date() < ((today or datetime.now(UTC)).date() - timedelta(days=1)):
        raise StaleData(f"{symbol}: newest closed bar is {day.date()}")
    require_continuous(frame)
    run = state[state != state.iloc[-1]]  # the streak began the day after the last opposite state
    since = state.index[state.index.get_loc(run.index[-1]) + 1] if len(run) else state.index[0]
    return {"symbol": symbol, "state": state.iloc[-1], "date": day.date().isoformat(), "close": float(close.iloc[-1]),
            "sma": float(level(frame, _rule(symbol)).iloc[-1]), "since": since.date().isoformat()}


def _state_path(book_id: str) -> str:
    """Each book its own file: the shadow book can never rewrite the SMA200 book's state."""
    return os.path.join(DIR, "state.json" if _rule(book_id) == SMA200 else f"state-{book_id.lower()}.json")


def _load(book_id: str = "BTCUSDT") -> dict:
    try:
        with open(_state_path(book_id), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _save(data: dict, book_id: str = "BTCUSDT") -> None:
    os.makedirs(DIR, exist_ok=True)
    path = _state_path(book_id)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(path + ".tmp", path)  # a crash mid-write leaves the old state, not half a file


def _log(event: str, **fields) -> None:
    os.makedirs(DIR, exist_ok=True)
    with open(os.path.join(DIR, "log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": event, "at": datetime.now(UTC).isoformat(timespec="seconds"), **fields}) + "\n")


def _esc(text: str) -> str:
    return _html(text, quote=False)  # Telegram only needs & < > escaped


def _pre(rows: list[tuple[str, str]]) -> str:
    """Label/value rows as one monospace block, values right-aligned, for Telegram HTML."""
    left, right = max(len(k) for k, _ in rows), max(len(v) for _, v in rows)
    return "<pre>" + "\n".join(_esc(f"{k:<{left}}  {v:>{right}}") for k, v in rows) + "</pre>"


def _levels(info: dict, lang: str = "en") -> str:
    return _pre([(f"{_t(lang, 'Close', 'Đóng cửa')} {info['date']}", f"{info['close']:,.0f}"),
                 (_level_name(info["symbol"], lang), f"{info['sma']:,.0f}"),
                 (_t(lang, "Gap", "Chênh lệch"), f"{info['close'] / info['sma'] - 1:+.1%}")])


def _split(weight: float, lang: str = "en") -> str:
    return _t(lang, f"Split: {weight:.0%} core BTC (never sold) · {1 - weight:.0%} follows the rule, not rebalanced.",
              f"Phân bổ: {weight:.0%} BTC lõi (không bao giờ bán) · {1 - weight:.0%} theo quy tắc, không tái cân bằng.")


def _now(state: str, lang: str = "en") -> str:
    return "🟢 " + _t(lang, "HOLD BTC", "GIỮ BTC") if state == HOLD else "🔴 " + _t(lang, "CASH", "TIỀN MẶT")


def _head(state: str, book_id: str = "BTCUSDT", lang: str = "en") -> str:
    move = _t(lang, "UP · HOLD BTC", "TĂNG · GIỮ BTC") if state == HOLD else _t(lang, "DOWN · CASH", "GIẢM · TIỀN MẶT")
    return f"{'🟢' if state == HOLD else '🔴'} <b>{_title(book_id, lang)} {move}</b>"


def alert_text(info: dict, detected_at: str, weight: float | None = None, lang: str = "en") -> str:
    return "\n".join([
        _head(info["state"], info["symbol"], lang),
        _t(lang, "<i>Paper signal only, no order was placed.</i>",
           "<i>Chỉ là tín hiệu giao dịch thử, không có lệnh nào được đặt.</i>"),
        _levels(info, lang),
        _t(lang, f"Detected {detected_at[11:16]} UTC. The backtest fills at the 00:00 UTC open; you cannot get that price, "
                 "so a real fill will differ.",
           f"Phát hiện lúc {detected_at[11:16]} UTC. Backtest khớp lệnh ở giá mở cửa 00:00 UTC; bạn không thể có giá đó, "
           "nên giá khớp thực tế sẽ khác."),
        _split(core_weight() if weight is None else weight, lang),
        "",
        _whipsaw(info["symbol"], lang),
    ])


def late_text(flips: list[dict], info: dict, outage: dict, detected_at: str, weight: float, lang: str = "en") -> str:
    """One message for signal changes found after the bot was offline: nobody could act on them on time."""
    n = len(flips)
    return "\n".join([
        _t(lang, f"⚠️ <b>{_title(info['symbol'])} signals found late</b>",
           f"⚠️ <b>{_title(info['symbol'], lang)}: tín hiệu phát hiện trễ</b>"),
        _t(lang, f"The check was offline from {outage['from']} to {outage['to']} ({_days(outage['days'])}). "
                 f"{n} signal change{'s' * (n != 1)} {'was' if n == 1 else 'were'} found late, at {detected_at[11:16]} UTC, "
                 "and could not be acted on in time:",
           f"Bot ngừng kiểm tra từ {outage['from']} đến {outage['to']} ({_days(outage['days'], lang)}). "
           f"{n} lần đổi tín hiệu được phát hiện trễ, lúc {detected_at[11:16]} UTC, và không thể hành động kịp:"),
        *[f"• {f['signal_date']} <b>{_state(f['state'], lang)}</b> · {_t(lang, 'close', 'đóng cửa')} {f['signal_close']:,.0f}"
          for f in flips],
        "",
        f"{_t(lang, 'Now', 'Hiện tại')}: {_now(info['state'], lang)}",
        _levels(info, lang),
        _t(lang, "/paper shows the rule (every change filled) and as operated (position held through the outage).",
           "/paper hiển thị theo quy tắc (khớp mọi lần đổi) và theo vận hành thực (giữ nguyên vị thế trong lúc bot ngừng)."),
        _split(weight, lang),
        "",
        _whipsaw(info["symbol"], lang),
    ])


def status_text(symbol: str = "BTCUSDT", lang: str = "en") -> str:
    symbol = book(symbol)
    if symbol not in BOOKS:
        return _t(lang, "The trend rule was only validated on BTCUSDT (/trend ichimoku is its shadow book).",
                  "Quy tắc xu hướng chỉ được kiểm chứng trên BTCUSDT (/trend ichimoku là sổ phụ của nó).")
    try:
        info = latest(symbol)
        weight = (_load(symbol).get(symbol) or {}).get("core", core_weight())
    except Exception as exc:  # report, keep serving
        return _t(lang, "⚠️ Trend data unavailable: ", "⚠️ Không lấy được dữ liệu xu hướng: ") + _esc(str(exc))
    if _rule(symbol) == ICHIMOKU:
        rule = _t(lang, "Rule: UTC close above the Ichimoku cloud (9/26/52) = hold BTC, otherwise cash. TradingView defaults, "
                        "not tuned; runs next to SMA200 and never changes it.",
                  "Quy tắc: giá đóng cửa UTC trên mây Ichimoku (9/26/52) = giữ BTC, ngược lại giữ tiền mặt. Thông số mặc "
                  "định TradingView, không tinh chỉnh; chạy song song với SMA200 và không bao giờ thay đổi nó.")
    else:
        rule = _t(lang, "Rule: UTC close &gt; SMA200 = hold BTC, otherwise cash. Fixed, no parameters.",
                  "Quy tắc: giá đóng cửa UTC &gt; SMA200 = giữ BTC, ngược lại giữ tiền mặt. Cố định, không có tham số.")
    return "\n".join([
        f"<b>{_title(symbol, lang)}: {_now(info['state'], lang)}</b> {_t(lang, '(paper)', '(giao dịch thử)')}",
        f"{_t(lang, 'Since', 'Từ')} {info['since']}",
        _levels(info, lang),
        rule,
        _split(weight, lang),
        "",
        _whipsaw(symbol, lang),
    ])


def _book(entry: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """The stored days as bars, the rule's state on each, and the state the operated book held."""
    days = pd.DataFrame(entry["days"]).set_index("date")
    days.index = pd.DatetimeIndex(days.index)
    held = []
    for state, live in zip(days["state"], days["live"], strict=True):
        held.append(state if live or not held else held[-1])  # a day processed late is not acted on
    return days[["Open", "Close"]].astype(float), days["state"], pd.Series(held, index=days.index)


def paper_text(symbol: str = "BTCUSDT", lang: str = "en") -> str:
    """The paper book since the first check, from the stored days: no fetch, no revision, no env."""
    symbol = book(symbol)
    entry = _load(symbol).get(symbol)
    if entry is None:
        return _t(lang, "No paper book yet: the first daily check has not run.",
                  "Chưa có sổ giao dịch thử: lần kiểm tra hằng ngày đầu tiên chưa chạy.")
    frame, rule, operated = _book(entry)
    weight = entry["core"]
    # shortcut: the event itself stays in English, as stored; translate at write time if Vietnamese chats need it
    notes = [f"⚠️ {_t(lang, 'Data check', 'Kiểm tra dữ liệu')}: {_esc(event)}" for event in entry["data_events"][-2:]]
    head = _t(lang, f"📒 <b>{_name(symbol)} paper book</b>", f"📒 <b>Sổ giao dịch thử {_name(symbol, lang)}</b>")
    if len(frame) < 2:
        return "\n".join([head, _t(lang, f"Opened {entry['start']['date']} as cash; no day has closed since, so no equity yet.",
                                   f"Mở ngày {entry['start']['date']} bằng tiền mặt; chưa có ngày nào đóng cửa kể từ đó "
                                   "nên chưa có lợi nhuận."), *notes])
    ideal, ops = simulate(frame, rule, weight), simulate(frame, operated, weight)

    def pct(equity: pd.Series) -> str:
        return f"{equity.iloc[-1] * 100 - 100:+.1f}%"

    in_btc = operated.shift(fill_value=CASH).iloc[-1] == HOLD  # filled: yesterday's signal, bought at today's open
    core, sleeve = weight * ops["hold"].iloc[-1], (1 - weight) * ops["trend"].iloc[-1] * in_btc
    total = ops["mix"].iloc[-1]
    mix = _t(lang, "Mix (core + rule)", "Kết hợp (lõi + quy tắc)")
    lines = [
        head,
        _t(lang, f"Since {entry['start']['date']} ({_days(len(frame) - 1)}, no real orders) · core {weight:.0%} fixed at the start",
           f"Từ {entry['start']['date']} ({_days(len(frame) - 1, lang)}, không có lệnh thật) · lõi {weight:.0%} cố định từ đầu"),
        "",
        _t(lang, f"<b>Position</b> · filled through {frame.index[-1].date()}", f"<b>Vị thế</b> · đã khớp đến {frame.index[-1].date()}"),
        _pre([("BTC", f"{(core + sleeve) / total:.1%}"), (_t(lang, "  core", "  lõi"), f"{core / total:.1%}"),
              (_t(lang, "  trend sleeve", "  phần theo xu hướng"), f"{sleeve / total:.1%}"),
              (_t(lang, "Cash", "Tiền mặt"), f"{1 - (core + sleeve) / total:.1%}")]),
    ]
    if operated.iloc[-1] != (HOLD if in_btc else CASH):
        lines.append(_t(lang, f"⏳ Pending at the next open: trend sleeve to {operated.iloc[-1]} (signal of {frame.index[-1].date()})",
                        f"⏳ Chờ khớp ở giá mở cửa kế tiếp: phần theo xu hướng chuyển sang {_state(operated.iloc[-1], lang)} "
                        f"(tín hiệu ngày {frame.index[-1].date()})"))
    lines += ["", _t(lang, "<b>Equity, rule</b> (every change filled)", "<b>Lợi nhuận, theo quy tắc</b> (khớp mọi lần đổi)"),
              _pre([(mix, pct(ideal["mix"])), (f"100% {_short(symbol)}", pct(ideal["trend"])),
                    (_t(lang, "Buy-and-hold", "Mua và giữ"), pct(ideal["hold"]))])]
    if entry["gaps"]:
        off = sum(g["days"] for g in entry["gaps"])
        lines += [_t(lang, f"<b>Equity, as operated</b> (⚠️ {_days(off)} offline, position held through them)",
                     f"<b>Lợi nhuận, theo vận hành thực</b> (⚠️ bot ngừng {_days(off, lang)}, giữ nguyên vị thế)"),
                  _pre([(mix, pct(ops["mix"])), (f"100% {_short(symbol)}", pct(ops["trend"]))])]
    if notes:
        lines += ["", *notes]
    lines += ["", _t(lang, "<b>Ledger</b> · signal close → next open fill (UTC)",
                     "<b>Sổ lệnh</b> · đóng cửa có tín hiệu → khớp ở giá mở cửa kế tiếp (UTC)")]
    for row in entry["ledger"][-5:]:
        fill = _t(lang, "pending", "chờ") if row["fill_open"] is None else f"{row['fill_open']:,.0f}"
        sent = min(row["sent"].values())[5:16].replace("T", " ") if row["sent"] else _t(lang, "not yet", "chưa")
        lines += [f"• <b>{row['signal_date']} {_state(row['state'], lang)}</b>{_t(lang, ' (late)', ' (trễ)') if row['late'] else ''} · "
                  + _t(lang, f"close {row['signal_close']:,.0f} → open {fill} ({row['fill_date'][5:]})",
                       f"đóng {row['signal_close']:,.0f} → mở {fill} ({row['fill_date'][5:]})"),
                  _t(lang, f"   detected {row['detected_at'][5:16].replace('T', ' ')} · sent {sent}",
                     f"   phát hiện {row['detected_at'][5:16].replace('T', ' ')} · đã gửi {sent}")]
        seen = row.get("quote")
        if seen:  # what the alert could really have traded at, against the open the backtest uses
            base = row["fill_open"] or seen["open"]
            lines.append(_t(lang, f"   price at send {seen['price']:,.0f} ({seen['price'] / base - 1:+.2%} vs open)",
                            f"   giá lúc gửi {seen['price']:,.0f} ({seen['price'] / base - 1:+.2%} so với giá mở)"))
    if not entry["ledger"]:
        lines.append(_t(lang, "No flips yet.", "Chưa có lần đổi tín hiệu nào."))
    lines.append(_t(lang, "<i>Sleeves start as cash, fill at the next open, 0.1% fee per side, no rebalancing.</i>",
                    "<i>Các phần bắt đầu bằng tiền mặt, khớp ở giá mở cửa kế tiếp, phí 0.1% mỗi chiều, không tái cân bằng.</i>"))
    return "\n".join(lines)


def quote(symbol: str) -> dict | None:
    """Price now and today's 00:00 UTC open from Binance, or None: it never blocks or fails a send."""
    base = crypto_base(symbol)
    for host in HOSTS if base else ():
        try:
            response = requests.get(f"{host}/api/v3/klines", params={"symbol": f"{base}USDT", "interval": "1d", "limit": 1},
                                    timeout=5)
            response.raise_for_status()
            bar = response.json()[-1]
            return {"price": float(bar[4]), "open": float(bar[1]), "at": datetime.now(UTC).isoformat(timespec="seconds")}
        except (requests.RequestException, ValueError, IndexError, TypeError) as exc:
            logger.warning("price quote failed via %s: %s", host, exc)
    return None


def _equity(entry: dict) -> dict | None:
    """Total return of the rule book's three variants since the start, from the stored days."""
    if len(entry["days"]) < 2:
        return None
    frame, rule, _ = _book(entry)
    book = simulate(frame, rule, entry["core"]).iloc[-1] - 1
    return {name: round(float(book[name]), 6) for name in ("hold", "trend", "mix")}


def _ops(**fields) -> None:
    """One row per completed daily run in ``ops.jsonl``, the record kept during the paper-trade period."""
    os.makedirs(DIR, exist_ok=True)
    with open(os.path.join(DIR, "ops.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": datetime.now(UTC).isoformat(timespec="seconds"), **fields}) + "\n")


def _ops_rows() -> list[dict]:
    try:
        with open(os.path.join(DIR, "ops.jsonl"), encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def overdue(entry: dict | None, now: datetime) -> bool:
    """After 00:30 UTC, no run has completed since today's midnight."""
    if (now.hour, now.minute) < DEADMAN_AT:
        return False
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    last = (entry or {}).get("last_run")
    return last is None or datetime.fromisoformat(last["at"]) < midnight


def health_text(symbol: str = "BTCUSDT", now: datetime | None = None, lang: str = "en") -> str:
    """Is the daily job alive: last completed run, queued alerts, recent runs and errors."""
    now = now or datetime.now(UTC)
    symbol = book(symbol)
    entry = _load(symbol).get(symbol)
    name = _name(symbol, lang)
    if entry is None:
        return _t(lang, f"{_esc(name)} job: no run has completed yet.", f"Tác vụ {_esc(name)}: chưa có lần chạy nào hoàn tất.")
    last = entry.get("last_run")
    if overdue(entry, now):
        head = _t(lang, f"🚨 <b>{name} job: OVERDUE</b>", f"🚨 <b>Tác vụ {name}: QUÁ HẠN</b>")
    else:
        head = _t(lang, f"✅ <b>{name} job: OK</b>", f"✅ <b>Tác vụ {name}: OK</b>")
    lines = [head + _t(lang, " · checked ", " · kiểm tra lúc ") + f"{now.strftime('%Y-%m-%d %H:%M')} UTC"]
    if last:
        age = int((now - datetime.fromisoformat(last["at"])).total_seconds() // 60)
        at = last["at"][:16].replace("T", " ")
        lines += [_t(lang, f"Last completed run {at} UTC ({age // 60}h {age % 60}m ago)",
                     f"Lần chạy xong gần nhất {at} UTC ({age // 60} giờ {age % 60} phút trước)"),
                  f"{_t(lang, 'Data through', 'Dữ liệu đến')} {last['data_date']} · {last['source']} · {_state(last['state'], lang)}"]
    else:
        lines.append(_t(lang, f"No run has completed; data processed through {entry['checked']}",
                        f"Chưa có lần chạy nào hoàn tất; dữ liệu đã xử lý đến {entry['checked']}"))
    lines += ["", _pre([(_t(lang, "Undelivered alerts", "Cảnh báo chưa gửi"), str(len(entry["pending"]))),
                        (_t(lang, "Outages", "Lần bot ngừng"), str(len(entry["gaps"]))),
                        (_t(lang, "Data events", "Sự kiện dữ liệu"), str(len(entry["data_events"]))),
                        (_t(lang, "Signals", "Tín hiệu"), str(len(entry["ledger"])))]),
              _t(lang, "<b>Recent runs</b>", "<b>Các lần chạy gần đây</b>")]
    rows = [row for row in _ops_rows() if row.get("symbol", "BTCUSDT") == symbol]
    for row in rows[-3:]:
        eq = row.get("equity")
        stats = f" · {_t(lang, 'mix', 'kết hợp')} {eq['mix']:+.1%}" if eq else ""
        lag = f" · {_t(lang, 'alert lag', 'trễ cảnh báo')} {row['lag_s']:.0f}s" if row.get("lag_s") is not None else ""
        lines.append(f"{'✅' if row['ok'] else _t(lang, '❌ FAILED', '❌ LỖI')} {row['at'][5:16].replace('T', ' ')} · "
                     f"{_t(lang, 'data', 'dữ liệu')} {row.get('data_date', '-')}{stats}{lag}")
        if row.get("errors"):
            lines.append(f"   {len(row['errors'])} {_t(lang, 'error(s), first', 'lỗi, lỗi đầu')}: "
                         f"<code>{_esc(row['errors'][0][:80])}</code>")
    if not rows:
        lines.append(_t(lang, "No ops rows yet.", "Chưa có bản ghi vận hành."))
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


def _verify_source(entry: dict, frame: pd.DataFrame, source: str, state: pd.Series, sma: pd.Series, new,
                   name: str = "SMA200") -> str:
    """Another source is used only if it reproduces the last stored day. Returns why not, or ''."""
    last = entry["days"][-1]
    stamp = pd.Timestamp(last["date"])
    if stamp not in frame.index or pd.isna(sma.get(stamp)):
        return f"{last['date']} is not in the {source} bars"
    close, level = float(frame.loc[stamp, "Close"]), float(sma[stamp])
    if abs(close / last["Close"] - 1) > REVISION or abs(level / last["sma"] - 1) > REVISION or state[stamp] != last["state"]:
        return (f"{source} has {last['date']} at close {close:,.1f}, {name} {level:,.1f}, {state[stamp]}; "
                f"the book stored {last['Close']:,.1f}, {last['sma']:,.1f}, {last['state']}")
    current = entry["state"]
    for date in new:  # a flip inside the gap between the two sources' SMAs may exist in one of them only
        if state[date] != current and abs(frame.loc[date, "Close"] / sma[date] - 1) <= REVISION:
            return f"{source} flips to {state[date]} on {date.date()} within {REVISION:.1%} of the {name}"
        current = state[date]
    return ""


def check(symbol: str, send, chats, today: datetime | None = None) -> bool:
    """Process every closed day since the last check and alert on flips.

    ``send(chat, text)`` delivers to one chat. True when an alert was newly delivered to every chat.
    """
    now = today or datetime.now(UTC)
    frame, source = bars(_pair(symbol), now)
    close = frame["Close"]
    state = rule_states(frame, _rule(symbol))
    sma = level(frame, _rule(symbol))
    if state.empty:
        raise StaleData(f"{symbol}: fewer than {WINDOW} closed daily bars")
    newest = state.index[-1]
    if newest.date() < now.date() - timedelta(days=1):
        raise StaleData(f"{symbol}: newest closed bar is {newest.date()}")
    data = _load(symbol)
    entry = data.get(symbol)
    if entry is None:  # first run: the book opens as cash at this close, and nothing is alerted
        require_continuous(frame)
        data[symbol] = entry = {
            "core": core_weight(), "source": source, "state": state[newest], "checked": newest.date().isoformat(),
            "start": {"date": newest.date().isoformat(), "close": float(close[newest])},
            "days": [_day(newest, frame, state, sma, source, True)], "ledger": [], "gaps": [], "data_events": [],
            "revised": [], "pending": []}
        entry["last_run"] = {"at": datetime.now(UTC).isoformat(timespec="seconds"), "data_date": entry["checked"], "source": source,
                             "state": entry["state"], "equity": None, "flips": [], "send_s": 0.0, "lag_s": None}
        _save(data, symbol)
        _log("baseline", book=symbol, source=source, date=newest.date().isoformat())
        return False

    new = state.index[state.index > pd.Timestamp(entry["checked"])]
    require_continuous(frame, len(new))
    if source != entry["source"]:
        why = _verify_source(entry, frame, source, state, sma, new, _level_name(symbol))
        note = f"source changed from {entry['source']} to {source}" + (f" and does not match: {why}" if why else " (verified)")
        if note not in entry["data_events"][-1:]:
            entry["data_events"].append(note)
            _log("data_event", note=note, source=source)
        if why:  # no signal from numbers the book cannot reproduce; the stored source returning ends this
            _save(data, symbol)
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
        texts = {lang: (late_text(rows, info, outage, detected, entry["core"], lang) if outage
                        else alert_text(info, detected, entry["core"], lang)) for lang in LANGS}
        # an undelivered older alert stays queued; each chat gets it in its own language
        entry["pending"].append({"text": texts["en"], "texts": texts, "rows": flips, "sent_to": []})
        _log("detected", detected_at=detected, flips=[r["signal_date"] for r in rows], late=bool(outage))
    _save(data, symbol)  # before sending: a crash now resends to the chats not yet reached, it cannot lose the flip
    stats = {"rows": []}
    started = time.monotonic()
    delivered = _deliver(data, entry, send, chats, symbol, stats)
    lags = [(datetime.fromisoformat(entry["ledger"][i]["sent"][c]) - datetime.fromisoformat(entry["ledger"][i]["fill_date"]).replace(tzinfo=UTC)).total_seconds()
            for i in stats["rows"] for c in entry["ledger"][i]["sent"]]
    entry["last_run"] = {"at": datetime.now(UTC).isoformat(timespec="seconds"), "data_date": entry["checked"], "source": entry["source"],
                         "state": entry["state"], "equity": _equity(entry), "flips": [entry["ledger"][i]["signal_date"] for i in stats["rows"]],
                         "send_s": round(time.monotonic() - started, 2), "lag_s": max(lags) if lags else None}
    _save(data, symbol)  # only a run that went through to the last chat counts as completed
    return delivered


def _deliver(data: dict, entry: dict, send, chats, symbol: str = "BTCUSDT", stats: dict | None = None) -> bool:
    """Send every queued alert, oldest first, to each chat that has not had it; a chat that fails one waits
    with the rest of its queue, so it never gets a later alert before an earlier one."""
    if not entry["pending"]:
        return False
    failed = []
    for pending in entry["pending"]:
        if "quote" not in pending:  # the price the alert could be acted on, at the first attempt, live signals only
            pending["quote"] = None if any(entry["ledger"][i]["late"] for i in pending["rows"]) else quote(_pair(symbol))
        for chat in chats:
            if chat in pending["sent_to"] or chat in failed:
                continue
            try:
                send(chat, pending.get("texts", {}).get(chat_lang(chat)) or pending["text"])  # alerts queued before /lang: English
            except Exception as exc:  # the other chats still get theirs; only this one is retried
                failed.append(chat)
                _log("send_failed", chat=chat, error=repr(exc))
                continue
            stamp = datetime.now(UTC).isoformat(timespec="seconds")
            pending["sent_to"].append(chat)
            for i in pending["rows"]:
                entry["ledger"][i]["sent"][chat] = stamp
                if pending["quote"]:
                    entry["ledger"][i].setdefault("quote", pending["quote"])
                if stats is not None and i not in stats["rows"]:
                    stats["rows"].append(i)
            _save(data, symbol)
            _log("sent", book=symbol, chat=chat, sent_at=stamp, rows=pending["rows"])
    entry["pending"] = [p for p in entry["pending"] if not set(chats) <= set(p["sent_to"])]
    _save(data, symbol)
    if failed:
        raise SendFailed(f"not delivered to {', '.join(failed)}")
    return True


def _seconds_to_run(now: datetime, at: tuple[int, int] = RUN_AT) -> float:
    run = now.replace(hour=at[0], minute=at[1], second=0, microsecond=0)
    return ((run if run > now else run + timedelta(days=1)) - now).total_seconds()


def watch(send, chats) -> None:
    """Check at start-up (catch-up), then every day just after 00:00 UTC. Never returns.

    Each book is checked on its own, the SMA200 book first, so a failing shadow book can neither delay
    nor fail it. Each book's run, retries included, leaves one row in ``ops.jsonl``: outcome, tries,
    errors, and on success the data date, the equity of the three variants and the alert latency.
    """
    while True:
        for symbol in BOOKS:
            errors = []
            for _ in range(RETRIES):
                try:
                    check(symbol, send, chats)
                    break
                except Exception as exc:  # a late bar, a data gap or a network error: try again shortly
                    errors.append(repr(exc))
                    logger.warning("trend check of %s failed: %r", symbol, exc)
                    time.sleep(RETRY_SECONDS)
            else:
                error = errors[-1]
                _log("check_failed", book=symbol, error=error)
                for chat in chats:
                    lang = chat_lang(chat)
                    try:
                        send(chat, _t(lang, f"⚠️ <b>{_title(symbol)} check failed</b> after {RETRIES} tries. No signal was made "
                                            "today; the next check is tomorrow.",
                                      f"⚠️ <b>{_title(symbol, lang)}: kiểm tra thất bại</b> sau {RETRIES} lần thử. Hôm nay không "
                                      "có tín hiệu; lần kiểm tra tiếp theo là ngày mai.") + f"\n<code>{_esc(error[:300])}</code>")
                    except Exception as exc:
                        logger.warning("could not report the failed check to %s: %r", chat, exc)
            ok = len(errors) < RETRIES
            run = (_load(symbol).get(symbol) or {}).get("last_run") or {}
            _ops(symbol=symbol, ok=ok, tries=len(errors) + ok, errors=errors,
                 **({k: run[k] for k in ("data_date", "source", "state", "equity", "flips", "send_s", "lag_s")} if ok and run else {}))
        time.sleep(_seconds_to_run(datetime.now(UTC)))


def deadman(send, chats) -> None:
    """A separate thread from ``watch``: if no run has completed by 00:30 UTC, tell the chats (once a day)."""
    while True:
        time.sleep(_seconds_to_run(datetime.now(UTC), DEADMAN_AT))
        for symbol in BOOKS:
            try:
                alert_overdue(symbol, send, chats, datetime.now(UTC))
            except Exception as exc:  # this thread must outlive any one failure
                logger.warning("dead-man check failed: %r", exc)


def alert_overdue(symbol: str, send, chats, now: datetime) -> bool:
    entry = _load(symbol).get(symbol)
    path = os.path.join(DIR, "deadman.json")
    try:
        with open(path, encoding="utf-8") as f:
            told = json.load(f)
    except FileNotFoundError:
        told = {}
    today = now.date().isoformat()
    if not overdue(entry, now) or told.get(symbol) == today:
        return False
    last = ((entry or {}).get("last_run") or {}).get("at", "never")
    since, clock = last[:16].replace("T", " "), now.strftime("%H:%M")
    for chat in chats:
        lang = chat_lang(chat)
        text = _t(lang, f"🚨 <b>{_title(symbol)} job is down</b>\nNo run has completed since {since} UTC, and it is "
                        f"{clock} UTC. Today's signal check did not happen. See /status.",
                  f"🚨 <b>Tác vụ {_title(symbol, lang)} đã ngừng</b>\nKhông có lần chạy nào hoàn tất kể từ "
                  f"{'(chưa từng)' if last == 'never' else since + ' UTC'}, và bây giờ là {clock} UTC. "
                  "Hôm nay chưa kiểm tra tín hiệu. Xem /status.")
        try:
            send(chat, text)
        except Exception as exc:
            _log("send_failed", chat=chat, error=repr(exc))
    told[symbol] = today
    os.makedirs(DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(told, f)
    _log("deadman", symbol=symbol, last=last)
    return True
