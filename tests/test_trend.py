"""BTC trend rule and paper book: flip dates against pandas, alert-once per chat across restarts and
outages, stored-day equity, data checks, first fill at the next open, backtest warm-up.

Prices are synthetic and the vendors are mocked, so these run without a network connection.
"""
import _socket
import importlib
import json
import re
import socket
import threading
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from tradingagents import trend, trend_backtest, trend_drill


def _telegram_html(text: str) -> bool:
    """Telegram rejects a message with an unknown or unclosed tag, or a bare < or &: that would be a lost alert."""
    tags = re.findall(r"<(/?)([a-z]*)[^>]*>", text)
    stack = []
    for close, name in tags:
        assert name in ("b", "i", "code", "pre"), name
        if close:
            assert stack.pop() == name
        else:
            stack.append(name)
    bare = re.sub(r"</?(?:b|i|code|pre)>", "", text)
    assert "<" not in bare and not re.search(r"&(?!lt;|gt;|amp;)", bare), text
    return not stack


def _btc_line(out: str) -> str:
    return re.search(r"<pre>BTC +([\d.]+%)", out).group(1)

CHATS = ["a", "b", "c"]


def _frame(close, open_=None):
    """Bars ending yesterday; each day opens at the previous close unless ``open_`` says otherwise."""
    close = np.asarray(close, float)
    open_ = np.r_[close[0], close[:-1]] if open_ is None else np.asarray(open_, float)
    end = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0).replace(tzinfo=None) - timedelta(days=1)
    return pd.DataFrame({"Open": open_, "Close": close}, index=pd.date_range(end=end, periods=len(close), freq="D"))


def _walk(days=900, seed=7):
    """A random walk that crosses its SMA200 many times."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.03, days)))
    return _frame(close, np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.005, days)))


def _crafted():
    """Rises for 260 days (HOLD), falls for 40 (CASH), rises again (HOLD): two flips after day 260."""
    return _frame(np.r_[100 + 0.5 * np.arange(260), 229.5 - 4 * np.arange(1, 41), 69.5 + 5 * np.arange(1, 81)])


@pytest.fixture(autouse=True)
def _state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(trend, "DIR", str(tmp_path))
    monkeypatch.setattr(trend, "quote", lambda symbol: None)  # no network; tests that need a price set their own
    monkeypatch.delenv("TREND_CORE_WEIGHT", raising=False)


def _events():
    with open(f"{trend.DIR}/log.jsonl", encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def _at(day):
    """When the check for the bar ``day`` runs: five minutes past the next midnight."""
    return datetime.combine(day.date() + timedelta(days=1), datetime.min.time(), UTC) + timedelta(minutes=5)


def _check(frame, day, sent, chats=("a",), source="binance", fail=()):
    """One check with bars up to ``day``; ``sent`` collects (chat, text); chats in ``fail`` raise."""
    def send(chat, text):
        if chat in fail:
            raise OSError(f"{chat} is down")
        sent.append((chat, text))

    with mock.patch.object(trend, "bars", return_value=(frame[:day], source)):
        return trend.check("BTCUSDT", send, list(chats), _at(day))


def _replay(frame, sent, first_day, restart_every=None, chats=("a",)):
    for n, day in enumerate(frame.index[frame.index >= first_day]):
        _check(frame, day, sent, chats)
        if restart_every and n % restart_every == 0:  # a restart re-checks the same bar
            _check(frame, day, sent, chats)


def _flips(frame):
    state = trend.states(frame["Close"])
    return state, state.index[state.ne(state.shift())][1:]


@pytest.mark.unit
def test_states_match_a_plain_pandas_computation():
    close = _walk()["Close"]
    expected = (close > close.rolling(200).mean())[close.rolling(200).mean().notna()]
    got = trend.states(close)
    assert (got == trend.HOLD).equals(expected) and len(got) == len(close) - 199


@pytest.mark.unit
def test_alerts_fire_on_exactly_the_flip_days_and_never_twice():
    frame = _walk()
    state, flips = _flips(frame)
    sent = []
    _replay(frame, sent, state.index[0], restart_every=3)
    assert len(flips) >= 6 and len(sent) == len(flips)
    for (_, text), day in zip(sent, flips, strict=True):
        assert f"Close {day.date()}" in text and _telegram_html(text)
        assert ("HOLD BTC" in text) == (state[day] == trend.HOLD)
    assert not trend._load()["BTCUSDT"]["gaps"]  # processed on time every day


@pytest.mark.unit
def test_first_run_opens_the_book_as_cash_and_sends_nothing(monkeypatch):
    monkeypatch.setenv("TREND_CORE_WEIGHT", "0.3")
    sent = []
    assert _check(_walk(), _walk().index[-1], sent) is False
    entry = trend._load()["BTCUSDT"]
    assert sent == [] and entry["pending"] == [] and entry["ledger"] == [] and entry["core"] == 0.3
    assert len(entry["days"]) == 1 and entry["start"]["date"] == entry["days"][0]["date"] and entry["source"] == "binance"


# --- finding 3: one delivery record per chat ---------------------------------------------------


@pytest.mark.unit
def test_one_failing_chat_blocks_no_other_and_only_it_is_retried():
    frame = _crafted()
    state, flips = _flips(frame)
    flip = flips[0]
    _check(frame, flip - timedelta(days=1), [])  # baseline the day before the first flip
    sent = []
    with pytest.raises(trend.SendFailed):
        _check(frame, flip, sent, CHATS, fail=("b",))
    assert sorted(c for c, _ in sent) == ["a", "c"]  # b's failure did not stop c
    assert _check(frame, flip, sent, CHATS) is True  # retry reaches only b
    assert sorted(c for c, _ in sent) == ["a", "b", "c"]
    assert _check(frame, flip, sent, CHATS) is False and len(sent) == 3  # nobody gets it twice
    row = trend._load()["BTCUSDT"]["ledger"][0]
    assert sorted(row["sent"]) == CHATS and trend._load()["BTCUSDT"]["pending"] == []


@pytest.mark.unit
def test_a_chat_added_later_still_gets_the_undelivered_alert():
    frame = _crafted()
    flip = _flips(frame)[1][0]
    _check(frame, flip - timedelta(days=1), [])
    sent = []
    with pytest.raises(trend.SendFailed):
        _check(frame, flip, sent, ("a", "b"), fail=("a", "b"))
    _check(frame, flip, sent, ("a", "b"))
    assert sorted(c for c, _ in sent) == ["a", "b"]


# --- finding 2: days missed while the bot was down ----------------------------------------------


@pytest.mark.unit
def test_an_outage_with_two_flips_is_flagged_alerted_once_and_kept_out_of_the_operated_book():
    frame = _crafted()
    state, flips = _flips(frame)
    start = flips[0] - timedelta(days=20)  # HOLD, before the fall
    _check(frame, start, [])
    sent = []
    last = frame.index[-1]  # HOLD again: the bot was down through HOLD -> CASH -> HOLD
    assert state[start] == state[last] == trend.HOLD
    _check(frame, last, sent, CHATS)
    entry = trend._load()["BTCUSDT"]
    assert len(sent) == 3 and len({t for _, t in sent}) == 1  # one combined message, to every chat
    text = sent[0][1]
    assert "offline from" in text and all(str(f.date()) in text for f in flips[:2]) and "could not be acted on in time" in text
    assert _telegram_html(text)
    assert [r["late"] for r in entry["ledger"]] == [True, True] and len(entry["ledger"]) == 2
    gap = entry["gaps"][0]
    assert gap["from"] == (start + timedelta(days=1)).date().isoformat() and gap["to"] == (last - timedelta(days=1)).date().isoformat()
    assert [e["event"] for e in _events()].count("outage") == 1
    out = trend.paper_text("BTCUSDT")
    assert "Equity, rule</b> (every change filled)" in out and "Equity, as operated" in out and "offline, position held through them" in out
    assert _telegram_html(out)
    frame_days, rule, operated = trend._book(entry)
    assert (operated == trend.HOLD).all() and (rule == trend.CASH).any()  # the operated book never left the market
    ideal, ops = trend.simulate(frame_days, rule), trend.simulate(frame_days, operated)
    assert ideal["trend"].iloc[-1] != pytest.approx(ops["trend"].iloc[-1])


@pytest.mark.unit
def test_an_outage_without_a_flip_is_recorded_but_sends_nothing():
    frame = _crafted()
    _check(frame, frame.index[250], [])
    sent = []
    _check(frame, frame.index[255], sent)  # five days offline, still HOLD
    entry = trend._load()["BTCUSDT"]
    assert sent == [] and entry["gaps"][0]["days"] == 4 and entry["ledger"] == []
    assert [d["live"] for d in entry["days"]] == [True, False, False, False, False, True]


@pytest.mark.unit
def test_the_operated_book_moves_when_the_bot_is_back_after_an_outage():
    frame = _crafted()
    state, flips = _flips(frame)
    _check(frame, flips[0] - timedelta(days=10), [])
    _check(frame, flips[0] + timedelta(days=5), [])  # back while CASH: the operated book follows the current state
    _, _, operated = trend._book(trend._load()["BTCUSDT"])
    assert operated.iloc[0] == trend.HOLD and operated.iloc[1:-1].eq(trend.HOLD).all() and operated.iloc[-1] == trend.CASH


# --- finding 1: the book's allocation is fixed at its start -------------------------------------


@pytest.mark.unit
def test_book_starts_as_cash_with_the_configured_split_and_ignores_later_env_changes(monkeypatch):
    monkeypatch.setenv("TREND_CORE_WEIGHT", "0.5")
    frame = _walk()
    state = trend.states(frame["Close"])
    start = state.index[400]
    sent = []
    _replay(frame, sent, start)
    entry = trend._load()["BTCUSDT"]
    days, rule, _ = trend._book(entry)
    book = trend.simulate(days, rule, entry["core"])
    assert book["mix"].iloc[0] == 1.0 and book["hold"].iloc[0] == 1.0  # all cash at the first close
    # the core sleeve is its own half of the capital, bought at the next open: exactly half the hold equity
    assert book["hold"].iloc[-1] == pytest.approx((1 - trend.FEE) * days["Close"].iloc[-1] / days["Open"].iloc[1])
    assert book["mix"].iloc[-1] == pytest.approx(0.5 * book["hold"].iloc[-1] + 0.5 * book["trend"].iloc[-1])
    before = trend.paper_text("BTCUSDT")
    monkeypatch.setenv("TREND_CORE_WEIGHT", "0.9")
    assert trend.paper_text("BTCUSDT") == before and "core 50% fixed at the start" in before
    assert "Split: 50% core BTC" in sent[0][1]


# --- finding 4: data continuity, source and revisions --------------------------------------------


@pytest.mark.unit
def test_a_missing_day_in_the_window_makes_no_signal_and_changes_nothing():
    frame = _crafted()
    _check(frame, frame.index[250], [])
    holed = frame.drop(frame.index[200])
    sent = []
    with pytest.raises(trend.DataGap):
        _check(holed, frame.index[270], sent)
    with mock.patch.object(trend, "bars", return_value=(holed[:frame.index[270]], "binance")), pytest.raises(trend.DataGap):
        trend.latest("BTCUSDT", _at(frame.index[270]), holed[:frame.index[270]])
    entry = trend._load()["BTCUSDT"]
    assert sent == [] and entry["checked"] == frame.index[250].date().isoformat() and len(entry["days"]) == 1


@pytest.mark.unit
def test_a_revised_stored_close_is_a_data_event_not_a_new_signal():
    frame = _crafted()
    state, flips = _flips(frame)
    day = flips[0] - timedelta(days=5)
    _replay(frame, [], frame.index[250])  # up to the day before the first flip would be... processed live
    sent = []
    # the vendor now serves a very different close for an already processed day, which would flip its state
    revised = frame.copy()
    target = frame.index[270]
    revised.loc[target, "Close"] = frame.loc[target, "Close"] * 0.5
    entry = trend._load()["BTCUSDT"]
    stored = next(d["Close"] for d in entry["days"] if d["date"] == target.date().isoformat())
    _check(revised, frame.index[-1], sent)
    entry = trend._load()["BTCUSDT"]
    assert next(d["Close"] for d in entry["days"] if d["date"] == target.date().isoformat()) == stored  # kept
    assert any("is now" in e and str(target.date()) in e for e in entry["data_events"])
    assert "Data check:" in trend.paper_text("BTCUSDT") and day  # shown to the user
    assert [e["event"] for e in _events()].count("data_event") >= 1
    assert all(r["signal_date"] != target.date().isoformat() for r in entry["ledger"])  # no flip from the revision


@pytest.mark.unit
def test_a_source_change_is_recorded_as_a_data_event():
    frame = _crafted()
    _check(frame, frame.index[250], [], source="binance")
    _check(frame, frame.index[251], [], source="yahoo")
    entry = trend._load()["BTCUSDT"]
    assert entry["source"] == "yahoo" and "source changed from binance to yahoo (verified)" in entry["data_events"]
    assert [d["source"] for d in entry["days"]] == ["binance", "yahoo"]


def _other_history(frame, day, factor=3.0):
    """Bars whose last 5 closes are the same as ``frame``'s but whose older history is another price level."""
    alt = frame[:day].copy()
    alt.loc[alt.index[:-5], ["Open", "Close"]] *= factor
    return alt


@pytest.mark.unit
def test_an_unverified_source_switch_makes_no_signal_and_the_stored_source_returning_ends_it():
    frame = _crafted()
    _check(frame, frame.index[250], [], CHATS)  # HOLD, from binance
    day = frame.index[251]
    alt = _other_history(frame, day)
    assert alt["Close"].iloc[-1] == frame.loc[day, "Close"]  # today's close is unchanged...
    assert trend.states(alt["Close"]).iloc[-1] == trend.CASH  # ...but the other history's SMA200 says CASH
    sent = []
    with mock.patch.object(trend, "bars", return_value=(alt, "yahoo")), pytest.raises(trend.SourceUnverified):
        trend.check("BTCUSDT", lambda c, t: sent.append((c, t)), list(CHATS), _at(day))
    entry = trend._load()["BTCUSDT"]
    assert sent == [] and entry["ledger"] == [] and entry["pending"] == [] and entry["state"] == trend.HOLD
    assert entry["source"] == "binance" and entry["checked"] == frame.index[250].date().isoformat() and len(entry["days"]) == 1
    assert "does not match" in entry["data_events"][-1] and "Data check:" in trend.paper_text("BTCUSDT")
    with mock.patch.object(trend, "bars", return_value=(alt, "yahoo")), pytest.raises(trend.SourceUnverified):
        trend.check("BTCUSDT", lambda c, t: sent.append((c, t)), list(CHATS), _at(day))  # a retry
    assert sent == [] and len(trend._load()["BTCUSDT"]["data_events"]) == 1  # logged once, not on every retry
    _check(frame, day, sent, CHATS)  # binance is back: the day is processed normally
    assert sent == [] and trend._load()["BTCUSDT"]["checked"] == day.date().isoformat()


@pytest.mark.unit
def test_a_source_switch_does_not_signal_a_flip_inside_the_sources_own_noise():
    frame = _crafted()
    flip = _flips(frame)[1][0]
    _check(frame, flip - timedelta(days=1), [], CHATS)
    near = frame[:flip].copy()
    sma = near["Close"].rolling(trend.WINDOW).mean()
    near.loc[flip, "Close"] = sma[flip] * (1 - trend.REVISION / 2)  # just under the SMA: CASH here, HOLD to a vendor 0.3% away
    assert trend.states(near["Close"]).iloc[-1] == trend.CASH
    sent = []
    with mock.patch.object(trend, "bars", return_value=(near, "yahoo")), pytest.raises(trend.SourceUnverified):
        trend.check("BTCUSDT", lambda c, t: sent.append((c, t)), list(CHATS), _at(flip))
    assert sent == [] and trend._load()["BTCUSDT"]["ledger"] == []


@pytest.mark.unit
def test_bars_fall_back_to_yahoo_without_gap_filling_and_without_todays_bar():
    today = datetime.now(UTC)
    days = pd.date_range(end=today.date(), periods=5, freq="D")
    yahoo = pd.DataFrame({"Date": days, "Open": [1.0, 2, 3, 4, 5], "Close": [1.5, 2, 3, 4, 5]})
    with mock.patch.object(trend, "fetch_daily_klines", return_value=None), \
            mock.patch.object(trend, "load_ohlcv", return_value=yahoo) as load:
        out, source = trend.bars("BTCUSDT", today)
    assert source == "yahoo" and load.call_args.kwargs["fill_gaps"] is False
    assert out.index[-1].date() < today.date() and list(out["Close"]) == [1.5, 2, 3, 4]


# --- finding 5: the first fill is at the next open ------------------------------------------------


@pytest.mark.unit
def test_the_first_trade_fills_at_the_next_open_not_at_the_signal_close():
    frame = pd.DataFrame({"Open": [110.0, 200.0, 200.0], "Close": [110.0, 200.0, 200.0]},
                         index=pd.date_range("2026-01-01", periods=3, freq="D"))
    state = pd.Series(trend.HOLD, index=frame.index)
    book = trend.simulate(frame, state, 0.5)
    assert book["trend"].iloc[0] == 1.0  # cash at the signal close
    assert book["trend"].tolist() == pytest.approx([1.0, 1 - trend.FEE, 1 - trend.FEE])  # not +81.8%
    assert book["hold"].tolist() == pytest.approx([1.0, 1 - trend.FEE, 1 - trend.FEE])


# --- finding 6: the backtest evaluates from the day asked ------------------------------------------


@pytest.mark.unit
def test_bitstamp_download_starts_the_warm_up_before_the_requested_day():
    calls = []

    def get(url, params, timeout):
        calls.append(params["start"])
        response = mock.Mock()
        first = int(datetime(2013, 6, 15, tzinfo=UTC).timestamp())
        rows = [{"timestamp": str(first + 86400 * i), "open": "1", "close": "1"} for i in range(5)]
        response.json.return_value = {"data": {"ohlc": rows if len(calls) == 1 else []}}
        return response

    with mock.patch.object(trend_backtest.requests, "get", side_effect=get):
        trend_backtest.bitstamp("2014-01-01")
    assert calls[0] == int((datetime(2014, 1, 1, tzinfo=UTC) - timedelta(days=199)).timestamp())


@pytest.mark.unit
def test_report_evaluates_from_the_requested_start_and_flags_one_without_warm_up():
    idx = pd.date_range("2013-06-15", periods=600, freq="D")
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.03, 600)))
    frame = pd.DataFrame({"Open": np.r_[close[0], close[:-1]], "Close": close}, index=idx)
    out = trend_backtest.report(frame, 0.5, "2014-01-01")
    assert "evaluated 2014-01-01 to" in out and "note:" not in out
    # data that begins on the start day has no SMA200 yet: the first evaluated day is 199 days later and says so
    bare = trend_backtest.report(frame.loc["2014-01-01":], 0.5, "2014-01-01")
    assert "note: no full SMA200 window at 2014-01-01" in bare and "evaluated 2014-07-19 to" in bare
    assert "100% SMA200 / cash (the validated rule)" in out and "50% core + 50% SMA200 (added strategy)" in out


# --- messages, ledger and the rest -------------------------------------------------------------------


@pytest.mark.unit
def test_alert_does_not_promise_the_open_price_and_warns_about_whipsaw():
    info = {"symbol": "BTCUSDT", "state": trend.HOLD, "date": "2026-10-21", "close": 84093.4, "sma": 72774.0, "since": "2026-10-21"}
    text = trend.alert_text(info, "2026-10-22T00:06:11+00:00", 0.5)
    assert "<pre>Close 2026-10-21  84,093\nSMA200            72,774\nGap               +15.6%</pre>" in text
    assert "Detected 00:06 UTC" in text and "no order was placed" in text and "you cannot get that price" in text
    assert "29 of 44 closed trades" in text and "Split: 50% core BTC (never sold) · 50% follows" in text and _telegram_html(text)


@pytest.mark.unit
def test_ledger_keeps_detection_and_send_time_apart_and_fills_the_open_from_the_stored_day():
    frame = _crafted()
    flip = _flips(frame)[1][0]
    _check(frame, flip - timedelta(days=1), [])
    _check(frame, flip, [])
    row = trend._load()["BTCUSDT"]["ledger"][0]
    assert row["signal_date"] == flip.date().isoformat() and row["signal_close"] == pytest.approx(frame.loc[flip, "Close"])
    assert row["fill_date"] == (flip + timedelta(days=1)).date().isoformat() and row["fill_open"] is None
    assert row["detected_at"] and row["sent"]["a"] >= row["detected_at"] and row["late"] is False
    nxt = flip + timedelta(days=1)
    _check(frame, nxt, [])
    assert trend._load()["BTCUSDT"]["ledger"][0]["fill_open"] == pytest.approx(frame.loc[nxt, "Open"])


@pytest.mark.unit
def test_the_streak_start_is_reported():
    frame = _walk()
    state, flips = _flips(frame)
    with mock.patch.object(trend, "bars", return_value=(frame, "binance")):
        assert trend.latest("BTCUSDT", datetime.now(UTC))["since"] == flips[-1].date().isoformat() and state.iloc[-1]


@pytest.mark.unit
def test_a_bar_older_than_yesterday_is_stale_and_changes_nothing():
    frame = _walk()
    with mock.patch.object(trend, "bars", return_value=(frame[:-3], "binance")), pytest.raises(trend.StaleData):
        trend.check("BTCUSDT", lambda c, t: None, ["a"], datetime.now(UTC))
    assert not trend._load()


@pytest.mark.unit
def test_status_text_only_serves_the_validated_symbol():
    assert "only validated on BTCUSDT" in trend.status_text("ETHUSDT")
    with mock.patch.object(trend, "bars", return_value=(_walk(), "binance")):
        out = trend.status_text("btcusdt")
    assert out.startswith("<b>BTCUSDT trend: ") and "SMA200" in out and "Whipsaw" in out and _telegram_html(out)


@pytest.mark.unit
def test_paper_text_shows_one_equity_line_without_an_outage_and_the_ledger():
    frame = _crafted()
    flip = _flips(frame)[1][0]
    _replay(frame, [], flip - timedelta(days=10))
    out = trend.paper_text("BTCUSDT")
    assert "paper book</b>" in out and "no real orders" in out and "Equity, rule</b>" in out and "Mix (core + rule)" in out
    assert "as operated" not in out and "Ledger" in out and "→ open" in out and "Whipsaw" not in out and _telegram_html(out)
    assert trend.paper_text("ETHUSDT") == "No paper book yet: the first daily check has not run."


@pytest.mark.unit
def test_seconds_to_next_run_is_just_after_midnight_utc():
    assert trend._seconds_to_run(datetime(2026, 10, 22, 0, 1, tzinfo=UTC)) == 4 * 60
    assert trend._seconds_to_run(datetime(2026, 10, 22, 0, 6, tzinfo=UTC)) == 24 * 3600 - 60


@pytest.mark.unit
def test_watch_retries_then_reports_a_check_that_never_succeeded(monkeypatch):
    calls, naps, sent = [], [], []
    monkeypatch.setattr(trend, "RETRIES", 2)

    def failing(symbol, send, chats):
        calls.append(symbol)
        raise trend.DataGap("1 day(s) missing")

    def nap(seconds):
        naps.append(seconds)
        if seconds != trend.RETRY_SECONDS:  # the sleep until tomorrow's run: stop the endless loop here
            raise SystemExit

    monkeypatch.setattr(trend, "check", failing)
    monkeypatch.setattr(trend.time, "sleep", nap)
    with pytest.raises(SystemExit):
        trend.watch(lambda chat, text: sent.append((chat, text)), ["a", "b"])
    assert calls == ["BTCUSDT", "BTCUSDT"] and naps[:2] == [trend.RETRY_SECONDS] * 2
    assert [c for c, _ in sent] == ["a", "b"] and "No signal was made today" in sent[0][1] and _telegram_html(sent[0][1])
    assert "check_failed" in [e["event"] for e in _events()]


@pytest.mark.unit
def test_watch_stays_quiet_when_a_retry_succeeds(monkeypatch):
    calls, sent = [], []

    def flaky(symbol, send, chats):
        calls.append(symbol)
        if len(calls) == 1:
            raise OSError("binance down")

    def nap(seconds):
        if seconds != trend.RETRY_SECONDS:
            raise SystemExit

    monkeypatch.setattr(trend, "check", flaky)
    monkeypatch.setattr(trend.time, "sleep", nap)
    with pytest.raises(SystemExit):
        trend.watch(lambda chat, text: sent.append(text), ["a"])
    assert calls == ["BTCUSDT", "BTCUSDT"] and sent == []


# --- the book model --------------------------------------------------------------------------------


def _loop_equity(frame, state, weight, fee):
    """The same book as an explicit cash-and-units loop: cash on day 0, the signal fills at the next open."""
    want = (state == trend.HOLD).tolist()
    core_cash, trend_cash, core_units, trend_units = weight, 1 - weight, 0.0, 0.0
    out = [1.0]
    for i in range(1, len(frame)):
        row = frame.iloc[i]
        if core_cash:  # the core sleeve buys at the first open and never sells
            core_units, core_cash = core_cash * (1 - fee) / row["Open"], 0.0
        target = want[i - 1]
        if target and trend_units == 0:
            trend_units, trend_cash = trend_cash * (1 - fee) / row["Open"], 0.0
        elif not target and trend_units > 0:
            trend_units, trend_cash = 0.0, trend_units * row["Open"] * (1 - fee)
        out.append(core_units * row["Close"] + trend_cash + trend_units * row["Close"])
    return pd.Series(out, index=frame.index)


@pytest.mark.unit
@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_simulate_matches_an_independent_loop(weight):
    frame = _walk()
    state = trend.states(frame["Close"])
    frame = frame.loc[state.index[0]:]
    got = trend.simulate(frame, state, weight)["mix"]
    assert got.to_numpy() == pytest.approx(_loop_equity(frame, state, weight, trend.FEE).to_numpy(), rel=1e-9)


@pytest.mark.unit
def test_core_weight_comes_from_the_environment_and_is_validated(monkeypatch):
    assert trend.core_weight() == 0.5
    monkeypatch.setenv("TREND_CORE_WEIGHT", "0.8")
    assert trend.core_weight() == 0.8
    monkeypatch.setenv("TREND_CORE_WEIGHT", "1.5")
    with pytest.raises(ValueError):
        trend.core_weight()


# --- /paper shows the real BTC weight, filled apart from pending -----------------------------------


def _paper_expected(frame, start, last, weight=0.5):
    """BTC value / book value at ``last``, from an explicit units loop, and whether the sleeve holds BTC."""
    part = frame.loc[start:last]
    state = trend.states(frame["Close"]).loc[start:last]
    total = _loop_equity(part, state, weight, trend.FEE).iloc[-1]
    core_units = weight * (1 - trend.FEE) / part["Open"].iloc[1]
    filled = (state.shift().iloc[-1] == trend.HOLD)
    sleeve_cash = (1 - weight) * (_loop_equity(part, state, 0.0, trend.FEE).iloc[-1])
    btc = core_units * part["Close"].iloc[-1] + (0 if not filled else sleeve_cash)
    return btc / total, filled


@pytest.mark.unit
def test_paper_text_reports_the_real_btc_weight_after_the_core_sleeve_drifted():
    frame = _crafted()
    state, flips = _flips(frame)
    start, last = frame.index[250], frame.index[299]  # the price fell 70% from its peak, sleeves are not rebalanced
    assert state[last] == trend.CASH and flips[0] < last - timedelta(days=2)
    _replay(frame[:last], [], start)
    expected, filled = _paper_expected(frame, start, last)
    out = trend.paper_text("BTCUSDT")
    assert not filled and expected < 0.45
    assert _btc_line(out) == f"{expected:.1%}" and "Pending at the next open" not in out


@pytest.mark.unit
def test_paper_text_keeps_a_pending_signal_apart_from_the_filled_position():
    frame = _crafted()
    flip = _flips(frame)[1][0]  # the first flip, to CASH
    start = frame.index[250]
    _replay(frame[:flip], [], start)  # stop on the signal day itself: still in BTC, the sale waits for the next open
    out = trend.paper_text("BTCUSDT")
    expected, filled = _paper_expected(frame, start, flip)
    assert filled and _btc_line(out) == f"{expected:.1%}"
    assert f"Pending at the next open: trend sleeve to CASH (signal of {flip.date()})" in out
    _check(frame, flip + timedelta(days=1), [])  # the next open has filled it
    out = trend.paper_text("BTCUSDT")
    expected, filled = _paper_expected(frame, start, flip + timedelta(days=1))
    assert not filled and _btc_line(out) == f"{expected:.1%}" and "Pending at the next open" not in out


# --- an undelivered alert stays queued when a newer one is made -------------------------------------


@pytest.mark.unit
def test_an_older_undelivered_alert_stays_queued_when_the_signal_reverses_next_day():
    close = np.r_[100 + 0.5 * np.arange(260), 150.0, 235.0, 236.0, 237.0]  # one day under the SMA200, then back above
    frame = _frame(close)
    state, flips = _flips(frame)
    assert [f.date() for f in flips] == [frame.index[260].date(), frame.index[261].date()]
    _check(frame, frame.index[259], [], CHATS)
    sent = []
    with pytest.raises(trend.SendFailed):
        _check(frame, frame.index[260], sent, CHATS, fail=("b",))  # b misses the CASH alert
    assert sorted(c for c, _ in sent) == ["a", "c"]
    _check(frame, frame.index[261], sent, CHATS)  # the next day reverses; b is reachable again
    to_b = [t for c, t in sent if c == "b"]
    assert len(to_b) == 2 and "DOWN" in to_b[0] and "UP" in to_b[1]  # both, in order
    assert [t for c, t in sent if c == "a"] == [t for c, t in sent if c == "c"] and len([1 for c, _ in sent if c == "a"]) == 2
    entry = trend._load()["BTCUSDT"]
    assert entry["pending"] == [] and all(sorted(r["sent"]) == CHATS for r in entry["ledger"])
    _check(frame, frame.index[261], sent, CHATS)
    assert len(sent) == 6  # nothing is repeated


# --- plurals ----------------------------------------------------------------------------------------


@pytest.mark.unit
def test_one_day_is_not_written_as_1_days():
    assert trend._days(1) == "1 day" and trend._days(2) == "2 days" and trend._days(0) == "0 days"
    info = {"symbol": "BTCUSDT", "state": trend.HOLD, "date": "2026-10-21", "close": 100.0, "sma": 90.0, "since": "2026-10-21"}
    row = {"signal_date": "2026-10-20", "state": trend.HOLD, "signal_close": 99.0}
    one = trend.late_text([row], info, {"from": "2026-10-20", "to": "2026-10-20", "days": 1}, "2026-10-22T00:05:00+00:00", 0.5)
    assert "(1 day)" in one and "1 signal change was found late" in one and "1 days" not in one
    two = trend.late_text([row, row], info, {"from": "2026-10-19", "to": "2026-10-20", "days": 2}, "2026-10-22T00:05:00+00:00", 0.5)
    assert "(2 days)" in two and "2 signal changes were found late" in two
    frame = _crafted()
    _check(frame, frame.index[250], [])
    _check(frame, frame.index[251], [])
    assert "(1 day, no real orders)" in trend.paper_text("BTCUSDT")
    with pytest.raises(trend.DataGap, match="1 day missing"):
        trend.require_continuous(frame.drop(frame.index[-3]))


# --- one ops row per run, health, dead-man -----------------------------------------------------------


def _watch_once(monkeypatch, frame, send=lambda chat, text: None):
    """One ``watch`` run on ``frame``'s bars; the sleep until tomorrow ends the loop."""
    monkeypatch.setattr(trend, "bars", lambda symbol, today=None: (frame, "binance"))

    def stop(seconds):
        raise SystemExit

    monkeypatch.setattr(trend.time, "sleep", stop)
    with pytest.raises(SystemExit):
        trend.watch(send, ["a"])


@pytest.mark.unit
def test_each_run_leaves_one_ops_row_with_the_data_date_equity_and_latency(monkeypatch):
    frame = _frame(np.r_[100 + 0.5 * np.arange(260), 150.0])  # the last closed bar dips under the SMA200: a flip to send
    flip = frame.index[-1]
    assert _flips(frame)[1][-1] == flip
    _check(frame, frame.index[-2], [])
    sent = []
    _watch_once(monkeypatch, frame, lambda chat, text: sent.append((chat, text)))
    (row,) = trend._ops_rows()
    assert row["ok"] is True and row["tries"] == 1 and row["errors"] == [] and row["symbol"] == "BTCUSDT"
    assert row["data_date"] == flip.date().isoformat() and row["state"] == trend.CASH and row["flips"] == [flip.date().isoformat()]
    assert set(row["equity"]) == {"hold", "trend", "mix"} and row["send_s"] >= 0 and row["lag_s"] >= 0
    entry = trend._load()["BTCUSDT"]
    assert entry["last_run"]["data_date"] == flip.date().isoformat() and len(sent) == 1
    # a second run the same day with nothing new is another row, with no flip and no latency
    _watch_once(monkeypatch, frame, lambda chat, text: sent.append((chat, text)))
    assert [r["flips"] for r in trend._ops_rows()] == [[flip.date().isoformat()], []] and trend._ops_rows()[1]["lag_s"] is None


@pytest.mark.unit
def test_a_run_that_never_succeeds_leaves_one_failed_row_with_its_errors(monkeypatch):
    monkeypatch.setattr(trend, "RETRIES", 2)
    monkeypatch.setattr(trend, "bars", mock.Mock(side_effect=OSError("no network")))
    naps = []

    def nap(seconds):
        naps.append(seconds)
        if seconds != trend.RETRY_SECONDS:
            raise SystemExit

    monkeypatch.setattr(trend.time, "sleep", nap)
    with pytest.raises(SystemExit):
        trend.watch(lambda chat, text: None, ["a"])
    (row,) = trend._ops_rows()
    assert row["ok"] is False and row["tries"] == 2 and len(row["errors"]) == 2 and "no network" in row["errors"][0]


@pytest.mark.unit
def test_health_shows_the_last_completed_run_and_flags_an_overdue_one():
    now = datetime(2026, 10, 23, 8, 0, tzinfo=UTC)
    assert "no run has completed yet" in trend.health_text("BTCUSDT", now)
    frame = _crafted()
    _check(frame, frame.index[250], [])
    entry = trend._load()["BTCUSDT"]
    ran = datetime.fromisoformat(entry["last_run"]["at"])
    today = ran.replace(hour=12, minute=0)
    text = trend.health_text("BTCUSDT", today)
    assert text.startswith("✅ <b>BTCUSDT job: OK</b>") and _telegram_html(text)
    assert "Last completed run" in text and f"Data through {frame.index[250].date()}" in text and "Undelivered alerts  0" in text
    tomorrow = today + timedelta(days=1)
    assert trend.overdue(entry, tomorrow.replace(hour=0, minute=29)) is False  # not yet due
    assert trend.overdue(entry, tomorrow.replace(hour=0, minute=31)) is True
    assert "OVERDUE" in trend.health_text("BTCUSDT", tomorrow.replace(hour=3))
    trend._ops(symbol="BTCUSDT", ok=False, tries=12, errors=["HTTPError('<Response [451]> & more')"])
    assert "&lt;Response [451]&gt; &amp; more" in (text := trend.health_text("BTCUSDT", today)) and _telegram_html(text)


@pytest.mark.unit
def test_dead_man_alerts_once_a_day_and_only_when_no_run_completed():
    sent = []
    frame = _crafted()
    _check(frame, frame.index[250], [])
    ran = datetime.fromisoformat(trend._load()["BTCUSDT"]["last_run"]["at"])
    day2 = (ran + timedelta(days=1)).replace(hour=0, minute=31)
    send = lambda chat, text: sent.append((chat, text))  # noqa: E731
    assert trend.alert_overdue("BTCUSDT", send, ["a", "b"], ran.replace(hour=0, minute=31)) is False  # a run completed today
    assert trend.alert_overdue("BTCUSDT", send, ["a", "b"], day2) is True
    assert sorted(c for c, _ in sent) == ["a", "b"] and "No run has completed since" in sent[0][1]
    assert trend.alert_overdue("BTCUSDT", send, ["a", "b"], day2.replace(minute=40)) is False and len(sent) == 2  # once a day
    assert trend.alert_overdue("BTCUSDT", send, ["a", "b"], day2 + timedelta(days=1)) is True  # and again tomorrow
    _check(frame, frame.index[251], [])  # the job completed a run: tomorrow-after-that stays quiet
    ran = datetime.fromisoformat(trend._load()["BTCUSDT"]["last_run"]["at"])
    assert trend.alert_overdue("BTCUSDT", send, ["a"], ran.replace(hour=0, minute=45)) is False


@pytest.mark.unit
def test_dead_man_alerts_when_the_job_never_completed_and_survives_a_dead_chat():
    def send(chat, text):
        if chat == "a":
            raise OSError("down")
        sent.append(chat)

    sent = []
    assert trend.alert_overdue("BTCUSDT", send, ["a", "b"], datetime(2026, 10, 23, 1, 0, tzinfo=UTC)) is True
    assert sent == ["b"] and any(e["event"] == "send_failed" for e in _events())


# --- the price at send time, next to the open the backtest uses -----------------------------------------


@pytest.mark.unit
def test_a_live_signal_records_the_price_at_send_next_to_the_open(monkeypatch):
    frame = _crafted()
    flip = _flips(frame)[1][0]
    _check(frame, flip - timedelta(days=1), [])
    open_ = float(frame.loc[flip, "Close"])
    monkeypatch.setattr(trend, "quote", lambda symbol: {"price": open_ * 1.004, "open": open_, "at": "2026-10-22T00:05:07+00:00"})
    sent = []
    _check(frame, flip, sent, CHATS)
    row = trend._load()["BTCUSDT"]["ledger"][0]
    assert row["quote"]["price"] == pytest.approx(open_ * 1.004) and row["quote"]["open"] == open_
    out = trend.paper_text("BTCUSDT")
    assert "price at send" in out and "+0.40% vs open" in out and _telegram_html(out)


@pytest.mark.unit
def test_no_price_never_blocks_the_alert_and_a_late_signal_gets_none(monkeypatch):
    frame = _crafted()
    flip = _flips(frame)[1][0]
    _check(frame, flip - timedelta(days=1), [])
    sent = []
    _check(frame, flip, sent, CHATS)  # quote() is None here, as when Binance cannot be reached
    assert len(sent) == 3 and "quote" not in trend._load()["BTCUSDT"]["ledger"][0]
    asked = []
    monkeypatch.setattr(trend, "quote", lambda symbol: asked.append(symbol) or {"price": 1.0, "open": 1.0, "at": "x"})
    _check(frame, frame.index[-1], [], CHATS)  # the second flip arrives after a 40-day outage: late, nobody could act on it
    assert trend._load()["BTCUSDT"]["ledger"][-1]["late"] is True and asked == []


@pytest.mark.unit
def test_quote_reads_the_price_and_the_open_and_returns_none_when_binance_is_down(monkeypatch):
    bar = [0, "84000.0", "0", "0", "84210.5", "0"]
    response = mock.Mock(**{"json.return_value": [bar], "raise_for_status.return_value": None})
    monkeypatch.undo()  # the autouse stub would hide quote
    monkeypatch.setattr(trend, "DIR", "/tmp")
    with mock.patch.object(trend.requests, "get", return_value=response) as get:
        out = trend.quote("BTCUSDT")
    assert out["price"] == 84210.5 and out["open"] == 84000.0 and get.call_args.kwargs["params"]["symbol"] == "BTCUSDT"
    with mock.patch.object(trend.requests, "get", side_effect=trend.requests.ConnectionError("down")):
        assert trend.quote("BTCUSDT") is None
    assert trend.quote("AAPL") is None  # not a crypto pair


# --- the delivery drill, through the real send path --------------------------------------------------------


class _FakeTelegram:
    """A local Telegram Bot API: records every sendMessage and can answer 500 for a chat."""

    def __init__(self):
        self.log, self.down, self.modes = [], set(), []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if str(body["chat_id"]) in outer.down:
                    self.send_response(500)
                    self.end_headers()
                    return
                outer.log.append((str(body["chat_id"]), body["text"]))
                outer.modes.append(body.get("parse_mode"))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok": true, "result": {}}')

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/bottest"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def telegram(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")  # a proxy in the environment must not sit between the test and its fake
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    # the suite blocks the network; open loopback only, so the fake is reachable and nothing else is
    real_connect, real_lookup = _socket.socket.connect, _socket.getaddrinfo

    def loopback_only(self, address):
        if address[0] != "127.0.0.1":
            raise OSError(f"test tried to reach the network: {address}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", loopback_only)

    def lookup(host, *args, **kwargs):
        if host != "127.0.0.1":
            raise OSError(f"test tried to reach the network: {host}")
        return real_lookup(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    import telegram_bot
    mod = importlib.reload(telegram_bot)
    fake = _FakeTelegram()
    monkeypatch.setattr(mod, "API", fake.url)
    yield mod, fake
    fake.server.shutdown()


@pytest.mark.unit
def test_the_delivery_drill_passes_through_the_real_send_path(telegram):
    mod, fake = telegram
    results = trend_drill.run(mod.say_html, ["111", "222"])
    assert all(ok for _, ok, _ in results), results
    for chat in ("111", "222"):  # the HTTP requests really arrived, in order, each exactly once
        texts = [t for c, t in fake.log if c == chat]
        assert len(texts) == 2 and "trend UP" in texts[0] and "trend DOWN" in texts[1] and texts[0].startswith(trend_drill.PREFIX)
    assert fake.modes and set(fake.modes) == {"HTML"}
    assert trend.DIR != "/tmp" and not any(x.endswith("state.json") for x in __import__("os").listdir(trend.DIR))  # the live book is untouched


@pytest.mark.unit
def test_the_drill_runs_with_a_single_test_chat_and_fails_loudly_on_a_bad_chat(telegram):
    mod, fake = telegram
    assert all(ok for _, ok, _ in trend_drill.run(mod.say, ["111"]))
    with pytest.raises(ValueError):
        trend_drill.run(mod.say, ["111"], fail_chat="999")


@pytest.mark.unit
def test_a_telegram_http_error_is_a_failed_send_that_is_retried_for_that_chat_only(telegram):
    mod, fake = telegram
    frame = _crafted()
    flip = _flips(frame)[1][0]
    _check(frame, flip - timedelta(days=1), [])
    fake.down = {"b"}  # the server answers 500 for chat b
    with mock.patch.object(trend, "bars", return_value=(frame[:flip], "binance")), pytest.raises(trend.SendFailed):
        trend.check("BTCUSDT", mod.say, ["a", "b"], _at(flip))
    assert [c for c, _ in fake.log] == ["a"] and trend._load()["BTCUSDT"]["pending"][0]["sent_to"] == ["a"]
    fake.down = set()
    with mock.patch.object(trend, "bars", return_value=(frame[:flip], "binance")):
        trend.check("BTCUSDT", mod.say, ["a", "b"], _at(flip))
    assert [c for c, _ in fake.log] == ["a", "b"] and trend._load()["BTCUSDT"]["pending"] == []


@pytest.mark.unit
def test_the_drill_command_prints_pass_for_every_step_and_exits_zero(telegram, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["trend_drill", "--chat", "111", "--chat", "222"])
    assert trend_drill.main() == 0
    out = capsys.readouterr().out
    assert out.count("PASS") == 7 and "FAIL" not in out
