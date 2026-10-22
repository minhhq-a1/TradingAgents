"""BTC trend rule and paper book: flip dates against pandas, alert-once per chat across restarts and
outages, stored-day equity, data checks, first fill at the next open, backtest warm-up.

Prices are synthetic and the vendors are mocked, so these run without a network connection.
"""
import json
from datetime import UTC, datetime, timedelta
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from tradingagents import trend, trend_backtest

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
        assert f"UTC close {day.date()}" in text
        assert ("HOLD BTC" in text) == (state[day] == trend.HOLD)
    assert not trend._load()["BTCUSDT"]["gaps"]  # processed on time every day


@pytest.mark.unit
def test_first_run_opens_the_book_as_cash_and_sends_nothing(monkeypatch):
    monkeypatch.setenv("TREND_CORE_WEIGHT", "0.3")
    sent = []
    assert _check(_walk(), _walk().index[-1], sent) is False
    entry = trend._load()["BTCUSDT"]
    assert sent == [] and entry["pending"] is None and entry["ledger"] == [] and entry["core"] == 0.3
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
    assert sorted(row["sent"]) == CHATS and trend._load()["BTCUSDT"]["pending"] is None


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
    assert [r["late"] for r in entry["ledger"]] == [True, True] and len(entry["ledger"]) == 2
    gap = entry["gaps"][0]
    assert gap["from"] == (start + timedelta(days=1)).date().isoformat() and gap["to"] == (last - timedelta(days=1)).date().isoformat()
    assert [e["event"] for e in _events()].count("outage") == 1
    out = trend.paper_text("BTCUSDT")
    assert "Equity, rule (every change filled)" in out and "Equity, as operated" in out and "offline day(s)" in out
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
    assert "Book split: 50%" in sent[0][1]


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
    assert entry["source"] == "yahoo" and "source changed from binance to yahoo" in entry["data_events"]
    assert [d["source"] for d in entry["days"]] == ["binance", "yahoo"]


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
    assert "84,093 vs SMA200 72,774 (+15.6%)" in text and "Detected 00:06 UTC" in text
    assert "no order was placed" in text and "you cannot get that price" in text
    assert "29 of 44 closed trades lost money" in text and "50% stays in BTC permanently, 50% follows" in text


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
    assert out.startswith("BTCUSDT trend (paper): ") and "SMA200" in out


@pytest.mark.unit
def test_paper_text_shows_one_equity_line_without_an_outage_and_the_ledger():
    frame = _crafted()
    flip = _flips(frame)[1][0]
    _replay(frame, [], flip - timedelta(days=10))
    out = trend.paper_text("BTCUSDT")
    assert "paper book since" in out and "no real orders" in out and "Equity, rule (every change filled): mix" in out
    assert "as operated" not in out and "Ledger" in out and "-> open" in out and "Reference only: Whipsaw warning" in out
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
    assert [c for c, _ in sent] == ["a", "b"] and "No signal was made today" in sent[0][1]
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
