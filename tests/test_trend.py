"""BTC trend rule and paper book: flip dates against a pure pandas computation, alert-once across
restarts, the ledger, and the equity model against an independent day-by-day loop.

Prices are synthetic and the vendors are mocked, so these run without a network connection.
"""
import json
from datetime import UTC, datetime, timedelta
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from tradingagents import trend, trend_backtest


def _bars(days=900, seed=7):
    """A random walk that crosses its SMA200 many times, ending yesterday; each day opens near the last close."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.03, days)))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.005, days))
    end = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0).replace(tzinfo=None) - timedelta(days=1)
    return pd.DataFrame({"Open": open_, "Close": close}, index=pd.date_range(end=end, periods=days, freq="D"))


@pytest.fixture(autouse=True)
def _state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(trend, "DIR", str(tmp_path))
    monkeypatch.delenv("TREND_CORE_WEIGHT", raising=False)


def _events():
    with open(f"{trend.DIR}/log.jsonl", encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def _at(day):
    """The moment the check for the bar ``day`` runs: five minutes past the next midnight."""
    return datetime.combine(day.date() + timedelta(days=1), datetime.min.time(), UTC) + timedelta(minutes=5)


def _replay(frame, notify, first_day, restart_every=None):
    """Run ``check`` once per day as the bar for that day becomes the newest closed bar."""
    for n, day in enumerate(frame.index[frame.index >= first_day]):
        with mock.patch.object(trend, "bars", return_value=frame[: day]):
            trend.check("BTCUSDT", notify, _at(day))
            if restart_every and n % restart_every == 0:  # a restart re-checks the same bar
                trend.check("BTCUSDT", notify, _at(day))


@pytest.mark.unit
def test_states_match_a_plain_pandas_computation():
    close = _bars()["Close"]
    expected = (close > close.rolling(200).mean())[close.rolling(200).mean().notna()]
    got = trend.states(close)
    assert (got == trend.HOLD).equals(expected) and len(got) == len(close) - 199


@pytest.mark.unit
def test_alerts_fire_on_exactly_the_flip_days_and_never_twice():
    frame = _bars()
    close = frame["Close"]
    expected = (close > close.rolling(200).mean())[close.rolling(200).mean().notna()]
    flips = expected.index[expected.ne(expected.shift())][1:]  # the first day only sets the baseline
    sent = []
    _replay(frame, sent.append, expected.index[0], restart_every=3)
    assert len(flips) >= 6 and len(sent) == len(flips)
    for text, day in zip(sent, flips, strict=True):
        assert f"UTC close {day.date()}" in text
        assert ("HOLD BTC" in text) == bool(expected[day])
    assert [e["event"] for e in _events()].count("sent") == len(flips)


@pytest.mark.unit
def test_first_run_records_a_baseline_and_sends_nothing():
    sent = []
    with mock.patch.object(trend, "bars", return_value=_bars()):
        assert trend.check("BTCUSDT", sent.append) is False
    saved = trend._load()["BTCUSDT"]
    assert sent == [] and saved["pending"] is None and saved["ledger"] == [] and saved["start"]["state"] == saved["state"]


@pytest.mark.unit
def test_a_failed_send_is_retried_once_then_not_repeated():
    frame = _bars()
    state = trend.states(frame["Close"])
    flip = state.index[state.ne(state.shift())][-1]  # the last flip in the series
    with mock.patch.object(trend, "bars", return_value=frame[: flip - timedelta(days=1)]):
        trend.check("BTCUSDT", lambda t: None, _at(flip - timedelta(days=1)))  # baseline the day before the flip
    sent = []

    def down(_):
        raise OSError("telegram down")

    with mock.patch.object(trend, "bars", return_value=frame[:flip]):
        with pytest.raises(OSError):
            trend.check("BTCUSDT", down, _at(flip))
        assert trend.check("BTCUSDT", sent.append, _at(flip))
        assert not trend.check("BTCUSDT", sent.append, _at(flip))
    assert len(sent) == 1
    assert [e["event"] for e in _events()] == ["baseline", "detected", "send_failed", "sent"]


@pytest.mark.unit
def test_ledger_keeps_detection_and_send_time_apart_and_fills_the_open_later():
    frame = _bars()
    state = trend.states(frame["Close"])
    flip = state.index[state.ne(state.shift())][-1]
    with mock.patch.object(trend, "bars", return_value=frame[: flip - timedelta(days=1)]):
        trend.check("BTCUSDT", lambda t: None, _at(flip - timedelta(days=1)))
    with mock.patch.object(trend, "bars", return_value=frame[:flip]):
        trend.check("BTCUSDT", lambda t: None, _at(flip))
    row = trend._load()["BTCUSDT"]["ledger"][0]
    assert row["signal_date"] == flip.date().isoformat() and row["signal_close"] == pytest.approx(frame.loc[flip, "Close"])
    assert row["fill_date"] == (flip + timedelta(days=1)).date().isoformat()
    assert row["fill_open"] is None and row["detected_at"] and row["sent_at"] and row["sent_at"] >= row["detected_at"]
    nxt = flip + timedelta(days=1)  # the next day's bar closes: its open is now known
    with mock.patch.object(trend, "bars", return_value=frame[:nxt]):
        trend.check("BTCUSDT", lambda t: None, _at(nxt))
    assert trend._load()["BTCUSDT"]["ledger"][0]["fill_open"] == pytest.approx(frame.loc[nxt, "Open"])


@pytest.mark.unit
def test_alert_does_not_promise_the_open_price_and_warns_about_whipsaw():
    info = {"symbol": "BTCUSDT", "state": trend.HOLD, "date": "2026-10-21", "close": 84093.4, "sma": 72774.0, "since": "2026-10-21"}
    text = trend.alert_text(info, "2026-10-22T00:06:11+00:00")
    assert "84,093 vs SMA200 72,774 (+15.6%)" in text and "Detected 00:06 UTC" in text
    assert "no order was placed" in text and "you cannot get that price" in text
    assert "29 of 44 closed trades lost money" in text and "50% stays in BTC permanently, 50% follows" in text


@pytest.mark.unit
def test_the_streak_start_is_reported():
    frame = _bars()
    state = trend.states(frame["Close"])
    expected_since = state.index[state.ne(state.shift())][-1].date().isoformat()
    with mock.patch.object(trend, "bars", return_value=frame):
        assert trend.latest("BTCUSDT", datetime.now(UTC))["since"] == expected_since


@pytest.mark.unit
def test_a_bar_older_than_yesterday_is_stale_and_changes_nothing():
    with mock.patch.object(trend, "bars", return_value=_bars()[:-3]), pytest.raises(trend.StaleData):
        trend.check("BTCUSDT", lambda t: None, datetime.now(UTC))
    assert not trend._load()


@pytest.mark.unit
def test_status_text_only_serves_the_validated_symbol():
    assert "only validated on BTCUSDT" in trend.status_text("ETHUSDT")
    with mock.patch.object(trend, "bars", return_value=_bars()):
        out = trend.status_text("btcusdt")
    assert out.startswith("BTCUSDT trend (paper): ") and "SMA200" in out


@pytest.mark.unit
def test_bars_fall_back_to_yahoo_without_todays_bar():
    today = datetime.now(UTC)
    days = pd.date_range(end=today.date(), periods=5, freq="D")
    yahoo = pd.DataFrame({"Date": days, "Open": [1.0, 2, 3, 4, 5], "Close": [1.5, 2, 3, 4, 5]})
    with mock.patch.object(trend, "fetch_daily_klines", return_value=None), \
            mock.patch.object(trend, "load_ohlcv", return_value=yahoo):
        out = trend.bars("BTCUSDT", today)
    assert out.index[-1].date() < today.date() and list(out["Close"]) == [1.5, 2, 3, 4]


@pytest.mark.unit
def test_seconds_to_next_run_is_just_after_midnight_utc():
    assert trend._seconds_to_run(datetime(2026, 10, 22, 0, 1, tzinfo=UTC)) == 4 * 60
    assert trend._seconds_to_run(datetime(2026, 10, 22, 0, 6, tzinfo=UTC)) == 24 * 3600 - 60


# --- the book model --------------------------------------------------------------------------


def _loop_equity(frame, weight, fee):
    """The same book written as an explicit day-by-day loop with cash and units."""
    state = trend.states(frame["Close"])
    frame = frame.loc[state.index[0]:]
    want = (state == trend.HOLD).tolist()
    first = frame["Close"].iloc[0]
    units_hold = weight * (1 - fee) / first
    cash, units = (1 - weight) * (0.0 if want[0] else 1.0), 0.0
    if want[0]:
        units = (1 - weight) * (1 - fee) / first
    out = [units_hold * first + cash + units * first]
    for i in range(1, len(frame)):
        row = frame.iloc[i]
        target = want[i - 1]  # the previous close's signal fills at today's open
        if target and units == 0:
            units, cash = cash * (1 - fee) / row["Open"], 0.0
        elif not target and units > 0:
            units, cash = 0.0, units * row["Open"] * (1 - fee)
        out.append(units_hold * row["Close"] + cash + units * row["Close"])
    return pd.Series(out, index=frame.index)


@pytest.mark.unit
@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_simulate_matches_an_independent_loop(weight):
    frame = _bars()
    got = trend.simulate(frame, weight)["mix"]
    assert got.to_numpy() == pytest.approx(_loop_equity(frame, weight, trend.FEE).to_numpy(), rel=1e-9)


@pytest.mark.unit
def test_hold_sleeve_is_buy_and_hold_after_one_fee_and_the_mix_is_the_weighted_sum():
    frame = _bars()
    book = trend.simulate(frame, 0.3)
    close = frame["Close"].loc[book.index]
    assert book["hold"].to_numpy() == pytest.approx((close / close.iloc[0] * (1 - trend.FEE)).to_numpy())
    assert book["mix"].to_numpy() == pytest.approx((0.3 * book["hold"] + 0.7 * book["trend"]).to_numpy())


@pytest.mark.unit
def test_core_weight_comes_from_the_environment_and_is_validated(monkeypatch):
    assert trend.core_weight() == 0.5
    monkeypatch.setenv("TREND_CORE_WEIGHT", "0.8")
    assert trend.core_weight() == 0.8
    monkeypatch.setenv("TREND_CORE_WEIGHT", "1.5")
    with pytest.raises(ValueError):
        trend.core_weight()


@pytest.mark.unit
def test_paper_text_shows_the_book_the_ledger_and_the_exposure():
    frame = _bars()
    state = trend.states(frame["Close"])
    flip = state.index[state.ne(state.shift())][-3]
    with mock.patch.object(trend, "bars", return_value=frame[: flip - timedelta(days=1)]):
        trend.check("BTCUSDT", lambda t: None, _at(flip - timedelta(days=1)))
    assert "no flips yet" in _paper(frame[: flip - timedelta(days=1)])
    _replay(frame, lambda t: None, flip)
    out = _paper(frame)
    assert "paper book since" in out and "no real orders" in out and "Equity: mix" in out
    assert out.count("sent ") >= 2 and "-> open" in out and "Reference only: Whipsaw warning" in out
    assert ("50% in BTC" in out) or ("100% in BTC" in out)


def _paper(frame):
    with mock.patch.object(trend, "bars", return_value=frame):
        return trend.paper_text("BTCUSDT", datetime.now(UTC) if frame.index[-1].date() >= datetime.now(UTC).date() - timedelta(days=1)
                                else _at(frame.index[-1]))


@pytest.mark.unit
def test_backtest_report_lists_the_three_books(tmp_path):
    path = tmp_path / "p.csv"
    frame = _bars().rename_axis("Date")
    frame.to_csv(path)
    out = trend_backtest.report(pd.read_csv(path, parse_dates=["Date"]).set_index("Date"), 0.5)
    assert "100% hold" in out and "100% SMA200" in out and "50% core + 50% SMA200" in out and "state changes" in out


@pytest.mark.unit
def test_watch_checks_at_start_up_retries_a_failure_then_sleeps_until_the_next_run(monkeypatch):
    calls, naps = [], []

    def flaky(symbol, notify):
        calls.append(symbol)
        if len(calls) == 1:
            raise OSError("binance down")

    def nap(seconds):
        naps.append(seconds)
        if seconds != trend.RETRY_SECONDS:  # the sleep until tomorrow's run: stop the endless loop here
            raise SystemExit

    monkeypatch.setattr(trend, "check", flaky)
    monkeypatch.setattr(trend.time, "sleep", nap)
    with pytest.raises(SystemExit):
        trend.watch(lambda text: None)
    assert calls == ["BTCUSDT", "BTCUSDT"] and naps[0] == trend.RETRY_SECONDS and 0 < naps[1] <= 24 * 3600
