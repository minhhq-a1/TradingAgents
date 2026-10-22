"""Crypto data fixes: Binance daily bars first, StockTwits status, crypto RSS news,
and the Trader's stop-side check. All network access is mocked.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from unittest import mock
from urllib.error import HTTPError

import pandas as pd
import pytest
import requests

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.agents.context import last_close_line
from tradingagents.agents.schemas import TraderAction, TraderProposal
from tradingagents.agents.trader import trader
from tradingagents.dataflows import router
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.vendors import binance_klines, crypto_news, stocktwits
from tradingagents.dataflows.vendors.yahoo import market, ohlcv


@pytest.fixture(autouse=True)
def _default_config():
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    yield
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


def _day_ms(day):
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)


def _kline(day, close):
    return [_day_ms(day), "1.30", "1.40", "1.20", str(close), "5000", 0, "6500.5", 10, "0", "0", "0"]


def _response(payload):
    response = mock.Mock()
    response.json.return_value = payload
    return response


# --- Binance daily bars -----------------------------------------------------


def _ago(days):
    return (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d")


@pytest.mark.unit
def test_klines_serve_every_completed_day_with_turnover_as_volume():
    days = [_ago(3), _ago(2), _ago(1)]
    rows = [_kline(d, 1.35) for d in days] + [_kline(_ago(0), 9.99)]  # the open day, partial
    with mock.patch.object(binance_klines.requests, "get", return_value=_response(rows)) as get:
        frame = binance_klines.fetch_daily_klines("XRPUSDT", days[0], _ago(0))
    assert get.call_args.kwargs["params"]["symbol"] == "XRPUSDT"
    assert [d.strftime("%Y-%m-%d") for d in frame.index] == days  # none dropped, the open day left out
    assert frame["Volume"].iloc[0] == 6500.5 and frame["Close"].iloc[0] == 1.35


@pytest.mark.unit
def test_klines_with_only_the_open_day_are_none():
    with mock.patch.object(binance_klines.requests, "get", return_value=_response([_kline(_ago(0), 1)])):
        assert binance_klines.fetch_daily_klines("XRPUSDT", _ago(0), _ago(0)) is None


@pytest.mark.unit
def test_klines_page_past_one_thousand_bars():
    first = [_kline("2026-01-01", 1)] * binance_klines.PAGE_LIMIT
    second = [_kline(_ago(1), 2)]
    with mock.patch.object(binance_klines.requests, "get",
                           side_effect=[_response(first), _response(second)]) as get:
        frame = binance_klines.fetch_daily_klines("BTC-USD", "2023-01-01", _ago(1))
    assert get.call_count == 2 and len(frame) == 1001


@pytest.mark.unit
def test_klines_fall_back_to_the_mirror_then_to_none():
    blocked = requests.HTTPError("451")
    with mock.patch.object(binance_klines.requests, "get",
                           side_effect=[blocked, _response([_kline(_ago(1), 1)])]) as get:
        assert binance_klines.fetch_daily_klines("ETHUSDT", _ago(1), _ago(1)) is not None
    assert get.call_args_list[1].args[0].startswith("https://data-api.binance.vision")
    with mock.patch.object(binance_klines.requests, "get", side_effect=blocked):
        assert binance_klines.fetch_daily_klines("ETHUSDT", _ago(1), _ago(1)) is None


@pytest.mark.unit
def test_klines_ignore_stocks():
    with mock.patch.object(binance_klines.requests, "get") as get:
        assert binance_klines.fetch_daily_klines("NVDA", "2026-10-22", "2026-10-22") is None
    get.assert_not_called()


@pytest.mark.unit
def test_get_stock_data_for_crypto_uses_binance_and_keeps_precision():
    bars = binance_klines.pd.DataFrame(
        {"Open": [1.3456], "High": [1.4], "Low": [1.2], "Close": [1.3512], "Volume": [10.0]},
        index=pd.DatetimeIndex([datetime.now() - timedelta(days=1)], name="Date").normalize(),
    )
    day = bars.index[0].strftime("%Y-%m-%d")
    with mock.patch.object(market, "fetch_daily_klines", return_value=bars), \
            mock.patch.object(market.yf, "Ticker") as yahoo:
        out = market.get_YFin_data_online("XRPUSDT", day, day)
    yahoo.assert_not_called()
    assert "1.3512" in out and "1.3456" in out  # not flattened to two places


@pytest.mark.unit
def test_get_stock_data_for_crypto_falls_back_to_yahoo():
    day = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    yahoo = pd.DataFrame(
        {"Open": [1.3456], "High": [1.4], "Low": [1.2], "Close": [1.3512], "Volume": [10.0]},
        index=pd.DatetimeIndex([day]),
    )
    with mock.patch.object(market, "fetch_daily_klines", return_value=None), \
            mock.patch.object(market, "yf_retry", return_value=yahoo):
        out = market.get_YFin_data_online("XRPUSDT", day, day)
    assert "1.35" in out and "1.3512" not in out  # Yahoo's rows, rounded as before


@pytest.mark.unit
def test_load_ohlcv_for_crypto_takes_binance_bars(tmp_path):
    config_module._config["data_cache_dir"] = str(tmp_path)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    bars = pd.DataFrame(
        {"Open": [1.0, 1.1], "High": [1.2, 1.3], "Low": [0.9, 1.0], "Close": [1.1, 1.2], "Volume": [5.0, 6.0]},
        index=pd.DatetimeIndex([today - timedelta(days=1), today], name="Date"),
    )
    with mock.patch.object(ohlcv, "fetch_daily_klines", return_value=bars), \
            mock.patch.object(ohlcv.yf, "Ticker") as yahoo:
        data = ohlcv.load_ohlcv("XRP-USD", today.strftime("%Y-%m-%d"))
    yahoo.assert_not_called()
    assert list(data["Close"]) == [1.1, 1.2]


# --- StockTwits ---------------------------------------------------------------


@pytest.mark.unit
def test_stocktwits_placeholder_names_the_http_status():
    error = HTTPError("url", 403, "Forbidden", {}, None)
    with mock.patch.object(stocktwits, "urlopen", side_effect=error):
        out = stocktwits.fetch_stocktwits_messages("XRPUSDT")
    assert out == "<stocktwits unavailable: HTTP 403>"


@pytest.mark.unit
def test_stocktwits_presents_a_browser_user_agent():
    with mock.patch.object(stocktwits, "urlopen", side_effect=OSError("down")) as opened:
        stocktwits.fetch_stocktwits_messages("XRPUSDT")
    assert opened.call_args.args[0].get_header("User-agent").startswith("Mozilla/5.0")


def _forbidden():
    return HTTPError("url", 403, "Forbidden", {}, None)


@pytest.mark.unit
def test_stocktwits_403_is_retried_with_a_browser_tls_fingerprint():
    reply = mock.Mock(status_code=200)
    reply.json.return_value = {"messages": [
        {"created_at": "2026-10-22T04:00:00Z", "body": "up", "user": {"username": "a"},
         "entities": {"sentiment": {"basic": "Bullish"}}}]}
    with mock.patch.object(stocktwits, "urlopen", side_effect=_forbidden()), \
            mock.patch.object(stocktwits.curl_requests, "get", return_value=reply) as curl:
        out = stocktwits.fetch_stocktwits_messages("XRPUSDT")
    assert curl.call_args.kwargs["impersonate"] == "chrome"
    assert "Bullish: 1" in out


@pytest.mark.unit
@pytest.mark.parametrize("retry", [mock.Mock(status_code=403), OSError("curl failed")])
def test_stocktwits_still_blocked_reports_the_original_403(retry):
    kwargs = {"side_effect": retry} if isinstance(retry, Exception) else {"return_value": retry}
    with mock.patch.object(stocktwits, "urlopen", side_effect=_forbidden()), \
            mock.patch.object(stocktwits.curl_requests, "get", **kwargs):
        assert stocktwits.fetch_stocktwits_messages("XRPUSDT") == "<stocktwits unavailable: HTTP 403>"


@pytest.mark.unit
def test_stocktwits_other_errors_are_not_retried():
    with mock.patch.object(stocktwits, "urlopen", side_effect=HTTPError("url", 429, "slow", {}, None)), \
            mock.patch.object(stocktwits.curl_requests, "get") as curl:
        assert stocktwits.fetch_stocktwits_messages("XRPUSDT") == "<stocktwits unavailable: HTTP 429>"
    curl.assert_not_called()


@pytest.mark.unit
def test_indicator_table_says_a_crypto_bar_is_not_closed_rather_than_a_holiday():
    yesterday, today = _ago(1), _ago(0)
    with mock.patch.object(market, "_get_stock_stats_bulk",
                           return_value={_ago(2): "1.0", yesterday: "2.0"}):
        crypto = market.get_stock_stats_indicators_window("XRPUSDT", "rsi", today, 2)
        stock = market.get_stock_stats_indicators_window("NVDA", "rsi", today, 2)
    assert f"{today}: N/A: bar not closed yet" in crypto and f"{yesterday}: 2.0" in crypto
    assert f"{today}: N/A: Not a trading day" in stock


# --- Crypto news RSS ------------------------------------------------------------


def _rss(*items):
    body = "".join(
        f"<item><title>{t}</title><description>&lt;p&gt;{d}&lt;/p&gt;</description>"
        f"<link>https://x/{i}</link><pubDate>{format_datetime(p)}</pubDate></item>"
        for i, (t, d, p) in enumerate(items)
    )
    return f"<?xml version='1.0'?><rss><channel>{body}</channel></rss>".encode()


def _feeds(coindesk, cointelegraph):
    def get(url, **_):
        response = mock.Mock()
        response.content = coindesk if "coindesk" in url else cointelegraph
        return response

    return mock.patch.object(crypto_news.requests, "get", side_effect=get)


NOW = datetime.now(UTC)
TODAY = NOW.strftime("%Y-%m-%d")
WEEK_AGO = (NOW - timedelta(days=6)).strftime("%Y-%m-%d")


@pytest.mark.unit
def test_rss_serves_articles_that_name_the_coin():
    coindesk = _rss(("XRP jumps after ruling", "Ripple wins", NOW - timedelta(hours=3)),
                    ("Bitcoin slips", "BTC drifts", NOW - timedelta(hours=2)))
    cointelegraph = _rss(("Ripple expands", "payments push", NOW - timedelta(hours=5)))
    with _feeds(coindesk, cointelegraph):
        out = crypto_news.get_news_crypto_rss("XRPUSDT", WEEK_AGO, TODAY)
    assert "XRP jumps after ruling" in out and "Ripple expands" in out
    assert "Bitcoin slips" not in out and "<p>" not in out
    assert out.index("XRP jumps") < out.index("Ripple expands")  # newest first


@pytest.mark.unit
def test_rss_one_feed_down_still_serves_the_other():
    cointelegraph = _rss(("XRP rallies", "x", NOW - timedelta(hours=1)))

    def get(url, **_):
        if "coindesk" in url:
            raise requests.ConnectionError("down")
        response = mock.Mock()
        response.content = cointelegraph
        return response

    with mock.patch.object(crypto_news.requests, "get", side_effect=get):
        assert "XRP rallies" in crypto_news.get_news_crypto_rss("XRP-USD", WEEK_AGO, TODAY)


@pytest.mark.unit
def test_rss_all_feeds_down_is_unavailable_and_stocks_have_no_data():
    with mock.patch.object(crypto_news.requests, "get", side_effect=requests.ConnectionError("down")), \
            pytest.raises(VendorUnavailableError):
        crypto_news.get_news_crypto_rss("XRP-USD", WEEK_AGO, TODAY)
    with pytest.raises(NoMarketDataError):
        crypto_news.get_news_crypto_rss("NVDA", WEEK_AGO, TODAY)


@pytest.mark.unit
def test_rss_window_it_never_observed_reads_as_unavailable_not_silence():
    recent = _rss(("XRP today", "x", NOW - timedelta(hours=1)))
    with _feeds(recent, recent):
        out = crypto_news.get_news_crypto_rss("XRP-USD", "2026-01-01", "2026-01-31")
    assert "unavailable" in out and "XRP today" not in out


@pytest.mark.unit
def test_router_moves_from_yahoo_with_no_tagged_news_to_the_rss_feeds():
    def yahoo(*_a, **_k):
        raise VendorUnavailableError("Yahoo Finance has no news articles tagged XRP-USD")

    with mock.patch.dict(router.VENDOR_METHODS["get_news"], {"yfinance": yahoo, "crypto_rss": lambda *a: "RSS_OK"}):
        assert router.route_to_vendor("get_news", "XRPUSDT", WEEK_AGO, TODAY) == "RSS_OK"


# --- Trader stop-side check -----------------------------------------------------


def _proposal(action, entry, stop):
    return TraderProposal(action=action, reasoning="r", entry_price=entry, stop_loss=stop)


def _bars(atr):
    """Fifteen bars whose true range is ``atr`` every day."""
    return pd.DataFrame({"High": [10 + atr] * 15, "Low": [10.0] * 15, "Close": [10.0] * 15})


@pytest.mark.unit
@pytest.mark.parametrize("action, entry, stop", [
    (TraderAction.BUY, 100.0, 95.0), (TraderAction.SELL, 100.0, 105.0),
    (TraderAction.HOLD, None, None),              # no order, nothing to check
    (TraderAction.SELL, None, 90.0), (TraderAction.BUY, 100.0, None),
])
def test_valid_or_absent_levels_are_untouched(action, entry, stop):
    proposal = _proposal(action, entry, stop)
    with mock.patch.object(trader, "load_ohlcv", side_effect=AssertionError("no data needed")):
        assert trader.check_levels(proposal, "XRPUSDT", "2026-10-22") is proposal


@pytest.mark.unit
@pytest.mark.parametrize("action, entry, stop, expected", [
    (TraderAction.SELL, 1.38, 1.29, 1.38 + 1.5 * 0.05),   # the XRP run: a long's stop on a Sell
    (TraderAction.BUY, 100.0, 105.0, 100.0 - 1.5 * 4.0),
    (TraderAction.SELL, 100.0, 100.0, 100.0 + 1.5 * 4.0),  # at entry fills at once too
])
def test_a_stop_on_the_wrong_side_is_recomputed_from_atr(action, entry, stop, expected):
    atr = 0.05 if entry < 10 else 4.0
    with mock.patch.object(trader, "load_ohlcv", return_value=_bars(atr)):
        fixed = trader.check_levels(_proposal(action, entry, stop), "XRPUSDT", "2026-10-22")
    assert fixed.stop_loss == pytest.approx(expected)
    assert "Level check" in fixed.reasoning and str(stop) in fixed.reasoning
    assert fixed.entry_price == entry and fixed.action is action


@pytest.mark.unit
def test_a_wrong_side_stop_is_dropped_when_there_is_no_price_history():
    with mock.patch.object(trader, "load_ohlcv", side_effect=NoMarketDataError("XRPUSDT")):
        fixed = trader.check_levels(_proposal(TraderAction.SELL, 1.38, 1.29), "XRPUSDT", "2026-10-22")
    assert fixed.stop_loss is None and "dropped" in fixed.reasoning


@pytest.mark.unit
def test_a_hold_carries_no_entry_or_stop():
    with mock.patch.object(trader, "load_ohlcv", side_effect=AssertionError("no data needed")):
        fixed = trader.check_levels(_proposal(TraderAction.HOLD, 82000.0, 78631.0), "BTCUSDT", "2026-10-22")
    assert fixed.entry_price is None and fixed.stop_loss is None
    assert "Hold places no order" in fixed.reasoning


# --- Reference price for the agents that never call a price tool ----------------


@pytest.mark.unit
def test_reference_close_line_states_the_last_completed_close():
    bars = pd.DataFrame({"Date": pd.to_datetime(["2026-10-20", "2026-10-21"]), "Close": [81000.0, 82761.4]})
    with mock.patch("tradingagents.dataflows.vendors.yahoo.ohlcv.load_ohlcv", return_value=bars):
        line = last_close_line("BTCUSDT", "2026-10-22")
    assert "82,761.40" in line and "2026-10-21" in line and "do not call the price unverified" in line


@pytest.mark.unit
def test_reference_close_line_is_empty_when_prices_are_unavailable():
    with mock.patch("tradingagents.dataflows.vendors.yahoo.ohlcv.load_ohlcv", side_effect=NoMarketDataError("X")):
        assert last_close_line("BTCUSDT", "2026-10-22") == ""
