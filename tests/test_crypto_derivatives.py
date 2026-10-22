"""Crypto perpetual-futures vendors: parsing, point-in-time cut, fallback, tool gating.

All API access is mocked, so these run without a network connection.
"""
import copy
from datetime import UTC, datetime
from unittest import mock

import pytest
import requests

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.agents import tools
from tradingagents.dataflows import router
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.vendors import crypto_derivatives as cd

DATE = "2026-08-14"


def _ms(day, hour=0):
    return int(datetime.strptime(day, "%Y-%m-%d").replace(hour=hour, tzinfo=UTC).timestamp() * 1000)


@pytest.fixture(autouse=True)
def _default_config():
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    yield
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


def _binance(url, **_):
    if url.endswith("/fapi/v1/fundingRate"):
        return [
            {"fundingTime": _ms("2026-08-13", 0), "fundingRate": "0.0001"},
            {"fundingTime": _ms("2026-08-13", 8), "fundingRate": "0.0002"},
            {"fundingTime": _ms("2026-08-14", 16), "fundingRate": "-0.0001"},
            {"fundingTime": _ms("2026-08-15", 0), "fundingRate": "0.0100"},  # after the run date
        ]
    if url.endswith("openInterestHist"):
        return [{"timestamp": _ms("2026-08-13"), "sumOpenInterestValue": "8000000000"},
                {"timestamp": _ms("2026-08-14"), "sumOpenInterestValue": "8800000000"}]
    if url.endswith("globalLongShortAccountRatio"):
        return [{"timestamp": _ms("2026-08-14"), "longShortRatio": "1.85"}]
    raise AssertionError(url)


@pytest.mark.unit
def test_binance_report_sums_daily_funding_and_cuts_at_the_run_date():
    now = datetime(2026, 8, 20, tzinfo=UTC)  # within Binance's 30-day OI retention
    with mock.patch.object(cd, "_get", side_effect=lambda _v, url, **k: _binance(url, **k)), \
            mock.patch.object(cd, "datetime", wraps=datetime) as dt:
        dt.now.return_value = now
        out = cd.get_binance("BTCUSDT", DATE, 7)
    assert "+0.0300%" in out            # 0.0001 + 0.0002 on 08-13
    assert "| 2026-08-14 | -0.0100%" in out
    assert "2026-08-15" not in out      # later print withheld
    assert "8,800,000,000" in out and "1.85" in out
    assert "Open interest +10.0%" in out


@pytest.mark.unit
def test_binance_skips_open_interest_beyond_its_retention():
    calls = []

    def fake(_v, url, **k):
        calls.append(url)
        return _binance(url, **k)

    with mock.patch.object(cd, "_get", side_effect=fake):
        out = cd.get_binance("BTC-USD", DATE, 7)  # the real clock is far past 30 days later
    assert not any("openInterestHist" in u for u in calls)
    assert "open interest, long/short ratio" in out  # named as unavailable


@pytest.mark.unit
def test_okx_parses_rubik_arrays_and_funding():
    def fake(_v, url, params=None, **_):
        if url.endswith("funding-rate-history"):
            data = [{"fundingTime": str(_ms("2026-08-14", 8)), "fundingRate": "0.0003"}]
        elif url.endswith("open-interest-volume"):
            data = [[str(_ms("2026-08-14")), "5000000000", "1"]]
        else:
            data = [[str(_ms("2026-08-14")), "0.92"]]
        return {"code": "0", "data": data}

    with mock.patch.object(cd, "_get", side_effect=fake):
        out = cd.get_okx("BTCUSDT", DATE, 7)
    assert "(OKX)" in out and "+0.0300%" in out and "5,000,000,000" in out and "0.92" in out


@pytest.mark.unit
def test_okx_error_code_is_unavailable():
    refused = {"code": "51001", "msg": "restricted", "data": []}
    with mock.patch.object(cd, "_get", return_value=refused), pytest.raises(VendorUnavailableError):
        cd.get_okx("BTCUSDT", DATE)


@pytest.mark.unit
def test_hyperliquid_serves_funding_only():
    rows = [{"time": _ms("2026-08-14", h), "fundingRate": "0.0000125"} for h in range(24)]
    with mock.patch.object(cd, "_get", return_value=rows):
        out = cd.get_hyperliquid("BTCUSDT", DATE, 7)
    assert "+0.0300%" in out and "open interest, long/short ratio" in out


@pytest.mark.unit
def test_http_failure_is_vendor_unavailable():
    response = mock.Mock()
    response.raise_for_status.side_effect = requests.HTTPError("451 restricted location")
    with mock.patch.object(cd.requests, "get", return_value=response), pytest.raises(VendorUnavailableError):
        cd.get_binance("BTCUSDT", DATE)


@pytest.mark.unit
def test_non_crypto_symbol_has_no_data():
    with pytest.raises(NoMarketDataError):
        cd.get_okx("NVDA", DATE)


@pytest.mark.unit
def test_router_falls_back_past_a_blocked_binance():
    def blocked(*_a, **_k):
        raise VendorUnavailableError("451")

    with mock.patch.dict(router.VENDOR_METHODS, {"get_crypto_derivatives": {
        "binance": blocked, "okx": lambda *a, **k: "OKX_OK", "hyperliquid": lambda *a, **k: "HL"}}):
        assert router.route_to_vendor("get_crypto_derivatives", "BTCUSDT", DATE, 30) == "OKX_OK"


@pytest.mark.unit
def test_router_degrades_when_every_venue_is_blocked():
    def blocked(*_a, **_k):
        raise VendorUnavailableError("451")

    with mock.patch.dict(router.VENDOR_METHODS, {"get_crypto_derivatives": {
        "binance": blocked, "okx": blocked, "hyperliquid": blocked}}):
        out = router.route_to_vendor("get_crypto_derivatives", "BTCUSDT", DATE, 30)
    assert out.startswith("DATA_UNAVAILABLE")


@pytest.mark.unit
def test_tool_skips_stocks_and_clamps_the_date_for_crypto():
    assert tools.get_crypto_derivatives.func("NVDA", DATE, trade_date=DATE).startswith("NOT_APPLICABLE")
    with mock.patch.object(tools, "route_to_vendor", return_value="ok") as route:
        tools.get_crypto_derivatives.func("BTCUSDT", "2026-09-30", trade_date=DATE)
    route.assert_called_once_with("get_crypto_derivatives", "BTCUSDT", DATE, 30)
