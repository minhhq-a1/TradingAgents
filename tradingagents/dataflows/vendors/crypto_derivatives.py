"""Perpetual-futures positioning for a crypto pair: funding, open interest, long/short.

Crypto has no statements to read, and its short-term moves are driven by
leveraged positioning, which OHLCV alone does not show. This serves the three
public, keyless series that describe it, one daily row each:

- funding: the sum of the day's funding payments (longs pay shorts when
  positive), so crowded longs show up as persistently positive funding;
- open interest (USD): the notional of open perpetual contracts;
- long/short account ratio: accounts net long over accounts net short.

Each exchange is its own vendor so the router can fall back between them.
Binance refuses requests from restricted regions (the US among them), so the
default chain is ``binance,okx,hyperliquid``. Hyperliquid serves funding only.

Rows are cut at the end of the analysis date (UTC), so a backtest never sees a
later funding print, and the still-open current day is left out as partial. Binance keeps open-interest and long/short history for 30
days only; older dates get funding alone from it.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import UTC, datetime, timedelta

import requests

from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.symbols import crypto_base

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 30
BINANCE = "https://fapi.binance.com"
OKX = "https://www.okx.com"
HYPERLIQUID = "https://api.hyperliquid.xyz/info"


def _window(curr_date: str, look_back_days: int) -> tuple[int, int]:
    """Millisecond bounds: the start of the first day to the end of ``curr_date`` (UTC)."""
    end = datetime.strptime(curr_date, "%Y-%m-%d").replace(tzinfo=UTC) + timedelta(days=1)
    start = end - timedelta(days=look_back_days)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000) - 1


def _day(ms) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, UTC).strftime("%Y-%m-%d")


def _get(vendor: str, url: str, **kwargs):
    """GET (or POST with ``json=``) a JSON body; any failure means the vendor is unavailable."""
    try:
        if "json" in kwargs:
            response = requests.post(url, timeout=REQUEST_TIMEOUT, **kwargs)
        else:
            response = requests.get(url, timeout=REQUEST_TIMEOUT, **kwargs)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as e:
        # A 451 here is Binance's regional block, not a statement about the coin.
        raise VendorUnavailableError(f"{vendor} request failed: {e}") from e


def _base(symbol: str) -> str:
    base = crypto_base(symbol)
    if base is None:
        raise NoMarketDataError(symbol, detail="not a crypto pair")
    return base


def _report(symbol: str, venue: str, funding: dict, oi: dict, ratio: dict, start_ms: int, end_ms: int) -> str:
    """One markdown table of daily rows inside the window, plus a short summary."""
    lo, hi = _day(start_ms), _day(end_ms)
    # The still-open UTC day is partial (funding not yet paid out, OI mid-move),
    # and the price bars stop at the last completed day, so these do too.
    open_day = datetime.now(UTC).strftime("%Y-%m-%d")
    days = sorted(d for d in set(funding) | set(oi) | set(ratio) if lo <= d <= hi and d < open_day)
    if not days:
        raise NoMarketDataError(symbol, detail=f"{venue} returned no derivatives rows up to {hi}")

    def pct(v):
        return "n/a" if v is None else f"{v * 100:+.4f}%"

    lines = [
        f"## {symbol} perpetual-futures positioning ({venue}), {days[0]} to {days[-1]}",
        "Funding is the day's summed rate (positive: longs pay shorts); annualized = daily x 365.",
        "",
        "| Date | Funding (day) | Funding (annualized) | Open interest (USD) | Long/short accounts |",
        "|---|---|---|---|---|",
    ]
    for d in days:
        f, o, r = funding.get(d), oi.get(d), ratio.get(d)
        lines.append(
            f"| {d} | {pct(f)} | {'n/a' if f is None else f'{f * 365 * 100:+.1f}%'} | "
            f"{'n/a' if o is None else f'{o:,.0f}'} | {'n/a' if r is None else f'{r:.2f}'} |"
        )

    summary = []
    recent = [funding[d] for d in days[-7:] if d in funding]
    if recent:
        summary.append(f"7-day average daily funding {pct(sum(recent) / len(recent))}.")
    oi_days = [d for d in days if d in oi]
    if len(oi_days) >= 2 and oi[oi_days[0]]:
        change = oi[oi_days[-1]] / oi[oi_days[0]] - 1
        summary.append(f"Open interest {change:+.1%} from {oi_days[0]} to {oi_days[-1]}.")
    missing = [name for name, series in (("funding", funding), ("open interest", oi), ("long/short ratio", ratio))
               if not any(lo <= d <= hi for d in series)]
    if missing:
        summary.append(f"Not available from {venue} for this window: {', '.join(missing)}; do not estimate them.")
    return "\n".join(lines + [""] + summary) + "\n"


def get_binance(symbol: str, curr_date: str, look_back_days: int = 30) -> str:
    pair = f"{_base(symbol)}USDT"
    start, end = _window(curr_date, look_back_days)
    span = {"symbol": pair, "startTime": start, "endTime": end}

    funding = defaultdict(float)
    for row in _get("binance", f"{BINANCE}/fapi/v1/fundingRate", params={**span, "limit": 1000}):
        funding[_day(row["fundingTime"])] += float(row["fundingRate"])

    oi, ratio = {}, {}
    # Binance serves these two for the last 30 days only; an older window gets none.
    if datetime.now(UTC).timestamp() * 1000 - start < 30 * 86_400_000:
        daily = {**span, "period": "1d", "limit": 500}
        for row in _get("binance", f"{BINANCE}/futures/data/openInterestHist", params=daily):
            oi[_day(row["timestamp"])] = float(row["sumOpenInterestValue"])
        for row in _get("binance", f"{BINANCE}/futures/data/globalLongShortAccountRatio", params=daily):
            ratio[_day(row["timestamp"])] = float(row["longShortRatio"])
    return _report(symbol, "Binance USD-M", dict(funding), oi, ratio, start, end)


def _okx_data(path: str, params: dict) -> list:
    body = _get("okx", f"{OKX}{path}", params=params)
    if str(body.get("code")) != "0":
        raise VendorUnavailableError(f"okx {path} answered code {body.get('code')}: {body.get('msg')}")
    return body.get("data") or []


def get_okx(symbol: str, curr_date: str, look_back_days: int = 30) -> str:
    base = _base(symbol)
    start, end = _window(curr_date, look_back_days)

    funding = defaultdict(float)
    # Newest first, at most 100 per page (about 33 days of 8-hour prints).
    for row in _okx_data("/api/v5/public/funding-rate-history",
                         {"instId": f"{base}-USDT-SWAP", "after": end + 1, "limit": 100}):
        if int(row["fundingTime"]) >= start:
            funding[_day(row["fundingTime"])] += float(row["fundingRate"])

    span = {"ccy": base, "period": "1D", "begin": start, "end": end}
    oi = {_day(r[0]): float(r[1]) for r in _okx_data("/api/v5/rubik/stat/contracts/open-interest-volume", span)}
    ratio = {_day(r[0]): float(r[1]) for r in _okx_data("/api/v5/rubik/stat/contracts/long-short-account-ratio", span)}
    return _report(symbol, "OKX", dict(funding), oi, ratio, start, end)


def get_hyperliquid(symbol: str, curr_date: str, look_back_days: int = 30) -> str:
    start, end = _window(curr_date, look_back_days)
    # shortcut: one call returns at most 500 hourly prints (about 20 days), so a
    # longer window starts late; page on the last ``time`` if longer windows matter.
    rows = _get("hyperliquid", HYPERLIQUID,
                json={"type": "fundingHistory", "coin": _base(symbol), "startTime": start, "endTime": end})
    funding = defaultdict(float)
    for row in rows:
        if int(row["time"]) <= end:
            funding[_day(row["time"])] += float(row["fundingRate"])
    return _report(symbol, "Hyperliquid", dict(funding), {}, {}, start, end)
