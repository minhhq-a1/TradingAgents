"""Daily spot candles for a crypto pair, from Binance's public market-data API.

Yahoo's crypto daily series has days missing, and a missing day reads as a
non-trading day to the indicators although crypto trades every day. Binance
serves every UTC day, so crypto symbols take their bars from here first and fall
back to Yahoo only when Binance cannot be reached (it refuses restricted regions
such as the US).

Bars are keyed on the UTC open date, like Yahoo's crypto bars. ``Volume`` is the
USDT turnover, which is what Yahoo reports for a crypto pair, not the base-coin
count Binance calls volume. A USD or USDC quote maps to the USDT pair, the only
dollar pair Binance lists across these coins.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pandas as pd
import requests

from tradingagents.dataflows.symbols import crypto_base

logger = logging.getLogger(__name__)

# Tried in order: the main API, then the public market-data mirror.
HOSTS = ("https://api.binance.com", "https://data-api.binance.vision")
REQUEST_TIMEOUT = 20
PAGE_LIMIT = 1000  # Binance's maximum bars per request
DAY_MS = 86_400_000


def _ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)


def _fetch(host: str, pair: str, start_ms: int, end_ms: int) -> list:
    rows, cursor = [], start_ms
    while cursor <= end_ms:
        response = requests.get(
            f"{host}/api/v3/klines",
            params={"symbol": pair, "interval": "1d", "startTime": cursor,
                    "endTime": end_ms, "limit": PAGE_LIMIT},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        page = response.json()
        rows += page
        if len(page) < PAGE_LIMIT:
            break
        cursor = page[-1][0] + DAY_MS
    return rows


def fetch_daily_klines(symbol: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """Daily bars from ``start_date`` to ``end_date`` inclusive, indexed by UTC ``Date``.

    None when ``symbol`` is not a crypto pair or Binance cannot serve it, so the
    caller can fall back to Yahoo.
    """
    base = crypto_base(symbol)
    if base is None:
        return None
    pair = f"{base}USDT"
    start_ms, end_ms = _ms(start_date), _ms(end_date) + DAY_MS - 1

    rows = None
    for host in HOSTS:
        try:
            rows = _fetch(host, pair, start_ms, end_ms)
            break
        except (requests.RequestException, ValueError) as e:
            logger.warning("Binance klines failed for %s via %s: %s", pair, host, e)
    if not rows:
        return None

    frame = pd.DataFrame({
        "Date": pd.to_datetime([r[0] for r in rows], unit="ms").normalize(),
        "Open": [float(r[1]) for r in rows],
        "High": [float(r[2]) for r in rows],
        "Low": [float(r[3]) for r in rows],
        "Close": [float(r[4]) for r in rows],
        "Volume": [float(r[7]) for r in rows],
    })
    return frame.set_index("Date")
