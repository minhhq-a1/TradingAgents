"""Crypto news from the public RSS feeds of CoinDesk and Cointelegraph.

Yahoo tags news to a symbol only for a few large coins (BTC, ETH); an altcoin
such as XRP gets none, which the news analyst reads as a data gap. These feeds
need no key and cover the whole market, so the articles that name the coin (by
ticker or full name) in the title or summary are served as its news.

A feed returns only its latest items, so a window it did not reach is reported
as unavailable rather than as silence, like the other live feeds.
"""

from __future__ import annotations

import contextlib
import html
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime

import requests

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.date_window import coverage_gap, in_window
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.symbols import crypto_base

logger = logging.getLogger(__name__)

FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
}
REQUEST_TIMEOUT = 20
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

# Names a coin goes by in headlines, besides its ticker.
_NAMES = {
    "BTC": ("bitcoin",), "ETH": ("ethereum", "ether"), "SOL": ("solana",),
    "XRP": ("ripple",), "ADA": ("cardano",), "DOGE": ("dogecoin",),
    "LTC": ("litecoin",), "BCH": ("bitcoin cash",), "DOT": ("polkadot",),
    "AVAX": ("avalanche",), "LINK": ("chainlink",),
}
_TAG = re.compile(r"<[^>]+>")


def _items(source: str, url: str) -> list[dict]:
    """The feed's articles; any failure to fetch or read it means it is unavailable."""
    try:
        response = requests.get(url, headers={"User-Agent": _UA}, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except (requests.RequestException, ET.ParseError) as e:
        raise VendorUnavailableError(f"{source} RSS failed: {e}") from e

    articles = []
    for item in root.iter("item"):
        pub = None
        with contextlib.suppress(TypeError, ValueError):
            pub = parsedate_to_datetime(item.findtext("pubDate") or "")
        summary = html.unescape(_TAG.sub("", item.findtext("description") or "")).strip()
        articles.append({
            "title": html.unescape((item.findtext("title") or "").strip()) or "No title",
            "summary": summary[:300],
            "publisher": source,
            "link": (item.findtext("link") or "").strip(),
            "pub_date": pub,
        })
    return articles


def get_news_crypto_rss(ticker: str, start_date: str, end_date: str) -> str:
    """Articles about the coin behind ``ticker`` published in the window."""
    base = crypto_base(ticker)
    if base is None:
        raise NoMarketDataError(ticker, detail="not a crypto pair")

    pattern = re.compile(
        r"\b(?:" + "|".join(map(re.escape, (base.lower(), *_NAMES.get(base, ())))) + r")\b",
        re.IGNORECASE,
    )
    articles, failures = [], []
    for source, url in FEEDS.items():
        try:
            articles += _items(source, url)
        except VendorUnavailableError as e:
            logger.warning("%s", e)
            failures.append(e)
    if len(failures) == len(FEEDS):
        raise failures[0]

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    about = [a for a in articles if pattern.search(f"{a['title']} {a['summary']}")]
    found = sorted(
        (a for a in about if in_window(a["pub_date"], start_dt, end_dt)),
        key=lambda a: str(a["pub_date"]), reverse=True,
    )[: get_config()["news_article_limit"]]

    if found:
        body = "".join(
            f"### {a['title']} (source: {a['publisher']})\n"
            + (f"{a['summary']}\n" if a["summary"] else "")
            + (f"Link: {a['link']}\n" if a["link"] else "")
            + "\n"
            for a in found
        )
        return f"## {ticker} News, from {start_date} to {end_date}:\n\n{body}"

    # Coverage is judged on every item the feeds returned, not just the matches.
    gap = coverage_gap(
        (a["pub_date"] for a in articles), start_date, end_date,
        "Crypto news RSS", f"news for {ticker}",
    )
    return gap or f"No news found for {ticker} between {start_date} and {end_date}"
