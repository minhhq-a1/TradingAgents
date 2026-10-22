"""Telegram front-end: `/analyze BTCUSDT [YYYY-MM-DD]` -> decision + full report.

Long-polling, so it needs no public URL. Env: TELEGRAM_BOT_TOKEN and
TELEGRAM_ALLOWED_CHAT_IDS (comma-separated; anyone else is ignored).
"""
import datetime
import os
import re
import tempfile

import requests

from cli.models import AnalystType
from cli.prompts import detect_asset_type, filter_analysts_for_asset_type
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

API = f"https://api.telegram.org/bot{os.environ.get('TELEGRAM_BOT_TOKEN', '')}"
ALLOWED = {c.strip() for c in os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", "").split(",") if c.strip()}
TICKER = re.compile(r"^[A-Za-z0-9.\-^=]{1,20}$")


def call(method, **kw):
    r = requests.post(f"{API}/{method}", timeout=70, **kw)
    r.raise_for_status()
    return r.json()["result"]


def say(chat, text):
    call("sendMessage", json={"chat_id": chat, "text": text[:4096]})


def summarize(ticker, date, decision, limit=4096):
    """Chat summary of the Portfolio Manager's decision, within Telegram's message cap.

    The decision is rendered by `render_pm_decision` with fixed `**Label**:` fields,
    so the fields are split by label; the thesis is cut at a line boundary to fit,
    but its "Data gaps" block is always kept.
    """
    parts = re.split(r"\*\*(Rating|Executive Summary|Investment Thesis|Price Target|Time Horizon)\*\*: ", decision)
    f = {k: v.strip() for k, v in zip(parts[1::2], parts[2::2], strict=True)}
    if "Rating" not in f:  # shortcut: unknown shape, send it raw; fix if render_pm_decision changes
        return decision[:limit]
    thesis = f.get("Investment Thesis", "")
    gaps = re.search(r"\*\*Data gaps:?\*\*:?.*?(?=\n\*\*|\Z)", thesis, re.DOTALL)
    if gaps:
        thesis = thesis.replace(gaps.group(0), "").strip()
    head = (f"{ticker} {date}: {f['Rating']}\n"
            f"Target: {f.get('Price Target', 'n/a')} | Horizon: {f.get('Time Horizon', 'n/a')}\n\n"
            f"{f.get('Executive Summary', '')}")
    tail = (gaps.group(0) if gaps else "") + "\n\nFull report attached."
    room = limit - len(head) - len(tail) - 20
    if room > 0 and thesis:
        if len(thesis) > room:
            thesis = thesis[:room].rsplit("\n", 1)[0] + "\n..."
        head += f"\n\n{thesis}"
    return (head + "\n\n" + tail.strip()).replace("**", "")[:limit]


def analyze(chat, ticker, date):
    kind = detect_asset_type(ticker)
    analysts = [a.value for a in filter_analysts_for_asset_type(list(AnalystType), kind)]
    graph = TradingAgentsGraph(selected_analysts=analysts, config=DEFAULT_CONFIG.copy())
    state, _ = graph.propagate(ticker, date, asset_type=kind.value)
    say(chat, summarize(ticker, date, state["final_trade_decision"]))
    # Telegram caps a message at 4096 chars, so the full report goes as a file.
    with tempfile.TemporaryDirectory() as tmp:
        report = graph.save_reports(state, ticker, tmp, html=False)
        with open(report, "rb") as f:
            call("sendDocument", data={"chat_id": chat}, files={"document": (f"{ticker}-{date}.md", f)})


def handle(msg):
    chat = str(msg["chat"]["id"])
    if chat not in ALLOWED:
        return
    parts = (msg.get("text") or "").split()
    if not parts or parts[0].split("@")[0] != "/analyze":
        return say(chat, "Usage: /analyze TICKER [YYYY-MM-DD]")
    ticker = parts[1] if len(parts) > 1 else ""
    date = parts[2] if len(parts) > 2 else datetime.date.today().isoformat()
    if not TICKER.match(ticker):
        return say(chat, "Usage: /analyze TICKER [YYYY-MM-DD]")
    say(chat, f"Analyzing {ticker} {date}, about 3 minutes...")
    try:
        analyze(chat, ticker, date)
    except Exception as exc:  # report and keep serving
        say(chat, f"Failed: {exc}")


def main():
    assert ALLOWED, "set TELEGRAM_ALLOWED_CHAT_IDS"
    offset = None
    while True:
        try:
            updates = call("getUpdates", json={"timeout": 50, "offset": offset})
        except requests.RequestException:
            continue
        for u in updates:
            offset = u["update_id"] + 1
            if "message" in u:
                handle(u["message"])  # sequential: one analysis at a time


if __name__ == "__main__":
    main()
