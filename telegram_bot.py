"""Telegram front-end: `/analyze BTCUSDT [YYYY-MM-DD]` -> decision + full report.

Long-polling, so it needs no public URL. Env: TELEGRAM_BOT_TOKEN and
TELEGRAM_ALLOWED_CHAT_IDS (comma-separated; anyone else is ignored).
"""
import datetime
import html
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


def say(chat, text, **kw):
    call("sendMessage", json={"chat_id": chat, "text": text[:4096], **kw})


EMOJI = {"Buy": "🟢", "Overweight": "🟢", "Hold": "🟡", "Underweight": "🔴", "Sell": "🔴"}


def clip(text, n):
    """Cut at a word boundary so a number is never split."""
    text = text.replace("**", "").strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


def summarize(ticker, date, decision, trade_plan=""):
    """Short HTML chat card from the Portfolio Manager's decision and the Trader's levels.

    Both are rendered with fixed `**Label**:` fields (render_pm_decision,
    render_trader_proposal), so levels are read by label and stay exact; only prose is
    clipped. The full text goes in the attached report.
    """
    labels = "Rating|Executive Summary|Investment Thesis|Price Target|Time Horizon"
    parts = re.split(rf"\*\*({labels})\*\*: ", decision)
    f = {k: v.strip() for k, v in zip(parts[1::2], parts[2::2], strict=True)}
    if "Rating" not in f:  # shortcut: unknown shape, send it raw; fix if render_pm_decision changes
        return html.escape(decision[:3500], quote=False)
    def e(x):
        return html.escape(x, quote=False)  # Telegram only needs & < > escaped

    rating = f["Rating"]
    out = [f"{EMOJI.get(rating, '⚪')} <b>{e(ticker)}</b> · {e(date)}", f"<b>{e(rating)}</b> · horizon {e(clip(f.get('Time Horizon', 'n/a'), 60))}"]

    t = dict(re.findall(r"\*\*(Entry Price|Stop Loss|Position Sizing)\*\*: (.+)", trade_plan))
    levels = {"Entry": t.get("Entry Price"), "Stop": t.get("Stop Loss"), "Target": f.get("Price Target"),
              "Size": t.get("Position Sizing")}
    # A long sizing paragraph belongs in the report; the card shows only a short one.
    levels = [f"{k} <b>{e(v.strip())}</b>" for k, v in levels.items()
              if v and v.strip() != "not provided" and len(v) <= 40]
    if levels:
        out += ["", "📍 " + " · ".join(levels)]

    out += ["", e(clip(f.get("Executive Summary", ""), 600))]

    # The thesis is bold-led blocks ("**Trend.** text" or "**Bullish evidence:**\n- item"):
    # show label + first bullet or sentence of the first three, and fold "Data gaps" into one line.
    blocks = re.findall(r"\*\*([^*\n]+?)[.:]?\*\*[.:]?\s*(.*?)(?=\n\*\*|\Z)", f.get("Investment Thesis", ""), re.DOTALL)
    gaps = [body for label, body in blocks if label.lower().startswith("data gaps")]
    bullets = []
    for label, body in [x for x in blocks if not x[0].lower().startswith("data gaps")]:
        body = body.strip()
        first = re.match(r"- (.+)", body) or re.match(r"(.+?[.!?])(?:\s|$)", body, re.DOTALL)
        if first and len(first.group(1)) > 25 and len(bullets) < 3:  # a bare "Aggressive:" sub-label says nothing
            bullets.append(f"• <b>{e(clip(label, 40))}</b> {e(clip(first.group(1), 150))}")
    if bullets:
        out += [""] + bullets
    if gaps:
        items = [x.strip("- ").strip().rstrip(".") for x in gaps[0].strip().splitlines() if x.strip()]
        out += ["", f"⚠️ <i>Gaps: {e(clip('; '.join(items), 260))}</i>"]
    out += ["", "📎 Full report attached."]
    return "\n".join(out)


def analyze(chat, ticker, date):
    kind = detect_asset_type(ticker)
    analysts = [a.value for a in filter_analysts_for_asset_type(list(AnalystType), kind)]
    graph = TradingAgentsGraph(selected_analysts=analysts, config=DEFAULT_CONFIG.copy())
    state, _ = graph.propagate(ticker, date, asset_type=kind.value)
    say(chat, summarize(ticker, date, state["final_trade_decision"], state.get("trader_investment_plan", "")),
        parse_mode="HTML")
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
