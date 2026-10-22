import importlib
from pathlib import Path

import pytest


@pytest.fixture
def bot(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "1")
    import telegram_bot
    mod = importlib.reload(telegram_bot)
    sent = []
    monkeypatch.setattr(mod, "say", lambda chat, text: sent.append(text))
    monkeypatch.setattr(mod, "analyze", lambda chat, t, d: sent.append(f"ran {t} {d}"))
    return mod, sent


def test_ignores_unlisted_chat(bot):
    mod, sent = bot
    mod.handle({"chat": {"id": 2}, "text": "/analyze BTCUSDT"})
    assert sent == []


def test_runs_allowed_chat(bot):
    mod, sent = bot
    mod.handle({"chat": {"id": 1}, "text": "/analyze BTCUSDT 2026-10-20"})
    assert sent[-1] == "ran BTCUSDT 2026-10-20"


def test_rejects_bad_ticker(bot):
    mod, sent = bot
    mod.handle({"chat": {"id": 1}, "text": "/analyze $(rm -rf)"})
    assert sent == ["Usage: /analyze TICKER [YYYY-MM-DD]"]


DECISION = """**Rating**: Hold

**Executive Summary**: Hold. Overweight above 84,300; underweight below 81,400.

**Investment Thesis**: Balanced.

**Bullish evidence:**
- Price above the 200 SMA and rising.

**Data gaps:**
- The 10-21 row is missing.
- FRED macro data was unavailable.

**What would change the view:**
- Close above 84,300.

**Price Target**: not provided

**Time Horizon**: 2-6 weeks"""


PLAN = """**Action**: Hold

**Entry Price**: 81400.0

**Stop Loss**: not provided

**Position Sizing**: 1% of book"""


def test_summary_card(bot):
    mod, _ = bot
    out = mod.summarize("BTCUSDT", "2026-10-22", DECISION, PLAN)
    assert out.startswith("🟡 <b>BTCUSDT</b> · 2026-10-22\n<b>Hold</b> · horizon 2-6 weeks")
    assert "Entry <b>81400.0</b> · Size <b>1% of book</b>" in out and "Stop" not in out
    assert "Overweight above 84,300" in out and "<b>Bullish evidence</b> Price above the 200 SMA and rising" in out
    assert "Gaps: The 10-21 row is missing. FRED macro data was unavailable." in out
    assert "**" not in out


def test_summary_escapes_html_and_stays_short(bot):
    mod, _ = bot
    out = mod.summarize("BTC", "d", DECISION.replace("Hold. Overweight", "a<b>& " + "x" * 5000 + " Overweight"), "")
    assert "a<b>&" not in out and "a&lt;b&gt;&amp;" in out and len(out) < 1500


def test_summary_paragraph_thesis_long_size_and_period_gaps(bot):
    mod, _ = bot
    decision = ("**Rating**: Hold\n\n**Executive Summary**: Hold.\n\n**Investment Thesis**: Narrow.\n\n"
                "**Trend (supports holding).** Price is 13.6% above the 200 SMA (72,843) today.\n\n"
                "**Data gaps.** No live price. Volume looks incomplete.\n\n"
                "**Price Target**: not provided\n\n**Time Horizon**: 1-3 weeks")
    plan = "**Stop Loss**: 78631.0\n\n**Position Sizing**: " + "long " * 20
    out = mod.summarize("BTC", "d", decision, plan)
    assert "<b>Trend (supports holding)</b> Price is 13.6% above the 200 SMA (72,843) today." in out
    assert "Gaps: No live price. Volume looks incomplete." in out
    assert "Stop <b>78631.0</b>" in out and "Size" not in out


def test_summary_plain_paragraph_thesis(bot):
    mod, _ = bot
    decision = ("**Rating**: Hold\n\n**Executive Summary**: Hold.\n\n"
                "**Investment Thesis**: The evidence is balanced after weighing, so Hold stands.\n\n"
                "For the bull: price is above the 10 EMA (82,733) and the uptrend is intact.\n\n"
                "Missing from the evidence: spot volume and ETF flow size. Sizing is relative.\n\n"
                "**Price Target**: not provided\n\n**Time Horizon**: 2-6 weeks")
    out = mod.summarize("BTC", "d", decision, "")
    assert "• The evidence is balanced after weighing, so Hold stands." in out
    assert "• <b>For the bull</b> price is above the 10 EMA (82,733)" in out
    assert "Missing: spot volume and ETF flow size. Sizing is relative." in out


def test_analyze_sends_card_then_report(bot, monkeypatch):
    """The whole path with a stub graph: HTML card first, then the full report as a file."""
    mod, _ = bot
    mod = importlib.reload(mod)  # undo the fixture's stubbed analyze
    sent = []
    monkeypatch.setattr(mod, "call", lambda method, **kw: sent.append((method, kw)))

    class Graph:
        def __init__(self, **kw):
            pass

        def propagate(self, ticker, date, asset_type):
            assert asset_type == "crypto"
            return {"final_trade_decision": DECISION, "trader_investment_plan": PLAN}, "Hold"

        def save_reports(self, state, ticker, path, html):
            report = f"{path}/complete_report.md"
            Path(report).write_text("# full report")
            return report

    monkeypatch.setattr(mod, "TradingAgentsGraph", Graph)
    mod.analyze("1", "BTCUSDT", "2026-10-22")
    (m1, kw1), (m2, kw2) = sent
    assert m1 == "sendMessage" and kw1["json"]["parse_mode"] == "HTML" and "<b>Hold</b>" in kw1["json"]["text"]
    assert "Entry <b>81400.0</b>" in kw1["json"]["text"]
    assert m2 == "sendDocument" and kw2["files"]["document"][0] == "BTCUSDT-2026-10-22.md"


def test_analyze_falls_back_to_plain_text_when_html_rejected(bot, monkeypatch):
    import requests

    mod, _ = bot
    mod = importlib.reload(mod)
    sent = []

    def call(method, **kw):
        sent.append((method, kw))
        if kw.get("json", {}).get("parse_mode"):
            raise requests.HTTPError("400")

    monkeypatch.setattr(mod, "call", call)

    class Graph:
        def __init__(self, **kw):
            pass

        def propagate(self, *a, **k):
            return {"final_trade_decision": DECISION}, "Hold"

        def save_reports(self, state, ticker, path, html):
            Path(f"{path}/r.md").write_text("x")
            return f"{path}/r.md"

    monkeypatch.setattr(mod, "TradingAgentsGraph", Graph)
    mod.analyze("1", "BTCUSDT", "2026-10-22")
    assert [m for m, _ in sent] == ["sendMessage", "sendMessage", "sendDocument"]
    assert sent[1][1]["json"]["text"].startswith("**Rating**: Hold")
