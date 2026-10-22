import importlib

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
- Price above the 200 SMA.

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
    assert "Overweight above 84,300" in out and "Price above the 200 SMA" in out
    assert "Gaps: The 10-21 row is missing; FRED macro data was unavailable" in out
    assert "**" not in out


def test_summary_escapes_html_and_stays_short(bot):
    mod, _ = bot
    out = mod.summarize("BTC", "d", DECISION.replace("Hold. Overweight", "a<b>& " + "x" * 5000 + " Overweight"), "")
    assert "a<b>&" not in out and "a&lt;b&gt;&amp;" in out and len(out) < 1500
