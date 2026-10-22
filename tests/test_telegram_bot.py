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


def test_summary_has_fields_and_keeps_gaps(bot):
    mod, _ = bot
    out = mod.summarize("BTCUSDT", "2026-10-22", DECISION)
    assert out.startswith("BTCUSDT 2026-10-22: Hold\nTarget: not provided | Horizon: 2-6 weeks")
    assert "Overweight above 84,300" in out and "The 10-21 row is missing" in out
    assert "**" not in out


def test_summary_fits_telegram_limit(bot):
    mod, _ = bot
    long = DECISION.replace("- Price above the 200 SMA.", "\n".join(f"- point {i} " + "x" * 80 for i in range(200)))
    out = mod.summarize("BTCUSDT", "2026-10-22", long)
    assert len(out) <= 4096 and "The 10-21 row is missing" in out and out.endswith("Full report attached.")
