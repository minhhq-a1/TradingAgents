import importlib
import json
import re
from pathlib import Path

import pytest


@pytest.fixture
def bot(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "1")
    import telegram_bot
    mod = importlib.reload(telegram_bot)
    sent = []
    monkeypatch.setattr(mod, "say", lambda chat, text, **kw: sent.append(text))
    monkeypatch.setattr(mod, "analyze", lambda chat, t, d, lang="en": sent.append(f"ran {t} {d}" + (f" {lang}" if lang != "en" else "")))
    monkeypatch.setattr(mod.trend, "DIR", str(tmp_path))  # the chats' /lang choices
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


@pytest.mark.parametrize("text", ["/start", "hello", "/start@my_bot", ""])
def test_unknown_message_lists_the_commands(bot, text):
    mod, sent = bot
    mod.handle({"chat": {"id": 1}, "text": text})
    assert sent == [mod.COMMANDS] and all(c in mod.COMMANDS for c in ("/analyze", "/trend", "/paper"))


def test_status_and_health_report_whether_the_job_is_alive(bot, monkeypatch):
    mod, sent = bot
    monkeypatch.setattr(mod.trend, "health_text", lambda symbol, lang="en": f"health of {symbol}")
    mod.handle({"chat": {"id": 1}, "text": "/status"})
    mod.handle({"chat": {"id": 1}, "text": "/health@MyBot ETHUSDT"})
    mod.handle({"chat": {"id": 2}, "text": "/status"})  # unlisted chat: ignored
    assert sent == ["health of BTCUSDT\n\nhealth of BTCUSDT-ICHIMOKU", "health of ETHUSDT"] and "/status" in mod.COMMANDS


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
    assert out.startswith("🟡 <b>BTCUSDT · Hold</b>\n2026-10-22 · horizon 2-6 weeks")
    assert "📍 <b>Levels</b>\nEntry <b>81400.0</b>\nSize <b>1% of book</b>" in out and "Stop" not in out
    assert "🎯 <b>Triggers</b>\n▲ close above <b>84,300</b>" in out and "• <b>Bullish evidence:</b> Price above the 200 SMA and rising." in out
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
    assert "<b>Trend (supports holding):</b> Price is 13.6% above the 200 SMA (72,843) today." in out
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
    assert "• <b>For the bull:</b> price is above the 10 EMA (82,733)" in out
    assert "Missing: spot volume and ETF flow size. Sizing is relative." in out


REPORT_4 = (
    "**Rating**: Hold\n\n**Executive Summary**: Hold the existing BTCUSDT position. The one level that would change "
    "the rating is a daily close below 80,191, which means reducing to Underweight.\n\n**Investment Thesis**: "
    "The trend is intact. Price is 84,093.42, about 2.9% above a rising 50-SMA (81,743). Every close since September 22 "
    "has held above 81,491. This blocks an Underweight call while price holds above the 50-SMA.\n\n"
    "The directional evidence for adding is weak. MACD is below its signal line, falling from about 2,487 to 181. "
    "Price has failed twice near 86.5k.\n\n"
    "The balanced range of roughly 80.2k to 86.5k therefore stands, and the evidence is too even for a directional call.\n\n"
    "Triggers: a daily close below 80,191 moves the rating to Underweight. The upgrade trigger is a daily close above "
    "86,620, which would end the lower-high pattern. Do not add on that single print. No on-chain data or ETF flow "
    "series were provided, and the institutional-demand claim rests on one headline.\n\n"
    "**Price Target**: not provided\n\n**Time Horizon**: not provided")


def test_summary_real_report_shape(bot):
    """A real PM report: no 'not provided' fields, both triggers, whole-sentence bullets, gaps line."""
    mod, _ = bot
    out = mod.summarize("BTCUSDT", "2026-10-22", REPORT_4, "")
    assert "not provided" not in out.split("⚠️")[0] and "horizon" not in out
    assert "🎯 <b>Triggers</b>\n▲ close above <b>86,620</b>\n▼ close below <b>80,191</b>\n" in out
    bullets = [x for x in out.splitlines() if x.startswith("• ")]
    assert len(bullets) == 3 and all(x.endswith(".") for x in bullets)
    assert "⚠️ <i>Gaps: No on-chain data or ETF flow series were provided" in out


def test_summary_one_paragraph_thesis_gives_three_bullets(bot):
    mod, _ = bot
    one = REPORT_4.replace("\n\n", " ").replace("**Rating**: Hold **Executive", "**Rating**: Hold\n\n**Executive")
    one = one.replace(" **Investment Thesis**:", "\n\n**Investment Thesis**:").replace(" **Price Target**", "\n\n**Price Target**")
    out = mod.summarize("BTCUSDT", "d", one, "")
    assert len([x for x in out.splitlines() if x.startswith("• ")]) >= 2 and "close above <b>86,620</b>" in out


def test_summary_always_has_a_gaps_line(bot):
    mod, _ = bot
    out = mod.summarize("BTC", "d", "**Rating**: Hold\n\n**Executive Summary**: Hold.\n\n**Investment Thesis**: Fine.",
                        "")
    assert "⚠️ <i>Gaps: none stated in the decision.</i>" in out


def test_summary_live_run_shapes(bot):
    """Shapes from live runs: a yield % is not a price trigger, the gaps line keeps only the gap clause,
    the horizon is the period, and "Why ... not" blocks give no key point."""
    mod, _ = bot
    out = mod.summarize("ETHUSDT", "d", (
        "**Rating**: Underweight\n\n**Executive Summary**: Trim. Add back on a daily close above 2,552.\n\n"
        "**Investment Thesis**: Price is below the 50 SMA and momentum is negative.\n\n"
        "Why the call is not Sell, and why 75% is too aggressive:\n- RSI never reached 30 so there was no capitulation.\n"
        "- If the 10-year closes above 5.31% and holds, don't add.\n\n"
        "Why not Sell: price is above a rising 200 SMA, the 10 EMA has been reclaimed, and the news feed is missing.\n"
        "CPI and core PCE are missing.\n\n"
        "**Price Target**: 2296.0\n\n**Time Horizon**: 2-6 weeks, reviewed on daily closes against 1.29 and 1.41"), "")
    assert "<b>ETHUSDT · Underweight</b>\nd · horizon 2-6 weeks\n" in out
    assert "▲ close above <b>2,552</b>\n" in out and "5.31" not in out.split("🎯")[1].split("\n\n")[0]
    assert "Key points" in out and "RSI never reached 30" not in out and "Why not Sell" not in out
    assert "⚠️ <i>Gaps: The news feed is missing. CPI and core PCE are missing.</i>" in out


def test_key_points_skip_rejected_case_and_levels(bot):
    """2026-10-24 BTCUSDT Hold: the card quoted "It wants a tranche added now", the Aggressive case the PM
    rejected under "Why not Buy", so it read as a buy call. Key points come from the PM's own reasoning."""
    mod, _ = bot
    thesis = (
        "The evidence is balanced, and neither the Aggressive nor the Conservative case wins decisively.\n\n"
        "Why not Buy or Overweight (the Aggressive case):\n- It wants a tranche added now with a stop near 76,904.\n"
        "- The close was 81,189.43, below the level the bounce claim needs.\n\n"
        "Why not Underweight or Sell (the Conservative case):\n- Price is still about 11.4% above a rising 200 SMA.\n\n"
        "What favors caution:\n- The close is below the 50 SMA, the 10 EMA and the Bollinger middle.\n\n"
        "Levels that change the rating:\n- Downgrade to Underweight on a daily close below 80,513.\n"
        "- The lower Bollinger band at 79,810 would then be the next checkpoint.\n\n"
        "Also avoid adding into weakness if open interest rises on down moves, because that signals new shorts.")
    for shape in (thesis, thesis.replace(":\n- ", ":\n\n").replace("\n- ", "\n\n"),  # labels on their own line
                  "\n".join(f"**{p.split(':')[0]}:**{p.split(':', 1)[1]}" if ":\n" in p else p
                            for p in thesis.split("\n\n"))):  # bold labels
        out = mod.summarize("BTCUSDT", "d", f"**Rating**: Hold\n\n**Executive Summary**: Hold.\n\n"
                                           f"**Investment Thesis**: {shape}\n\n**Price Target**: not provided", "")
        bullets = out.split("<b>Key points</b>\n")[1].split("\n\n")[0]
        assert "tranche" not in bullets and "81,189" not in bullets and "11.4%" not in bullets
        assert "79,810" not in bullets and "▼ close below <b>80,513</b>" in out
        assert "What favors caution:</b> The close is below the 50 SMA" in bullets
        assert "Also avoid adding" in bullets or shape != thesis  # in the other shapes it sits under the levels label


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
    monkeypatch.setattr(mod, "chart", lambda ticker, date, card, lang="en": b"png")
    mod.analyze("1", "BTCUSDT", "2026-10-22")
    (m1, kw1), (m2, kw2), (m3, kw3) = sent
    assert m1 == "sendMessage" and kw1["json"]["parse_mode"] == "HTML" and "<b>BTCUSDT · Hold</b>" in kw1["json"]["text"]
    assert "Entry <b>81400.0</b>" in kw1["json"]["text"]
    assert m2 == "sendPhoto" and kw2["files"]["photo"] == ("chart.png", b"png")
    assert m3 == "sendDocument" and kw3["files"]["document"][0] == "BTCUSDT-2026-10-22.md"

    sent.clear()  # a failing chart is skipped: the card and the report still go out
    monkeypatch.setattr(mod, "chart", lambda *a: 1 / 0)
    mod.analyze("1", "BTCUSDT", "2026-10-22")
    assert [m for m, _ in sent] == ["sendMessage", "sendDocument"]


def test_chart_draws_the_card_levels(bot, monkeypatch):
    import io

    import pandas as pd
    from PIL import Image

    mod, _ = bot
    close = pd.Series([100 + i * 0.1 for i in range(300)])
    df = pd.DataFrame({"Date": pd.date_range("2025-12-01", periods=300), "Open": close - 1, "High": close + 2,
                       "Low": close - 2, "Close": close})
    monkeypatch.setattr(mod, "load_ohlcv", lambda ticker, date: df)
    card = "🎯 <b>Triggers</b>\n▲ close above <b>$135</b>\n▼ close below <b>1,200</b>\n📍 <b>Levels</b>\nStop <b>118.5</b>\nTarget <b>n/a</b>"
    png = Image.open(io.BytesIO(mod.chart("NVDA", "2026-09-26", card)))
    assert png.size == (1000, 560)
    assert [mod.level(x) for x in ("$135", "84,093.42", "84.5k", "~2500.0", "1.39 (trim)")] == [
        135, 84093.42, 84500, 2500, None]
    card_vi = "▲ đóng cửa trên <b>135,5</b>\n▼ đóng cửa dưới <b>1.200</b>\nCắt lỗ <b>118.5</b>\nMục tiêu <b>n/a</b>"
    assert Image.open(io.BytesIO(mod.chart("NVDA", "2026-09-26", card_vi, "vi"))).size == (1000, 560)
    # Vietnamese writes 82.626 and 80.888,16; a level the program wrote ("82626.0", "118.5") reads as it does in English
    assert [mod.level(x, "vi") for x in ("82.626", "80.888,16", "$1.200", "82626.0", "118.5", "2,5")] == [
        82626, 80888.16, 1200, 82626, 118.5, 2.5]
    assert mod.level("82.626") == 82.626  # an English card is read as before


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
    monkeypatch.setattr(mod, "chart", lambda ticker, date, card, lang="en": b"png")
    mod.analyze("1", "BTCUSDT", "2026-10-22")
    assert [m for m, _ in sent] == ["sendMessage", "sendMessage", "sendPhoto", "sendDocument"]
    assert sent[1][1]["json"]["text"].startswith("**Rating**: Hold")


def test_trend_command_replies_with_the_trend_state(bot, monkeypatch):
    mod, sent = bot
    monkeypatch.setattr(mod.trend, "status_text", lambda symbol, lang="en": f"state of {symbol}")
    mod.handle({"chat": {"id": 1}, "text": "/trend"})
    mod.handle({"chat": {"id": 1}, "text": "/trend@MyBot ETHUSDT"})
    mod.handle({"chat": {"id": 2}, "text": "/trend"})  # unlisted chat: ignored
    assert sent == ["state of BTCUSDT", "state of ETHUSDT"] and "/trend ichimoku" in mod.COMMANDS


def test_paper_command_and_analysis_label(bot, monkeypatch):
    mod, sent = bot
    monkeypatch.setattr(mod.trend, "paper_text", lambda symbol, lang="en": f"book of {symbol}")
    mod.handle({"chat": {"id": 1}, "text": "/paper"})
    assert sent == ["book of BTCUSDT"]
    assert "không phải tín hiệu giao dịch" in mod.REFERENCE["vi"] and "not a trading signal" in mod.REFERENCE["en"]


def test_lang_sets_each_chats_language_and_every_command_follows_it(bot, monkeypatch):
    mod, sent = bot
    monkeypatch.setattr(mod, "ALLOWED", {"1", "3"})
    seen = []
    for name in ("status_text", "paper_text", "health_text"):
        monkeypatch.setattr(mod.trend, name, lambda symbol, lang="en", name=name: seen.append((name, lang)) or f"{name} {lang}")
    mod.handle({"chat": {"id": 1}, "text": "/lang"})
    assert sent[-1].startswith("Language: English") and "/lang vi" in sent[-1]
    mod.handle({"chat": {"id": 1}, "text": "/lang@MyBot VI"})
    assert sent[-1].startswith("✅ Ngôn ngữ: tiếng Việt") and mod.trend.chat_lang("1") == "vi"
    for text in ("/trend", "/paper", "/status", "/analyze BTCUSDT 2026-10-20", "/analyze $(x)", "/help"):
        mod.handle({"chat": {"id": 1}, "text": text})
    assert seen == [("status_text", "vi"), ("paper_text", "vi"), ("health_text", "vi"), ("health_text", "vi")]
    assert sent[-4:] == ["Đang phân tích BTCUSDT 2026-10-20, khoảng 3 phút...", "ran BTCUSDT 2026-10-20 vi",
                         "Cách dùng: /analyze MÃ [YYYY-MM-DD]", mod.COMMANDS_VI]
    mod.handle({"chat": {"id": 3}, "text": "/trend"})  # another chat keeps English
    assert seen[-1] == ("status_text", "en")
    mod.handle({"chat": {"id": 1}, "text": "/lang fr"})
    assert sent[-1].startswith("Ngôn ngữ: tiếng Việt. Cách dùng") and mod.trend.chat_lang("1") == "vi"
    mod.handle({"chat": {"id": 1}, "text": "/lang en"})
    assert sent[-1].startswith("✅ Language: English") and mod.trend.chat_lang("1") == "en"
    mod.handle({"chat": {"id": 2}, "text": "/lang vi"})  # an unlisted chat cannot write the file
    assert mod.trend.chat_lang("2") == "en"
    assert [line.split(":")[0] for line in mod.COMMANDS.splitlines()[1:]] == [
        line.split(":")[0].replace("MÃ", "TICKER" if "analyze" in line else "SYMBOL") for line in mod.COMMANDS_VI.splitlines()[1:]]


def test_analyze_in_vietnamese_asks_the_agents_for_vietnamese_and_labels_everything_in_it(bot, monkeypatch):
    mod, _ = bot
    mod = importlib.reload(mod)
    sent, configs = [], []
    monkeypatch.setattr(mod, "call", lambda method, **kw: sent.append((method, kw)))

    class Graph:
        def __init__(self, config, **kw):
            configs.append(config)

        def propagate(self, ticker, date, asset_type):
            return {"final_trade_decision": DECISION, "trader_investment_plan": PLAN}, "Hold"

        def save_reports(self, state, ticker, path, html):
            Path(f"{path}/r.md").write_text("x")
            return f"{path}/r.md"

    monkeypatch.setattr(mod, "TradingAgentsGraph", Graph)
    monkeypatch.setattr(mod, "chart", lambda ticker, date, card, lang="en": b"png")
    mod.analyze("1", "BTCUSDT", "2026-10-22", "vi")
    mod.analyze("1", "BTCUSDT", "2026-10-22")
    assert configs[0]["output_language"] == "Vietnamese" and configs[1]["output_language"] == "English"
    assert mod.DEFAULT_CONFIG["output_language"] == "English"  # a Vietnamese run never leaks into the next one
    card_vi, card_en = sent[0][1]["json"]["text"], sent[3][1]["json"]["text"]
    assert "<b>BTCUSDT · Giữ</b>" in card_vi and "Vào lệnh <b>81400.0</b>" in card_vi and "📍 <b>Mức giá</b>" in card_vi
    assert "🎯 <b>Ngưỡng kích hoạt</b>" in card_vi and "▲ đóng cửa trên <b>84,300</b>" in card_vi
    assert "Báo cáo tham khảo" in card_vi and sent[2][1]["data"]["caption"].startswith("Báo cáo tham khảo")
    assert "<b>BTCUSDT · Hold</b>" in card_en and "Reference report" in card_en and sent[5][1]["data"]["caption"].startswith("Reference")


# Real /analyze runs with output_language Vietnamese (2026-10-24, vilao); XRPUSDT is the PM's free-text
# fallback, which heads each field on its own line and repeats the trader's FINAL TRANSACTION PROPOSAL.
VI_RUNS = json.loads((Path(__file__).parent / "data" / "telegram_vi_decisions.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("ticker", sorted(VI_RUNS))
def test_vietnamese_reports_give_a_short_vietnamese_card_with_their_own_levels(bot, ticker):
    from tests.test_trend import _telegram_html

    mod, _ = bot
    run = VI_RUNS[ticker]
    card = mod.summarize(ticker, "2026-10-24", run["decision"], run["plan"], "vi")
    head = card.splitlines()[0]
    assert _telegram_html(card) and len(card) < 1500 and "FINAL TRANSACTION" not in card, card
    assert head == f"🔴 <b>{ticker} · Giảm tỷ trọng</b>"  # every run was Underweight
    for label in ("📍 <b>Mức giá</b>", "📝 <b>Tóm tắt</b>", "🎯 <b>Ngưỡng kích hoạt</b>", "<b>Ý chính</b>", "⚠️ <i>Thiếu",
                  "📎 Báo cáo đầy đủ đính kèm."):
        assert label in card, (label, card)
    assert not re.search(r"\b(?:Summary|Levels|Triggers|Key points|Gaps|close above|close below|Entry|Stop|Target)\b", card)
    # every trigger is a level the PM wrote, copied verbatim, and the chart reads it at the right size
    triggers = re.findall(r"đóng cửa (?:trên|dưới) <b>([^<]+)</b>", card)
    assert len(triggers) >= 2 and all(t in run["decision"] for t in triggers)
    entry = mod.level(re.search(r"Vào lệnh <b>([^<]+)</b>", card)[1])
    assert all(entry / 1.5 < mod.level(t, "vi") < entry * 1.5 for t in triggers), (entry, triggers)
    bullets = [x for x in card.splitlines() if x.startswith("• ")]
    assert 1 <= len(bullets) <= 3 and not any(re.match(r"• (?:<b>[^<]*</b> )?(?:Vì vậy|Do đó|Đây|Ba yếu tố này)", x) for x in bullets)
