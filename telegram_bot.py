"""Telegram front-end: `/analyze BTCUSDT [YYYY-MM-DD]` -> decision + full report;
`/trend [BTCUSDT]` -> paper trend state, plus an alert when it flips; `/paper` -> the paper book
(tradingagents/trend.py). An /analyze report is reference material, not a trading signal.
`/lang vi|en` sets the chat's language for every reply, alert and /analyze report (English by default).

Long-polling, so it needs no public URL. Env: TELEGRAM_BOT_TOKEN and
TELEGRAM_ALLOWED_CHAT_IDS (comma-separated; anyone else is ignored).
"""
import datetime
import html
import io
import math
import os
import re
import tempfile
import threading

import requests
from PIL import Image, ImageDraw, ImageFont

from cli.models import AnalystType
from cli.prompts import detect_asset_type, filter_analysts_for_asset_type
from tradingagents import trend
from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

API = f"https://api.telegram.org/bot{os.environ.get('TELEGRAM_BOT_TOKEN', '')}"
ALLOWED = {c.strip() for c in os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", "").split(",") if c.strip()}
REFERENCE = {"en": "Reference report, not a trading signal.", "vi": "Báo cáo tham khảo, không phải tín hiệu giao dịch."}
COMMANDS = ("Commands:\n/analyze TICKER [YYYY-MM-DD]: full analysis report\n"
            "/trend [SYMBOL]: SMA200 trend rule, current state (paper)\n/paper [SYMBOL]: paper book, equity and ledger\n"
            "/trend ichimoku, /paper ichimoku: the Ichimoku shadow book next to it\n"
            "/status: is the daily trend job alive (also /health)\n/lang vi|en: reply language")
COMMANDS_VI = ("Các lệnh:\n/analyze MÃ [YYYY-MM-DD]: báo cáo phân tích đầy đủ\n"
               "/trend [MÃ]: quy tắc xu hướng SMA200, trạng thái hiện tại (giao dịch thử)\n"
               "/paper [MÃ]: sổ giao dịch thử, lợi nhuận và sổ lệnh\n"
               "/trend ichimoku, /paper ichimoku: sổ phụ Ichimoku chạy song song\n"
               "/status: tác vụ xu hướng hằng ngày còn chạy không (hoặc /health)\n/lang vi|en: ngôn ngữ trả lời")
TICKER = re.compile(r"^[A-Za-z0-9.\-^=]{1,20}$")


def call(method, **kw):
    r = requests.post(f"{API}/{method}", timeout=70, **kw)
    r.raise_for_status()
    return r.json()["result"]


def say(chat, text, **kw):
    call("sendMessage", json={"chat_id": chat, "text": text[:4096], **kw})


def say_html(chat, text):
    say(chat, text, parse_mode="HTML")  # every trend message is Telegram HTML


EMOJI = {"Buy": "🟢", "Overweight": "🟢", "Hold": "🟡", "Underweight": "🔴", "Sell": "🔴"}
RATING_VI = {"Buy": "Mua", "Overweight": "Tăng tỷ trọng", "Hold": "Giữ", "Underweight": "Giảm tỷ trọng", "Sell": "Bán"}
# A /lang vi chat gets its report written in Vietnamese (output_language), so the patterns below read both languages.
NUM = r"~?\$?\d+(?:,\d{3})*(?:\.\d+)?k?"
TRIGGER = re.compile(rf"(?:\bcloses?\b(?: a day| back| decisively)?|\bđóng cửa(?: ngày| lại| hẳn| dứt khoát)?(?: ở)?(?: vượt)?)"
                     rf"\s+(above|below|trên|dưới)\s+(?:about |roughly |~|mức |khoảng |ngưỡng )?({NUM})(?![\d.,]*%)", re.I)  # a % is a yield, not a price
UP = ("above", "trên")
GAPS_LABEL = re.compile(r"data (?:conflict and )?gaps?|gaps?|missing\b|(?:khoảng trống|thiếu hụt|thiếu|hạn chế) dữ liệu|dữ liệu (?:còn )?thiếu|thiếu\b", re.I)
GAP_SENTENCE = re.compile(
    r"^(?:missing|data gaps?|gaps?)\b|^no\b[^.]{0,80}\b(?:data|figures?|feeds?|series|inputs?|information)\b"
    r"|\b(?:was|were) not provided\b|\bunavailable\b|\b(?:data|key|feeds?|series|figures?)\b[^.]{0,40}\bmissing\b"
    r"|\b(?:is|are)(?: also)? missing\b|^no\b[^.]{0,80}\b(?:news|fundamentals)\b"
    r"|\bholdings are unknown\b"
    r"|^thiếu\b|\bkhông có (?:sẵn )?(?:dữ liệu|số liệu|thông tin)\b|\bkhông được cung cấp\b|\bkhông khả dụng\b"
    r"|\b(?:dữ liệu|số liệu)\b[^.]{0,40}\b(?:bị thiếu|còn thiếu|không có sẵn)\b", re.I)
DANGLING = re.compile(r"(?:[^:.\n]{3,60}:\s+)?(?:It|This|That|They|These|Those|Điều này|Điều đó|Nó|Những điều này)\b")  # needs the sentence before it, which the card drops
# "Why not Buy", "Why the call is not Sell"; not "Why Hold and not a tilt", which argues for the call
REJECTION = re.compile(r"why\s+(?:not\b|(?:[\w']+\s+){1,3}(?:is|are)\s+not\b)|(?:tại sao|vì sao|lý do)\s+không\b", re.I)
LEVELS_LABEL = re.compile(r"(?:\w+ )?(?:triggers?|levels?)\b|what would change|.{0,30}\bchanges? the (?:view|rating|call)"
                          r"|(?:các )?(?:mức giá|ngưỡng|điểm kích hoạt)\b|điều gì (?:sẽ )?(?:làm )?thay đổi"
                          r"|.{0,30}\bthay đổi (?:quan điểm|đánh giá|khuyến nghị|nhận định)", re.I)
CONCLUSION = re.compile(r"\b(?:therefore|stands|intact|weak|not supported|too even|favou?rs?|wins?|decisive|balanced"
                        r"|do đó|vì vậy|vẫn giữ|còn nguyên|yếu|không được ủng hộ|nghiêng về|chiếm ưu thế|cân bằng)\b", re.I)


INLINE_LABEL = re.compile(r"([A-ZÀ-Ỹ][^:.\d]{2,40}):\s+(\S.*)", re.S)


def sentences(text):
    """Whole sentences, list items and markdown stripped; the card never cuts one."""
    out = []
    for line in text.replace("**", "").splitlines():
        line = re.sub(r"^\s*(?:[-•]|\d+[.)])\s+", "", line).strip()
        # À-Ỹ: Vietnamese capitals (Đ, Ư, Ở...); the range has lower-case letters too, which only an odd "x. đ" splits
        out += re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ỹ$\d(\"'])", line) if line else []
    return out


def fit(parts, limit, sep=" "):
    """Join whole parts in order while they fit; a part that does not fit is skipped, not cut."""
    kept = []
    for part in parts:
        if len(sep.join(kept + [part])) <= limit:
            kept.append(part)
    return sep.join(kept)


def summarize(ticker, date, decision, trade_plan="", lang="en"):
    """Short HTML chat card from the Portfolio Manager's decision and the Trader's levels.

    Both are rendered with fixed `**Label**:` fields (render_pm_decision,
    render_trader_proposal), so levels are read by label and stay exact. Prose is only ever
    chosen by whole sentence, never cut; the full text goes in the attached report.
    """
    labels = "Rating|Executive Summary|Investment Thesis|Price Target|Time Horizon"
    parts = re.split(rf"\*\*({labels})\*\*: ", decision)
    f = {k: v.strip() for k, v in zip(parts[1::2], parts[2::2], strict=True)}
    if "Rating" not in f:  # shortcut: unknown shape, send it raw; fix if render_pm_decision changes
        return html.escape(decision[:3500], quote=False)

    def e(x):
        return html.escape(x, quote=False)  # Telegram only needs & < > escaped

    def given(value):
        return value if value and value.strip().lower() not in ("not provided", "không có", "không được cung cấp") else None

    def tr(en, vi):
        return trend._t(lang, en, vi)

    rating, horizon = f["Rating"], given(f.get("Time Horizon"))
    if horizon:  # "2-6 weeks, reviewed on ..." -> the period itself
        horizon = horizon.split(",")[0]
    shown = RATING_VI.get(rating, rating) if lang == "vi" else rating
    out = [f"{EMOJI.get(rating, '⚪')} <b>{e(ticker)} · {e(shown)}</b>",
           e(date) + (f" · {tr('horizon', 'kỳ hạn')} {e(horizon)}" if horizon else "")]

    t = dict(re.findall(r"\*\*(Entry Price|Stop Loss|Position Sizing)\*\*: (.+)", trade_plan))
    levels = {tr("Entry", "Vào lệnh"): t.get("Entry Price"), tr("Stop", "Cắt lỗ"): t.get("Stop Loss"),
              tr("Target", "Mục tiêu"): f.get("Price Target"), tr("Size", "Khối lượng"): t.get("Position Sizing")}
    # A long sizing paragraph belongs in the report; the card shows only a short one.
    levels = [f"{k} <b>{e(v.strip())}</b>" for k, v in levels.items() if given(v) and len(v) <= 40]
    if levels:
        out += ["", tr("📍 <b>Levels</b>", "📍 <b>Mức giá</b>"), *levels]

    summary = sentences(f.get("Executive Summary", ""))
    # shortcut: a lone sentence over the limit is cut at a word (rare); fix if the PM starts writing run-ons
    out += ["", tr("📝 <b>Summary</b>", "📝 <b>Tóm tắt</b>"),
            e(fit(summary, 600) or summary[0][:600].rsplit(" ", 1)[0] + "…" if summary else "")]

    # The thesis is bold-led blocks ("**Trend.** text", "**Evidence:**\n- item") or plain paragraphs
    # ("For the bull: text"). A block is (label, sentences).
    thesis = f.get("Investment Thesis", "")
    raw = re.findall(r"\*\*([^*\n]+?)[.:]?\*\*[.:]?\s*(.*?)(?=\n\*\*|\Z)", thesis, re.DOTALL)
    if not raw:
        heading = ""  # a label alone on its line ("Why not Buy:") heads the paragraphs under it
        for para in re.split(r"\n\s*\n", thesis):
            m = re.match(rf"((?:{GAPS_LABEL.pattern})[^.:\n]*|[^.:\n]{{3,60}})\s*[.:]\s*(.*)", para.strip(), re.DOTALL | re.I)
            ok = m and (GAPS_LABEL.match(m.group(1)) or para.strip()[len(m.group(1))] == ":")
            heading = (m.group(1) if not m.group(2).strip() else "") if ok else heading
            raw.append(m.groups() if ok else (heading, para.strip()))
    blocks = [(label, sentences(body)) for label, body in raw]
    prose = [x for label, ss in blocks if not GAPS_LABEL.match(label) for x in ss]
    # A "Why not Buy" block argues the case the PM rejected, and a levels block's actions ("Add one tranche")
    # lose their condition; alone on the card either reads as advice. The levels are in Triggers.
    blocks = [(label, ss) for label, ss in blocks
              if not REJECTION.match(label) and not LEVELS_LABEL.match(label) and not (ss and REJECTION.match(ss[0]))]
    while blocks and sum(not GAPS_LABEL.match(label) for label, _ in blocks) < 3:  # few blocks: halve the longest
        i = max(range(len(blocks)), key=lambda j: len(blocks[j][1]) if not GAPS_LABEL.match(blocks[j][0]) else 0)
        label, ss = blocks[i]
        if GAPS_LABEL.match(label) or len(ss) < 2:
            break
        blocks[i:i + 1] = [(label, ss[:len(ss) // 2]), ("", ss[len(ss) // 2:])]

    # Trigger levels, as their own line: every "close above/below N" the PM states (summary first).
    triggers = {}
    for sentence in summary + prose:
        for direction, number in TRIGGER.findall(sentence):
            if len(re.sub(r"\D", "", number)) >= 3 or number.lower().endswith("k"):
                triggers.setdefault((direction.lower() in UP, number.lstrip("~")), None)
    ups = [tr(f"close above <b>{e(n)}</b>", f"đóng cửa trên <b>{e(n)}</b>") for up, n in triggers if up]
    downs = [tr(f"close below <b>{e(n)}</b>", f"đóng cửa dưới <b>{e(n)}</b>") for up, n in triggers if not up]
    if ups or downs:
        out += ["", tr("🎯 <b>Triggers</b>", "🎯 <b>Ngưỡng kích hoạt</b>"), *[f"▲ {x}" for x in ups[:2]], *[f"▼ {x}" for x in downs[:2]]]

    # Up to three conclusions: per block, a whole sentence that is short enough and not already a trigger,
    # a gap or the summary; one that states a conclusion is preferred.
    bullets = []
    for label, ss in blocks:
        if GAPS_LABEL.match(label) or len(bullets) == 3:
            continue
        usable = [x for x in ss if 20 <= len(x) <= 170 and x[-1] != ":" and not TRIGGER.search(x)
                  and not GAP_SENTENCE.search(x) and not REJECTION.match(x) and not DANGLING.match(x)
                  and x not in summary]  # a trailing ":" is a lead-in
        if usable:
            label = re.split(r",(?![^(]*\))", label)[0].strip()  # "Trend, momentum and flows" -> "Trend"
            pick = next((x for x in usable if CONCLUSION.search(x)), usable[0])
            if len(pick) < 40:  # a bare "The trend is intact." says little: add the sentence after it
                nxt = usable[usable.index(pick) + 1:][:1] or [x for x in usable if x != pick][:1]
                pick = fit([pick] + nxt, 170)
            if not label and (m := INLINE_LABEL.match(pick)):  # "Trend: text" written inline
                label, pick = m.groups()
            tag = f"<b>{e(label)}:</b> " if label and len(label) <= 40 and label.lower() != "rating" else ""
            bullets.append(f"• {tag}{e(pick)}")
    if bullets:
        out += ["", tr("<b>Key points</b>", "<b>Ý chính</b>"), *bullets]

    # Gaps: a labelled block's items, else any sentence that says something was not provided.
    gap_text = ""
    for label, ss in blocks:
        if GAPS_LABEL.match(label) and (items := fit([x if x[-1] in ".!?" else x + "." for x in ss], 260)):
            gap_text = (tr("Missing: ", "Thiếu: ") if label.lower().startswith(("missing", "thiếu"))
                        else tr("Gaps: ", "Thiếu dữ liệu: ")) + items
            break
    if not gap_text:  # only the clause that names the gap: "Why not Sell: price is up, and the feed is missing."
        clauses = [re.sub(r"^(?:and|but|so|while)\s+", "", c).rstrip(".")
                   for x in prose for c in re.split(r"[,;:]\s+(?![^(]*\))", x)]
        found = [c[0].upper() + c[1:] + "." for c in clauses if GAP_SENTENCE.search(c)]
        gap_text = tr("Gaps: ", "Thiếu dữ liệu: ") + fit(found, 254) if found else ""
    out += ["", f"⚠️ <i>{e(gap_text) or tr('Gaps: none stated in the decision.', 'Thiếu dữ liệu: quyết định không nêu.')}</i>"]
    out += ["", tr("📎 Full report attached.", "📎 Báo cáo đầy đủ đính kèm.")]
    return "\n".join(out)


def level(text):
    """'$84,093.42' / '84.5k' / '2500.0' -> float; None for anything else."""
    m = re.fullmatch(r"~?\$?(\d+(?:,\d{3})*(?:\.\d+)?)(k?)", text.strip())
    return m and float(m[1].replace(",", "")) * (1000 if m[2] else 1)


def chart(ticker, date, card):
    """PNG of the last 90 daily bars with SMA 50/200, Bollinger 20/2 and the card's own levels.

    Prices come from load_ohlcv, which the analysis has just cached, so this makes no extra
    request. Levels are read back from the rendered card, so the chart shows exactly the card's.
    """
    df = load_ohlcv(ticker, date)
    close = df["Close"]
    df = df.assign(sma50=close.rolling(50).mean(), sma200=close.rolling(200).mean(),
                   mid=close.rolling(20).mean(), sd=close.rolling(20).std())
    df = df.assign(up=df["mid"] + 2 * df["sd"], low=df["mid"] - 2 * df["sd"]).tail(90).reset_index(drop=True)
    last = float(df["Close"].iloc[-1])
    lines = [("up" if d in UP else "down", level(v), (22, 163, 74) if d in UP else (220, 38, 38))
             for d, v in re.findall(r"(?:close|đóng cửa) (above|below|trên|dưới) <b>([^<]+)</b>", card)]
    named = {"Cắt lỗ": "Stop", "Mục tiêu": "Target"}  # the chart's font has no Vietnamese accents
    lines += [(named.get(k, k), level(v), (100, 116, 139)) for k, v in re.findall(r"(Stop|Target|Cắt lỗ|Mục tiêu) <b>([^<]+)</b>", card)]
    lines = [x for x in lines if x[1] and last / 3 < x[1] < last * 3]  # a misread unit would flatten the chart

    W, H, L, R, T, B = 1000, 560, 10, 110, 40, 30
    lo = min([df["Low"].min(), df["low"].min()] + [v for _, v, _ in lines])
    hi = max([df["High"].max(), df["up"].max()] + [v for _, v, _ in lines])
    lo, hi = lo - (hi - lo) * 0.03, hi + (hi - lo) * 0.03
    def y(v):
        return T + (hi - v) / (hi - lo) * (H - T - B)
    step = (W - L - R) / len(df)
    def x(i):
        return L + (i + 0.5) * step

    img = Image.new("RGB", (W, H), "white")
    g = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=14)
    tick = 10 ** math.floor(math.log10((hi - lo) / 5))
    tick *= next(k for k in (1, 2, 5, 10) if (hi - lo) / (tick * k) <= 6)
    for i in range(math.ceil(lo / tick), math.floor(hi / tick) + 1):  # price gridlines
        g.line([(L, y(i * tick)), (W - R, y(i * tick))], fill=(241, 245, 249))
        g.text((L + 2, y(i * tick) - 16), f"{i * tick:,.{2 if tick < 1 else 0}f}", fill=(148, 163, 184), font=font)
    for i, r in df.iterrows():  # month gridlines
        if i and r["Date"].month != df["Date"][i - 1].month:
            g.line([(x(i), T), (x(i), H - B)], fill=(235, 235, 235))
            g.text((x(i) + 3, H - B + 6), r["Date"].strftime("%b"), fill=(120, 120, 120), font=font)
    for col, color in [("up", (148, 163, 184)), ("low", (148, 163, 184)), ("mid", (203, 213, 225)),
                       ("sma50", (234, 88, 12)), ("sma200", (124, 58, 237))]:
        pts = [(x(i), y(v)) for i, v in df[col].items() if v == v]
        if len(pts) > 1:
            g.line(pts, fill=color, width=2)
    for i, r in df.iterrows():
        color = (22, 163, 74) if r["Close"] >= r["Open"] else (220, 38, 38)
        g.line([(x(i), y(r["High"])), (x(i), y(r["Low"]))], fill=color)
        top, bot = sorted([y(r["Open"]), y(r["Close"])])
        g.rectangle([x(i) - step * 0.35, top, x(i) + step * 0.35, max(bot, top + 1)], fill=color)
    for tag, v, color in lines + [("", last, (15, 23, 42))]:
        for x0 in range(L, W - R, 12):  # dashed
            g.line([(x0, y(v)), (min(x0 + 6, W - R), y(v))], fill=color, width=2 if tag else 1)
        tx, ty = W - R + 4, y(v)
        if tag in ("up", "down"):  # the default font has no ▲▼, so draw them
            d = -5 if tag == "up" else 5
            g.polygon([(tx, ty - d), (tx + 10, ty - d), (tx + 5, ty + d)], fill=color)
            tag, tx = "", tx + 14
        g.text((tx, ty - 8), f"{tag} {v:,.{2 if v < 100 else 0}f}".strip(), fill=color, font=font)
    g.text((L, 8), f"{ticker} · daily to {df['Date'].iloc[-1]:%Y-%m-%d}", fill=(15, 23, 42), font=font)
    g.text((L + 330, 8), "SMA50", fill=(234, 88, 12), font=font)
    g.text((L + 390, 8), "SMA200", fill=(124, 58, 237), font=font)
    g.text((L + 460, 8), "Bollinger 20/2", fill=(100, 116, 139), font=font)
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()


def analyze(chat, ticker, date, lang="en"):
    kind = detect_asset_type(ticker)
    analysts = [a.value for a in filter_analysts_for_asset_type(list(AnalystType), kind)]
    config = DEFAULT_CONFIG.copy()
    if lang == "vi":  # every agent writes in Vietnamese; the labelled lines the card reads stay in English
        config["output_language"] = "Vietnamese"
    graph = TradingAgentsGraph(selected_analysts=analysts, config=config)
    state, _ = graph.propagate(ticker, date, asset_type=kind.value)
    card = summarize(ticker, date, state["final_trade_decision"], state.get("trader_investment_plan", ""), lang)
    try:
        say(chat, f"{card}\n\nℹ️ <i>{REFERENCE[lang]}</i>", parse_mode="HTML")
    except requests.HTTPError:  # Telegram rejected the markup: still deliver the decision, as plain text
        say(chat, f"{state['final_trade_decision']}\n\n{REFERENCE[lang]}")
    try:  # the chart is extra: if it fails, the card and the report still go out
        call("sendPhoto", data={"chat_id": chat}, files={"photo": ("chart.png", chart(ticker, date, card))})
    except Exception as exc:
        print(f"chart failed for {ticker} {date}: {exc!r}")
    # Telegram caps a message at 4096 chars, so the full report goes as a file.
    with tempfile.TemporaryDirectory() as tmp:
        report = graph.save_reports(state, ticker, tmp, html=False)
        with open(report, "rb") as f:
            call("sendDocument", data={"chat_id": chat, "caption": REFERENCE[lang]}, files={"document": (f"{ticker}-{date}.md", f)})


def handle(msg):
    chat = str(msg["chat"]["id"])
    if chat not in ALLOWED:
        return
    parts = (msg.get("text") or "").split()
    command = parts[0].split("@")[0] if parts else ""
    lang = trend.chat_lang(chat)

    def tr(en, vi):
        return trend._t(lang, en, vi)

    if command == "/lang":
        choice = parts[1].lower() if len(parts) > 1 else ""
        if choice not in trend.LANGS:
            return say(chat, tr("Language: English. Usage: /lang vi or /lang en",
                                "Ngôn ngữ: tiếng Việt. Cách dùng: /lang vi hoặc /lang en"))
        trend.set_chat_lang(chat, choice)
        return say(chat, trend._t(choice, "✅ Language: English. Replies, alerts and /analyze reports are now in English.",
                                  "✅ Ngôn ngữ: tiếng Việt. Câu trả lời, cảnh báo và báo cáo /analyze từ giờ bằng tiếng Việt."))
    if command in ("/status", "/health"):
        books = parts[1:2] or list(trend.BOOKS)  # no argument: every book the daily job runs
        return say_html(chat, "\n\n".join(trend.health_text(name, lang=lang) for name in books))
    if command in ("/trend", "/paper"):
        report = trend.status_text if command == "/trend" else trend.paper_text
        return say_html(chat, report(parts[1] if len(parts) > 1 else "BTCUSDT", lang=lang))
    if command != "/analyze":  # anything else, /status or /help included, gets the list of commands
        return say(chat, tr(COMMANDS, COMMANDS_VI))
    ticker = parts[1] if len(parts) > 1 else ""
    date = parts[2] if len(parts) > 2 else datetime.date.today().isoformat()
    if not TICKER.match(ticker):
        return say(chat, tr("Usage: /analyze TICKER [YYYY-MM-DD]", "Cách dùng: /analyze MÃ [YYYY-MM-DD]"))
    say(chat, tr(f"Analyzing {ticker} {date}, about 3 minutes...", f"Đang phân tích {ticker} {date}, khoảng 3 phút..."))
    try:
        analyze(chat, ticker, date, lang)
    except Exception as exc:  # report and keep serving
        say(chat, f"{tr('Failed', 'Lỗi')}: {exc}")


def main():
    assert ALLOWED, "set TELEGRAM_ALLOWED_CHAT_IDS"
    # Own thread: an analysis blocks this loop for minutes, which must not delay a trend alert.
    for job in (trend.watch, trend.deadman):  # the dead-man alert must not share a thread with the job it watches
        threading.Thread(target=job, args=(say_html, sorted(ALLOWED)), daemon=True).start()
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
