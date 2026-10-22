"""Telegram front-end: `/analyze BTCUSDT [YYYY-MM-DD]` -> decision + full report.

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

import requests
from PIL import Image, ImageDraw, ImageFont

from cli.models import AnalystType
from cli.prompts import detect_asset_type, filter_analysts_for_asset_type
from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv
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
NUM = r"~?\$?\d+(?:,\d{3})*(?:\.\d+)?k?"
TRIGGER = re.compile(rf"\bcloses?\b(?: a day| back| decisively)?\s+(above|below)\s+(?:about |roughly |~)?({NUM})(?![\d.,]*%)", re.I)  # a % is a yield, not a price
GAPS_LABEL = re.compile(r"data (?:conflict and )?gaps?|gaps?|missing\b", re.I)
GAP_SENTENCE = re.compile(
    r"^(?:missing|data gaps?|gaps?)\b|^no\b[^.]{0,80}\b(?:data|figures?|feeds?|series|inputs?|information)\b"
    r"|\b(?:was|were) not provided\b|\bunavailable\b|\b(?:data|key|feeds?|series|figures?)\b[^.]{0,40}\bmissing\b"
    r"|\b(?:is|are)(?: also)? missing\b|^no\b[^.]{0,80}\b(?:news|fundamentals)\b"
    r"|\bholdings are unknown\b", re.I)
CONCLUSION = re.compile(r"\b(?:therefore|stands|intact|weak|not supported|too even|favou?rs?|wins?|decisive|balanced)\b", re.I)


INLINE_LABEL = re.compile(r"([A-Z][^:.\d]{2,40}):\s+(\S.*)", re.S)


def sentences(text):
    """Whole sentences, list items and markdown stripped; the card never cuts one."""
    out = []
    for line in text.replace("**", "").splitlines():
        line = re.sub(r"^\s*(?:[-•]|\d+[.)])\s+", "", line).strip()
        out += re.split(r"(?<=[.!?])\s+(?=[A-Z$\d(\"'])", line) if line else []
    return out


def fit(parts, limit, sep=" "):
    """Join whole parts in order while they fit; a part that does not fit is skipped, not cut."""
    kept = []
    for part in parts:
        if len(sep.join(kept + [part])) <= limit:
            kept.append(part)
    return sep.join(kept)


def summarize(ticker, date, decision, trade_plan=""):
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
        return value if value and value.strip().lower() != "not provided" else None

    rating, horizon = f["Rating"], given(f.get("Time Horizon"))
    if horizon:  # "2-6 weeks, reviewed on ..." -> the period itself
        horizon = horizon.split(",")[0]
    out = [f"{EMOJI.get(rating, '⚪')} <b>{e(ticker)}</b> · {e(date)}",
           f"<b>{e(rating)}</b>" + (f" · horizon {e(horizon)}" if horizon else "")]

    t = dict(re.findall(r"\*\*(Entry Price|Stop Loss|Position Sizing)\*\*: (.+)", trade_plan))
    levels = {"Entry": t.get("Entry Price"), "Stop": t.get("Stop Loss"), "Target": f.get("Price Target"),
              "Size": t.get("Position Sizing")}
    # A long sizing paragraph belongs in the report; the card shows only a short one.
    levels = [f"{k} <b>{e(v.strip())}</b>" for k, v in levels.items() if given(v) and len(v) <= 40]
    if levels:
        out += ["", "📍 " + " · ".join(levels)]

    summary = sentences(f.get("Executive Summary", ""))
    # shortcut: a lone sentence over the limit is cut at a word (rare); fix if the PM starts writing run-ons
    out += ["", e(fit(summary, 600) or summary[0][:600].rsplit(" ", 1)[0] + "…" if summary else "")]

    # The thesis is bold-led blocks ("**Trend.** text", "**Evidence:**\n- item") or plain paragraphs
    # ("For the bull: text"). A block is (label, sentences).
    thesis = f.get("Investment Thesis", "")
    raw = re.findall(r"\*\*([^*\n]+?)[.:]?\*\*[.:]?\s*(.*?)(?=\n\*\*|\Z)", thesis, re.DOTALL)
    if not raw:
        for para in re.split(r"\n\s*\n", thesis):
            m = re.match(rf"((?:{GAPS_LABEL.pattern})[^.:\n]*|[^.:\n]{{3,60}})\s*[.:]\s*(.*)", para.strip(), re.DOTALL | re.I)
            ok = m and (GAPS_LABEL.match(m.group(1)) or para.strip()[len(m.group(1))] == ":")
            raw.append(m.groups() if ok else ("", para.strip()))
    blocks = [(label, sentences(body)) for label, body in raw]
    prose = [x for label, ss in blocks if not GAPS_LABEL.match(label) for x in ss]
    while sum(not GAPS_LABEL.match(label) for label, _ in blocks) < 3:  # few blocks: halve the longest
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
                triggers.setdefault((direction.lower(), number.lstrip("~")), None)
    ups = [f"close above <b>{e(n)}</b>" for d, n in triggers if d == "above"]
    downs = [f"close below <b>{e(n)}</b>" for d, n in triggers if d == "below"]
    if ups or downs:
        out += ["", "🎯 " + " · ".join([f"▲ {x}" for x in ups[:2]] + [f"▼ {x}" for x in downs[:2]])]

    # Up to three conclusions: per block, a whole sentence that is short enough and not already a trigger,
    # a gap or the summary; one that states a conclusion is preferred.
    bullets = []
    for label, ss in blocks:
        if GAPS_LABEL.match(label) or len(bullets) == 3:
            continue
        usable = [x for x in ss if 20 <= len(x) <= 170 and x[-1] != ":" and not TRIGGER.search(x)
                  and not GAP_SENTENCE.search(x) and x not in summary]  # a trailing ":" is a lead-in
        if usable:
            label = label.split(",")[0].strip()  # "Why not Sell, and why not 75%" -> "Why not Sell"
            pick = next((x for x in usable if CONCLUSION.search(x)), usable[0])
            if len(pick) < 40:  # a bare "The trend is intact." says little: add the sentence after it
                nxt = usable[usable.index(pick) + 1:][:1] or [x for x in usable if x != pick][:1]
                pick = fit([pick] + nxt, 170)
            if not label and (m := INLINE_LABEL.match(pick)):  # "Why not Sell: text" written inline
                label, pick = m.groups()
            tag = f"<b>{e(label)}:</b> " if label and len(label) <= 40 and label.lower() != "rating" else ""
            bullets.append(f"• {tag}{e(pick)}")
    if bullets:
        out += [""] + bullets

    # Gaps: a labelled block's items, else any sentence that says something was not provided.
    gap_text = ""
    for label, ss in blocks:
        if GAPS_LABEL.match(label) and (items := fit([x if x[-1] in ".!?" else x + "." for x in ss], 260)):
            gap_text = ("Missing: " if label.lower().startswith("missing") else "Gaps: ") + items
            break
    if not gap_text:  # only the clause that names the gap: "Why not Sell: price is up, and the feed is missing."
        clauses = [re.sub(r"^(?:and|but|so|while)\s+", "", c).rstrip(".")
                   for x in prose for c in re.split(r"[,;:]\s+(?![^(]*\))", x)]
        found = [c[0].upper() + c[1:] + "." for c in clauses if GAP_SENTENCE.search(c)]
        gap_text = "Gaps: " + fit(found, 254) if found else ""
    out += ["", f"⚠️ <i>{e(gap_text) or 'Gaps: none stated in the decision.'}</i>"]
    out += ["", "📎 Full report attached."]
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
    lines = [("up" if d == "above" else "down", level(v), (22, 163, 74) if d == "above" else (220, 38, 38))
             for d, v in re.findall(r"close (above|below) <b>([^<]+)</b>", card)]
    lines += [(k, level(v), (100, 116, 139)) for k, v in re.findall(r"(Stop|Target) <b>([^<]+)</b>", card)]
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


def analyze(chat, ticker, date):
    kind = detect_asset_type(ticker)
    analysts = [a.value for a in filter_analysts_for_asset_type(list(AnalystType), kind)]
    graph = TradingAgentsGraph(selected_analysts=analysts, config=DEFAULT_CONFIG.copy())
    state, _ = graph.propagate(ticker, date, asset_type=kind.value)
    card = summarize(ticker, date, state["final_trade_decision"], state.get("trader_investment_plan", ""))
    try:
        say(chat, card, parse_mode="HTML")
    except requests.HTTPError:  # Telegram rejected the markup: still deliver the decision, as plain text
        say(chat, state["final_trade_decision"])
    try:  # the chart is extra: if it fails, the card and the report still go out
        call("sendPhoto", data={"chat_id": chat}, files={"photo": ("chart.png", chart(ticker, date, card))})
    except Exception as exc:
        print(f"chart failed for {ticker} {date}: {exc!r}")
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
