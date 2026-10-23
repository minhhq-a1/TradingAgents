"""End-to-end drill of the alert path on a test chat: entry, one failed send, restart, exit.

    python -m tradingagents.trend_drill --chat TEST_CHAT_ID [--chat SECOND_TEST_CHAT_ID]

It sends through the bot's real ``say`` (token from ``TELEGRAM_BOT_TOKEN``) and runs the real
``trend.check`` on synthetic prices in a throw-away state directory, so the live paper book is
untouched and the chat gets "🧪 DRILL" messages, never a real alert. The first send to the last chat
at the entry is made to fail, as a network error would; the bot is then "restarted" (state read
back from disk only) and the exit signal comes the next day. It checks that the messages reach each
chat in order, that only the failed chat is retried, and that the state survives.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from unittest import mock

import numpy as np
import pandas as pd

from tradingagents import trend

PREFIX = "🧪 DRILL (synthetic prices, not a signal)\n"


def synthetic() -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """A fall, one jump above the SMA200 (entry) and a crash back under it the next day (exit)."""
    close = np.r_[200 - 0.4 * np.arange(230), 300.0, 60.0]
    end = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None) - timedelta(days=1)
    frame = pd.DataFrame({"Open": np.r_[close[0], close[:-1]], "Close": close},
                         index=pd.date_range(end=end, periods=len(close), freq="D"))
    state = trend.states(frame["Close"])
    flips = state.index[state.ne(state.shift())][1:]
    assert [state[f] for f in flips] == [trend.HOLD, trend.CASH], "the drill prices must make one entry and one exit"
    assert flips[1] - flips[0] == timedelta(days=1)
    return frame, flips[0], flips[1]


def run(say, chats: list[str], fail_chat: str | None = None) -> list[tuple[str, bool, str]]:
    """Run the drill with ``say(chat, text)``; the first send to ``fail_chat`` (default: the last chat) raises."""
    fail_chat = fail_chat or chats[-1]
    if fail_chat not in chats:
        raise ValueError("fail_chat must be one of the chats")
    frame, entry_day, exit_day = synthetic()
    received: list[tuple[str, str]] = []
    failed_once = []

    def send(chat, text):
        if chat == fail_chat and not failed_once:
            failed_once.append(chat)
            raise OSError("injected network failure")
        say(chat, PREFIX + text)
        received.append((chat, text))

    def check(day, expect_fail=False):
        with mock.patch.object(trend, "bars", return_value=(frame[:day], "drill")), \
                mock.patch.object(trend, "quote", return_value={"price": float(frame.loc[day, "Close"]), "open": float(frame.loc[day, "Open"]),
                                                                  "at": datetime.now(UTC).isoformat(timespec="seconds")}):
            try:
                trend.check("BTCUSDT", send, chats, datetime.combine(day.date() + timedelta(days=1), datetime.min.time(), UTC) + timedelta(minutes=5))
            except trend.SendFailed:
                if not expect_fail:
                    raise
                return True
        return False

    results: list[tuple[str, bool, str]] = []

    def verify(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    old_dir = trend.DIR
    with tempfile.TemporaryDirectory() as tmp:
        trend.DIR = tmp
        try:
            check(entry_day - timedelta(days=1))  # baseline
            verify("entry: the failing chat raises SendFailed, the others are served",
                   check(entry_day, expect_fail=True) and sorted(c for c, _ in received) == sorted(set(chats) - {fail_chat}),
                   f"received by {sorted(c for c, _ in received)}")
            with open(os.path.join(tmp, "state.json"), encoding="utf-8") as f:  # the restart: only the file survives
                on_disk = json.load(f)["BTCUSDT"]
            verify("restart: the undelivered entry alert is still queued for exactly the failed chat",
                   [p["sent_to"] for p in on_disk["pending"]] == [sorted(set(chats) - {fail_chat})] if len(chats) > 1 else len(on_disk["pending"]) == 1)
            check(exit_day)  # the next day reverses; the failed chat is reachable again
            texts = {c: [t for ch, t in received if ch == c] for c in chats}
            verify("order: every chat got the entry (UP) before the exit (DOWN)",
                   all(len(v) == 2 and "UP" in v[0] and "DOWN" in v[1] for v in texts.values()), str({c: len(v) for c, v in texts.items()}))
            verify("retry: only the failed chat was retried, nobody got a message twice",
                   len(received) == 2 * len(chats) and all(len(set(v)) == 2 for v in texts.values()))
            entry = trend._load()["BTCUSDT"]
            verify("state: both signals are in the ledger, sent to every chat, and the queue is empty",
                   [r["state"] for r in entry["ledger"]] == [trend.HOLD, trend.CASH] and entry["pending"] == []
                   and all(sorted(r["sent"]) == sorted(chats) for r in entry["ledger"]))
            verify("price: the price at send was recorded next to the open", all(r.get("quote") for r in entry["ledger"]))
            check(exit_day)
            verify("a repeated check sends nothing more", len(received) == 2 * len(chats))
        finally:
            trend.DIR = old_dir
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chat", action="append", required=True, help="test chat id; repeat for a second chat")
    args = parser.parse_args()
    import telegram_bot  # next to the bot script, which holds the real send path

    results = run(telegram_bot.say, args.chat)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}{f' ({detail})' if detail and not ok else ''}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
