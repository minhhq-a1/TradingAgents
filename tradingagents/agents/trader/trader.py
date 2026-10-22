"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import pandas as pd
from langchain_core.messages import AIMessage

from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
)
from tradingagents.agents.schemas import TraderAction, TraderProposal, render_trader_proposal
from tradingagents.agents.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.errors import VendorError
from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv

STOP_ATR_MULTIPLE = 1.5
ATR_WINDOW = 14


def _atr(symbol: str, trade_date: str) -> float:
    """Average true range over the last ``ATR_WINDOW`` bars up to ``trade_date``."""
    bars = load_ohlcv(symbol, trade_date)
    close_before = bars["Close"].shift()
    true_range = pd.concat(
        [bars["High"] - bars["Low"], (bars["High"] - close_before).abs(),
         (bars["Low"] - close_before).abs()], axis=1,
    ).max(axis=1)
    return float(true_range.tail(ATR_WINDOW).mean())


def check_levels(proposal: TraderProposal, symbol: str, trade_date: str) -> TraderProposal:
    """A proposal whose stop sits on the losing side of its entry.

    A Buy loses below its entry and a Sell above it, so a Buy stop at or above
    entry (or a Sell stop at or below) is a contradictory order that would fill
    at once. The model writes these (a trim of a long reads as a Sell with the
    long's stop), so the code, not the model, fixes the level: the stop is
    recomputed as entry -/+ ``STOP_ATR_MULTIPLE`` x ATR, or dropped when no
    price history is available, and the correction is stated in the reasoning.
    """
    entry, stop, action = proposal.entry_price, proposal.stop_loss, proposal.action
    if entry is None or stop is None or action is TraderAction.HOLD:
        return proposal
    buying = action is TraderAction.BUY
    if stop < entry if buying else stop > entry:
        return proposal

    side = "below" if buying else "above"
    try:
        atr = _atr(symbol, trade_date)
    except VendorError:
        atr = None
    if atr is None or pd.isna(atr):
        fixed, how = None, "dropped it, as no price history was available to recompute it"
    else:
        distance = STOP_ATR_MULTIPLE * atr
        fixed = round(entry - distance if buying else entry + distance, 6)
        how = f"replaced it with {fixed} ({STOP_ATR_MULTIPLE} x ATR({ATR_WINDOW}) = {distance:.6g} {side} entry)"
    note = (
        f" [Level check: the proposed stop-loss {stop} is not {side} the {entry} entry "
        f"of a {action.value}, so the order contradicted itself; the system {how}.]"
    )
    return proposal.model_copy(update={"stop_loss": fixed, "reasoning": proposal.reasoning + note})


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state):
        company_name = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        investment_plan = state["investment_plan"]
        # The research plan digests the debate but loses exact price structure;
        # give the Trader the technical market report so entry/stop levels are
        # grounded in real ATR / support-resistance / current price (#1167). The
        # report is empty when the user did not select the market analyst, so
        # only offer it (and the grounding instruction) when it has content.
        market_report = (state["market_report"] or "").strip()
        portfolio_context = get_portfolio_context_from_state(state)

        if market_report:
            grounding = (
                "Ground concrete price levels (entry, stop-loss, position sizing) in the technical "
                "market report's price structure -- current price, support/resistance, ATR, and "
                "volatility -- and use the research plan for direction and strategy. "
            )
            report_section = f"Technical Market Report:\n{market_report}\n\n"
        else:
            grounding = ""
            report_section = ""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    + grounding
                    # Entry/stop are numeric price fields. Asking for concrete
                    # levels invites a percentage ("15%"), which is not a price
                    # and fails the structured parse (#1288).
                    + "State entry price and stop-loss as absolute price levels in the "
                    "instrument's quote currency (for example 189.5), never a percentage "
                    "or a range; convert a percentage distance to the price level it "
                    "implies, or omit the field if you cannot state a number. "
                    "A Buy's stop-loss is below its entry price and a Sell's is above it. "
                    + NO_EXTERNAL_TOOLS
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here is the research team's investment plan for {company_name}. "
                    f"{instrument_context}\n\n"
                    f"{report_section}"
                    f"{portfolio_context}\n\n"
                    f"Proposed Investment Plan:\n{investment_plan}\n\n"
                    "Make an informed, strategic trading decision.\n\n"
                    "## Output\n\n"
                    "Write these sections, in this order, starting with the action "
                    "on its own line:\n\n"
                    "- **Action**: exactly one of Buy / Hold / Sell. A research "
                    "recommendation of Overweight is a Buy and Underweight is a Sell, "
                    "sized by how strong the case is; conflict alone is not a Hold.\n"
                    "- **Reasoning**: why, against the plan and the price structure\n"
                    "- **Entry Price**, **Stop Loss**, **Position Sizing**: when you can state them"
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            # shortcut: a free-text fallback reply is not parsed, so its levels go unchecked.
            lambda proposal: render_trader_proposal(
                check_levels(proposal, company_name, state["trade_date"])
            ),
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
        }

    return trader_node
