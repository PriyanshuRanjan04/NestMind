"""
Safety Guard — runs synchronously BEFORE any LLM call.

Logic specification implemented here:
  BLOCK only when a query contains BOTH:
    (a) A harmful keyword from a category, AND
    (b) Clear action intent (user is trying to DO the thing, not understand it)

  EDUCATIONAL PASS-THROUGH:
    Queries that ask "what is X", "explain X", "how does X work", "penalties for X",
    "how do regulators detect X", compliance questions, or historical examples are
    never blocked — regardless of whether they mention harmful keywords.

  ABSOLUTE BLOCKS:
    A small set of patterns so unambiguous (both keyword + intent in one phrase)
    that no educational framing can save them.

  PERFORMANCE:
    All patterns compiled at module load time. Target: < 5 ms for any input.
    Typical: < 1 ms.
"""
from __future__ import annotations

import re
import time

from src.schemas import SafetyFlag, SafetyGuardResult

# ---------------------------------------------------------------------------
# Educational escape signals — if any match, conditional blocks are lifted.
#
# Rule (from spec):
#   "what is X" / "explain X" / "how does X work" / "what are the penalties for X"
#   / "how do regulators detect X" / compliance obligations / historical examples
#   / "illegal" + "?" / "short selling" (always legal)
# ---------------------------------------------------------------------------

_EDUCATIONAL_SIGNALS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # interrogative framing
        r"\bwhat\s+is\b",
        r"\bwhat\s+are\b",
        r"\bwhat\s+does\b",
        r"\bwhat.{0,10}difference\b",
        r"\bhow\s+does\b",
        r"\bhow\s+do\b",
        r"\bhow\s+should\b",
        r"\bhow\s+would\b",
        r"\bwhy\s+is\b",
        r"\bwhy\s+are\b",
        r"\bwhen\s+is\b",
        r"\bcan\s+you\s+explain\b",
        r"\bexplain\b",
        r"\bdescribe\b",
        r"\bdefine\b",
        r"\bwhat.{0,20}means?\b",
        # regulatory / detection / penalty framing
        r"\bpenalt(y|ies)\b",
        r"\bregulat\w*\b",
        r"\bdetect\b",
        r"\binvestigat\w+\b",
        r"\bcatch\b",
        r"\bscreen\b",
        r"\bcompliance\b",
        r"\bobligations?\b",
        r"\bdisclose\b",
        r"\bdisclosure\b",
        r"\brequirements?\b",
        r"\brules?\b",
        # educational intent keywords
        r"\bhistorical\b",
        r"\bexample\b",
        r"\blegitimate\b",
        r"\bred\s+flag\b",
        r"\bfraud\s+indicator\b",
        r"\bAML\b",
        r"\bSEC\b",
        r"\bFCA\b",
        r"\bOFAC\b",
        r"\bCFTC\b",
        r"\bstages?\s+of\b",
        r"\bponzi\b",
        r"\bfront.?running\b",
        r"\blayering\s+and\s+integration\b",
        r"\bplacement.+layer\b",
        # "is X illegal?" pattern = educational (asking, not doing)
        r"\billegal\b",
        # short selling is always legal — any mention should pass
        r"\bshort\s+sell(ing)?\b",
        # factors/risks/how-to-think-about framing
        r"\bfactors?\s+should\b",
        r"\bconsider\b",
        r"\brisks?\s+of\b",
        r"\brelative\s+to\b",
        r"\ballocat\w+\s+to\b",
        r"\brisk.?adjusted\b",
        r"\bever\s+legitimate\b",
    ]
]

# ---------------------------------------------------------------------------
# Category definitions
#
# Structure per category:
#   (SafetyFlag, [absolute_pattern_strings], [conditional_pattern_strings])
#
# absolute  → block regardless of any educational framing
# conditional → block ONLY when no educational signal is present
#
# Keyword-AND-intent rule from spec:
#   harmful keyword must co-occur with action intent within the same query.
#   Patterns are written to enforce this co-occurrence.
# ---------------------------------------------------------------------------

_CATEGORIES: list[tuple[SafetyFlag, list[str], list[str]]] = [

    # =========================================================================
    # INSIDER TRADING
    # Block if: insider/non-public/confidential/merger keyword + buy/sell/trade
    # intent in the same query.
    # =========================================================================
    (
        SafetyFlag.insider_trading,
        # absolute — keyword + intent so tightly coupled that edu framing cannot save them
        [
            # "i work at [company] and know about X, when should i / buy / sell"
            r"i\s+work\s+at\s+\w+.{0,80}(buy|sell|trade|invest|load\s+up|when\s+should\s+i)",
            # tip from an insider + action
            r"(tip|told\s+me).{0,80}(before\s+the\s+(call|announcement|earnings|release)|load\s+up|buy|sell)",
            # explicit: "trade on this confidential / insider / non-public / merger"
            r"trade\s+on\s+(this\s+)?(confidential|insider|non.?public|merger|acqui)",
            r"help\s+me\s+trade\s+on\s+.{0,40}(confidential|merger|acquisition)",
            # confidential merger/deal/earnings news + buy/sell action
            r"confidential\s+(merger|acquisition|deal|news|information).{0,80}(buy|sell|trade|invest|load\s+up)",
            # "i need to know the earnings before [tomorrow's/the] announcement"
            r"i\s+need\s+to\s+know\s+the\s+(earnings|results|numbers|report)\s+before",
            # "know about acquisition" + when/buy/sell intent
            r"know\s+about\s+(an?\s+)?(acquisition|merger|deal|announcement).{0,60}(buy|sell|trade|when\s+should)",
        ],
        # conditional — only block when no educational signal present
        [
            # non-public/material non-public info + trade intent
            r"(non.?public|material\s+non.?public|unpublished|confidential)\s+(information|data|news|tip).{0,80}(trade|invest|buy|sell|when\s+should)",
            # unannounced event + trade action
            r"unannounced\s+(acquisition|merger|deal|earnings).{0,80}(buy|sell|trade|invest|when)",
            # "before the announcement" + trade action
            r"(trade|invest|buy|sell|load\s+up).{0,80}(before\s+the\s+(announcement|earnings\s+call|call|release))",
            # insider tip/info + trade action
            r"insider\s+(tip|info|information|knowledge|news).{0,80}(buy|sell|trade|invest|load\s+up|when\s+should)",
            # "earnings before" + action (e.g. "load up on earnings before the call")
            r"earnings\s+before.{0,40}(call|announcement|release).{0,40}(buy|sell|load|invest)",
        ],
    ),

    # =========================================================================
    # MARKET MANIPULATION
    # Block if: pump/dump/coordinate/wash-trade/spoofing keyword + action intent.
    # =========================================================================
    (
        SafetyFlag.market_manipulation,
        # absolute
        [
            # "pump up the price ... sell / dump"
            r"pump\s+up\s+the\s+price.{0,80}(sell|dump|before|exit)",
            # "how can i pump up the price of" + any stock
            r"how\s+can\s+i\s+pump\s+(up\s+)?the\s+price\s+of",
            # explicit coordinated buying scheme
            r"design\s+a\s+coordinated\s+(buying|selling|trading)\s+scheme",
            r"(coordinated|coordinate).{0,60}(buy|sell|scheme|buying|selling).{0,60}move\s+(this\s+)?stock",
            # explicit wash trading with intent to create volume / deceive
            r"help\s+me\s+wash\s+trade",
            r"wash\s+trad\w{0,10}(between\s+two\s+accounts?|to\s+create\s+(false\s+)?volume)",
            # "create volume" / "create false volume" with manipulation intent
            r"create\s+(false\s+|fake\s+)?volume\s+(in|for|on)\s+\w",
            # "move the stock" / "move this stock" with explicit action
            r"(help\s+me\s+|i\s+want\s+to\s+)?(pump|inflate|spike|manipulat\w*)\s+.{0,30}(stock|ticker|share|price)",
            # spoofing with action intent ("how can i" / "help me" + "spoof")
            r"(how\s+can\s+i|help\s+me|i\s+want\s+to)\s+.{0,20}spoof",
        ],
        # conditional
        [
            # pump + dump/sell sequence with intent framing
            r"(pump|inflate|push\s+up|move|spike).{0,80}(price|stock|ticker|share).{0,80}(sell|dump|exit|before)",
            # explicit manipulation actions
            r"(manipulat\w*|spoof|layer\s+orders?|front.?run).{0,80}(order|market|stock|trade|price)",
            # "coordinate" + "buy/sell" + price-moving intent
            r"coordinate.{0,60}(buy|sell|purchase).{0,60}(price|stock|move|spike)",
        ],
    ),

    # =========================================================================
    # MONEY LAUNDERING
    # Block if: launder/hide/obscure/structure + action intent.
    # =========================================================================
    (
        SafetyFlag.money_laundering,
        # absolute
        [
            # "move $Xk in cash through stock trading without reporting"
            r"move\s+\$?[\d,]+\s*(k|m|million|thousand)?\s*in\s+cash.{0,80}without\s+reporting",
            # structuring to avoid the 10k threshold
            r"structure\s+deposits.{0,60}avoid\s+the\s+(10k|10,000|\$10)",
            r"help\s+me\s+structure\s+deposits.{0,60}avoid",
            # "avoid the 10k reporting threshold" with structuring intent
            r"avoid\s+the\s+10.?000.{0,30}reporting",
            r"avoid\s+the\s+10k\s+reporting",
            # layer trades to obscure source
            r"layer\s+(my\s+)?trades?.{0,80}obscure\s+the\s+source",
            r"how\s+can\s+i\s+layer\s+(my\s+)?trades?\s+to\s+obscure",
            # hide trading profits from tax authorities
            r"help\s+me\s+hide\s+trading\s+profits?\s+from",
            r"hide\s+(my\s+)?trading\s+profits?\s+from\s+(the\s+)?(tax|HMRC|IRS)",
            # "move 500k in cash through stock trading without reporting"
            r"(cash|money|funds).{0,60}(stock\s+trading|trading|investments?).{0,60}without\s+report",
        ],
        # conditional
        [
            # move/transfer/hide/launder cash + illegal/without-reporting
            r"(move|transfer|hide|launder|obscure|disguise).{0,80}(cash|money|funds).{0,80}(without\s+report|illegal|tax.{0,20}author)",
            # avoid/evade AML / reporting thresholds
            r"(avoid|evade|bypass).{0,60}(reporting\s+threshold|AML|CTR|SAR|10k|10,000)",
            # source of funds concealment
            r"(hide|conceal|obscure).{0,60}(source\s+of\s+funds|origin\s+of\s+funds|where\s+the\s+money\s+comes?\s+from)",
            # structuring / smurfing
            r"\b(smurfi|smurf|structur\w+\s+cash|structuring\s+deposits)\b",
            # "avoid reporting" standalone with financial context
            r"avoid\s+reporting.{0,40}(cash|fund|deposit|threshold)",
        ],
    ),

    # =========================================================================
    # GUARANTEED RETURNS
    # Block if: guarantee/promise/100% certain/foolproof keyword + specific
    # return claim or demand.
    # =========================================================================
    (
        SafetyFlag.guaranteed_returns,
        # absolute
        [
            # "guarantee me 30% returns"
            r"guarantee\s+(me|my|a|us|the)\s+.{0,30}(return|profit|gain|\d+\s*%)",
            r"guarantee\s+me\s+\d+%",
            # "promise me my money will double / triple"
            r"promise\s+me\s+(my\s+money|it\s+will|you\s+will).{0,80}(double|triple|guarantee|certain)",
            # "100% certain / foolproof / guaranteed [way to make / to go up]"
            r"(100\s*%\s*certain|foolproof|guaranteed)\s+(way\s+to\s+make|to\s+go\s+up|return|profit)",
            # "tell me a stock that's 100% certain to go up"
            r"tell\s+me\s+a\s+stock\s+that.{0,40}(100\s*%|certain|guaranteed|definitely).{0,20}(go\s+up|increase|rise)",
            # "what's a foolproof way to make X% in a month"
            r"what.{0,20}foolproof\s+way\s+to\s+make\s+\d+",
            # "will definitely double in" / "certain to go up"
            r"will\s+definitely\s+(double|triple|go\s+up|rise|increase)\s+in",
            r"certain\s+to\s+go\s+up",
            # "double in a year" as a demand/promise
            r"(double|triple)\s+in\s+(a\s+)?(year|month|week)",
        ],
        # conditional
        [
            # guarantee/promise/risk-free + return (demand or claim)
            r"(guarantee|promised?|certain(ly)?|risk.?free).{0,80}(return|profit|gain|make\s+money)",
            # "always / never loses / impossible to lose" profit claim
            r"(always|never\s+loses?|impossible\s+to\s+lose).{0,80}(profit|return|gain|make\s+money)",
            # "100% certain" paired with investment context
            r"100\s*%\s+certain.{0,40}(invest|stock|crypto|portfolio|fund)",
        ],
    ),

    # =========================================================================
    # RECKLESS ADVICE
    # Four patterns from spec:
    #   A. retirement savings + all-in + high-risk asset
    #   B. margin/leverage for single stock
    #   C. emergency fund + high-risk
    #   D. mortgage for stocks
    # =========================================================================
    (
        SafetyFlag.reckless_advice,
        # absolute
        [
            # Pattern A: all retirement + crypto/options/margin
            r"(i.?m|i\s+am)\s+\d{2}.{0,80}(all\s+(my|the)\s+retirement|everything\s+in\s+crypto|all.{0,30}savings\s+in\s+crypto)",
            r"\d{2}\s*(years?\s*old|yr\s*old).{0,80}(put\s+all|everything|retirement\s+savings).{0,80}crypto",
            # "put all my retirement savings in crypto / options / single stock"
            r"(all\s+my|entire|all\s+the)\s+retirement\s+savings?.{0,60}(crypto|options?|single\s+stock|margin)",
            # Pattern B: "tell me to take a margin loan to buy more [X]"
            r"tell\s+me\s+to\s+take\s+a\s+margin\s+loan.{0,80}(buy\s+more|invest|load\s+up)",
            r"margin\s+loan\s+to\s+buy\s+(more\s+)?\w",
            r"leverage\s+to\s+buy\s+(more\s+)?\w+.{0,40}(single\s+stock|one\s+stock|all.?in)",
            # Pattern C: emergency fund + options/crypto/single stock
            r"(put|place|move)\s+(my\s+)?(entire|all(\s+my)?)\s+emergency\s+fund\s+into\s+(options?|crypto|stocks?|single)",
            r"should\s+i\s+put\s+my\s+entire\s+emergency\s+fund\s+into",
            r"emergency\s+fund.{0,40}(options?|crypto|single\s+stock|penny)",
            # Pattern D: mortgage house for stocks
            r"tell\s+me\s+which\s+stock\s+to\s+(mortgage\s+my\s+house|use\s+my\s+house)",
            r"mortgage\s+my\s+house.{0,60}(stock|invest|buy|trade)",
            r"remortgage.{0,60}(stock|invest|buy|trade)",
        ],
        # conditional
        [
            # mortgage/borrow-against-house + invest
            r"(mortgage\s+(my|the)\s+house|borrow\s+against\s+(my|the)\s+house|home\s+equity).{0,80}(stock|invest|buy|trade)",
            # all/entire + emergency/retirement/life savings + high-risk
            r"(all|entire|100\s*%).{0,80}(emergency\s+fund|retirement\s+savings?|life\s+savings?).{0,80}(option|crypto|speculative|single\s+stock|penny|margin)",
            # margin loan/leverage + double down / all-in / single stock
            r"(margin\s+loan|leverage).{0,80}(buy\s+more|double\s+down|go\s+all\s+in|all.?in).{0,80}(single\s+stock|crypto|option)",
            # bet/gamble/yolo + life savings / retirement / emergency
            r"(bet|gamble|yolo).{0,80}(life\s+savings?|retirement|emergency\s+fund)",
        ],
    ),

    # =========================================================================
    # SANCTIONS EVASION
    # Block if: sanction/OFAC/bypass/evade/shell company + action intent to evade.
    # =========================================================================
    (
        SafetyFlag.sanctions_evasion,
        # absolute
        [
            # "route trades through shell company to bypass OFAC sanctions"
            r"route\s+trades?.{0,80}(shell\s+company|nominee|bypass|circumvent|evade).{0,80}(OFAC|sanctions?|restrict)",
            r"help\s+me\s+route\s+trades?.{0,60}(shell|bypass|evade|sanction)",
            # "invest in a sanctioned [country/company] without it being traced"
            r"invest\s+in\s+(a\s+)?(sanctioned|banned|restricted)\s+\w+.{0,80}without\s+(it\s+being\s+traced|detection|getting\s+caught|trace)",
            r"how\s+do\s+i\s+invest\s+in\s+a\s+sanctioned.{0,60}without",
            # "bypass OFAC sanctions"
            r"bypass\s+OFAC\s+sanctions?",
            # "evade / circumvent / avoid + OFAC/sanctions + invest/trade/route"
            r"(evade|circumvent|avoid).{0,80}(OFAC|sanctions?).{0,80}(invest|trade|buy|route)",
            # "without it being traced" / "without being traced" (evasion signal)
            r"(invest|trade|buy|route|transfer).{0,60}without\s+(it\s+being\s+traced|being\s+traced|detection)",
        ],
        # conditional
        [
            # sanctioned country/company + invest/get-around
            r"(sanctioned|restricted\s+country|banned\s+firm).{0,80}(invest|buy|trade|get\s+around|without)",
            # shell company/nominee + trade/invest/sanction
            r"(shell\s+company|nominee|front\s+company).{0,80}(trade|invest|sanction|OFAC|restriction)",
        ],
    ),

    # =========================================================================
    # FRAUD
    # Block if: fake contract / fake statement / forged / fabricate /
    # claim false losses / false losses keywords present.
    # (spec: no separate "AND intent" — the keywords themselves are the intent)
    # =========================================================================
    (
        SafetyFlag.fraud,
        # absolute
        [
            # "draft a fake contract note to claim losses"
            r"draft\s+(a\s+)?(fake|false|forged|fraudulent).{0,80}(contract|note|document|statement|record)",
            # fake contract note / statement used to claim/submit
            r"fake\s+(contract\s+note|statement|document|invoice|record).{0,80}(claim|loss|profit|tax|insurance|submit)",
            # forge a document/statement
            r"forge\s+(a\s+)?(document|statement|note|record|contract)",
            # create fake/false trading records
            r"create\s+(fake|false|fraudulent).{0,80}(trading\s+record|statement|invoice|document)",
            # fabricate financial records
            r"\bfabricat\w+.{0,60}(loss(es)?|record|statement|document|contract|note)\b",
            # claim false losses
            r"claim\s+false\s+losses?",
            # false losses (standalone — strong fraud signal regardless of framing)
            r"\bfalse\s+losses?\b",
        ],
        # conditional
        [
            # fake/false/forged document + submit/file/use
            r"(fake|false|forged|fraudulent).{0,60}(document|contract|statement|note|invoice).{0,60}(claim|submit|file|send|use)",
        ],
    ),
]

# ---------------------------------------------------------------------------
# Pre-compile all patterns at module import time (< 5 ms budget).
# ---------------------------------------------------------------------------
_COMPILED: list[
    tuple[SafetyFlag, list[re.Pattern[str]], list[re.Pattern[str]]]
] = [
    (
        flag,
        [re.compile(p, re.IGNORECASE | re.DOTALL) for p in abs_pats],
        [re.compile(p, re.IGNORECASE | re.DOTALL) for p in cond_pats],
    )
    for flag, abs_pats, cond_pats in _CATEGORIES
]

# ---------------------------------------------------------------------------
# Per-category distinct refusal messages — exact wording from spec.
# ---------------------------------------------------------------------------

_REFUSAL_MESSAGES: dict[SafetyFlag, str] = {
    SafetyFlag.insider_trading: (
        "I'm not able to assist with trading on non-public or confidential "
        "information. This falls under insider trading regulations and carries "
        "serious legal consequences including criminal prosecution. If you have "
        "questions about insider trading regulations, I'm happy to explain them."
    ),
    SafetyFlag.market_manipulation: (
        "I can't help with coordinating trades to artificially move prices or "
        "create false volume. This constitutes market manipulation, which is a "
        "serious financial crime. I'm happy to explain how regulators detect and "
        "prosecute these schemes if that's helpful."
    ),
    SafetyFlag.money_laundering: (
        "I'm unable to assist with concealing the origin of funds or structuring "
        "transactions to avoid regulatory reporting. These activities constitute "
        "money laundering or structuring offences with severe legal penalties. "
        "I can explain AML regulations and compliance requirements if you're interested."
    ),
    SafetyFlag.guaranteed_returns: (
        "No investment can guarantee returns, and making or demanding such claims "
        "is both misleading and potentially fraudulent. All investments carry risk. "
        "I can help you understand realistic historical return ranges and how to "
        "evaluate risk-adjusted performance."
    ),
    SafetyFlag.reckless_advice: (
        "This approach carries risks that could seriously damage your financial "
        "security. Concentrating funds critical to your livelihood — retirement "
        "savings, emergency reserves, or home equity — in high-risk or leveraged "
        "positions is something I'm not able to recommend. I'd be glad to discuss "
        "risk-appropriate allocation strategies instead."
    ),
    SafetyFlag.sanctions_evasion: (
        "I'm not able to assist with circumventing sanctions or structuring trades "
        "to evade regulatory oversight. Sanctions violations carry severe criminal "
        "and civil penalties. I can explain how sanctions compliance works for "
        "brokers and investors if that's useful."
    ),
    SafetyFlag.fraud: (
        "I'm not able to help create fraudulent financial documents or false "
        "records. This constitutes financial fraud and is a criminal offence. "
        "Please reach out if you have a legitimate financial question I can help with."
    ),
}


def check(query: str) -> SafetyGuardResult:
    """
    Run the safety guard against *query*.

    Returns a SafetyGuardResult. If blocked=True, the caller should
    emit the message to the user and halt the pipeline immediately.

    Decision algorithm:
      1. Detect educational signals (one pass over _EDUCATIONAL_SIGNALS).
      2. For each category:
         a. Check absolute patterns — block regardless of educational framing.
         b. If no educational signal: check conditional patterns — block on match.
      3. If no pattern fires → return blocked=False.

    Timing: compiled patterns + single pass per pattern = < 1 ms in practice.
    """
    _t0 = time.perf_counter()

    # One-pass educational signal detection
    has_educational = any(sig.search(query) for sig in _EDUCATIONAL_SIGNALS)

    for flag, abs_patterns, cond_patterns in _COMPILED:
        # Absolute blocks — educational escape cannot lift these
        for pat in abs_patterns:
            if pat.search(query):
                _elapsed_ms = (time.perf_counter() - _t0) * 1000
                assert _elapsed_ms < 50, f"Safety guard too slow: {_elapsed_ms:.1f}ms"
                return SafetyGuardResult(
                    blocked=True,
                    category=flag,
                    message=_REFUSAL_MESSAGES[flag],
                )

        # Conditional blocks — lifted when educational framing is present
        if not has_educational:
            for pat in cond_patterns:
                if pat.search(query):
                    _elapsed_ms = (time.perf_counter() - _t0) * 1000
                    assert _elapsed_ms < 50, f"Safety guard too slow: {_elapsed_ms:.1f}ms"
                    return SafetyGuardResult(
                        blocked=True,
                        category=flag,
                        message=_REFUSAL_MESSAGES[flag],
                    )

    return SafetyGuardResult(blocked=False)
