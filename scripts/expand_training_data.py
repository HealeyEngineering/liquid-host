"""Expand and fix training data for Aiera MCP tool-calling fine-tuning.

Phase 1: Fix citations on existing examples that lack them (re-run Claude rewrite)
Phase 2: Generate ~210 new diverse training examples using Claude API
Phase 3: Validate all output and save

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...

    # Phase 1: Fix citations on existing examples
    python scripts/expand_training_data.py --phase fix --input data/training/aiera_tools_v6.jsonl --hf-token <token>

    # Phase 2: Generate new examples
    python scripts/expand_training_data.py --phase generate --input data/training/aiera_tools_v6.jsonl --hf-token <token>

    # Phase 3: Combine, validate, and push
    python scripts/expand_training_data.py --phase finalize --hf-token <token>

    # All phases
    python scripts/expand_training_data.py --phase all --input data/training/aiera_tools_v6.jsonl --hf-token <token>
"""

import argparse
import json
import logging
import os
import re
import time
from copy import deepcopy
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("expand_training")

HF_REPO = "bryanhealey/my-aiera-finetune-v7-data"
CHECKPOINT_DIR = Path("/tmp/training_expansion")
FIXED_FILE = CHECKPOINT_DIR / "fixed_examples.jsonl"
GENERATED_FILE = CHECKPOINT_DIR / "generated_examples.jsonl"
FINAL_FILE = Path("data/training/aiera_tools_v7.jsonl")

# ── Required tool parameters ──────────────────────────────────────

TOOLS_WITH_PAGE_SIZE = {
    "find_events", "find_filings", "find_equities", "find_company_docs",
    "find_research", "find_conferences", "find_third_bridge_events",
    "get_upcoming_events", "get_index_constituents", "get_watchlist_constituents",
}

TOOLS_WITH_INSTRUCTIONS_PARAMS = {
    "find_events", "find_filings", "find_equities", "find_company_docs",
    "find_research", "find_conferences", "find_third_bridge_events",
    "get_event", "get_filing", "get_company_doc", "get_third_bridge_event",
    "get_research", "get_upcoming_events", "get_equity_summaries",
    "get_financials", "get_ratios", "get_kpis_and_segments",
    "get_index_constituents", "get_watchlist_constituents",
    "get_available_indexes", "get_available_watchlists",
    "get_sectors_and_subsectors", "search_transcripts", "search_filings",
    "search_research", "search_company_docs", "search_thirdbridge",
    "get_company_doc_categories", "get_company_doc_keywords",
    "get_research_providers", "get_research_authors",
    "get_research_asset_classes", "get_research_asset_types",
    "get_research_subjects", "get_research_product_focuses",
    "get_research_discipline_types", "get_research_region_types",
    "get_research_country_codes", "trusted_web_search",
}


def fix_tool_call_params(content: str) -> str:
    """Add required parameters to tool call JSON."""
    def replace_tool_call(match):
        json_str = match.group(1)
        try:
            call = json.loads(json_str)
        except json.JSONDecodeError:
            return match.group(0)
        name = call.get("name", "")
        args = call.get("arguments", {})
        if name in TOOLS_WITH_PAGE_SIZE and "page_size" not in args:
            args["page_size"] = 100
        if name in TOOLS_WITH_INSTRUCTIONS_PARAMS:
            args.setdefault("exclude_instructions", True)
            args.setdefault("self_identification", "aierachat")
        call["arguments"] = args
        return f"<tool_call>\n{json.dumps(call)}\n</tool_call>"
    return re.sub(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", replace_tool_call, content, flags=re.DOTALL)


def extract_citations(messages: list[dict]) -> str:
    """Build citation markers from tool results."""
    citations = {}
    for msg in messages:
        if msg["role"] != "tool":
            continue
        try:
            data = json.loads(msg["content"])
        except (json.JSONDecodeError, TypeError):
            continue

        def walk(obj):
            if isinstance(obj, dict):
                for key in ["event_id", "filing_id", "research_id", "company_doc_id",
                            "thirdbridge_event_id", "equity_id"]:
                    if key in obj:
                        ctype = key.replace("_id", "")
                        citations[f"{ctype}_{obj[key]}"] = obj
                if "transcript_section" in obj or "transcript" in obj:
                    eid = obj.get("event_id", "")
                    if eid:
                        citations[f"transcript_{eid}"] = obj
                # For search results with chunk IDs
                if "company_doc_id" in obj:
                    citations[f"company_doc_{obj['company_doc_id']}"] = obj
                if "thirdbridge_event_id" in obj:
                    citations[f"thirdbridge_event_{obj['thirdbridge_event_id']}"] = obj
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)
        walk(data)

    if not citations:
        return "No citation IDs found in tool results."
    lines = []
    for marker, data in citations.items():
        title = data.get("title", data.get("name", data.get("company_name", "")))
        date = data.get("event_date", data.get("published_date", data.get("filing_date", "")))
        desc = f"{title}" + (f" ({date})" if date else "")
        lines.append(f"  [{marker}] — {desc}")
    return "\n".join(lines)


# ── Rewrite prompt ──────────────────────────────────────────────

REWRITE_PROMPT = """You are rewriting the final assistant response in a training example for a financial AI.

## Rules
1. Professional institutional tone — write for portfolio managers and buy-side analysts
2. Direct answer first — 1-2 sentences with key figures, no header
3. EVERY factual claim MUST have an inline citation immediately after it: [event_XXXXX], [transcript_XXXXX], [filing_XXXXX], [research_XXXXX], [company_doc_XXXXX], [thirdbridge_event_XXXXX]
4. ONLY use citation markers from the AVAILABLE CITATIONS list
5. Organize sections with ### headers (Events, Filings, Financials, Research, etc.)
6. Use markdown tables, **bold** key figures, period labels (Q4 FY25), currency/units ($10.8B USD)
7. DO NOT: offer to search more, use first-person, add disclaimers, include Sources section, use strikethrough
8. Simple queries (listing events/indexes): concise answer + table/list with citations

Return ONLY the rewritten response text."""


def rewrite_final_response(example: list[dict], client) -> list[dict]:
    """Re-run Claude rewrite on the final assistant response."""
    result = deepcopy(example)
    available_citations = extract_citations(result)

    user_questions = [m["content"] for m in result if m["role"] == "user"]
    tool_results = [m["content"] for m in result if m["role"] == "tool"]

    # Find last non-tool-call assistant message
    final_idx = None
    for i in range(len(result) - 1, -1, -1):
        if result[i]["role"] == "assistant" and "<tool_call>" not in result[i]["content"]:
            final_idx = i
            break

    if final_idx is None:
        return result

    prompt = f"""## User Question
{' → '.join(user_questions)}

## Tool Results
{chr(10).join(f'--- Result {i+1} ---{chr(10)}{tr[:3000]}' for i, tr in enumerate(tool_results))}

## Available Citations
{available_citations}

## Current Response
{result[final_idx]['content']}

Rewrite following the rules. Return ONLY the rewritten text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=REWRITE_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    result[final_idx]["content"] = response.content[0].text
    return result


# ── Example generation ──────────────────────────────────────────

GENERATION_SYSTEM = """You generate training data for a financial AI assistant that uses Aiera MCP tools.

Each example is a multi-turn conversation in JSON format with a messages array. Messages use roles: user, assistant, tool.

## Format Rules

1. **Assistant tool-calling turns** must follow this EXACT format:
```
<think>Brief reasoning, 2-3 sentences max.</think>
Short status message describing the action...
<tool_call>
{"name":"tool_name","arguments":{...}}
</tool_call>
```

2. **Tool result turns** (role: "tool") contain raw JSON matching the Aiera API schema.

3. **Final assistant turns** (no <tool_call>) use institutional financial analyst style:
   - Direct answer first (1-2 sentences, key figures)
   - Organized sections with ### headers
   - EVERY claim has inline citation: [event_XXXXX], [transcript_XXXXX], [filing_XXXXX], [research_XXXXX], [company_doc_XXXXX], [thirdbridge_event_XXXXX]
   - Markdown tables, **bold** figures, period labels (Q4 FY25), currency/units ($10.8B USD)
   - NO "Would you like...", NO first-person, NO disclaimers, NO Sources section

4. **Required tool parameters** (add to ALL tool calls):
   - `page_size: 100` for find_* tools, get_upcoming_events, get_*_constituents
   - `exclude_instructions: true` for ALL tools
   - `self_identification: "aierachat"` for ALL tools

5. Tool results must NOT contain an "instructions" field.

6. Citation IDs in the final response must match IDs found in tool results (event_id, filing_id, research_id, company_doc_id, thirdbridge_event_id).

7. Use realistic financial data, company names, tickers (Bloomberg format: AAPL:US, MSFT:US), dates in 2025-2026 range.

Return ONLY valid JSON: {"messages": [...]}"""


# Query templates for new examples
QUERY_TEMPLATES = [
    # ── Events & Transcripts ──────────────────────────────────
    {
        "category": "earnings_single",
        "count": 8,
        "description": "Single-company earnings call query → find_events → get_event → final answer",
        "example_queries": [
            "Summarize the latest AMZN earnings call",
            "What did Google report in Q3 2025?",
            "Get me the JPMorgan Q4 2025 earnings summary",
        ],
        "tools": ["find_events", "get_event"],
        "turns": 6,
    },
    {
        "category": "earnings_multi",
        "count": 6,
        "description": "Multi-company earnings comparison → find_events with multiple tickers → final answer with comparison table",
        "example_queries": [
            "Compare the latest earnings for AMD and NVDA",
            "How did the big banks do last quarter? (JPM, GS, MS)",
        ],
        "tools": ["find_events"],
        "turns": 4,
    },
    {
        "category": "transcript_search",
        "count": 10,
        "description": "Search transcripts for specific topics/quotes → search_transcripts → final answer with transcript citations",
        "example_queries": [
            "What did NFLX CEO say about the ad tier on the latest call?",
            "Find management comments about AI spending from MSFT earnings",
            "What guidance did TSLA give for 2026 deliveries?",
            "Search for any mentions of tariff impact in recent semiconductor earnings calls",
        ],
        "tools": ["search_transcripts"],
        "turns": 4,
    },
    {
        "category": "upcoming_events",
        "count": 6,
        "description": "Upcoming events query → get_upcoming_events → final answer listing events",
        "example_queries": [
            "What earnings calls are coming up this week?",
            "Any upcoming biotech events in the next 2 weeks?",
        ],
        "tools": ["get_upcoming_events"],
        "turns": 4,
    },
    {
        "category": "conferences",
        "count": 5,
        "description": "Conference search → find_conferences → final answer",
        "example_queries": [
            "What tech conferences are happening in Q2 2026?",
            "Find upcoming healthcare investor conferences",
        ],
        "tools": ["find_conferences"],
        "turns": 4,
    },
    # ── Financials & Ratios ──────────────────────────────────
    {
        "category": "financials_quarterly",
        "count": 8,
        "description": "Quarterly financials → get_financials → final answer with table",
        "example_queries": [
            "AAPL revenue and EPS for the last 4 quarters",
            "Show me MSFT cloud revenue breakdown by quarter",
            "What were META's operating margins last year by quarter?",
        ],
        "tools": ["get_financials"],
        "turns": 4,
    },
    {
        "category": "financials_comparison",
        "count": 6,
        "description": "Multi-company financial comparison → get_financials × 2-3 → comparison table",
        "example_queries": [
            "Compare AAPL and MSFT revenue and margins for FY2025",
            "Show me NVDA vs AMD revenue growth over the last year",
        ],
        "tools": ["get_financials"],
        "turns": 6,
    },
    {
        "category": "ratios",
        "count": 8,
        "description": "Financial ratios → get_ratios → final answer with metrics table",
        "example_queries": [
            "What are GOOGL's profitability ratios?",
            "Compare P/E ratios for the FAANG stocks",
            "Show me NFLX valuation metrics",
        ],
        "tools": ["get_ratios"],
        "turns": 4,
    },
    {
        "category": "kpis_segments",
        "count": 6,
        "description": "KPIs and segments → get_kpis_and_segments → final answer",
        "example_queries": [
            "What are NFLX's key subscriber metrics?",
            "Show me AMZN's segment breakdown — AWS vs retail",
            "CRM segment revenue by product line",
        ],
        "tools": ["get_kpis_and_segments"],
        "turns": 4,
    },
    # ── Filings ──────────────────────────────────────────────
    {
        "category": "filing_lookup",
        "count": 6,
        "description": "Filing lookup → find_filings → get_filing → final answer",
        "example_queries": [
            "Show me TSLA's latest 10-K filing",
            "Get the most recent AAPL 10-Q",
            "Find any 8-K filings from NVDA this year",
        ],
        "tools": ["find_filings", "get_filing"],
        "turns": 6,
    },
    {
        "category": "filing_search",
        "count": 8,
        "description": "Search filings for specific topics → find_filings → search_filings → final answer",
        "example_queries": [
            "What risk factors does MSFT mention in their latest 10-K?",
            "Find AI-related disclosures in GOOGL's SEC filings",
            "Search for supply chain risks in AAPL's most recent annual report",
            "What does AMZN's 10-K say about antitrust?",
        ],
        "tools": ["find_filings", "search_filings"],
        "turns": 6,
    },
    # ── Research ─────────────────────────────────────────────
    {
        "category": "research_lookup",
        "count": 8,
        "description": "Research report lookup → find_research → get_research → final answer",
        "example_queries": [
            "Find Goldman Sachs reports on NVDA",
            "Any recent Morgan Stanley research on semiconductors?",
            "Get me the latest J.P. Morgan analysis of AAPL",
        ],
        "tools": ["find_research", "get_research"],
        "turns": 6,
    },
    {
        "category": "research_search",
        "count": 6,
        "description": "Search research for topics → search_research → final answer",
        "example_queries": [
            "What are analysts saying about the AI capex cycle?",
            "Find research on interest rate impact on tech valuations",
        ],
        "tools": ["search_research"],
        "turns": 4,
    },
    {
        "category": "research_metadata",
        "count": 8,
        "description": "Research metadata queries → get_research_providers/authors/asset_classes/subjects → final answer",
        "example_queries": [
            "What research providers are available?",
            "Show me research authors covering semiconductors at Morgan Stanley",
            "What asset classes does the research cover?",
            "List all research subject types",
        ],
        "tools": ["get_research_providers", "get_research_authors", "get_research_asset_classes",
                   "get_research_subjects", "get_research_product_focuses",
                   "get_research_discipline_types", "get_research_region_types",
                   "get_research_country_codes", "get_research_asset_types"],
        "turns": 4,
    },
    # ── Company Docs ─────────────────────────────────────────
    {
        "category": "company_docs",
        "count": 6,
        "description": "Company document lookup → find_company_docs → get_company_doc → final answer",
        "example_queries": [
            "Find NVDA's latest earnings press release",
            "Show me AMZN investor presentations from 2026",
            "Get the latest AAPL press releases",
        ],
        "tools": ["find_company_docs", "get_company_doc"],
        "turns": 6,
    },
    {
        "category": "company_docs_search",
        "count": 5,
        "description": "Search company docs → search_company_docs → final answer",
        "example_queries": [
            "Search MSFT company documents for Azure AI announcements",
            "Find press releases mentioning layoffs from META",
        ],
        "tools": ["search_company_docs"],
        "turns": 4,
    },
    {
        "category": "company_docs_metadata",
        "count": 5,
        "description": "Company doc categories/keywords → get_company_doc_categories or get_company_doc_keywords → final answer",
        "example_queries": [
            "What company document categories are available?",
            "Show me the available keywords for filtering company documents",
        ],
        "tools": ["get_company_doc_categories", "get_company_doc_keywords"],
        "turns": 4,
    },
    # ── Third Bridge ─────────────────────────────────────────
    {
        "category": "thirdbridge_lookup",
        "count": 5,
        "description": "Third Bridge expert event → find_third_bridge_events → get_third_bridge_event → final answer",
        "example_queries": [
            "Find expert calls about NVDA's competitive position",
            "Any Third Bridge forums on cloud infrastructure trends?",
        ],
        "tools": ["find_third_bridge_events", "get_third_bridge_event"],
        "turns": 6,
    },
    {
        "category": "thirdbridge_search",
        "count": 5,
        "description": "Search Third Bridge → search_thirdbridge → final answer",
        "example_queries": [
            "Search Third Bridge for expert views on EV battery technology",
            "What do industry experts say about semiconductor pricing?",
        ],
        "tools": ["search_thirdbridge"],
        "turns": 4,
    },
    # ── Equities & Indexes ───────────────────────────────────
    {
        "category": "equity_lookup",
        "count": 5,
        "description": "Equity lookup → find_equities → final answer",
        "example_queries": [
            "Look up the ticker for Palantir",
            "Find semiconductor companies in the database",
            "What's the bloomberg ticker for Toyota?",
        ],
        "tools": ["find_equities"],
        "turns": 4,
    },
    {
        "category": "equity_summary",
        "count": 6,
        "description": "Equity summary with analyst ratings → get_equity_summaries → final answer",
        "example_queries": [
            "Give me an overview of NVDA — price, analyst ratings, targets",
            "Compare analyst sentiment for CRM vs NOW",
            "What's the consensus on TSLA right now?",
        ],
        "tools": ["get_equity_summaries"],
        "turns": 4,
    },
    {
        "category": "indexes",
        "count": 5,
        "description": "Index listing or constituents → get_available_indexes → get_index_constituents → final answer",
        "example_queries": [
            "What indexes are available?",
            "Show me the top constituents of the S&P 500",
            "List NASDAQ 100 members",
        ],
        "tools": ["get_available_indexes", "get_index_constituents"],
        "turns": 4,
    },
    {
        "category": "watchlists",
        "count": 5,
        "description": "Watchlist queries → get_available_watchlists → get_watchlist_constituents → final answer",
        "example_queries": [
            "Show me my watchlists",
            "What's in the Mega Cap Tech watchlist?",
        ],
        "tools": ["get_available_watchlists", "get_watchlist_constituents"],
        "turns": 4,
    },
    # ── Sectors & Web Search ─────────────────────────────────
    {
        "category": "sectors",
        "count": 5,
        "description": "Sector/subsector listing → get_sectors_and_subsectors → final answer",
        "example_queries": [
            "What sectors and subsectors are available?",
            "Show me the technology sector breakdown",
            "List all healthcare subsectors",
        ],
        "tools": ["get_sectors_and_subsectors"],
        "turns": 4,
    },
    {
        "category": "web_search",
        "count": 5,
        "description": "Web search for news/current events → trusted_web_search → final answer (only after stating domain tools weren't sufficient)",
        "example_queries": [
            "What's the latest news on the Fed rate decision?",
            "Any recent headlines about NVDA and export controls?",
        ],
        "tools": ["trusted_web_search"],
        "turns": 4,
    },
    # ── Complex Multi-Tool ───────────────────────────────────
    {
        "category": "deep_dive",
        "count": 10,
        "description": "Deep company analysis → find_events → get_event + get_financials + search_transcripts → comprehensive answer",
        "example_queries": [
            "Give me a full analysis of NFLX's latest quarter — earnings, financials, and key management commentary",
            "Deep dive on MSFT — latest results, margins, and what management said about AI",
            "Comprehensive look at AMZN Q4 — revenue breakdown, AWS growth, and forward guidance",
        ],
        "tools": ["find_events", "get_event", "get_financials", "search_transcripts"],
        "turns": 10,
    },
    {
        "category": "company_overview",
        "count": 8,
        "description": "Company overview → get_equity_summaries + get_financials + find_research → overview with analyst views",
        "example_queries": [
            "Give me a complete overview of NVDA — price, financials, and recent research",
            "Full picture on CRM — analyst ratings, latest quarter, and research coverage",
        ],
        "tools": ["get_equity_summaries", "get_financials", "find_research"],
        "turns": 8,
    },
    {
        "category": "cross_reference",
        "count": 5,
        "description": "Cross-reference filings with transcripts → find_filings + search_filings + search_transcripts → synthesized answer",
        "example_queries": [
            "Compare what AAPL's 10-K says about services revenue vs what management said on the earnings call",
            "Cross-reference TSLA's risk factors with recent management commentary",
        ],
        "tools": ["find_filings", "search_filings", "search_transcripts"],
        "turns": 8,
    },
]

# Calculate total
TOTAL_NEW = sum(t["count"] for t in QUERY_TEMPLATES)


def generate_examples_batch(template: dict, count: int, client, existing_queries: set) -> list[dict]:
    """Generate a batch of training examples for a given template."""
    prompt = f"""Generate exactly {count} complete training examples for this category:

**Category:** {template['category']}
**Description:** {template['description']}
**Tools to use:** {', '.join(template['tools'])}
**Number of turns:** {template['turns']} messages minimum
**Example queries (generate DIFFERENT ones, don't reuse these):** {json.dumps(template['example_queries'])}

IMPORTANT:
- Generate {count} DIFFERENT user queries, each as a separate complete example
- Each example must be a valid JSON object with a "messages" array
- Use diverse companies (not just FAANG — include financials like JPM/GS, healthcare like JNJ/PFE, industrials like CAT/BA, energy like XOM/CVX, etc.)
- Use realistic dates in 2025-2026 range
- Tool results should have realistic data with proper IDs (event_id, filing_id, research_id, etc.)
- Final answers MUST have inline citations matching tool result IDs
- ALL tool calls MUST include: exclude_instructions: true, self_identification: "aierachat"
- find_*/get_upcoming_events/get_*_constituents calls MUST also include: page_size: 100

Return a JSON array of {count} objects, each with a "messages" key.
Format: [{{"messages": [...]}}, {{"messages": [...]}}, ...]"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        system=GENERATION_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    # Extract JSON from response (may be wrapped in ```json blocks)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    try:
        examples = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON for category %s: %s", template["category"], e)
        logger.error("Response text (first 500): %s", text[:500])
        return []

    if not isinstance(examples, list):
        examples = [examples]

    # Validate and fix each example
    valid = []
    for ex in examples:
        msgs = ex.get("messages", [])
        if len(msgs) < 3:
            logger.warning("Skipping example with only %d messages", len(msgs))
            continue

        # Ensure tool call params are correct
        for msg in msgs:
            if msg["role"] == "assistant" and "<tool_call>" in msg.get("content", ""):
                msg["content"] = fix_tool_call_params(msg["content"])
            # Strip instructions from tool results
            if msg["role"] == "tool":
                try:
                    data = json.loads(msg["content"])
                    if isinstance(data, dict) and "instructions" in data:
                        del data["instructions"]
                        msg["content"] = json.dumps(data)
                except (json.JSONDecodeError, TypeError):
                    pass

        # Check user query uniqueness
        user_q = next((m["content"] for m in msgs if m["role"] == "user"), "")
        if user_q.lower().strip() in existing_queries:
            logger.warning("Duplicate query, skipping: %s", user_q[:60])
            continue
        existing_queries.add(user_q.lower().strip())

        valid.append({"messages": msgs})

    return valid


# ── Validation ──────────────────────────────────────────────────

def validate_example(example: dict, idx: int) -> list[str]:
    """Validate a training example, return list of issues."""
    issues = []
    msgs = example.get("messages", [])

    if len(msgs) < 3:
        issues.append(f"[{idx}] Too few messages: {len(msgs)}")
        return issues

    # Check roles sequence
    if msgs[0]["role"] != "user":
        issues.append(f"[{idx}] First message is not user: {msgs[0]['role']}")

    # Check tool call format
    for j, msg in enumerate(msgs):
        if msg["role"] == "assistant" and "<tool_call>" in msg.get("content", ""):
            content = msg["content"]
            # Must have think block
            if "<think>" not in content:
                issues.append(f"[{idx}] msg[{j}] tool-calling turn missing <think> block")
            # Must have status text
            think_removed = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            before_tc = think_removed.split("<tool_call>")[0].strip()
            if not before_tc:
                issues.append(f"[{idx}] msg[{j}] tool-calling turn missing status text")
            # Validate JSON in tool call
            for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL):
                try:
                    call = json.loads(m.group(1))
                    if "name" not in call:
                        issues.append(f"[{idx}] msg[{j}] tool call missing 'name'")
                    if "arguments" not in call:
                        issues.append(f"[{idx}] msg[{j}] tool call missing 'arguments'")
                    # Check required params
                    args = call.get("arguments", {})
                    name = call.get("name", "")
                    if name in TOOLS_WITH_INSTRUCTIONS_PARAMS:
                        if "exclude_instructions" not in args:
                            issues.append(f"[{idx}] msg[{j}] {name} missing exclude_instructions")
                        if "self_identification" not in args:
                            issues.append(f"[{idx}] msg[{j}] {name} missing self_identification")
                    if name in TOOLS_WITH_PAGE_SIZE and "page_size" not in args:
                        issues.append(f"[{idx}] msg[{j}] {name} missing page_size")
                except json.JSONDecodeError:
                    issues.append(f"[{idx}] msg[{j}] invalid JSON in tool call")

        # Check tool results are valid JSON
        if msg["role"] == "tool":
            try:
                data = json.loads(msg["content"])
                if isinstance(data, dict) and "instructions" in data:
                    issues.append(f"[{idx}] msg[{j}] tool result contains 'instructions' (should be stripped)")
            except json.JSONDecodeError:
                issues.append(f"[{idx}] msg[{j}] tool result is not valid JSON")

    # Check final answer has citations (if tool results contain IDs)
    last_assistant = None
    for msg in msgs:
        if msg["role"] == "assistant" and "<tool_call>" not in msg.get("content", ""):
            last_assistant = msg

    if last_assistant:
        cits = extract_citations(msgs)
        if "No citation IDs" not in cits:
            has_citation = re.search(
                r"\[(event|transcript|filing|research|company_doc|thirdbridge_event)_[\w-]+\]",
                last_assistant["content"],
            )
            if not has_citation:
                issues.append(f"[{idx}] final answer has no citations but tool results have IDs")

        # Check for anti-patterns
        content = last_assistant["content"]
        if "Would you like" in content or "would you like" in content:
            issues.append(f"[{idx}] final answer contains 'Would you like'")
        if "I'll " in content or "I will " in content or "Let me " in content:
            issues.append(f"[{idx}] final answer contains first-person language")

    # Check for base model format leakage
    for msg in msgs:
        if msg["role"] == "assistant":
            if "<|tool_call_start|>" in msg.get("content", ""):
                issues.append(f"[{idx}] contains base model format <|tool_call_start|>")

    return issues


# ── Phase implementations ───────────────────────────────────────

def phase_fix(input_path: str, client):
    """Phase 1: Fix citations on existing examples."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    with open(input_path) as f:
        examples = [json.loads(line) for line in f]

    logger.info("Loaded %d existing examples", len(examples))

    # Find examples needing citation fixes
    needs_fix = []
    for i, ex in enumerate(examples):
        msgs = ex["messages"]
        last_a = None
        for msg in msgs:
            if msg["role"] == "assistant" and "<tool_call>" not in msg.get("content", ""):
                last_a = msg
        if last_a:
            cits = extract_citations(msgs)
            has_citation = re.search(
                r"\[(event|transcript|filing|research|company_doc|thirdbridge_event)_[\w-]+\]",
                last_a["content"],
            )
            if not has_citation and "No citation IDs" not in cits:
                needs_fix.append(i)

    logger.info("Found %d examples needing citation fixes", len(needs_fix))

    # Load checkpoint if exists
    fixed = {}
    if FIXED_FILE.exists():
        with open(FIXED_FILE) as f:
            for line in f:
                obj = json.loads(line)
                fixed[obj["index"]] = obj["messages"]
        logger.info("Resuming from checkpoint: %d already fixed", len(fixed))

    for i, idx in enumerate(needs_fix):
        if idx in fixed:
            continue

        user_q = next((m["content"][:80] for m in examples[idx]["messages"] if m["role"] == "user"), "?")
        logger.info("Fixing %d/%d (example %d): %s", i + 1, len(needs_fix), idx, user_q)

        try:
            fixed_msgs = rewrite_final_response(examples[idx]["messages"], client)
            fixed[idx] = fixed_msgs

            with open(FIXED_FILE, "a") as f:
                f.write(json.dumps({"index": idx, "messages": fixed_msgs}) + "\n")

        except Exception as e:
            logger.error("Failed to fix example %d: %s", idx, e)

        time.sleep(0.5)

    # Apply fixes to original data
    result = []
    for i, ex in enumerate(examples):
        if i in fixed:
            result.append({"messages": fixed[i]})
        else:
            result.append(ex)

    # Save fixed data
    output = CHECKPOINT_DIR / "v6_fixed.jsonl"
    with open(output, "w") as f:
        for ex in result:
            f.write(json.dumps(ex) + "\n")

    logger.info("Saved %d fixed examples to %s", len(result), output)
    return result


def phase_generate(input_path: str, client):
    """Phase 2: Generate new training examples."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect existing queries to avoid duplicates
    existing_queries = set()
    if input_path and Path(input_path).exists():
        with open(input_path) as f:
            for line in f:
                ex = json.loads(line)
                for msg in ex["messages"]:
                    if msg["role"] == "user":
                        existing_queries.add(msg["content"].lower().strip())

    logger.info("Existing queries: %d (will avoid duplicates)", len(existing_queries))
    logger.info("Target: %d new examples across %d categories", TOTAL_NEW, len(QUERY_TEMPLATES))

    # Load checkpoint
    generated = []
    completed_categories = set()
    if GENERATED_FILE.exists():
        with open(GENERATED_FILE) as f:
            for line in f:
                obj = json.loads(line)
                generated.append(obj)
                completed_categories.add(obj.get("_category", ""))
        logger.info("Resuming: %d examples already generated", len(generated))

    for template in QUERY_TEMPLATES:
        cat = template["category"]
        if cat in completed_categories:
            logger.info("Skipping category %s (already done)", cat)
            continue

        count = template["count"]
        logger.info("Generating %d examples for category: %s", count, cat)

        try:
            batch = generate_examples_batch(template, count, client, existing_queries)
            logger.info("  Got %d valid examples", len(batch))

            for ex in batch:
                ex["_category"] = cat
                generated.append(ex)
                with open(GENERATED_FILE, "a") as f:
                    f.write(json.dumps(ex) + "\n")

            completed_categories.add(cat)

        except Exception as e:
            logger.error("Failed to generate for category %s: %s", cat, e)

        time.sleep(1.0)  # Rate limit between categories

    logger.info("Generated %d total new examples", len(generated))
    return generated


def phase_finalize(hf_token: str, dry_run: bool = False):
    """Phase 3: Combine, validate, and optionally push."""
    # Load fixed existing data
    fixed_path = CHECKPOINT_DIR / "v6_fixed.jsonl"
    if not fixed_path.exists():
        logger.error("Fixed data not found at %s — run --phase fix first", fixed_path)
        return

    existing = []
    with open(fixed_path) as f:
        for line in f:
            existing.append(json.loads(line))

    # Load generated data
    generated = []
    if GENERATED_FILE.exists():
        with open(GENERATED_FILE) as f:
            for line in f:
                obj = json.loads(line)
                # Remove internal metadata
                obj.pop("_category", None)
                generated.append(obj)

    logger.info("Combining: %d existing + %d new = %d total", len(existing), len(generated), len(existing) + len(generated))

    all_examples = existing + generated

    # Validate all
    total_issues = 0
    issue_counts = {"missing_citations": 0, "missing_think": 0, "missing_status": 0,
                    "bad_json": 0, "missing_params": 0, "anti_pattern": 0}

    for i, ex in enumerate(all_examples):
        issues = validate_example(ex, i)
        if issues:
            total_issues += len(issues)
            for issue in issues:
                if "no citations" in issue:
                    issue_counts["missing_citations"] += 1
                elif "missing <think>" in issue:
                    issue_counts["missing_think"] += 1
                elif "missing status" in issue:
                    issue_counts["missing_status"] += 1
                elif "invalid JSON" in issue or "not valid JSON" in issue:
                    issue_counts["bad_json"] += 1
                elif "missing exclude_instructions" in issue or "missing page_size" in issue:
                    issue_counts["missing_params"] += 1
                elif "Would you like" in issue or "first-person" in issue:
                    issue_counts["anti_pattern"] += 1
                if i < 5 or "base model format" in issue:
                    logger.warning(issue)

    logger.info("Validation: %d total issues across %d examples", total_issues, len(all_examples))
    for k, v in issue_counts.items():
        logger.info("  %s: %d", k, v)

    # Tool coverage
    tools_used = set()
    from collections import Counter
    tool_freq = Counter()
    for ex in all_examples:
        for msg in ex["messages"]:
            if msg["role"] == "assistant" and "<tool_call>" in msg.get("content", ""):
                for m in re.finditer(r'"name"\s*:\s*"([^"]+)"', msg["content"]):
                    tools_used.add(m.group(1))
                    tool_freq[m.group(1)] += 1

    logger.info("Tool coverage: %d unique tools", len(tools_used))
    for tool, count in tool_freq.most_common():
        logger.info("  %s: %d", tool, count)

    # Citation coverage
    has_citations = 0
    for ex in all_examples:
        last_a = None
        for msg in ex["messages"]:
            if msg["role"] == "assistant" and "<tool_call>" not in msg.get("content", ""):
                last_a = msg
        if last_a and re.search(r"\[(event|transcript|filing|research|company_doc|thirdbridge_event)_[\w-]+\]", last_a["content"]):
            has_citations += 1
    logger.info("Citation coverage: %d/%d (%.1f%%)", has_citations, len(all_examples), 100 * has_citations / len(all_examples))

    # Save final JSONL
    FINAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_FILE, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    logger.info("Saved %d examples to %s", len(all_examples), FINAL_FILE)

    if dry_run:
        logger.info("DRY RUN — not pushing to HuggingFace")
        return

    # Push to HuggingFace
    from datasets import Dataset
    records = [{"messages": ex["messages"]} for ex in all_examples]
    ds = Dataset.from_list(records)
    logger.info("Pushing %d examples to %s...", len(ds), HF_REPO)
    ds.push_to_hub(HF_REPO, split="train", private=False, token=hf_token)
    logger.info("Done! Dataset pushed to %s", HF_REPO)


def main():
    parser = argparse.ArgumentParser(description="Expand and fix training data")
    parser.add_argument("--phase", choices=["fix", "generate", "finalize", "all"], required=True)
    parser.add_argument("--input", type=str, default="data/training/aiera_tools_v6.jsonl")
    parser.add_argument("--hf-token", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Limit examples per category (for testing)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if args.phase in ("fix", "generate", "all") and not api_key:
        logger.error("ANTHROPIC_API_KEY required for fix/generate phases")
        return

    client = None
    if api_key:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

    if args.limit:
        for t in QUERY_TEMPLATES:
            t["count"] = min(t["count"], args.limit)

    if args.phase == "fix" or args.phase == "all":
        phase_fix(args.input, client)

    if args.phase == "generate" or args.phase == "all":
        phase_generate(args.input, client)

    if args.phase == "finalize" or args.phase == "all":
        phase_finalize(args.hf_token, args.dry_run)


if __name__ == "__main__":
    main()
