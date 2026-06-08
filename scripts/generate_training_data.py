"""Generate training data using real Aiera MCP tool results and Claude API.

Generates ~612 new training examples by:
1. Using Claude to generate diverse user prompts
2. Calling real Aiera MCP tools via the deployed endpoint
3. Using Claude to write final assistant responses with proper citations

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/generate_training_data.py \
        --endpoint https://YOUR-ENDPOINT.cloud \
        --output data/training/aiera_tools_v7.jsonl \
        --target 1000 \
        --hf-token <token> \
        --hf-repo bryanhealey/my-aiera-finetune-v7-data
"""

import argparse
import json
import logging
import os
import random
import re
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_training")

CHECKPOINT_DIR = Path("/tmp/training_generation")
CHECKPOINT_FILE = CHECKPOINT_DIR / "generated_checkpoint.jsonl"

# ── System prompt (must match inference) ─────────────────────────

SYSTEM_PROMPT = """You are an institutional financial research assistant. Reason step-by-step, plan tool usage, and call tools efficiently.

## Persona & Audience

Provide rigorous, data-driven analysis for an institutional audience — portfolio managers, buy-side analysts, and investment committees. Focus on key financial metrics, margin trends, revenue drivers, competitive positioning, and forward guidance.

## Thinking

Keep your <think> block extremely brief — 2-3 short sentences maximum. State only what you need to do, then act immediately.

## Tool Calling

You may call multiple tools in a single response. Call as many as needed. Always use the XML JSON format shown below.

Before each tool call, write a short status message (under 100 characters) describing what you are doing.

Example:
Searching for recent NFLX earnings events...
<tool_call>
{"name": "find_events", "arguments": {"bloomberg_ticker": "NFLX:US", "event_type": "earnings", "start_date": "2025-01-01", "end_date": "2026-03-19", "page_size": 100, "exclude_instructions": true, "self_identification": "aierachat"}}
</tool_call>

### Date Range Rules for find_events

- start_date and end_date are REQUIRED for find_events.
- When the user says "latest" or "most recent", use start_date at least 6 months back.
- When unsure, default to start_date = 12 months ago, end_date = today.
- NEVER use a date range shorter than 3 months for "latest" or "recent" queries.

### Tool Selection by Query Type

| Query Type | Tool |
|---|---|
| Financial metrics (revenue, EPS, etc.) | get_financials |
| Ratios, margins, ROE, P/E | get_ratios |
| Segment data, KPIs | get_kpis_and_segments |
| Management quotes/commentary | search_transcripts |
| Full earnings call or document summary | get_event or get_filing |
| Analyst/research reports | find_research + get_research |

## Response Structure

1. Direct answer first — 1-2 sentences with key figures, no header
2. Supporting detail — organized under clear markdown headers
3. Label all periods clearly (e.g., Q3 FY24). Include YoY comparisons. State currency and units.
4. Only state facts directly supported by tool results. Never infer or extrapolate specific numbers."""

# ── Companies pool ───────────────────────────────────────────────

COMPANIES = [
    # Mega-cap tech
    ("AAPL:US", "Apple"), ("MSFT:US", "Microsoft"), ("GOOGL:US", "Alphabet/Google"),
    ("AMZN:US", "Amazon"), ("META:US", "Meta"), ("NVDA:US", "NVIDIA"),
    ("TSLA:US", "Tesla"), ("AVGO:US", "Broadcom"), ("CRM:US", "Salesforce"),
    ("ORCL:US", "Oracle"), ("ADBE:US", "Adobe"), ("AMD:US", "AMD"),
    ("INTC:US", "Intel"), ("NFLX:US", "Netflix"), ("CSCO:US", "Cisco"),
    # Financials
    ("JPM:US", "JPMorgan"), ("GS:US", "Goldman Sachs"), ("MS:US", "Morgan Stanley"),
    ("BAC:US", "Bank of America"), ("WFC:US", "Wells Fargo"), ("C:US", "Citigroup"),
    ("BLK:US", "BlackRock"), ("SCHW:US", "Charles Schwab"),
    # Healthcare
    ("JNJ:US", "Johnson & Johnson"), ("UNH:US", "UnitedHealth"), ("PFE:US", "Pfizer"),
    ("LLY:US", "Eli Lilly"), ("ABBV:US", "AbbVie"), ("MRK:US", "Merck"),
    ("TMO:US", "Thermo Fisher"), ("ABT:US", "Abbott"),
    # Industrials
    ("CAT:US", "Caterpillar"), ("BA:US", "Boeing"), ("HON:US", "Honeywell"),
    ("GE:US", "GE Aerospace"), ("LMT:US", "Lockheed Martin"), ("RTX:US", "RTX"),
    ("DE:US", "Deere & Company"), ("UPS:US", "UPS"),
    # Consumer
    ("PG:US", "Procter & Gamble"), ("KO:US", "Coca-Cola"), ("PEP:US", "PepsiCo"),
    ("COST:US", "Costco"), ("WMT:US", "Walmart"), ("MCD:US", "McDonald's"),
    ("NKE:US", "Nike"), ("SBUX:US", "Starbucks"),
    # Energy
    ("XOM:US", "ExxonMobil"), ("CVX:US", "Chevron"), ("COP:US", "ConocoPhillips"),
    ("SLB:US", "Schlumberger"),
    # Telecom/Media
    ("DIS:US", "Disney"), ("CMCSA:US", "Comcast"), ("T:US", "AT&T"),
    ("VZ:US", "Verizon"),
]

# ── Tool call templates ──────────────────────────────────────────

CATEGORIES = [
    {
        "category": "earnings_summary",
        "count": 30,
        "description": "User asks to summarize a company's latest/recent earnings call. Use find_events to locate the call, then get_event for the summary.",
        "tools_sequence": [
            {"tool": "find_events", "args_template": {"bloomberg_ticker": "{ticker}", "event_type": "earnings", "start_date": "{start_date}", "end_date": "{end_date}", "page_size": 100, "exclude_instructions": True, "self_identification": "aierachat"}},
            {"tool": "get_event", "args_template": {"event_id": "{event_id}", "exclude_instructions": True, "self_identification": "aierachat"}},
        ],
        "prompt_templates": [
            "Can you summarize the latest {company} earnings call?",
            "What were the key takeaways from {company}'s most recent quarterly earnings?",
            "Summarize {company}'s latest earnings results",
            "How did {company}'s latest quarter go?",
            "Give me a rundown of {ticker}'s most recent earnings call",
            "What did {company} report in their latest earnings?",
            "Pull up the latest {company} earnings call summary",
            "I need a summary of {company}'s most recent earnings results",
        ],
    },
    {
        "category": "transcript_search",
        "count": 30,
        "description": "User asks what management said about a topic. Use find_events to get event ID, then search_transcripts for the topic.",
        "tools_sequence": [
            {"tool": "find_events", "args_template": {"bloomberg_ticker": "{ticker}", "event_type": "earnings", "start_date": "{start_date}", "end_date": "{end_date}", "page_size": 100, "exclude_instructions": True, "self_identification": "aierachat"}},
            {"tool": "search_transcripts", "args_template": {"query_text": "{query}", "event_ids": ["{event_id}"], "size": 10, "exclude_instructions": True, "self_identification": "aierachat"}},
        ],
        "prompt_templates": [
            "What did {company}'s CEO say about {topic} on their latest earnings call?",
            "Search for {company} management commentary on {topic}",
            "Find what {company} said about {topic} in their most recent earnings call",
            "I want to see what {company} management discussed regarding {topic}",
        ],
        "topics": [
            "AI strategy", "margins and profitability", "capital expenditure", "hiring and headcount",
            "competitive landscape", "supply chain", "international expansion", "share buybacks",
            "forward guidance", "pricing strategy", "new product launches", "regulatory risks",
            "cloud growth", "subscription metrics", "advertising revenue", "cost reduction",
        ],
    },
    {
        "category": "financials",
        "count": 30,
        "description": "User asks for financial metrics. Use get_financials directly.",
        "tools_sequence": [
            {"tool": "get_financials", "args_template": {"bloomberg_ticker": "{ticker}", "period": "quarterly", "exclude_instructions": True, "self_identification": "aierachat"}},
        ],
        "prompt_templates": [
            "Show me {company}'s quarterly financials",
            "What are {company}'s latest financial results?",
            "{ticker} revenue and earnings for the last few quarters",
            "Pull up {company}'s income statement data",
            "I need {company}'s quarterly revenue, EPS, and margins",
        ],
    },
    {
        "category": "ratios",
        "count": 25,
        "description": "User asks for valuation or profitability ratios. Use get_ratios directly.",
        "tools_sequence": [
            {"tool": "get_ratios", "args_template": {"bloomberg_ticker": "{ticker}", "exclude_instructions": True, "self_identification": "aierachat"}},
        ],
        "prompt_templates": [
            "What are {company}'s current valuation ratios?",
            "Show me {ticker} P/E, P/S, and other multiples",
            "Give me {company}'s profitability ratios",
            "How is {company} valued relative to its fundamentals?",
        ],
    },
    {
        "category": "filings",
        "count": 25,
        "description": "User asks about SEC filings. Use find_filings to locate, then get_filing or search_filings for details.",
        "tools_sequence": [
            {"tool": "find_filings", "args_template": {"bloomberg_ticker": "{ticker}", "start_date": "{start_date}", "end_date": "{end_date}", "page_size": 100, "exclude_instructions": True, "self_identification": "aierachat"}},
        ],
        "prompt_templates": [
            "Find {company}'s recent SEC filings",
            "Show me {ticker}'s latest 10-K or 10-Q filing",
            "What SEC filings has {company} submitted recently?",
            "Pull up recent filings for {company}",
        ],
    },
    {
        "category": "equity_summary",
        "count": 25,
        "description": "User asks for company overview with analyst ratings. Use get_equity_summaries.",
        "tools_sequence": [
            {"tool": "get_equity_summaries", "args_template": {"bloomberg_ticker": "{ticker}", "exclude_instructions": True, "self_identification": "aierachat"}},
        ],
        "prompt_templates": [
            "Give me an overview of {company} — price, analyst ratings, and targets",
            "What's the analyst consensus on {ticker}?",
            "Show me {company}'s equity summary and analyst sentiment",
            "{company} stock overview and analyst recommendations",
        ],
    },
    {
        "category": "research",
        "count": 25,
        "description": "User asks about analyst research. Use find_research or search_research.",
        "tools_sequence": [
            {"tool": "find_research", "args_template": {"bloomberg_ticker": "{ticker}", "start_date": "{start_date}", "end_date": "{end_date}", "page_size": 100, "exclude_instructions": True, "self_identification": "aierachat"}},
        ],
        "prompt_templates": [
            "Find recent analyst research on {company}",
            "Any research reports on {ticker} from the past few months?",
            "What are analysts writing about {company}?",
            "Show me recent research coverage for {company}",
        ],
    },
    {
        "category": "kpis_segments",
        "count": 20,
        "description": "User asks for KPIs or segment data. Use get_kpis_and_segments.",
        "tools_sequence": [
            {"tool": "get_kpis_and_segments", "args_template": {"bloomberg_ticker": "{ticker}", "exclude_instructions": True, "self_identification": "aierachat"}},
        ],
        "prompt_templates": [
            "Show me {company}'s key operating metrics and segment breakdown",
            "What are {ticker}'s KPIs and business segments?",
            "{company} segment revenue breakdown",
            "Give me {company}'s operating metrics by segment",
        ],
    },
    {
        "category": "company_docs",
        "count": 20,
        "description": "User asks for company documents like press releases. Use find_company_docs.",
        "tools_sequence": [
            {"tool": "find_company_docs", "args_template": {"bloomberg_ticker": "{ticker}", "start_date": "{start_date}", "end_date": "{end_date}", "page_size": 100, "exclude_instructions": True, "self_identification": "aierachat"}},
        ],
        "prompt_templates": [
            "Find {company}'s recent press releases and investor presentations",
            "Show me {company}'s latest company documents",
            "Any recent investor presentations from {ticker}?",
        ],
    },
    {
        "category": "thematic_transcript_search",
        "count": 25,
        "description": "User asks a broad thematic question across multiple companies. Use search_transcripts with topic query, no specific company.",
        "tools_sequence": [
            {"tool": "search_transcripts", "args_template": {"query_text": "{query}", "event_type": "earnings", "start_date": "{start_date}", "end_date": "{end_date}", "size": 10, "exclude_instructions": True, "self_identification": "aierachat"}},
        ],
        "prompt_templates": [
            "What are CEOs saying about {topic} across recent earnings calls?",
            "Search recent earnings transcripts for discussion of {topic}",
            "Find management commentary on {topic} from recent earnings calls",
            "Any mentions of {topic} in recent quarterly earnings transcripts?",
        ],
        "topics": [
            "tariffs and trade policy", "AI capital expenditure and data center buildout",
            "consumer spending and demand trends", "interest rates and monetary policy impact",
            "supply chain resilience", "workforce and hiring plans",
            "cloud infrastructure demand", "electric vehicle adoption",
            "drug pricing and pharma regulation", "energy transition and renewables",
            "cybersecurity spending", "M&A and deal pipeline",
            "inflation and input costs", "China market exposure",
            "share repurchase programs", "ESG and sustainability initiatives",
        ],
    },
    {
        "category": "deep_dive",
        "count": 30,
        "description": "User asks for comprehensive analysis. Use multiple tools: find_events + get_event + get_financials.",
        "tools_sequence": [
            {"tool": "find_events", "args_template": {"bloomberg_ticker": "{ticker}", "event_type": "earnings", "start_date": "{start_date}", "end_date": "{end_date}", "page_size": 100, "exclude_instructions": True, "self_identification": "aierachat"}},
            {"tool": "get_event", "args_template": {"event_id": "{event_id}", "exclude_instructions": True, "self_identification": "aierachat"}},
            {"tool": "get_financials", "args_template": {"bloomberg_ticker": "{ticker}", "period": "quarterly", "exclude_instructions": True, "self_identification": "aierachat"}},
        ],
        "prompt_templates": [
            "Give me a full analysis of {company}'s latest quarter — earnings call summary and financial data",
            "Deep dive on {company} — latest results, what management said, and financial metrics",
            "Comprehensive look at {ticker}'s most recent quarter with earnings and financials",
            "I need a complete picture of {company}'s latest quarter — call highlights and numbers",
        ],
    },
    {
        "category": "find_equities",
        "count": 15,
        "description": "User doesn't know a ticker or wants to look up a company. Use find_equities.",
        "tools_sequence": [
            {"tool": "find_equities", "args_template": {"search": "{search_term}", "page_size": 100, "exclude_instructions": True, "self_identification": "aierachat"}},
        ],
        "prompt_templates": [
            "Look up the ticker for {company}",
            "What's the Bloomberg ticker for {company}?",
            "Can you find {company} in the database?",
            "I need to look up {search_term} — what companies match?",
        ],
    },
    {
        "category": "upcoming_events",
        "count": 20,
        "description": "User asks about upcoming events. Use get_upcoming_events.",
        "tools_sequence": [
            {"tool": "get_upcoming_events", "args_template": {"bloomberg_ticker": "{ticker}", "page_size": 100, "exclude_instructions": True, "self_identification": "aierachat"}},
        ],
        "prompt_templates": [
            "When is {company}'s next earnings call?",
            "What upcoming events does {company} have?",
            "Show me {ticker}'s upcoming investor events",
            "Any scheduled events for {company} in the next few months?",
        ],
    },
    {
        "category": "sectors",
        "count": 10,
        "description": "User asks about sectors. Use get_sectors_and_subsectors.",
        "tools_sequence": [
            {"tool": "get_sectors_and_subsectors", "args_template": {"exclude_instructions": True, "self_identification": "aierachat"}},
        ],
        "prompt_templates": [
            "What sectors and subsectors are available?",
            "Show me the sector breakdown",
            "List all available sectors in the database",
        ],
    },
    {
        "category": "comparison",
        "count": 25,
        "description": "User compares two companies' financials. Call get_financials twice.",
        "tools_sequence": [
            {"tool": "get_financials", "args_template": {"bloomberg_ticker": "{ticker1}", "period": "quarterly", "exclude_instructions": True, "self_identification": "aierachat"}},
            {"tool": "get_financials", "args_template": {"bloomberg_ticker": "{ticker2}", "period": "quarterly", "exclude_instructions": True, "self_identification": "aierachat"}},
        ],
        "prompt_templates": [
            "Compare {company1} and {company2}'s latest quarterly financials",
            "How do {company1} and {company2} stack up financially?",
            "{ticker1} vs {ticker2} — compare their recent financial performance",
            "Side-by-side comparison of {company1} and {company2}'s latest results",
        ],
    },
    {
        "category": "filing_search",
        "count": 20,
        "description": "User wants to search within filings for a topic. Use find_filings then search_filings.",
        "tools_sequence": [
            {"tool": "find_filings", "args_template": {"bloomberg_ticker": "{ticker}", "start_date": "{start_date}", "end_date": "{end_date}", "page_size": 100, "exclude_instructions": True, "self_identification": "aierachat"}},
            {"tool": "search_filings", "args_template": {"query_text": "{query}", "bloomberg_ticker": "{ticker}", "start_date": "{start_date}", "end_date": "{end_date}", "size": 10, "exclude_instructions": True, "self_identification": "aierachat"}},
        ],
        "prompt_templates": [
            "Search {company}'s recent filings for discussion of {topic}",
            "What does {company}'s latest 10-K say about {topic}?",
            "Find {topic}-related disclosures in {ticker}'s SEC filings",
        ],
        "topics": [
            "risk factors", "revenue recognition", "goodwill impairment",
            "executive compensation", "related party transactions",
            "legal proceedings", "supply chain", "cybersecurity",
            "climate risk", "AI and machine learning", "antitrust",
        ],
    },
    {
        "category": "web_search",
        "count": 12,
        "description": "User asks for news or current events. Use trusted_web_search.",
        "tools_sequence": [
            {"tool": "trusted_web_search", "args_template": {"query": "{query}", "exclude_instructions": True, "self_identification": "aierachat"}},
        ],
        "prompt_templates": [
            "What's the latest news on {company}?",
            "Any recent headlines about {topic}?",
            "Search for news about {company} and {topic}",
        ],
        "topics": [
            "FDA approval", "merger acquisition", "stock split",
            "dividend announcement", "management change", "product launch",
            "export controls", "antitrust investigation", "earnings surprise",
        ],
    },
]

TOTAL_TARGET = sum(c["count"] for c in CATEGORIES)


def call_tool(endpoint: str, tool_name: str, arguments: dict) -> str:
    """Call a tool via the endpoint API."""
    try:
        resp = requests.post(
            f"{endpoint}/api/tools/call",
            json={"name": tool_name, "arguments": arguments},
            timeout=60,
        )
        if resp.ok:
            data = resp.json()
            return data.get("result", json.dumps(data))
        return json.dumps({"error": f"HTTP {resp.status_code}: {resp.text[:200]}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def extract_citations(tool_results: list[str]) -> str:
    """Build citation context from tool results."""
    citations = {}
    for result_text in tool_results:
        try:
            data = json.loads(result_text)
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


RESPONSE_SYSTEM = """You are writing the final assistant response for a financial AI training example.

Rules:
1. Professional institutional tone for portfolio managers and buy-side analysts
2. Direct answer first — 1-2 sentences with key figures, no header
3. EVERY factual claim MUST have an inline citation: [event_XXXXX], [transcript_XXXXX], [filing_XXXXX], [research_XXXXX], [company_doc_XXXXX]
4. ONLY use citation markers from the AVAILABLE CITATIONS list
5. Organize with ### headers (Events, Filings, Financials, Research, etc.)
6. Use **bold** key figures, period labels (Q4 FY25), currency/units ($10.8B USD)
7. DO NOT: offer to search more, use first-person, add disclaimers, include Sources section
8. If tool results are empty or error, state "Data not available" concisely

Return ONLY the response text."""


def generate_response(client, user_prompt: str, tool_calls_and_results: list, available_citations: str) -> str:
    """Use Claude to generate the final assistant response."""
    tool_context = ""
    for tc_name, tc_args, tc_result in tool_calls_and_results:
        tool_context += f"\n--- Tool: {tc_name}({json.dumps(tc_args)}) ---\n{tc_result[:3000]}\n"

    prompt = f"""## User Question
{user_prompt}

## Tool Results
{tool_context}

## Available Citations
{available_citations}

Write the final assistant response following the rules. Return ONLY the response text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=RESPONSE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def build_think_block(user_prompt: str, tools: list[str]) -> str:
    """Generate a brief think block."""
    tool_list = ", ".join(tools)
    return f"<think>Need to use {tool_list} to answer this query.</think>"


def build_status_message(tool_name: str, company: str = "") -> str:
    """Generate a status message for a tool call."""
    msgs = {
        "find_events": f"Searching for {company} earnings events...",
        "get_event": f"Retrieving earnings call details...",
        "get_financials": f"Pulling {company} financial data...",
        "get_ratios": f"Fetching {company} valuation ratios...",
        "get_kpis_and_segments": f"Retrieving {company} KPIs and segments...",
        "get_equity_summaries": f"Getting {company} equity summary...",
        "find_filings": f"Searching for {company} SEC filings...",
        "get_filing": f"Retrieving filing details...",
        "search_filings": f"Searching {company} filings for relevant content...",
        "find_research": f"Looking for analyst research on {company}...",
        "search_transcripts": f"Searching earnings transcripts...",
        "search_research": f"Searching research reports...",
        "find_company_docs": f"Finding {company} company documents...",
        "get_upcoming_events": f"Checking {company} upcoming events...",
        "find_equities": f"Looking up company information...",
        "get_sectors_and_subsectors": "Fetching available sectors...",
        "trusted_web_search": "Searching for recent news...",
    }
    return msgs.get(tool_name, f"Calling {tool_name}...")


def strip_instructions(result_text: str) -> str:
    """Remove instructions array from tool results."""
    try:
        data = json.loads(result_text)
        if isinstance(data, dict) and "instructions" in data:
            data["instructions"] = []
        return json.dumps(data)
    except (json.JSONDecodeError, TypeError):
        return result_text


def generate_example(category: dict, endpoint: str, client, used_prompts: set) -> dict | None:
    """Generate a single training example with real tool results."""
    # Pick a company
    ticker, company = random.choice(COMPANIES)
    ticker2, company2 = random.choice([c for c in COMPANIES if c[0] != ticker])

    # Date range (wide, 12 months back)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    # Pick a topic if applicable
    topics = category.get("topics", [])
    topic = random.choice(topics) if topics else ""

    # Pick and fill a prompt template
    template = random.choice(category["prompt_templates"])
    search_term = company.split("/")[0] if "/" in company else company
    user_prompt = template.format(
        company=company, ticker=ticker, topic=topic,
        company1=company, company2=company2,
        ticker1=ticker, ticker2=ticker2,
        search_term=search_term,
        start_date=start_date, end_date=end_date,
    )

    # Dedup
    if user_prompt.lower().strip() in used_prompts:
        return None
    used_prompts.add(user_prompt.lower().strip())

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": user_prompt})

    tool_calls_and_results = []
    event_id = None

    for step_idx, step in enumerate(category["tools_sequence"]):
        tool_name = step["tool"]
        args = dict(step["args_template"])

        # Fill in dynamic values
        for k, v in list(args.items()):
            if v == "{ticker}":
                args[k] = ticker
            elif v == "{ticker1}":
                args[k] = ticker
            elif v == "{ticker2}":
                args[k] = ticker2
            elif v == "{start_date}":
                args[k] = start_date
            elif v == "{end_date}":
                args[k] = end_date
            elif v == "{query}":
                args[k] = topic if topic else user_prompt[:100]
            elif v == "{search_term}":
                args[k] = search_term
            elif v == "{event_id}":
                if event_id:
                    args[k] = event_id
                else:
                    continue  # Skip if we don't have an event_id
            elif isinstance(v, list) and len(v) == 1 and v[0] == "{event_id}":
                if event_id:
                    args[k] = [event_id]
                else:
                    continue

        # Build the assistant tool-calling message
        think = build_think_block(user_prompt, [tool_name]) if step_idx == 0 else ""
        status = build_status_message(tool_name, company)
        tool_call_json = json.dumps({"name": tool_name, "arguments": args})

        assistant_content = f"{think}\n{status}\n\n<tool_call>\n{tool_call_json}\n</tool_call>" if think else f"{status}\n\n<tool_call>\n{tool_call_json}\n</tool_call>"
        messages.append({"role": "assistant", "content": assistant_content.strip()})

        # Call the real tool
        logger.debug("Calling %s with %s", tool_name, json.dumps(args)[:200])
        result_text = call_tool(endpoint, tool_name, args)
        result_text = strip_instructions(result_text)

        # Extract event_id for subsequent calls
        if tool_name == "find_events" and event_id is None:
            try:
                result_data = json.loads(result_text)
                # Navigate nested structure: {response: {data: [...]}} or {data: [...]}
                response_data = result_data.get("response", result_data)
                if isinstance(response_data, dict):
                    events = response_data.get("data", [])
                else:
                    events = []
                if not events and isinstance(result_data, dict):
                    events = result_data.get("events", [])
                if events and isinstance(events, list) and len(events) > 0:
                    eid = events[0].get("event_id")
                    if eid:
                        event_id = eid
                        logger.info("Extracted event_id=%s from find_events result", event_id)
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                logger.debug("Failed to extract event_id: %s", e)

        messages.append({"role": "tool", "content": result_text})
        tool_calls_and_results.append((tool_name, args, result_text))

    # Check if we got any meaningful results
    def _is_empty_result(result_text: str) -> bool:
        try:
            data = json.loads(result_text)
            # Check for actual error messages (not just "error": null)
            if isinstance(data, dict):
                err = data.get("error")
                if err and err is not None:
                    return True
                # Check nested response.data
                resp = data.get("response", data)
                if isinstance(resp, dict):
                    d = resp.get("data")
                    if isinstance(d, list) and len(d) == 0:
                        return True
            return False
        except (json.JSONDecodeError, TypeError):
            return "error" in result_text.lower()

    all_empty = all(_is_empty_result(r) for _, _, r in tool_calls_and_results)
    if all_empty:
        logger.warning("All tool results empty/error for: %s", user_prompt[:60])
        return None

    # Generate final response with Claude
    citations = extract_citations([r for _, _, r in tool_calls_and_results])
    try:
        final_response = generate_response(client, user_prompt, tool_calls_and_results, citations)
        messages.append({"role": "assistant", "content": final_response})
    except Exception as e:
        logger.error("Claude response generation failed: %s", e)
        return None

    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser(description="Generate training data with real MCP tool results")
    parser.add_argument("--endpoint", required=True, help="Endpoint URL")
    parser.add_argument("--output", default="data/training/aiera_tools_v7.jsonl", help="Output JSONL file")
    parser.add_argument("--target", type=int, default=1000, help="Target total examples")
    parser.add_argument("--hf-token", default=None, help="HF token for pushing")
    parser.add_argument("--hf-repo", default="bryanhealey/my-aiera-finetune-v7-data", help="HF dataset repo")
    parser.add_argument("--dry-run", action="store_true", help="Don't push to HF")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        return

    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)

    # Load existing examples
    existing = []
    if Path(args.output).exists():
        with open(args.output) as f:
            existing = [json.loads(l) for l in f if l.strip()]
    logger.info("Existing examples: %d", len(existing))

    needed = args.target - len(existing)
    if needed <= 0:
        logger.info("Already have %d examples (target: %d). Nothing to do.", len(existing), args.target)
        return

    logger.info("Need to generate %d new examples to reach %d", needed, args.target)

    # Collect existing user prompts for dedup
    used_prompts = set()
    for ex in existing:
        for m in ex.get("messages", []):
            if m["role"] == "user":
                used_prompts.add(m["content"].lower().strip())

    # Load checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    new_examples = []
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            new_examples = [json.loads(l) for l in f if l.strip()]
        for ex in new_examples:
            for m in ex.get("messages", []):
                if m["role"] == "user":
                    used_prompts.add(m["content"].lower().strip())
        logger.info("Loaded %d examples from checkpoint", len(new_examples))

    remaining = needed - len(new_examples)
    if remaining <= 0:
        logger.info("Checkpoint has enough examples. Skipping generation.")
    else:
        # Scale category counts to remaining
        total_template = sum(c["count"] for c in CATEGORIES)
        scale = remaining / total_template

        for category in CATEGORIES:
            cat_count = max(1, round(category["count"] * scale))
            logger.info("Generating %d examples for category: %s", cat_count, category["category"])

            generated = 0
            attempts = 0
            max_attempts = cat_count * 3

            while generated < cat_count and attempts < max_attempts:
                attempts += 1
                try:
                    example = generate_example(category, args.endpoint, client, used_prompts)
                    if example:
                        new_examples.append(example)
                        generated += 1

                        # Checkpoint every 10
                        if len(new_examples) % 10 == 0:
                            with open(CHECKPOINT_FILE, "w") as f:
                                for ex in new_examples:
                                    f.write(json.dumps(ex) + "\n")
                            logger.info("Checkpoint saved: %d new examples total", len(new_examples))

                        # Rate limit
                        time.sleep(0.5)
                except Exception as e:
                    logger.error("Error generating example: %s", e)
                    time.sleep(2)

            logger.info("  Generated %d/%d for %s (%d attempts)", generated, cat_count, category["category"], attempts)

    # Final checkpoint
    with open(CHECKPOINT_FILE, "w") as f:
        for ex in new_examples:
            f.write(json.dumps(ex) + "\n")

    # Combine and write
    all_examples = existing + new_examples
    logger.info("Total examples: %d (existing: %d, new: %d)", len(all_examples), len(existing), len(new_examples))

    with open(args.output, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    logger.info("Written to %s", args.output)

    # Push to HF
    if not args.dry_run and args.hf_token:
        from datasets import Dataset
        ds = Dataset.from_list(all_examples)
        logger.info("Pushing %d examples to %s...", len(ds), args.hf_repo)
        ds.push_to_hub(args.hf_repo, split="train", private=True, token=args.hf_token)
        logger.info("Pushed to HuggingFace!")
    elif args.dry_run:
        logger.info("Dry run — not pushing to HF")


if __name__ == "__main__":
    main()
