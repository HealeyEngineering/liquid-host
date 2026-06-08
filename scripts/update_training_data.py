"""Update training data to match institutional financial analyst style.

Transforms all 208 training examples to include:
- Inline citations ([transcript_XXXXX], [event_XXXXX], [filing_XXXXX], etc.)
- Professional institutional tone (no "Would you like...", no offers)
- Proper response structure (direct answer first, organized sections)
- Required tool parameters (page_size=100, exclude_instructions=true, self_identification="aierachat")
- Period labeling (Q3 FY24, not just Q3), currency/units, YoY comparisons

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/update_training_data.py --hf-token <token> --dry-run
    python scripts/update_training_data.py --hf-token <token>
"""

import argparse
import json
import logging
import os
import re
import time
from copy import deepcopy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("update_training_data")

HF_REPO = "bryanhealey/my-aiera-finetune-v6-data"

# ── Tool parameter fixes ────────────────────────────────────────────

REQUIRED_TOOL_PARAMS = {
    "page_size": 100,
    "exclude_instructions": True,
    "self_identification": "aierachat",
}

# Tools that accept these params (based on the Aiera MCP schema)
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
    "get_research_country_codes", "trusted_web_search", "ping", "debug_auth",
}


def fix_tool_call_params(content: str) -> str:
    """Add required parameters to tool call JSON in assistant messages."""
    def replace_tool_call(match):
        full_match = match.group(0)
        json_str = match.group(1)
        try:
            call = json.loads(json_str)
        except json.JSONDecodeError:
            return full_match

        name = call.get("name", "")
        args = call.get("arguments", {})

        if name in TOOLS_WITH_PAGE_SIZE and "page_size" not in args:
            args["page_size"] = REQUIRED_TOOL_PARAMS["page_size"]

        if name in TOOLS_WITH_INSTRUCTIONS_PARAMS:
            if "exclude_instructions" not in args:
                args["exclude_instructions"] = REQUIRED_TOOL_PARAMS["exclude_instructions"]
            if "self_identification" not in args:
                args["self_identification"] = REQUIRED_TOOL_PARAMS["self_identification"]

        call["arguments"] = args
        return f"<tool_call>\n{json.dumps(call)}\n</tool_call>"

    return re.sub(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
        replace_tool_call,
        content,
        flags=re.DOTALL,
    )


# ── Citation extraction ──────────────────────────────────────────────

def extract_citations_from_tool_result(tool_content: str) -> dict:
    """Parse tool result JSON and extract all referenceable IDs."""
    citations = {}
    try:
        data = json.loads(tool_content)
    except (json.JSONDecodeError, TypeError):
        return citations

    def walk(obj, path=""):
        if isinstance(obj, dict):
            # Look for ID fields
            for key in ["event_id", "filing_id", "research_id", "company_doc_id",
                        "thirdbridge_event_id", "equity_id", "index_id",
                        "watchlist_id"]:
                if key in obj:
                    # Map to citation type
                    ctype = key.replace("_id", "")
                    cid = str(obj[key])
                    citations[f"{ctype}_{cid}"] = obj
            # Check for transcript sections
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
    return citations


def build_citations_summary(messages: list[dict]) -> str:
    """Build a summary of available citations from all tool results in the conversation."""
    all_citations = {}
    for msg in messages:
        if msg["role"] == "tool":
            cits = extract_citations_from_tool_result(msg["content"])
            all_citations.update(cits)

    if not all_citations:
        return "No citations available."

    lines = []
    for marker, data in all_citations.items():
        # Build a short description
        title = data.get("title", data.get("name", data.get("company_name", "")))
        date = data.get("event_date", data.get("published_date", data.get("filing_date", "")))
        desc = f"{title}" + (f" ({date})" if date else "")
        lines.append(f"  [{marker}] — {desc}")

    return "\n".join(lines)


# ── Response rewriting via Claude API ────────────────────────────────

REWRITE_SYSTEM_PROMPT = """You are rewriting training data for a financial AI model. You will receive:
1. The original user question
2. Tool results (raw JSON) that were retrieved
3. The current assistant response that needs to be rewritten
4. Available citation markers extracted from the tool results

Your job is to rewrite ONLY the final assistant response to match the institutional financial analyst style described below. Do NOT change tool-calling assistant messages (those with <tool_call> tags).

## Style Requirements

1. **Professional institutional tone** — Write for portfolio managers and buy-side analysts. No casual language.

2. **Direct answer first** — Start with 1-2 sentences containing the key answer/numbers. No header on this opening.

3. **Inline citations** — EVERY factual claim, quote, and data point MUST have a citation immediately after it.
   - Format: [content_type_id] (e.g., [event_98432], [transcript_99120], [filing_45678], [research_10945])
   - Place citations right after the claim, not batched at paragraph end
   - Multiple sources: [transcript_123], [filing_456]
   - ONLY use citation markers from the AVAILABLE CITATIONS list provided

4. **Organized sections** — After the direct answer, use markdown headers for detail:
   - ### Events / ### Earnings & Events (for earnings calls, transcripts)
   - ### Research / ### Research Insights (for analyst reports)
   - ### Filings (for SEC filings)
   - ### Financials (for financial statements)
   - Only include sections that have data. Skip empty sections.

5. **Data presentation**:
   - Use markdown tables, bullet points, **bold** key figures
   - Label periods clearly: "Q4 FY25" not just "Q4"
   - Include YoY comparisons where data allows
   - Always state currency and units: "$10.8B USD"
   - Revenue: 1-2 decimal places, EPS: 2 decimal places, margins: 1-2 decimal places

6. **DO NOT**:
   - Offer to search for more ("Would you like me to...")
   - Invite document uploads
   - Include disclaimers about data limitations
   - Use strikethrough markup
   - Add a References/Sources section at the end
   - Comment on citation availability
   - Include "I'll", "Let me", or first-person language in the final response

7. **Simple queries** (like listing events or indexes): Keep it concise — direct answer + table/list with citations. No need for elaborate sections.

8. **Transcript/quote citations**: Prefer [transcript_XXXXX] for specific quotes and statements. Use [event_XXXXX] only for general event metadata (date, title, participants).

Return ONLY the rewritten response text. No preamble, no explanation."""


def rewrite_response_with_claude(
    user_question: str,
    tool_results: list[str],
    current_response: str,
    available_citations: str,
    client,
) -> str:
    """Use Claude to rewrite a training response with citations and proper style."""
    user_prompt = f"""## User Question
{user_question}

## Tool Results
{chr(10).join(f'--- Tool Result {i+1} ---{chr(10)}{tr[:3000]}' for i, tr in enumerate(tool_results))}

## Available Citations
{available_citations}

## Current Response (to rewrite)
{current_response}

Rewrite the above response following the style requirements. Return ONLY the rewritten text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=REWRITE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


# ── Main transformation ──────────────────────────────────────────────

def transform_example(messages: list[dict], client) -> list[dict]:
    """Transform a single training example."""
    result = deepcopy(messages)

    # 1. Fix tool call parameters in all assistant messages
    for msg in result:
        if msg["role"] == "assistant" and "<tool_call>" in msg["content"]:
            msg["content"] = fix_tool_call_params(msg["content"])

    # 2. Strip "instructions" from tool results (simulate exclude_instructions=true)
    for msg in result:
        if msg["role"] == "tool":
            try:
                data = json.loads(msg["content"])
                if isinstance(data, dict) and "instructions" in data:
                    del data["instructions"]
                    msg["content"] = json.dumps(data)
            except (json.JSONDecodeError, TypeError):
                pass

    # 3. Build available citations
    available_citations = build_citations_summary(result)

    # 4. Collect context for rewriting
    user_questions = [m["content"] for m in result if m["role"] == "user"]
    tool_results = [m["content"] for m in result if m["role"] == "tool"]

    # 5. Find the final assistant response (the one without <tool_call>)
    final_idx = None
    for i in range(len(result) - 1, -1, -1):
        if result[i]["role"] == "assistant" and "<tool_call>" not in result[i]["content"]:
            final_idx = i
            break

    if final_idx is None:
        logger.warning("No final assistant response found, skipping rewrite")
        return result

    # 6. Rewrite the final response
    current_response = result[final_idx]["content"]
    full_question = " → ".join(user_questions)

    rewritten = rewrite_response_with_claude(
        user_question=full_question,
        tool_results=tool_results,
        current_response=current_response,
        available_citations=available_citations,
        client=client,
    )

    result[final_idx]["content"] = rewritten
    return result


def main():
    parser = argparse.ArgumentParser(description="Update training data to match institutional style")
    parser.add_argument("--hf-token", required=True, help="HuggingFace token")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without pushing")
    parser.add_argument("--limit", type=int, default=0, help="Process only N examples (0=all)")
    parser.add_argument("--start", type=int, default=0, help="Start from example N")
    parser.add_argument("--output", type=str, default="/tmp/updated_training_data.json", help="Save intermediate output")
    args = parser.parse_args()

    # Check for Anthropic API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable required")
        return

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    # Load dataset
    from datasets import load_dataset, Dataset
    logger.info("Loading dataset from %s...", HF_REPO)
    ds = load_dataset(HF_REPO, split="train", token=args.hf_token)
    logger.info("Loaded %d examples", len(ds))

    # Check for existing progress
    updated_examples = []
    if os.path.exists(args.output):
        with open(args.output) as f:
            updated_examples = json.load(f)
        logger.info("Resuming from checkpoint: %d examples already processed", len(updated_examples))

    start = args.start if args.start else len(updated_examples)
    end = min(start + args.limit, len(ds)) if args.limit else len(ds)

    # If we're resuming but haven't reached start yet, fill in from checkpoint
    if len(updated_examples) < start:
        # Backfill with originals (shouldn't happen normally)
        while len(updated_examples) < start:
            updated_examples.append(ds[len(updated_examples)]["messages"])

    logger.info("Processing examples %d to %d", start, end - 1)

    for i in range(start, end):
        messages = ds[i]["messages"]
        user_preview = next((m["content"][:80] for m in messages if m["role"] == "user"), "?")

        logger.info("Processing example %d/%d: %s", i + 1, len(ds), user_preview)

        try:
            transformed = transform_example(messages, client)
            updated_examples.append(transformed)

            # Show diff preview
            orig_final = [m for m in messages if m["role"] == "assistant" and "<tool_call>" not in m["content"]]
            new_final = [m for m in transformed if m["role"] == "assistant" and "<tool_call>" not in m["content"]]
            if orig_final and new_final:
                logger.info("  Original: %s...", orig_final[-1]["content"][:120])
                logger.info("  Rewritten: %s...", new_final[-1]["content"][:120])

        except Exception as e:
            logger.error("  Failed: %s — keeping original", e)
            updated_examples.append(messages)

        # Save checkpoint every 10 examples
        if (i + 1) % 10 == 0:
            with open(args.output, "w") as f:
                json.dump(updated_examples, f, indent=2)
            logger.info("Checkpoint saved: %d examples", len(updated_examples))

        # Rate limiting
        time.sleep(0.5)

    # Final save
    with open(args.output, "w") as f:
        json.dump(updated_examples, f, indent=2)
    logger.info("Saved %d updated examples to %s", len(updated_examples), args.output)

    if args.dry_run:
        logger.info("DRY RUN — not pushing to HuggingFace")
        logger.info("Review the output at %s", args.output)
        return

    # Push to HuggingFace
    logger.info("Pushing updated dataset to %s...", HF_REPO)
    records = [{"messages": msgs} for msgs in updated_examples]
    new_ds = Dataset.from_list(records)
    new_ds.push_to_hub(HF_REPO, split="train", private=False, token=args.hf_token)
    logger.info("Done! %d examples pushed to %s", len(updated_examples), HF_REPO)


if __name__ == "__main__":
    main()
