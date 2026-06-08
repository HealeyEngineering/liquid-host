# Training Data Guide

This document describes how to create, update, and maintain the fine-tuning training data for Liquid Host's Aiera MCP tool-calling capability.

## Overview

Training data teaches the LFM2-24B-A2B model to use Aiera MCP tools for institutional financial research. Each example is a multi-turn conversation demonstrating tool selection, tool calling, and professional response generation.

## Data Format

Each example is a JSON object with a `messages` array. Messages use four roles: `user`, `assistant`, `tool`, and `system`.

```json
{
  "messages": [
    {"role": "user", "content": "Summarize the latest NFLX earnings call"},
    {"role": "assistant", "content": "<think>Need to find recent NFLX earnings events.</think>\nSearching for recent NFLX earnings events...\n<tool_call>\n{\"name\":\"find_events\",\"arguments\":{\"bloomberg_ticker\":\"NFLX:US\",\"event_type\":\"earnings\",\"start_date\":\"2025-01-01\",\"end_date\":\"2026-03-11\",\"page_size\":100,\"exclude_instructions\":true,\"self_identification\":\"aierachat\"}}\n</tool_call>"},
    {"role": "tool", "content": "{\"events\":[{\"event_id\":99120,\"title\":\"Netflix Inc. Q4 2025 Earnings Call\",\"event_date\":\"2026-01-21\",\"event_type\":\"earnings\",\"bloomberg_ticker\":\"NFLX:US\",\"status\":\"transcribed\"}]}"},
    {"role": "assistant", "content": "Netflix reported Q4 FY25 revenue of $10.8B USD, up 14% YoY, with 19.2M net subscriber additions [event_99120].\n\n### Earnings & Events\n\n..."}
  ]
}
```

### Message Roles

| Role        | Purpose                                                                                     |
|-------------|---------------------------------------------------------------------------------------------|
| `user`      | The user's question or follow-up                                                            |
| `assistant` | Model's response — either a tool-calling turn (with `<tool_call>` markup) or a final answer |
| `tool`      | Raw JSON result from an MCP tool call                                                       |
| `system`    | (Optional) System prompt — injected by the server at runtime                                |

### Assistant Message Types

**Tool-calling turns** contain thinking, a status message, and tool call markup:

```
<think>Brief reasoning (2-3 sentences max).</think>
Searching for NFLX earnings events...
<tool_call>
{"name":"find_events","arguments":{...}}
</tool_call>
```

**Final answer turns** contain the formatted response with citations (no `<tool_call>` markup).

## Style Requirements

All training data must follow these standards. These match the production system prompt and the reference implementation in `sample.py`.

### 1. Inline Citations

Every factual claim, quote, and data point MUST have a citation immediately after it.

**Format:** `[content_type_id]`

| Content Type      | Citation Format             | Example                     |
|-------------------|-----------------------------|-----------------------------|
| Event metadata    | `[event_XXXXX]`             | `[event_99120]`             |
| Transcript quotes | `[transcript_XXXXX]`        | `[transcript_99120]`        |
| SEC filings       | `[filing_XXXXX]`            | `[filing_45678]`            |
| Research reports  | `[research_XXXXX]`          | `[research_10945]`          |
| Company documents | `[company_doc_XXXXX]`       | `[company_doc_8901]`        |
| Third Bridge      | `[thirdbridge_event_XXXXX]` | `[thirdbridge_event_5432]`  |

**Rules:**
- Place citations immediately after each claim, not batched at paragraph end
- Multiple sources: `[transcript_123], [filing_456]`
- Prefer `[transcript_XXXXX]` for specific quotes/statements; use `[event_XXXXX]` only for general event metadata
- Only cite IDs that appear in the tool results — never fabricate
- Do NOT include a References/Sources section at the end

**Example:**
```
Revenue grew 15% year-over-year [transcript_123456], driven by subscriber additions [transcript_234567] and pricing improvements [filing_789012].
```

### 2. Professional Institutional Tone

Write for portfolio managers, buy-side analysts, and investment committees.

**DO:**
- Present analysis as definitive within available data
- Use precise financial terminology
- State "Data not available" when information is missing

**DO NOT:**
- Offer to search for more: "Would you like me to..."
- Invite document uploads
- Include disclaimers about data limitations
- Use casual language or first-person ("I'll look that up")
- Use strikethrough markup (`~~text~~`)

### 3. Response Structure

1. **Direct answer first** — 1-2 sentences with key figures, no header
2. **Organized sections** — Use markdown headers by content type (only include sections with data):
   - `### Research` or `### Research Insights`
   - `### Events` or `### Earnings & Events`
   - `### Filings`
   - `### Financials`, `### Company Documents`, `### News`, etc.
3. **Additional context** — Brief explanations, drivers, or background when helpful

**Simple queries** (listing events, indexes, etc.): Direct answer + table/list with citations is sufficient.

### 4. Data Presentation

- **Markdown:** `###` headers, tables, bullet points, **bold** key figures
- **Period labels:** "Q4 FY25" not just "Q4"
- **Quarterly data:** Most recent first (chronological descending)
- **Currency/units:** Always state them — "$10.8B USD", "€4.2B EUR"
- **Precision:** Revenue 1-2 decimals, EPS 2 decimals, margins 1-2 decimals
- **YoY comparisons:** Include where data allows
- **Table citations:** Only include a Sources column if citations exist for each row

### 5. Required Tool Call Parameters

ALL tool calls in training data must include these parameters:

| Parameter              | Value         | Applies To                                                      |
|------------------------|---------------|-----------------------------------------------------------------|
| `page_size`            | `100`         | All `find_*` tools, `get_upcoming_events`, `get_*_constituents` |
| `exclude_instructions` | `true`        | All Aiera tools                                                 |
| `self_identification`  | `"aierachat"` | All Aiera tools                                                 |

### 6. Tool Results

Tool result JSON should NOT contain the `instructions` array (this simulates `exclude_instructions=true` behavior in production).

### 7. Thinking Blocks

Keep `<think>` blocks brief — 2-3 short sentences maximum. State only what needs to be done, then act. No restating the question.

```
<think>Need to find NFLX earnings events and then get the transcript.</think>
```

## Creating New Training Examples

### Coverage Goals

Training data should cover all 41 Aiera MCP tools across these categories:

| Category           | Tools                                                                      | Example Queries                                                     |
|--------------------|----------------------------------------------------------------------------|---------------------------------------------------------------------|
| Events             | `find_events`, `get_event`, `get_upcoming_events`, `find_conferences`      | "Summarize NFLX latest earnings", "What conferences are coming up?" |
| Filings            | `find_filings`, `get_filing`                                               | "Show TSLA's latest 10-K risk factors"                              |
| Equities           | `find_equities`, `get_equity_summaries`                                    | "Look up Toyota's ticker", "Give me an AAPL overview"               |
| Financials         | `get_financials`, `get_ratios`, `get_kpis_and_segments`                    | "AAPL revenue last 4 quarters", "Compare AAPL vs MSFT margins"      |
| Transcripts        | `search_transcripts`                                                       | "What did NFLX CEO say about ad tier?"                              |
| Filings Search     | `search_filings`                                                           | "Find AI risk factors in MSFT 10-K"                                 |
| Company Docs       | `find_company_docs`, `get_company_doc`                                     | "AMZN logistics press releases"                                     |
| Research           | `find_research`, `get_research`                                            | "Goldman Sachs reports on AAPL"                                     |
| Third Bridge       | `find_third_bridge_events`, `get_third_bridge_event`, `search_thirdbridge` | "Expert opinions on NVDA vs AMD"                                    |
| Indexes/Watchlists | `get_available_indexes`, `get_index_constituents`, etc.                    | "List S&P 500 constituents"                                         |
| Metadata           | `get_sectors_and_subsectors`, `get_research_providers`, etc.               | "What research providers are available?"                            |
| Web Search         | `trusted_web_search`                                                       | "Latest NVDA news" (only after domain tools exhausted)              |

### Multi-Turn Examples

Include examples with multiple tool-calling rounds (6+ messages):
- First turn: user asks, model calls a discovery tool (e.g., `find_events`)
- Second turn: model uses the discovered ID to call a detail tool (e.g., `get_event`)
- Final turn: model synthesizes the results into a professional response

### Tool Call Patterns to Demonstrate

1. **Search over retrieval:** Use `search_transcripts` instead of multiple `get_event` calls
2. **Skip find_equities for known tickers:** Use `AAPL:US`, `MSFT:US`, etc. directly
3. **ID dependencies:** Show the chain: `find_events` → `get_event`, `find_filings` → `search_filings`
4. **Error recovery:** Broaden parameters, try alternative tools, state "Data not available"
5. **Multi-company comparisons:** Parallel tool calls, aligned periods, comparison tables

## Updating Training Data

### Automated Update Script

`scripts/update_training_data.py` transforms all examples to match the current style requirements. It uses Claude API to rewrite final assistant responses.

```bash
# Install dependency
pip install anthropic

# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Preview changes (5 examples)
python scripts/update_training_data.py \
  --hf-token $HF_TOKEN \
  --dry-run \
  --limit 5

# Full run, save locally without pushing
python scripts/update_training_data.py \
  --hf-token $HF_TOKEN \
  --dry-run

# Full run and push to HuggingFace
python scripts/update_training_data.py \
  --hf-token $HF_TOKEN
```

The script:
1. Adds `page_size`, `exclude_instructions`, `self_identification` to all tool calls
2. Strips `instructions` arrays from tool results
3. Extracts citation IDs from tool results (event_id, filing_id, etc.)
4. Sends each example to Claude Sonnet for response rewriting with citations and proper tone
5. Checkpoints every 10 examples to `/tmp/updated_training_data.json` (resumable)
6. Pushes to HuggingFace when not in `--dry-run` mode

### Manual Editing

The web-based training data editor at `/training` on any running Liquid Host instance allows browsing, editing, adding, and deleting individual examples, with sync back to HuggingFace Hub.

## Fine-Tuning

After updating training data, run a new fine-tuning job:

```bash
# Pull updated data to local JSONL
python -c "
from datasets import load_dataset
import json
ds = load_dataset('bryanhealey/my-aiera-finetune-v7-data', split='train', token='$HF_TOKEN')
with open('data/training/aiera_tools_v7.jsonl', 'w') as f:
    for row in ds:
        f.write(json.dumps({'messages': row['messages']}) + '\n')
"

# Launch remote fine-tune on A100
liquid-host finetune lfm2-24b-a2b data/training/aiera_tools_v7.jsonl \
  --remote \
  --backend a100-large \
  --quantize-4bit \
  --project-name my-aiera-finetune-v7 \
  --hf-token $HF_TOKEN \
  --epochs 2 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --max-seq-length 4096
```

### Recommended Hyperparameters (v7)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA rank | 32 | Increased from 16 — more capacity for diverse tool-calling patterns |
| LoRA alpha | 64 | Maintains alpha/rank ratio of 2 |
| Epochs | 2 | Reduced from 3 — larger dataset reduces overfitting risk |
| Batch size | 2 | Reduced to fit longer sequences in memory |
| Gradient accum | 8 | Effective batch size of 16 (same as before) |
| Learning rate | 1e-4 | Reduced from 2e-4 — gentler updates prevent format regression |
| Max seq length | 4096 | Increased from 2048 — accommodates richer multi-turn examples |
| Warmup ratio | 0.03 | Slightly reduced for faster convergence |

## Expanding Training Data

Use `scripts/expand_training_data.py` to generate new examples:

```bash
# Phase 1: Fix citations on existing examples
python scripts/expand_training_data.py --phase fix --input data/training/aiera_tools_v7.jsonl --hf-token $HF_TOKEN

# Phase 2: Generate new examples across all tool categories
python scripts/expand_training_data.py --phase generate --input data/training/aiera_tools_v7.jsonl --hf-token $HF_TOKEN

# Phase 3: Combine, validate, and push to HuggingFace
python scripts/expand_training_data.py --phase finalize --hf-token $HF_TOKEN

# Or run all phases at once
python scripts/expand_training_data.py --phase all --input data/training/aiera_tools_v7.jsonl --hf-token $HF_TOKEN
```

## Version History

| Version  | Examples  | Changes                                                                            |
|----------|-----------|------------------------------------------------------------------------------------|
| v4       | 286       | Initial dataset, 39 tools, basic style                                             |
| v5       | 208       | Refined examples, 41 tools (dev endpoint), cleaner format                          |
| v6       | 208       | Institutional style with inline citations, required tool params, professional tone |
| v7       | ~420      | 2X expansion, citation fixes, better tool coverage, tuned hyperparameters          |

## Reference

The target style is defined in `sample.py`, which contains the system prompt templates from the reference Aiera chat implementation. Key templates:

- `financial_react_prompt_template` — Tool-calling prompt (tool selection rules, ID dependencies, execution rules)
- `enhanced_financial_analyst_prompt` — Final answer prompt (citation standards, response structure, anti-hallucination rules)
