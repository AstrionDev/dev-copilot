# Verified Dev Copilot (LangChain + optional LangGraph)

## Overview
A developer-focused chatbot agent that answers technical questions using:
- DuckDuckGo web search
- StackOverflow search (StackExchange API)
- Official documentation discovery (site-biased search)
- Page fetching for grounded excerpts
- Human approval loop before finalizing

Designed as a portfolio-ready “agentic research” demo for Upwork profiles.

## What it Returns
JSON with:
- `answer`: final response (includes a `Sources:` section)
- `sources`: URLs found in the answer (filtered to fetched URLs)

## Project Layout
- `app/agent.py`: CLI entrypoint (basic engine + optional LangGraph engine).
- `app/tools.py`: Search + fetch tools (DuckDuckGo, StackOverflow, docs search).
- `examples/questions/`: Sample inputs.

## Run
```bash
python app/agent.py --question-file examples/questions/async_fastapi.txt
```

Optional:
```bash
python app/agent.py --engine basic --question "How do I fix pip SSL errors on macOS?"
```

## Interactive CLI
```bash
python app/agent.py --interactive
```

## Notes
- DuckDuckGo + StackOverflow require outbound network access.
- The default LLM config targets a local OpenAI-compatible endpoint:
  `LM_STUDIO_BASE_URL="http://localhost:1234/v1"` (e.g., LM Studio).
- Set `LM_STUDIO_API_KEY` if your local endpoint requires a key (default: `lm-studio`).
- Override the model with `LM_STUDIO_MODEL` (default: `qwen/qwen-3-1.7b`).
- `.env` is supported if you install `python-dotenv`.
- The agent asks for human input when it needs missing details (OS, versions, errors).
- The agent asks for approval only when it wants to use web/tools for research.
- Install `questionary` for nicer interactive prompts (arrow-key selection + input highlight).
- Install `rich` for panelled markdown output in interactive mode.
