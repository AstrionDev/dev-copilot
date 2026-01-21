from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

from pydantic import SecretStr

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass
from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from tools import duckduckgo_search, fetch_url, official_docs_search, stackoverflow_search

try:
    from langgraph.graph import END, StateGraph

    _HAS_LANGGRAPH = True
except Exception:
    END = None  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]
    _HAS_LANGGRAPH = False

_questionary: Any = None
try:
    import questionary as _questionary

    _HAS_QUESTIONARY = True
except Exception:
    _HAS_QUESTIONARY = False


SYSTEM_PROMPT = """You are Verified Dev Copilot, a developer-focused assistant.

You will be given:
- a user question
- web search results
- fetched excerpts from official docs and other pages
- StackOverflow matches

Requirements:
- Answer with concrete steps and minimal speculation.
- If there are multiple plausible interpretations, ask 1-2 clarifying questions.
- Prefer official documentation; use StackOverflow as supporting evidence.
- Never invent citations. Only cite URLs that appear in the provided context.
- End with a short Sources list containing the exact URLs you relied on.
"""

ANALYSIS_PROMPT = """You are a planning module for a developer assistant.

Decide whether you need more info from the user before answering, and whether to use web research tools.

Return ONLY valid JSON with this schema:
{
  "clarifying_questions": ["..."],
  "do_research": true/false,
  "search_queries": ["..."]
}

Rules:
- Ask clarifying questions only if required to avoid wrong advice (versions, OS, error messages, stack trace, minimal repro, framework versions).
- If the question is general, you can set do_research=true with 1-3 targeted queries.
- Keep clarifying_questions to <= 3. Keep search_queries to <= 3.
- No extra keys, no markdown.
"""


@dataclass
class AgentResponse:
    answer: str
    sources: list[str]


def _dedupe_urls(items: list[dict[str, object]]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for item in items:
        url = item.get("url")
        if not isinstance(url, str) or not url:
            continue
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def _build_context(
    question: str,
    so_results: list[dict[str, object]],
    docs_results: list[dict[str, object]],
    web_results: list[dict[str, object]],
    fetched_pages: list[dict[str, str]],
    max_page_excerpt_chars: int = 900,
) -> str:
    def fmt_results(label: str, results: list[dict[str, object]]) -> str:
        lines = [f"## {label}"]
        for idx, item in enumerate(results, start=1):
            title = item.get("title", "")
            url = item.get("url", "")
            snippet = item.get("snippet", "")
            answer_excerpt = item.get("answer_excerpt", "")
            lines.append(f"{idx}. {title}".strip())
            lines.append(f"   URL: {url}".strip())
            if snippet:
                lines.append(f"   Snippet: {snippet}".strip())
            if answer_excerpt:
                lines.append(f"   Answer excerpt: {answer_excerpt}".strip())
        return "\n".join(lines)

    def fmt_pages(pages: list[dict[str, str]]) -> str:
        lines = ["## Fetched Excerpts"]
        for idx, page in enumerate(pages, start=1):
            url = page.get("url", "")
            content_type = page.get("content_type", "")
            text = page.get("text", "")
            lines.append(f"{idx}. URL: {url}")
            if content_type:
                lines.append(f"   Content-Type: {content_type}")
            if text:
                excerpt = text[:max_page_excerpt_chars]
                if len(text) > max_page_excerpt_chars:
                    excerpt += "..."
                lines.append(f"   Excerpt: {excerpt}")
        return "\n".join(lines)

    parts = [
        f"# Question\n{question}",
        fmt_results("Official Docs Search", docs_results),
        fmt_results("StackOverflow Search", so_results),
        fmt_results("Web Search", web_results),
        fmt_pages(fetched_pages),
    ]
    return "\n\n".join(parts).strip()


def _extract_sources(text: str, allowed_urls: list[str]) -> list[str]:
    sources: list[str] = []
    for url in allowed_urls:
        if url and url in text and url not in sources:
            sources.append(url)
    return sources


def _clean_answer(text: str) -> str:
    if not text:
        return text
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", text)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned.strip()


def _prompt_choice(prompt: str, choices: list[str], default_index: int = 0) -> str:
    if _HAS_QUESTIONARY:
        assert _questionary is not None
        default_choice = choices[default_index] if choices else None
        answer = _questionary.select(
            prompt, choices=choices, default=default_choice
        ).ask()
        return answer if isinstance(answer, str) and answer in choices else choices[default_index]
    while True:
        print(prompt)
        for idx, option in enumerate(choices, start=1):
            marker = "*" if idx - 1 == default_index else " "
            print(f" {marker} {idx}. {option}")
        raw = input(f"Select [1-{len(choices)}]: ").strip()
        if not raw:
            return choices[default_index]
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(choices):
                return choices[index]
        print("Please enter a valid number.")


def _prompt_yes_no(prompt: str, default_no: bool = True) -> bool:
    choices = ["Use tools", "Skip tools"]
    default_index = 1 if default_no else 0
    choice = _prompt_choice(prompt, choices, default_index=default_index)
    return choice == "Use tools"


def _collect_human_info(questions: list[str]) -> dict[str, str]:
    answers: dict[str, str] = {}
    for question in questions:
        if _HAS_QUESTIONARY:
            assert _questionary is not None
            answer = _questionary.text(
                question,
                validate=lambda text: True if text.strip() else "Please enter a response.",
            ).ask()
            if answer is None:
                raise KeyboardInterrupt
            answers[question] = answer.strip()
        else:
            while True:
                answer = input(f"{question}\n> ").strip()
                if answer:
                    answers[question] = answer
                    break
                print("Please enter a response (or Ctrl+C to exit).")
    return answers


def _build_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("LM_STUDIO_MODEL", "qwen/qwen-3-1.7b"),
        base_url=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
        api_key=SecretStr(os.getenv("LM_STUDIO_API_KEY", "lm-studio")),
        temperature=0.1,
        max_tokens=600,  # pyright: ignore[reportCallIssue]
    )


def _safe_parse_json(text: str) -> dict[str, object]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {}
        return {}


def _analyze_question(model: ChatOpenAI, question: str, extra_context: str = "") -> dict[str, object]:
    content = model.invoke(
        [
            {"role": "system", "content": ANALYSIS_PROMPT},
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nExtra context:\n{extra_context}".strip(),
            },
        ]
    ).content
    payload = content if isinstance(content, str) else str(content)
    data = _safe_parse_json(payload)

    clarifying = data.get("clarifying_questions", [])
    if not isinstance(clarifying, list):
        clarifying = []
    clarifying_questions = [str(x).strip() for x in clarifying if str(x).strip()][:3]

    do_research = bool(data.get("do_research", True))

    queries = data.get("search_queries", [])
    if not isinstance(queries, list):
        queries = []
    search_queries = [str(x).strip() for x in queries if str(x).strip()][:3]
    if not search_queries:
        search_queries = [question]

    return {
        "clarifying_questions": clarifying_questions,
        "do_research": do_research,
        "search_queries": search_queries,
    }


def _draft_answer(model: ChatOpenAI, context: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Use the following context to answer.\n\n"
                f"{context}\n\n"
                "Do not output <think> or hidden reasoning.\n"
                "Return a complete answer followed by:\n"
                "Sources:\n"
                "- <url>\n"
                "- <url>\n"
            ),
        },
    ]
    content = model.invoke(messages).content
    return content if isinstance(content, str) else str(content)


def run_agent_basic(
    question: str,
    max_sources: int = 3,
    max_page_chars: int = 2000,
    max_context_chars: int = 12000,
    require_approval: bool = True,
) -> dict[str, object]:
    model = _build_model()

    analysis = _analyze_question(model, question)
    clarifying_questions = analysis.get("clarifying_questions", [])
    clarifying_answers: dict[str, str] = {}
    if isinstance(clarifying_questions, list) and clarifying_questions:
        clarifying_answers = _collect_human_info([str(q) for q in clarifying_questions])

    question_with_context = question
    if clarifying_answers:
        details = "\n".join([f"- {k} {v}" for k, v in clarifying_answers.items()])
        question_with_context = f"{question}\n\nUser-provided details:\n{details}"

    do_research = bool(analysis.get("do_research", True))
    tools_approved = True
    if do_research and require_approval:
        tools_approved = _prompt_yes_no(
            "Use web/tools (DuckDuckGo + StackOverflow + official docs fetch) for this answer?"
        )

    so_results: list[dict[str, object]] = []
    docs_results: list[dict[str, object]] = []
    web_results: list[dict[str, object]] = []
    allowed_urls: list[str] = []
    fetched_pages: list[dict[str, str]] = []

    if do_research and tools_approved:
        search_queries = analysis.get("search_queries", [])
        primary_query = question_with_context
        if isinstance(search_queries, list) and search_queries and isinstance(search_queries[0], str):
            primary_query = search_queries[0].strip() or primary_query

        so_results = stackoverflow_search.invoke({"query": primary_query, "max_results": 3})
        docs_results = official_docs_search.invoke({"query": primary_query, "max_results": 5})
        web_results = duckduckgo_search.invoke({"query": primary_query, "max_results": 5})

        candidate_urls: list[str] = []
        candidate_urls.extend(_dedupe_urls(docs_results))
        candidate_urls.extend(_dedupe_urls(so_results))
        candidate_urls.extend(_dedupe_urls(web_results))
        allowed_urls = candidate_urls[: max(1, max_sources)]

        for url in allowed_urls:
            fetched_pages.append(fetch_url.invoke({"url": url, "max_chars": max_page_chars}))

    context = _build_context(
        question=question_with_context,
        so_results=so_results,
        docs_results=docs_results,
        web_results=web_results,
        fetched_pages=fetched_pages,
        max_page_excerpt_chars=max(300, max_page_chars // 2),
    )
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n\n[Context truncated]"
    draft = _draft_answer(model, context)

    response = AgentResponse(
        answer=_clean_answer(draft),
        sources=_extract_sources(draft, allowed_urls),
    )
    return {"answer": response.answer, "sources": response.sources}


def run_agent_langgraph(
    question: str,
    max_sources: int = 3,
    max_page_chars: int = 2000,
    max_context_chars: int = 12000,
    require_approval: bool = True,
) -> dict[str, object]:
    if not _HAS_LANGGRAPH:
        raise RuntimeError(
            "LangGraph is not installed. Install it (pip install langgraph) "
            "or run with --engine basic."
        )

    model = _build_model()

    def analyze_node(state: dict[str, object]) -> dict[str, object]:
        next_state = dict(state)
        q = str(next_state.get("question", "")).strip()
        analysis = _analyze_question(model, q, extra_context=str(next_state.get("extra_context", "")))
        next_state.update({"analysis": analysis})
        return next_state

    def need_clarification(state: dict[str, object]) -> str:
        analysis = state.get("analysis", {})
        if not isinstance(analysis, dict):
            return "approve_tools"
        questions = analysis.get("clarifying_questions", [])
        if isinstance(questions, list) and any(str(q).strip() for q in questions):
            return "clarify"
        return "approve_tools"

    def clarify_node(state: dict[str, object]) -> dict[str, object]:
        next_state = dict(state)
        analysis = next_state.get("analysis", {})
        questions: list[str] = []
        if isinstance(analysis, dict):
            raw = analysis.get("clarifying_questions", [])
            if isinstance(raw, list):
                questions = [str(q).strip() for q in raw if str(q).strip()][:3]
        if questions:
            answers = _collect_human_info(questions)
            details = "\n".join([f"- {k} {v}" for k, v in answers.items()])
            q = str(next_state.get("question", "")).strip()
            if details:
                q = f"{q}\n\nUser-provided details:\n{details}"
            next_state.update({"question": q, "extra_context": details})
        # Clear questions so we don't re-ask in a loop.
        next_state.update({"analysis": {"clarifying_questions": [], "do_research": True, "search_queries": []}})
        return next_state

    def approve_tools_node(state: dict[str, object]) -> dict[str, object]:
        next_state = dict(state)
        analysis = next_state.get("analysis", {})
        do_research = True
        if isinstance(analysis, dict):
            do_research = bool(analysis.get("do_research", True))
        if not do_research:
            next_state.update({"tools_approved": False})
            return next_state
        if not require_approval:
            next_state.update({"tools_approved": True})
            return next_state
        next_state.update(
            {
                "tools_approved": _prompt_yes_no(
                    "Use web/tools (DuckDuckGo + StackOverflow + official docs fetch) for this answer?"
                )
            }
        )
        return next_state

    def route_after_approval(state: dict[str, object]) -> str:
        return "research" if state.get("tools_approved") else "draft"

    def research_node(state: dict[str, object]) -> dict[str, object]:
        next_state = dict(state)
        q = state["question"]
        assert isinstance(q, str)
        analysis = next_state.get("analysis", {})
        query = q
        if isinstance(analysis, dict):
            queries = analysis.get("search_queries", [])
            if isinstance(queries, list) and queries and isinstance(queries[0], str) and queries[0].strip():
                query = queries[0].strip()
        so = stackoverflow_search.invoke({"query": query, "max_results": 3})
        docs = official_docs_search.invoke({"query": query, "max_results": 5})
        web = duckduckgo_search.invoke({"query": query, "max_results": 5})
        urls: list[str] = []
        urls.extend(_dedupe_urls(docs))
        urls.extend(_dedupe_urls(so))
        urls.extend(_dedupe_urls(web))
        allowed = urls[: max(1, max_sources)]
        next_state.update(
            {
                "question": q,
                "so_results": so,
                "docs_results": docs,
                "web_results": web,
                "allowed_urls": allowed,
            }
        )
        return next_state

    def fetch_node(state: dict[str, object]) -> dict[str, object]:
        next_state = dict(state)
        urls = state.get("allowed_urls", [])
        if not isinstance(urls, list):
            urls = []
        pages: list[dict[str, str]] = []
        for url in urls:
            if isinstance(url, str) and url:
                pages.append(fetch_url.invoke({"url": url, "max_chars": max_page_chars}))
        next_state.update({"fetched_pages": pages})
        return next_state

    def draft_node(state: dict[str, object]) -> dict[str, object]:
        next_state = dict(state)
        q = state.get("question")
        if not isinstance(q, str) or not q:
            raise KeyError(
                "Missing 'question' in LangGraph state. "
                "This usually indicates a state merge/overwrite issue."
            )
        context = _build_context(
            question=q,
            so_results=state.get("so_results", []),  # type: ignore[arg-type]
            docs_results=state.get("docs_results", []),  # type: ignore[arg-type]
            web_results=state.get("web_results", []),  # type: ignore[arg-type]
            fetched_pages=state.get("fetched_pages", []),  # type: ignore[arg-type]
            max_page_excerpt_chars=max(300, max_page_chars // 2),
        )
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "\n\n[Context truncated]"
        draft = _draft_answer(model, context)
        next_state.update({"question": q, "context": context, "draft": draft})
        return next_state

    graph = StateGraph(dict)
    graph.add_node("analyze", analyze_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("approve_tools", approve_tools_node)
    graph.add_node("research", research_node)
    graph.add_node("fetch", fetch_node)
    graph.add_node("draft", draft_node)
    graph.set_entry_point("analyze")
    graph.add_conditional_edges("analyze", need_clarification)
    graph.add_edge("clarify", "approve_tools")
    graph.add_conditional_edges("approve_tools", route_after_approval)
    graph.add_edge("research", "fetch")
    graph.add_edge("fetch", "draft")
    graph.add_edge("draft", END)

    final_state = graph.compile().invoke({"question": question})
    answer = _clean_answer(str(final_state.get("draft", "")))
    urls = final_state.get("allowed_urls", [])
    allowed_urls: list[str] = []
    if isinstance(urls, list):
        allowed_urls = [u for u in urls if isinstance(u, str)]
    return {"answer": answer, "sources": _extract_sources(answer, allowed_urls)}


def _render_markdown(text: str) -> None:
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel

        console = Console()
        console.print(Panel(Markdown(text), title="Answer", border_style="green"))
    except Exception:
        print(text)


def _run_interactive(engine: str, max_sources: int, max_page_chars: int, max_context_chars: int) -> None:
    print("Verified Dev Copilot (interactive). Type '/exit' to quit, '/help' for commands.")
    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return
        if not prompt:
            continue
        if prompt.lower() in {"/exit", "/quit"}:
            print("Bye.")
            return
        if prompt == "/help":
            print("Commands: /help, /exit, /engine basic|langgraph")
            continue
        if prompt.startswith("/engine"):
            parts = prompt.split()
            if len(parts) == 2 and parts[1] in {"basic", "langgraph"}:
                engine = parts[1]
                print(f"Engine set to: {engine}")
            else:
                print("Usage: /engine basic|langgraph")
            continue

        if engine == "langgraph":
            result = run_agent_langgraph(
                question=prompt,
                max_sources=max(1, max_sources),
                max_page_chars=max(500, max_page_chars),
                max_context_chars=max(4000, max_context_chars),
                require_approval=True,
            )
        else:
            result = run_agent_basic(
                question=prompt,
                max_sources=max(1, max_sources),
                max_page_chars=max(500, max_page_chars),
                max_context_chars=max(4000, max_context_chars),
                require_approval=True,
            )
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        if isinstance(answer, str):
            _render_markdown(answer)
        if isinstance(sources, list) and sources:
            try:
                from rich.console import Console
                from rich.panel import Panel

                console = Console()
                source_text = "\n".join(f"- {src}" for src in sources)
                console.print(Panel(source_text, title="Sources", border_style="cyan"))
            except Exception:
                print("\nSources:")
                for src in sources:
                    print(f"- {src}")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--question", help="User question")
    group.add_argument("--question-file", help="Path to a text file with the question")
    group.add_argument("--interactive", action="store_true", help="Start interactive CLI")
    parser.add_argument("--max-sources", type=int, default=3)
    parser.add_argument("--max-page-chars", type=int, default=2000)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument(
        "--engine",
        choices=("langgraph", "basic"),
        default="langgraph" if _HAS_LANGGRAPH else "basic",
        help="Execution engine (langgraph recommended)",
    )
    parser.add_argument("--no-approval", action="store_true", help="Skip human approval loop")
    args = parser.parse_args()

    if args.interactive:
        _run_interactive(
            engine=args.engine,
            max_sources=args.max_sources,
            max_page_chars=args.max_page_chars,
            max_context_chars=args.max_context_chars,
        )
        return

    if args.question_file:
        question = open(args.question_file, encoding="utf-8").read().strip()
    else:
        question = args.question.strip()

    if args.engine == "langgraph":
        result = run_agent_langgraph(
            question=question,
            max_sources=max(1, args.max_sources),
            max_page_chars=max(500, args.max_page_chars),
            max_context_chars=max(4000, args.max_context_chars),
            require_approval=not args.no_approval,
        )
    else:
        result = run_agent_basic(
            question=question,
            max_sources=max(1, args.max_sources),
            max_page_chars=max(500, args.max_page_chars),
            max_context_chars=max(4000, args.max_context_chars),
            require_approval=not args.no_approval,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
