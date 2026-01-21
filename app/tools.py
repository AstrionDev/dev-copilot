from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from typing import Any

from langchain.tools import tool

DEFAULT_SEARCH_TIMEOUT = 15
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; VerifiedDevCopilot/1.0)"


def _strip_tags(raw_html: str) -> str:
    raw_html = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", raw_html)
    raw_html = re.sub(r"(?is)<[^>]+>", " ", raw_html)
    raw_html = html.unescape(raw_html)
    raw_html = re.sub(r"\\s+", " ", raw_html).strip()
    return raw_html


def _decode_ddg_redirect(url: str) -> str:
    if not url.startswith("//duckduckgo.com/l/?"):
        return url
    parsed = urllib.parse.urlparse("https:" + url)
    params = urllib.parse.parse_qs(parsed.query)
    uddg = params.get("uddg", [])
    if not uddg:
        return url
    return urllib.parse.unquote(uddg[0])


def _http_get(url: str, timeout: int = DEFAULT_SEARCH_TIMEOUT) -> tuple[str, str]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        payload = response.read()
    text = payload.decode("utf-8", errors="ignore")
    return text, content_type


@tool
def duckduckgo_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search the web via DuckDuckGo HTML and return title/url/snippet results."""
    if max_results < 1:
        return []
    params = urllib.parse.urlencode({"q": query})
    url = f"https://duckduckgo.com/html/?{params}"
    payload, _ = _http_get(url)

    results: list[dict[str, str]] = []
    for match in re.finditer(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        payload,
        re.IGNORECASE | re.DOTALL,
    ):
        raw_url = html.unescape(match.group(1))
        resolved_url = _decode_ddg_redirect(raw_url)
        title = _strip_tags(match.group(2))
        if not title:
            continue

        snippet = ""
        after = payload[match.end() : match.end() + 1800]
        snippet_match = re.search(
            r'class="result__snippet"[^>]*>(.*?)</',
            after,
            re.IGNORECASE | re.DOTALL,
        )
        if snippet_match:
            snippet = _strip_tags(snippet_match.group(1))[:280]

        results.append({"title": title, "url": resolved_url, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results


def _guess_official_doc_sites(query: str) -> list[str]:
    q = query.lower()
    sites: list[str] = []
    mapping = [
        (["python"], "docs.python.org"),
        (["javascript", "js", "html", "css"], "developer.mozilla.org"),
        (["react"], "react.dev"),
        (["node", "nodejs"], "nodejs.org"),
        (["kubernetes", "k8s"], "kubernetes.io"),
        (["terraform"], "developer.hashicorp.com"),
        (["aws"], "docs.aws.amazon.com"),
        (["gcp", "google cloud"], "cloud.google.com"),
        (["azure"], "learn.microsoft.com"),
        (["postgres", "postgresql"], "www.postgresql.org"),
        (["docker"], "docs.docker.com"),
        (["fastapi"], "fastapi.tiangolo.com"),
    ]
    for needles, site in mapping:
        if any(token in q for token in needles):
            sites.append(site)
    if not sites:
        sites = ["developer.mozilla.org", "docs.python.org", "learn.microsoft.com"]
    return sites[:3]


@tool
def official_docs_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search likely official documentation sites for a query."""
    sites = _guess_official_doc_sites(query)
    merged: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    per_site = max(2, max_results // max(1, len(sites)))
    for site in sites:
        results = duckduckgo_search.invoke(
            {"query": f"site:{site} {query}", "max_results": per_site}
        )
        for item in results:
            url = item.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            merged.append(item)
            if len(merged) >= max_results:
                return merged
    return merged


@tool
def stackoverflow_search(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    """Search StackOverflow via the StackExchange API and return top Q/A matches."""
    if max_results < 1:
        return []
    params = {
        "order": "desc",
        "sort": "relevance",
        "q": query,
        "site": "stackoverflow",
        "pagesize": str(max_results),
        "accepted": "True",
    }
    url = "https://api.stackexchange.com/2.3/search/advanced?" + urllib.parse.urlencode(
        params
    )
    payload, _ = _http_get(url)
    data = json.loads(payload)
    questions = data.get("items", [])
    if not questions:
        return []

    ids = [str(q.get("question_id")) for q in questions if q.get("question_id")]
    answers_by_question: dict[str, list[dict[str, Any]]] = {}
    if ids:
        answers_url = (
            "https://api.stackexchange.com/2.3/questions/"
            + ";".join(ids)
            + "/answers?"
            + urllib.parse.urlencode(
                {
                    "order": "desc",
                    "sort": "votes",
                    "site": "stackoverflow",
                    "filter": "withbody",
                    "pagesize": "5",
                }
            )
        )
        answers_payload, _ = _http_get(answers_url)
        answers_data = json.loads(answers_payload)
        for answer in answers_data.get("items", []):
            question_id = str(answer.get("question_id", ""))
            answers_by_question.setdefault(question_id, []).append(answer)

    results: list[dict[str, Any]] = []
    for question in questions:
        question_id = str(question.get("question_id", ""))
        accepted_id = question.get("accepted_answer_id")
        answers = answers_by_question.get(question_id, [])
        best = None
        if accepted_id is not None:
            best = next((a for a in answers if a.get("answer_id") == accepted_id), None)
        if best is None and answers:
            best = answers[0]
        excerpt = ""
        if best and isinstance(best.get("body"), str):
            excerpt = _strip_tags(best["body"])[:300]

        results.append(
            {
                "title": html.unescape(question.get("title", "")),
                "url": question.get("link", ""),
                "score": question.get("score", 0),
                "is_answered": bool(question.get("is_answered")),
                "answer_excerpt": excerpt,
            }
        )
    return results


@tool
def fetch_url(url: str, max_chars: int = 2500) -> dict[str, str]:
    """Fetch a URL and return a text-only excerpt for grounding/citations."""
    payload, content_type = _http_get(url)
    text = _strip_tags(payload)
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return {"url": url, "content_type": content_type, "text": text}
