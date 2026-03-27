"""
Tech Newsletter Agent
=====================
LangGraph-based multi-node agent that:
  Node 1 -- pick_topic:          LLM + web search identifies today's trending tech topic
  Node 2 -- find_articles:       Searches for 12+ articles on the topic
  Node 3 -- summarize_articles:  Summarizes each article individually
  Node 4 -- write_editorial:     Synthesizes cross-article insights into an editorial
  Node 5 -- format_newsletter:   Assembles the final Markdown newsletter
  Node 6 -- save_and_log:        Writes file + updates covered-topics history

Usage:
  python agent.py
  python agent.py --topic "Quantum Computing Breakthroughs"   # force a topic
"""

import os, sys, json, datetime, time, re, argparse
from pathlib import Path
from typing import TypedDict, List, Optional

import anthropic
from langgraph.graph import StateGraph, END

# -- Config ---------------------------------------------------------------------

HISTORY_FILE = Path("covered_topics.json")
OUTPUT_DIR   = Path("newsletters")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL = "claude-sonnet-4-20250514"
client = anthropic.Anthropic()

WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}

# -- Helpers --------------------------------------------------------------------

def load_covered() -> List[str]:
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text())
    return []

def save_covered(topic: str):
    topics = load_covered()
    if topic not in topics:
        topics.append(topic)
    HISTORY_FILE.write_text(json.dumps(topics, indent=2))

def extract_text(content_blocks) -> str:
    return "\n".join(
        b.text for b in content_blocks if hasattr(b, "text") and b.type == "text"
    ).strip()

def llm_with_search(system: str, user: str, max_tokens: int = 2000) -> str:
    for attempt in range(4):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=system,
                tools=[WEB_SEARCH_TOOL],
                messages=[{"role": "user", "content": user}],
            )
            return extract_text(resp.content)
        except anthropic.RateLimitError:
            wait = 60 * (attempt + 1)
            print(f"  Rate limit hit, waiting {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Rate limit retries exhausted")

def llm_plain(system: str, user: str, max_tokens: int = 2000) -> str:
    for attempt in range(4):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return extract_text(resp.content)
        except anthropic.RateLimitError:
            wait = 60 * (attempt + 1)
            print(f"  Rate limit hit, waiting {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Rate limit retries exhausted")

def parse_json_block(text: str):
    text = re.sub(r"```json\s*|```", "", text).strip()
    start = min(
        (text.find(c) for c in ["[", "{"] if text.find(c) != -1),
        default=0
    )
    return json.loads(text[start:])

# -- State ----------------------------------------------------------------------

class NewsletterState(TypedDict):
    today:             str
    covered_topics:    List[str]
    forced_topic:      Optional[str]
    chosen_topic:      str
    raw_article_list:  List[dict]
    article_summaries: List[dict]
    editorial:         str
    newsletter_md:     str

# -- Nodes ----------------------------------------------------------------------

def node_pick_topic(state: NewsletterState) -> dict:
    if state.get("forced_topic"):
        print(f"  -> Forced topic: {state['forced_topic']}")
        return {"chosen_topic": state["forced_topic"]}

    print("  Picking today's trending tech topic via web search...")
    covered_block = "\n".join(f"- {t}" for t in state["covered_topics"]) or "None."

    system = (
        "You are a senior tech editor. Use web search to find what is actually trending "
        "in tech RIGHT NOW. Focus on breakthroughs, new releases, major research, or "
        "significant industry events happening today."
    )
    user = f"""Today is {state['today']}.

Already covered topics -- DO NOT repeat these:
{covered_block}

Steps:
1. Search 'trending tech news {state['today']}'
2. Search 'AI machine learning news today'
3. Pick ONE specific, newsworthy topic dominating coverage right now.

Reply ONLY with valid JSON (no markdown fences):
{{"chosen_topic": "...", "reasoning": "..."}}"""

    raw = llm_with_search(system, user)
    try:
        data = parse_json_block(raw)
        topic = data["chosen_topic"]
    except Exception:
        m = re.search(r'"chosen_topic"\s*:\s*"([^"]+)"', raw)
        topic = m.group(1) if m else "AI Model Releases and Benchmarks"

    print(f"  -> Topic: {topic}")
    return {"chosen_topic": topic}


def node_find_articles(state: NewsletterState) -> dict:
    topic = state["chosen_topic"]
    print(f"  Finding articles on: {topic}...")

    system = (
        "You are a research assistant. Search the web thoroughly to find real, current "
        "articles on the given tech topic. Return only factual results with real URLs."
    )
    user = f"""Topic: "{topic}"
Today: {state['today']}

Do 3-4 web searches using different queries to find at least 12 distinct recent articles
from reputable tech sources (TechCrunch, Wired, The Verge, Ars Technica, MIT Technology
Review, VentureBeat, Nature, IEEE, Reuters, Bloomberg Tech, etc.).

Return ONLY a raw JSON array -- no prose, no markdown fences:
[
  {{"title": "...", "url": "...", "snippet": "...", "source": "..."}},
  ...
]

Requirements: minimum 12 items, real URLs, no duplicates, most recent first."""

    raw = llm_with_search(system, user, max_tokens=3000)
    try:
        articles = parse_json_block(raw)
        if not isinstance(articles, list):
            raise ValueError("Not a list")
    except Exception as e:
        print(f"  WARNING: JSON parse error ({e}), extracting manually...")
        matches = re.findall(r'\{[^{}]*?"title"[^{}]*?"url"[^{}]*?\}', raw, re.DOTALL)
        articles = []
        for m in matches:
            try:
                articles.append(json.loads(m))
            except Exception:
                pass

    # Deduplicate by URL
    seen, deduped = set(), []
    for a in articles:
        url = a.get("url", "")
        if url and url not in seen:
            seen.add(url)
            deduped.append(a)

    # Cap at 10 to stay within rate limits
    deduped = deduped[:10]

    print(f"  -> {len(deduped)} articles found")
    return {"raw_article_list": deduped}


def node_summarize_articles(state: NewsletterState) -> dict:
    topic    = state["chosen_topic"]
    articles = state["raw_article_list"]
    print(f"  Summarizing {len(articles)} articles...")

    summaries = []
    for i, art in enumerate(articles):
        print(f"  [{i+1}/{len(articles)}] {art.get('title','?')[:65]}...")
        system = (
            "You are a precise tech journalist. Write tight, informative summaries "
            "based on the title and snippet provided."
        )
        user = f"""Newsletter topic: "{topic}"
Article title:  {art.get('title', 'N/A')}
Article URL:    {art.get('url', 'N/A')}
Snippet:        {art.get('snippet', 'N/A')}

Write a 3-4 sentence summary covering:
- Core finding or announcement
- Why it matters for the topic
- Key specifics (stats, names, numbers, dates)

Plain prose only. No bullet points. No "This article..." opener."""

        summary = llm_plain(system, user, max_tokens=400)
        summaries.append({
            "title":   art.get("title", "Untitled"),
            "url":     art.get("url", "#"),
            "source":  art.get("source", "Unknown"),
            "summary": summary,
        })
        time.sleep(10)

    return {"article_summaries": summaries}


def node_write_editorial(state: NewsletterState) -> dict:
    topic     = state["chosen_topic"]
    summaries = state["article_summaries"]
    print("  Writing editorial synthesis...")

    summaries_text = "\n\n".join(
        f"**{s['title']}** ({s['source']})\n{s['summary']}" for s in summaries
    )

    system = (
        "You are a sharp, opinionated tech editor at a top publication. "
        "You synthesize information into insight -- not just summary. "
        "Write with authority, precision, and a clear editorial point of view."
    )
    user = f"""Today: {state['today']}
Topic: "{topic}"

Article summaries:
{summaries_text}

Write four Markdown sections:

## Why This Matters Today
2-3 sentences. Why is coverage exploding RIGHT NOW? What triggered today's wave?

## Key Themes Across Sources
3-5 bullet points. Cross-cutting insights no single article captured alone.
Each bullet = a synthesis, not a repetition.

## The Bigger Picture
1 paragraph. Longer-term implications and what to watch next.

## Editor's Take
2-3 punchy sentences. Opinionated conclusion. What should readers think or do?

Be specific. No filler. No "It remains to be seen."."""

    editorial = llm_plain(system, user, max_tokens=1200)
    return {"editorial": editorial}


def node_format_newsletter(state: NewsletterState) -> dict:
    topic     = state["chosen_topic"]
    today     = state["today"]
    articles  = state["article_summaries"]
    editorial = state.get("editorial", "")

    date_fmt = datetime.date.fromisoformat(today).strftime("%B %d, %Y")
    lines = [
        "# Tech Pulse",
        f"### {date_fmt}  -  Today's Focus: *{topic}*",
        "",
        "---",
        "",
        editorial,
        "",
        "---",
        "",
        "## Article-by-Article Roundup",
        "",
        f"> **{len(articles)} articles** curated and summarized on: **{topic}**",
        "",
    ]

    for i, art in enumerate(articles, 1):
        lines += [
            f"### {i}. [{art['title']}]({art['url']})",
            f"> *{art['source']}*",
            "",
            art["summary"],
            "",
        ]

    lines += [
        "---",
        "",
        f"*Tech Pulse - Generated by LangGraph Newsletter Agent - {today}*  ",
        f"*Articles sourced: {len(articles)}*",
    ]

    return {"newsletter_md": "\n".join(lines)}


def node_save_and_log(state: NewsletterState) -> dict:
    today = state["today"]
    slug  = re.sub(r"[^a-z0-9]+", "_", state["chosen_topic"].lower())[:40]
    fname = OUTPUT_DIR / f"{today}_{slug}.md"
    fname.write_text(state["newsletter_md"], encoding="utf-8")
    save_covered(state["chosen_topic"])
    print(f"\n  Newsletter saved -> {fname}")
    return {}

# -- Graph ----------------------------------------------------------------------

def build_graph():
    g = StateGraph(NewsletterState)

    g.add_node("pick_topic",         node_pick_topic)
    g.add_node("find_articles",      node_find_articles)
    g.add_node("summarize_articles", node_summarize_articles)
    g.add_node("write_editorial",    node_write_editorial)
    g.add_node("format_newsletter",  node_format_newsletter)
    g.add_node("save_and_log",       node_save_and_log)

    g.set_entry_point("pick_topic")
    g.add_edge("pick_topic",         "find_articles")
    g.add_edge("find_articles",      "summarize_articles")
    g.add_edge("summarize_articles", "write_editorial")
    g.add_edge("write_editorial",    "format_newsletter")
    g.add_edge("format_newsletter",  "save_and_log")
    g.add_edge("save_and_log",       END)

    return g.compile()

# -- Entry ----------------------------------------------------------------------

def run(forced_topic: str = None):
    graph = build_graph()
    today = datetime.date.today().isoformat()

    init: NewsletterState = {
        "today":             today,
        "covered_topics":    load_covered(),
        "forced_topic":      forced_topic,
        "chosen_topic":      "",
        "raw_article_list":  [],
        "article_summaries": [],
        "editorial":         "",
        "newsletter_md":     "",
    }

    print("Tech Pulse Newsletter Agent starting...\n")
    result = graph.invoke(init)
    preview = result["newsletter_md"][:2500]
    print("\n" + "=" * 60)
    print(preview)
    if len(result["newsletter_md"]) > 2500:
        print("\n... [truncated -- see output file for full newsletter]")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tech Pulse Newsletter Agent")
    parser.add_argument("--topic", default=None,
                        help="Force a topic instead of auto-discovering")
    args = parser.parse_args()
    run(forced_topic=args.topic)