# Tech Pulse Newsletter Agent

An autonomous AI agent that generates a daily tech newsletter. It identifies the most trending topic in tech that has not been previously covered, finds ten real articles on that topic, summarizes each one individually, synthesizes cross-article insights into an editorial, and assembles everything into a clean Markdown file. The agent runs end to end from a single command with no manual input required.

---

## Table of Contents

1. Project Overview
2. Architecture and Agent Design
3. How Each Node Works
4. Models Used and Why
5. Rate Limit Strategy
6. Project Structure
7. Prerequisites
8. Installation
9. Setting Up Your API Key
10. Running the Agent
11. Output Format
12. Topic History and Deduplication
13. Customization
14. Pushing to GitHub
15. Troubleshooting

---

## 1. Project Overview

Tech Pulse is a LangGraph-based agentic pipeline built on top of the Anthropic API. Every time it runs, it does the following automatically:

- Searches the web for what is trending in tech right now
- Picks the single most newsworthy topic that has not already been covered in a previous run
- Finds ten real articles from reputable tech publications on that topic
- Summarizes each article individually using a lightweight language model
- Writes an editorial synthesis with cross-article insights using a more powerful model
- Assembles the full newsletter into a Markdown file and saves it to disk
- Records the topic so it is never repeated in a future run

The agent is built for daily use. Each run produces one newsletter file. Running it again the next day produces a fresh newsletter on a different topic.

---

## 2. Architecture and Agent Design

The agent is implemented as a directed acyclic graph using LangGraph. LangGraph is a framework built on top of LangChain that lets you define multi-step agents as explicit state machines. Each step is a node in the graph. Nodes pass data to each other through a shared state object called NewsletterState. The graph executes nodes in sequence, and each node reads from the state and writes back to it.

The execution order is fixed and linear:

```
pick_topic -> find_articles -> summarize_articles -> write_editorial -> format_newsletter -> save_and_log
```

This design makes the pipeline easy to debug because you can inspect the state at any node, easy to extend because you can add nodes without touching existing ones, and easy to test because each node is a pure Python function.

The shared state is typed using Python TypedDict, which means every field has a declared type and the structure is enforced at runtime. This prevents silent data errors between nodes.

---

## 3. How Each Node Works

**Node 1: pick_topic**

This node uses Claude Sonnet with the web search tool enabled. It runs two searches: one for general trending tech news for today's date, and one specifically for AI and machine learning news. It reads the covered_topics.json file and passes the list of previously covered topics directly into the prompt, explicitly instructing the model not to repeat any of them. The model returns a JSON object with a chosen_topic field. If JSON parsing fails, a regex fallback extracts the topic from raw text. If that also fails, a hardcoded default is used. This three-layer fallback ensures the pipeline never crashes at this step.

**Node 2: find_articles**

This node uses Claude Sonnet with web search enabled. It instructs the model to run three to four different search queries on the chosen topic and return at least ten articles as a raw JSON array. Each article object contains a title, url, snippet, and source. The node deduplicates by URL to prevent the same article from appearing twice. It caps the final list at ten articles to stay within API rate limits. If JSON parsing fails, a regex pattern extracts individual JSON objects from the raw response as a fallback.

**Node 3: summarize_articles**

This node iterates over each article one at a time. For each article, it sends the title and snippet to Claude Haiku and asks for a two to three sentence summary covering the core finding, why it matters, and any key specifics like statistics or names. It uses llm_plain, which does not enable web search, keeping each call fast and cheap. A 15 second sleep is inserted between each article call to avoid hitting the API rate limit. Haiku is used here instead of Sonnet because summarization is a straightforward task that does not require deep reasoning, and Haiku has a much higher rate limit at a fraction of the cost.

**Node 4: write_editorial**

This node uses Claude Sonnet without web search. It receives all ten article summaries concatenated into a single prompt and asks the model to write four sections: Why This Matters Today, Key Themes Across Sources, The Bigger Picture, and Editor's Take. The key constraint in the prompt is that Key Themes must be cross-cutting insights that no single article captured on its own. This forces the model to synthesize rather than repeat, which is the core value of the editorial section.

**Node 5: format_newsletter**

This node is pure Python with no API calls. It assembles the newsletter Markdown by combining the editorial content with the per-article summaries. It adds a header with the date and topic, section dividers, and a footer with metadata. Each article is formatted with its title as a hyperlink to the original URL and the source publication name below it.

**Node 6: save_and_log**

This node writes the final Markdown content to a file in the newsletters directory. The filename is constructed from today's date and a URL-safe slug of the topic name. It then appends the chosen topic to covered_topics.json so it will be skipped in all future runs.

---

## 4. Models Used and Why

The agent uses two different Claude models strategically to balance quality, speed, and cost.

**Claude Sonnet (claude-sonnet-4-20250514)** is used for pick_topic and write_editorial. These are the two steps that require real reasoning. Picking a topic requires the model to evaluate news significance, avoid repetition, and return structured JSON reliably. Writing the editorial requires the model to synthesize across ten sources and form an opinion. Sonnet handles both well.

**Claude Haiku (claude-haiku-4-5-20251001)** is used for all ten article summaries. Summarization is a pattern-matching task that does not require deep reasoning. Haiku is significantly faster than Sonnet, costs less per token, and has a higher rate limit. Using Haiku for the summarization loop is what makes it possible to process ten articles without exhausting the free tier rate limit.

---

## 5. Rate Limit Strategy

The Anthropic free tier allows 30,000 input tokens per minute. The summarization loop is the most token-intensive part of the pipeline because it makes ten API calls in sequence. Three strategies are used to prevent rate limit errors.

First, a 15 second sleep is inserted between each article summary call. This spaces out the calls and prevents token usage from spiking within a single minute window.

Second, both llm_with_search and llm_plain implement retry logic with exponential backoff. If a rate limit error occurs, the function waits 60 seconds before the first retry, 120 seconds before the second, 180 before the third, and so on up to five attempts. This means a single rate limit hit does not crash the pipeline.

Third, Haiku is used for summaries instead of Sonnet. Haiku processes fewer tokens per call because the prompts are shorter and the responses are capped at 300 tokens. This reduces overall token consumption across the summarization loop.

---

## 6. Project Structure

```
Agent_NewsLetter/
    agent.py                   Main agent file containing all nodes and graph logic
    covered_topics.json        Auto-generated. Stores topics already covered to prevent repeats
    newsletters/               Auto-generated directory. All newsletter output files go here
        2026-03-27_meta_and_google_landmark_social_media_li.md
    venv/                      Python virtual environment (not committed to git)
    .gitignore                 Excludes venv, covered_topics.json, newsletters, and secrets
    README.md                  This file
```

---

## 7. Prerequisites

- Python 3.10 or higher
- An Anthropic API key (get one at console.anthropic.com)
- pip and venv available in your terminal
- A terminal on macOS, Linux, or Windows with WSL

---

## 8. Installation

Open your terminal and run the following commands in order.

Clone or create your project directory and navigate into it:

```
cd path/to/Agent_NewsLetter
```

Create a virtual environment:

```
python3 -m venv venv
```

Activate the virtual environment:

```
source venv/bin/activate
```

On Windows use:

```
venv\Scripts\activate
```

Install all required packages:

```
pip install langgraph langchain-anthropic anthropic feedparser httpx
```

---

## 9. Setting Up Your API Key

The agent reads your Anthropic API key from the environment variable ANTHROPIC_API_KEY. You need to set this before running the agent.

To set it permanently on macOS or Linux, add the following line to your shell configuration file. For macOS with zsh (the default), that file is ~/.zshrc:

```
echo 'export ANTHROPIC_API_KEY="sk-ant-your-key-here"' >> ~/.zshrc
source ~/.zshrc
```

Replace sk-ant-your-key-here with your actual key from console.anthropic.com under API Keys.

To verify it is set correctly:

```
echo $ANTHROPIC_API_KEY
```

It should print your key. If it prints nothing, the key was not saved correctly. Repeat the step above.

Important: never paste your API key directly into agent.py or any Python file. Never commit it to git. Treat it like a password.

---

## 10. Running the Agent

Make sure your virtual environment is activated (you should see (venv) at the start of your terminal prompt).

To run the agent and let it automatically pick today's trending topic:

```
python agent.py
```

To force a specific topic instead of auto-discovering one:

```
python agent.py --topic "OpenAI GPT-5 Release"
```

The agent will print its progress to the terminal as it works through each node. Total runtime is approximately 4 to 6 minutes depending on rate limit waits. When it finishes you will see:

```
Newsletter saved -> newsletters/2026-03-27_your_topic_slug.md
```

To run it automatically every day at 7am using cron on macOS:

```
crontab -e
```

Add this line (replace the path with your actual project path):

```
0 7 * * * cd /Users/yourname/Agent_NewsLetter && source venv/bin/activate && python agent.py
```

---

## 11. Output Format

Each newsletter is saved as a Markdown file in the newsletters directory. The filename format is:

```
YYYY-MM-DD_topic_slug_truncated_to_40_chars.md
```

The newsletter structure is as follows:

```
# Tech Pulse
### March 27, 2026  -  Today's Focus: Topic Name Here

---

## Why This Matters Today
Two to three sentences explaining why this topic is dominating coverage today.

## Key Themes Across Sources
Bullet points of synthesized cross-article insights.

## The Bigger Picture
One paragraph on longer-term implications.

## Editor's Take
Two to three opinionated concluding sentences.

---

## Article-by-Article Roundup

10 articles curated and summarized on: Topic Name Here

### 1. Article Title
Source Publication Name

Two to three sentence summary of the article.

### 2. Article Title
...

---

Tech Pulse - Generated by LangGraph Newsletter Agent - 2026-03-27
Articles sourced: 10
```

You can open and read the file in any Markdown viewer. In VSCode, right-click the file and select Open Preview to see it rendered with formatting.

---

## 12. Topic History and Deduplication

Every time the agent successfully completes a run, it writes the chosen topic to covered_topics.json. On the next run, this list is passed directly into the topic selection prompt. The model is explicitly instructed not to pick any topic from that list.

The covered_topics.json file looks like this:

```
[
  "Meta and Google Landmark Social Media Liability Verdicts",
  "OpenAI o3 Reasoning Model Benchmarks"
]
```

To reset the history and allow any topic to be picked again, delete the file or clear its contents:

```
echo "[]" > covered_topics.json
```

Note: covered_topics.json is excluded from git via .gitignore because it is a local runtime file that changes every day. Each person running the agent will build their own history independently.

---

## 13. Customization

**Change the tech domain focus**

By default the agent searches broadly across all tech news. To focus on a specific area, edit the search queries inside node_pick_topic in agent.py:

```python
user = f"""
1. Search 'AI safety research news {state['today']}'
2. Search 'large language model releases today'
...
"""
```

**Change the number of articles**

The article cap is set in node_find_articles. Find this line and change the number:

```python
deduped = deduped[:10]
```

Increasing this will increase runtime and token usage. On the free tier, keep it at 10 or lower.

**Adjust sleep time between summaries**

If you are on a paid Anthropic plan with higher rate limits, you can reduce the sleep time in node_summarize_articles:

```python
time.sleep(15)   # reduce this if you have a higher rate limit
```

**Add email delivery**

After the newsletter is generated, you can send it by email by adding a step after node_save_and_log. Use Python's smtplib with a Gmail app password or a service like SendGrid. The newsletter content is available as state["newsletter_md"].

**Add Slack delivery**

Post the editorial section to a Slack channel using the Slack webhook API. The editorial content is available as state["editorial"].

---

## 14. Pushing to GitHub

Follow these steps to push the project to a new GitHub repository.

Create a .gitignore file in your project directory to exclude files that should not be committed:

```
venv/
__pycache__/
*.pyc
.env
covered_topics.json
newsletters/
.DS_Store
```

Initialize a git repository:

```
git init
```

Stage all files:

```
git add .
```

Make your first commit:

```
git commit -m "Initial commit: Tech Pulse LangGraph newsletter agent"
```

Go to github.com, create a new repository, and copy the repository URL. Then connect your local repo to GitHub and push:

```
git remote add origin https://github.com/yourusername/your-repo-name.git
git branch -M main
git push -u origin main
```

Never commit your API key. If you accidentally commit it, rotate it immediately at console.anthropic.com and generate a new one.

---

## 15. Troubleshooting

**The agent crashes with a rate limit error even after retries**

Your account is on the free tier and has hit its per-minute token limit. Wait five minutes and run again. You can also increase the sleep time in node_summarize_articles to 20 or 25 seconds to reduce the chance of hitting the limit.

**The API key is not being picked up**

Run echo $ANTHROPIC_API_KEY in your terminal. If it prints nothing, open a new terminal window and run source ~/.zshrc before running the agent. The key must be set in the same terminal session where you run the agent.

**The topic keeps picking the same thing**

The covered_topics.json file stores past topics. If the file is empty or missing, the agent picks freely. If the same topic keeps appearing, it means the model is choosing a very generic topic. Force a specific topic using the --topic flag to override the auto-selection.

**The newsletters directory is empty**

The agent only writes the file after all nodes complete successfully. If any node crashes, no file is written. Check the terminal output for error messages from earlier nodes.

**JSON parse errors during article finding**

Occasionally the model returns malformed JSON. The agent has a regex-based fallback that extracts individual article objects from raw text. If both fail, re-run the agent. This is rare.

**The venv is not activating**

Make sure you are in the correct directory where the venv folder exists. Run ls to confirm. If the venv folder is missing, recreate it with python3 -m venv venv and reinstall dependencies.