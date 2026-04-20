# ovos-wikipedia-plugin

[![PyPI](https://img.shields.io/pypi/v/ovos-wikipedia-plugin)](https://pypi.org/project/ovos-wikipedia-plugin/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/)

Wikipedia integration for [OpenVoiceOS](https://openvoiceos.org). Provides a **retrieval engine** for RAG pipelines and
an **agent toolbox** for tool-using agents, both as standard OPM plugins.

---

## Installation

```bash
pip install ovos-wikipedia-plugin
```

---

## OPM Entry Points

| Entry point                                      | Class                      | Use case                                                 |
|--------------------------------------------------|----------------------------|----------------------------------------------------------|
| `opm.agents.retrieval` — `ovos-wikipedia-plugin` | `WikipediaRetrievalEngine` | Retrieval — returns ranked `(passage, score)` tuples     |
| `opm.agents.toolbox` — `ovos-wikipedia-tool`     | `WikipediaToolbox`         | Agent tool use — exposes `search_wikipedia`              |

---

## Retrieval Engine

`WikipediaRetrievalEngine` implements the `RetrievalEngine` OPM interface. It searches Wikipedia, fetches article
summaries in parallel, scores them for relevance, and optionally applies extractive QA and reranking sub-plugins.

```python
from ovos_wikipedia import WikipediaRetrievalEngine

engine = WikipediaRetrievalEngine()

# Returns List[WikipediaResult] sorted by relevance
for result in engine.search("Ada Lovelace", lang="en"):
    print(result.title, result.conf)
    print(result.summary)
    if result.image:
        print(result.image)

# RAG interface: List[Tuple[str, float]]  (passage, score)
passages = engine.query("who was Ada Lovelace", lang="en", k=3)
```

### Optional sub-plugins

| Config key          | OPM type                   | Effect                                                    |
|---------------------|----------------------------|-----------------------------------------------------------|
| `extractive_qa`     | `opm.agents.extractive_qa` | Extract the best passage from each article summary        |
| `reranker`          | `opm.agents.reranker`      | Re-score results with a cross-encoder                     |
| `keyword_extractor` | `opm.plugin.keywords`      | Rewrite the query when the initial search returns nothing |

```python
engine = WikipediaRetrievalEngine(config={
    "extractive_qa": "ovos-bm25-solver",
    "reranker": "ovos-bm25-reranker",
    "keyword_extractor": "ovos-rake-keywords",
})
```

---

## Agent Toolbox

`WikipediaToolbox` exposes a single `search_wikipedia` tool that any OPM-compatible agent loop (
e.g. [ovos-agentic-loop](https://github.com/TigreGotico/ovos-agentic-loop)) can discover and call.

### Loading via entry point (recommended)

Any agent loop that supports `opm.agents.toolbox` entry points will auto-discover this toolbox by name:

```json
{
  "name": "ResearchAgent",
  "solvers": [
    "ovos-react-loop"
  ],
  "ovos-react-loop": {
    "brain": "ovos-chat-openai-plugin",
    "ovos-chat-openai-plugin": {
      "api_url": "http://localhost:11434/v1/chat/completions"
    },
    "toolboxes": [
      "ovos-wikipedia-tool"
    ]
  }
}
```

### Direct usage

```python
from ovos_wikipedia import WikipediaToolbox, SearchWikipediaArgs

tb = WikipediaToolbox()

# Discover the tool (used internally by agent loops)
tools = tb.discover_tools()
# [AgentTool(name="search_wikipedia", description="Search Wikipedia for information about a topic...")]

# Call directly
output = tb.search_wikipedia(SearchWikipediaArgs(query="Ada Lovelace", lang="en"))
for title, summary in output.results:
    print(title)
    print(summary)
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
