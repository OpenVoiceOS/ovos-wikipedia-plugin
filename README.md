# ovos-wikipedia-plugin

[![PyPI](https://img.shields.io/pypi/v/ovos-wikipedia-plugin)](https://pypi.org/project/ovos-wikipedia-plugin/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/)

This plugin connects [OpenVoiceOS](https://openvoiceos.org) to Wikipedia. It provides a retrieval engine for RAG
pipelines and an agent toolbox for tool-using agents. Both ship as standard OPM plugins.

---

## Installation

```bash
pip install ovos-wikipedia-plugin
```

---

## OPM Entry Points

| Entry point                                      | Class                      | Use case                                                 |
|--------------------------------------------------|----------------------------|----------------------------------------------------------|
| `opm.agents.retrieval` (`ovos-wikipedia-plugin`) | `WikipediaRetrievalEngine` | Retrieval. Returns ranked `(passage, score)` tuples       |
| `opm.agents.toolbox` (`ovos-wikipedia-tool`)     | `WikipediaToolbox`         | Agent tool use. Exposes `search_wikipedia`                |

---

## Retrieval Engine

`WikipediaRetrievalEngine` implements the `RetrievalEngine` OPM interface. It searches Wikipedia, fetches article
summaries in parallel, and scores them for relevance. It can also apply extractive QA and reranking sub-plugins.

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

`WikipediaToolbox` exposes a single `search_wikipedia` tool. Any OPM-compatible agent loop, such as
[ovos-agentic-loop](https://github.com/TigreGotico/ovos-agentic-loop), can discover and call it.

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

## Related Projects

- [OpenVoiceOS/ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos-plugin-manager): the plugin manager that discovers OPM entry points
- [TigreGotico/ovos-agentic-loop](https://github.com/TigreGotico/ovos-agentic-loop): agent loop that consumes toolbox plugins like this one

---

## Docker

This repo publishes `ghcr.io/openvoiceos/ovos-wikipedia-plugin`, a standalone
`ovos-persona-server` that serves one persona, `WikiBot`, backed by the
`WikipediaRetrievalEngine` in this plugin. Wikipedia's public API needs no
key, so the image works with no configuration at all.

```bash
docker run -p 8390:8337 ghcr.io/openvoiceos/ovos-wikipedia-plugin:dev
```

```bash
curl http://localhost:8390/v1/models
curl http://localhost:8390/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "WikiBot", "messages": [{"role": "user", "content": "what is the capital of Portugal?"}]}'
```

Known issue as of `ovos-persona` 0.9.0a9 through 0.9.0a16 (the latest published
alpha): `Persona.chat` passes `sess.lang` and `sess.system_unit` into
`chat_completion` in the wrong order, so the retrieval engine receives the
unit string (`"metric"`) where it expects a language code, and the request
above answers `500 Internal Server Error` with `Persona chat failed:
'NoneType' object has no attribute 'split'`. This is a bug in `ovos-persona`
itself, not in this plugin or this image -- calling
`WikipediaRetrievalEngine().query(...)` directly returns a correct answer.
It will clear up once a fixed `ovos-persona` is published; there is nothing
to configure around it from this image.

A compose snippet:

```yaml
services:
  ovos-wikipedia-persona:
    image: ghcr.io/openvoiceos/ovos-wikipedia-plugin:dev
    ports:
      - "8390:8337"
    restart: unless-stopped
```

The image builds on every pull request touching `Dockerfile`,
`.dockerignore`, `pyproject.toml`, or the docker workflow itself (build only,
no push), and publishes on pushes to `master` (`latest`), `dev` (`dev`), and
version tags. See the [`docker` workflow](.github/workflows/docker.yml).

## License

Apache 2.0. See [LICENSE](LICENSE).
