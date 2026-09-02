# ovos-persona-server serving a persona backed by this plugin's retrieval
# engine (opm.agents.retrieval, WikipediaRetrievalEngine). No API key or
# other external credential is needed -- the plugin talks to the public
# Wikipedia API directly.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Prerelease floor: ovos-persona needs a version new enough to resolve a
# retrieval-engine entry point as a persona handler (find_retrieval_plugins).
RUN pip install --no-cache-dir "ovos-persona>=0.9.0a9"

# This plugin is installed from PyPI rather than the local checkout, so the
# image always matches whatever release is public -- there is no local
# source dependency to build from here.
RUN pip install --no-cache-dir ovos-wikipedia-plugin

# [mcp] mounts the MCP tool endpoint. From 0.17.0a1 that mount is opt-in --
# installing the extra alone no longer flips it on, so --mcp below is
# required, matching ovos-plugin-linguonnx's image.
RUN pip install --no-cache-dir "ovos-persona-server[mcp]>=0.17.0a1"

# The persona JSON: "handlers" (the modern config key; "solvers" is kept as
# a legacy alias) points at this plugin by its opm.agents.retrieval id.
RUN mkdir -p /personas && printf '%s\n' \
    '{' \
    '  "name": "WikiBot",' \
    '  "handlers": ["ovos-wikipedia-plugin"]' \
    '}' > /personas/wikibot.json

EXPOSE 8337

ENTRYPOINT ["ovos-persona-server", "--personas-dir", "/personas", "--mcp", \
            "--port", "8337", "--host", "0.0.0.0"]
