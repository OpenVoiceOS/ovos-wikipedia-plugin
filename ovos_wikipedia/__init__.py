# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import concurrent.futures
import dataclasses
from functools import lru_cache
from typing import Optional, Tuple, Dict, Any, List, Type
from urllib.parse import urlencode

import requests
from ovos_plugin_manager.agents import (
    load_extractive_qa_plugin,
    load_reranker_plugin,
)
from ovos_plugin_manager.keywords import load_keyword_extract_plugin
from ovos_plugin_manager.templates.agent_tools import AgentTool, ToolBox, ToolOutput, ToolArguments
from ovos_plugin_manager.templates.agents import ExtractiveQAEngine, ReRankerEngine
from ovos_plugin_manager.templates.agents import RetrievalEngine
from ovos_plugin_manager.templates.keywords import KeywordExtractor
from ovos_utils.log import LOG
from ovos_utils.parse import fuzzy_match, MatchStrategy
from ovos_utils.text_utils import rm_parentheses
from ovos_wikipedia.version import VERSION_BUILD, VERSION_MAJOR, VERSION_MINOR
from pydantic import Field


@dataclasses.dataclass
class WikipediaResult:
    """
    Data container for a Wikipedia search result.

    Attributes:
        page_id: The unique Wikipedia page identifier.
        lang: BCP-47 language code of the page.
        title: Title of the Wikipedia article.
        summary: Plain-text introductory summary.
        image: URL to the primary article image.
        conf: Confidence score of the result (0.0 to 1.0).
        query: The original search query that produced this result.
        best_passage: The specific snippet extracted by a QA engine, if available.
    """
    page_id: str
    lang: str
    title: str
    summary: str
    image: Optional[str] = None
    conf: float = 0.5
    query: Optional[str] = None
    best_passage: Optional[str] = None


class WikipediaRetrievalEngine(RetrievalEngine):
    """
    A solver for answering questions using Wikipedia search and summaries.
    Utilizes keyword extraction, extractive QA, and reranking plugins.
    """
    USER_AGENT: str = (
        f"ovos-wikipedia-solver/{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_BUILD} "
        f"(https://github.com/OpenVoiceOS/ovos-wikipedia-solver)"
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the Wikipedia engine and loads sub-plugins.

        Args:
            config: Configuration mapping for the specific engine.
        """
        super().__init__(config)
        self._kword_extractor: Optional[KeywordExtractor] = self._load_keyword_extractor()
        self._extractive_qa: Optional[ExtractiveQAEngine] = self._load_extractive_qa()
        self._reranker: Optional[ReRankerEngine] = self._load_reranker()
        LOG.debug(f"KeywordExtractor: {self._kword_extractor}")
        LOG.debug(f"Reranker: {self._reranker}")
        LOG.debug(f"ExtractiveQA: {self._extractive_qa}")

    def query(self, query: str, lang: Optional[str] = None, k: int = 3) -> List[Tuple[str, float]]:
        """
        Searches the knowledge base for relevant documents or data.

        Args:
            query: The search string.
            lang: BCP-47 language code.
            k: The maximum number of results to return.

        Returns:
            List of tuples containing (content, score) for the top k matches.
        """
        return [(page.best_passage or page.summary, page.conf)
                for page in self.search(query, lang, k)]

    ###########################
    # plugin loading
    ###########################
    def _load_extractive_qa(self) -> Optional[ExtractiveQAEngine]:
        """
        Loads the Extractive QA plugin defined in the config.

        Returns:
            An instance of ExtractiveQAEngine or None if loading fails.
        """
        plugin_name: Optional[str] = self.config.get("extractive_qa")
        if plugin_name:
            cls: Optional[Type[ExtractiveQAEngine]] = load_extractive_qa_plugin(plugin_name)
            if cls:
                try:
                    return cls()
                except Exception as e:
                    LOG.warning(f"Extractive QA plugin '{plugin_name}' failed to load ({e}), using BM25")
        return None

    def _load_reranker(self) -> Optional[ReRankerEngine]:
        """
        Loads the ReRanker plugin defined in the config.

        Returns:
            An instance of ReRankerEngine or None if loading fails.
        """
        plugin_name: Optional[str] = self.config.get("reranker")
        if plugin_name:
            cls: Optional[Type[ReRankerEngine]] = load_reranker_plugin(plugin_name)
            if cls:
                try:
                    return cls()
                except Exception as e:
                    LOG.warning(f"Reranker plugin '{plugin_name}' failed to load ({e}), using BM25")
        return None

    def _load_keyword_extractor(self) -> Optional[KeywordExtractor]:
        """
        Loads the Keyword Extractor plugin defined in the config.

        Returns:
            An instance of KeywordExtractor or None if loading fails.
        """
        plugin_name: Optional[str] = self.config.get("keyword_extractor")
        if plugin_name:
            cls: Optional[Type[KeywordExtractor]] = load_keyword_extract_plugin(plugin_name)
            if cls:
                try:
                    return cls()
                except Exception as e:
                    LOG.warning(f"KeywordExtractor plugin '{plugin_name}' failed to load ({e}), using RAKE")
        return None

    ###########################
    # wikipedia integration
    ###########################
    def search(self, query: str, lang: Optional[str] = None, top_k: int = 3) -> List[WikipediaResult]:
        """
        Performs a Wikipedia search, fetches page content, and optionally reranks results.

        Args:
            query: The search string.
            lang: BCP-47 language code (e.g., "en-US").
            top_k: Number of search results to process.

        Returns:
            A list of WikipediaResult objects sorted by confidence.
        """
        LOG.debug(f"WikiSolver query: {query}")
        lang = (lang or self.lang).split("-")[0]
        search_params = urlencode({
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
        })
        search_url = f"https://{lang}.wikipedia.org/w/api.php?{search_params}"
        try:
            search_results: List[Dict[str, Any]] = requests.get(
                search_url,
                timeout=5,
                headers={"User-Agent": self.USER_AGENT}
            ).json().get("query", {}).get("search", [])
        except Exception as e:
            LOG.error(f"Error fetching search results: {e}")
            search_results = []

        if not search_results:
            if self._kword_extractor:
                keywords: List[str] = self._kword_extractor.extract(query, lang=lang)
                if keywords:
                    fallback_query = max(keywords, key=len)
                    if fallback_query and fallback_query != query:
                        LOG.debug(f"WikiSolver Fallback, new query: {fallback_query}")
                        return self.search(fallback_query, lang=lang, top_k=top_k)
            return []

        LOG.debug(f"Matched {len(search_results)} Wikipedia pages, using top {top_k}")
        search_results = search_results[:top_k]

        # Prepare for parallel fetch and maintain original order
        summaries: List[Optional[WikipediaResult]] = [None] * len(search_results)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_idx = {
                executor.submit(self._get_page_data, str(r["pageid"]), lang, query, idx): idx
                for idx, r in enumerate(search_results)
                if "(disambiguation)" not in r["title"]
            }

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                summaries[idx] = future.result()

        res: List[WikipediaResult] = [s for s in summaries if s is not None]

        if self._extractive_qa:
            for page in res:
                page.best_passage = self._extractive_qa.get_best_passage(
                    evidence=page.summary,
                    question=query,
                    lang=lang
                ).strip()

        return self._rerank_pages(query, res, lang)

    @classmethod
    def _get_page_data(cls, pid: str, lang: str, query: Optional[str] = None, idx: int = 0) -> Optional[
        WikipediaResult]:
        """
        Fetch the title, plain-text intro summary, and an image URL for a Wikipedia page.

        Args:
            pid: Wikipedia pageid.
            lang: Language code (e.g., "en", "pt").
            query: The original query for scoring purposes.
            idx: The original search index for scoring purposes.

        Returns:
            A WikipediaResult object if successful, else None.
        """
        url = (
            f"https://{lang}.wikipedia.org/w/api.php?format=json&action=query&"
            f"prop=extracts|pageimages&exintro&explaintext&redirects=1&pageids={pid}"
        )
        try:
            disambiguation_indicators = ["may refer to:", "refers to:"]
            response = requests.get(url, timeout=5, headers={"User-Agent": cls.USER_AGENT}).json()
            page = response["query"]["pages"][pid]

            summary = rm_parentheses(page.get("extract", ""))
            if any(i in summary for i in disambiguation_indicators):
                return None  # Disambiguation list page

            img: Optional[str] = None
            if "thumbnail" in page:
                thumbnail = page["thumbnail"]["source"]
                parts = thumbnail.split("/")[:-1]
                img = "/".join(part for part in parts if part != "thumb")

            result = WikipediaResult(
                lang=lang,
                title=page["title"],
                summary=summary.strip(),
                image=img,
                page_id=pid,
                conf=cls._score_page(query or "", page["title"], summary, idx),
                query=query
            )
            return result
        except Exception as e:
            LOG.error(f"Error fetching page data for PID {pid}: {e}")
            return None

    ###########################
    # helpers to score results
    ###########################
    @staticmethod
    @lru_cache(maxsize=128)
    def _score_page(query: str, title: str, summary: str, idx: int) -> float:
        """
        Compute a relevance score for a Wikipedia page given a search query.

        Args:
            query: The user's search query.
            title: The page title.
            summary: The page summary text.
            idx: The page's index in the original search results.

        Returns:
            Relevance score where higher values indicate greater relevance.
        """
        page_mod = 1 - (idx * 0.05)  # Favor original order returned by Wikipedia
        title_score = fuzzy_match(query, rm_parentheses(title), MatchStrategy.TOKEN_SET_RATIO)
        summary_score = fuzzy_match(summary, query, MatchStrategy.TOKEN_SET_RATIO)
        return title_score * summary_score * page_mod

    def _rerank_pages(self, query: str, results: List[WikipediaResult], lang: str) -> List[WikipediaResult]:
        """
        Uses the loaded ReRanker plugin to re-score the list of results.

        Args:
            query: The user's search query.
            results: List of fetched WikipediaResult objects.
            lang: Language code.

        Returns:
            List of WikipediaResult objects sorted by updated confidence scores.
        """
        if self._reranker is not None:
            ranked = self._reranker.rerank(query, [r.best_passage or r.summary for r in results], lang)
            for idx, (score, summary) in enumerate(ranked):
                LOG.debug(f"{results[idx].title}: {results[idx].conf} -> {score}")
                results[idx].conf = score
        return sorted(results, key=lambda k: k.conf, reverse=True)


class SearchWikipediaArgs(ToolArguments):
    """Input arguments for the search_wikipedia tool."""
    query: str = Field(..., description="The search query to look up on Wikipedia.")
    lang: str = Field(..., description="BCP-47 language code, e.g. 'en', 'pt', 'de'.")


class SearchWikipediaOutput(ToolOutput):
    """Output from the search_wikipedia tool."""
    results: List[Tuple[str, str]] = Field(
        ...,
        description="List of (title, summary) pairs for the top matching Wikipedia articles."
    )


class WikipediaToolbox(ToolBox):
    toolbox_id = "ovos-wikipedia-tools"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialise the toolbox.

        Args:
            config: Plugin configuration dict (currently unused).
        """
        self.wiki = WikipediaRetrievalEngine(config)
        super().__init__(toolbox_id=self.toolbox_id)

    def search_wikipedia(self, args: SearchWikipediaArgs) -> SearchWikipediaOutput:
        """
        Search Wikipedia for articles matching the given query.

        Args:
            args: Validated ``SearchWikipediaArgs`` with ``query`` and ``lang`` fields.

        Returns:
            ``SearchWikipediaOutput`` with a list of (title, summary) pairs.
        """
        return SearchWikipediaOutput(
            results=[(res.title, res.summary)
                     for res in self.wiki.search(query=args.query, lang=args.lang)],
        )

    def discover_tools(self) -> List[AgentTool]:
        """
        Abstract method to be implemented by concrete ToolBox plugins.

        This method must define and return the list of AgentTools provided by this plugin.
        The implementation should be idempotent (safe to call multiple times).

        Returns:
            A list of instantiated AgentTool objects.
        """
        return [
            AgentTool(
                name="search_wikipedia",
                description=(
                    "Search Wikipedia for information about a topic. "
                    "Returns a list of matching article titles and their introductory summaries."
                ),
                argument_schema=SearchWikipediaArgs,
                output_schema=SearchWikipediaOutput,
                tool_call=self.search_wikipedia,
            )
        ]


if __name__ == "__main__":
    LOG.set_level("ERROR")

    s = WikipediaRetrievalEngine()

    data = s.search("who is venus")
    for r in data:
        print(r)
