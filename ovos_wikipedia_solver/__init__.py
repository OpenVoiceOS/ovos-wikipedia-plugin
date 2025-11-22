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
from functools import lru_cache
from typing import Optional, Tuple, Dict, Any, List

import requests
from ovos_plugin_manager.keywords import load_keyword_extract_plugin
from ovos_plugin_manager.templates.agents import RetrievalEngine
from ovos_plugin_manager.templates.keywords import KeywordExtractor
from ovos_utils.log import LOG
from ovos_utils.text_utils import rm_parentheses

from ovos_wikipedia_solver.version import VERSION_BUILD, VERSION_MAJOR, VERSION_MINOR


class WikipediaSolver(RetrievalEngine):
    """
    A solver for answering questions using Wikipedia search and summaries.
    """
    USER_AGENT = f"ovos-wikipedia-solver/{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_BUILD} (https://github.com/OpenVoiceOS/ovos-wikipedia-solver)"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a WikipediaSolver instance, configure base QuestionSolver, and prepare plugin caches.

        Parameters:
            config (dict | None): Optional configuration for the solver.

        Detailed behavior:
            - Calls the superclass initializer with the provided parameters and a fixed priority of 40.
            - Creates empty caches for per-language keyword extractors (`kword_extractors`) and summarizers (`summarizers`).
        """
        super().__init__(config)
        self.kword_extractors: Dict[str, KeywordExtractor] = {}

    def get_keyword_extractor(self, lang: str) -> Optional[KeywordExtractor]:
        """
        Get a keyword extractor instance for the given language, creating and caching a plugin instance when needed.

        Returns:
            KeywordExtractor | None: A `KeywordExtractor` configured for `lang`, or `None` if the configured plugin cannot be loaded.
        """
        if lang not in self.summarizers:
            kw_plugin: str = self.config.get("keyword_extractor") or "ovos-rake-keyword-extractor"
            kword_extractor_class = load_keyword_extract_plugin(kw_plugin)
            if not kword_extractor_class:
                return None
            self.kword_extractors[lang] = kword_extractor_class()
        return self.kword_extractors[lang]

    @classmethod
    @lru_cache(maxsize=128)
    def get_page_data(cls, pid: str, lang: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Fetch the title, plain-text intro summary, and an image URL for a Wikipedia page.

        Parameters:
            pid (str): Wikipedia pageid.
            lang (str): Language code (e.g., "en", "pt"); used as the wiki subdomain.

        Returns:
            Tuple[Optional[str], Optional[str], Optional[str]]: (title, summary, image_url).
                - title: Page title or `None` if not available.
                - summary: Plain-text intro with parenthetical content removed, or `None` when the page is a disambiguation list or on error.
                - image_url: Derived image URL (thumbnail path normalized by removing "thumb" segments) or `None` if no image is available.
                Returns (None, None, None) on disambiguation pages or on request/parse errors.
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
                return None, None, None  # Disambiguation list page
            img = None
            if "thumbnail" in page:
                thumbnail = page["thumbnail"]["source"]
                parts = thumbnail.split("/")[:-1]
                img = "/".join(part for part in parts if part != "thumb")
            return page["title"], summary, img
        except Exception as e:
            LOG.error(f"Error fetching page data for PID {pid}: {e}")
            return None, None, None

    def get_data(self, query: str, lang: Optional[str] = None, top_k = 5):
        """
        Searches Wikipedia for a query, fetches page extracts in the requested language, generates a short answer, and returns the best-matching result.

        Performs a site search using the language's Wikipedia, falls back to a keyword-extracted query if no results are found, concurrently fetches page data (skipping disambiguation pages), creates a short summary using the configured summarizer plugin, scores and re-ranks candidates, and returns the top entry.

        Parameters:
            query (str): User query to search on Wikipedia.
            lang (Optional[str]): Language code or locale to use (e.g., "en" or "pt-BR"); defaults to the solver's default language if omitted.

        Returns:
            dict: A dictionary with keys:
                - "title": selected page title (str)
                - "short_answer": concise summary of the page (str)
                - "summary": full page extract (str)
                - "img": image URL if available (str or None)
            Returns an empty dict if no suitable page data could be retrieved.
        """
        LOG.debug(f"WikiSolver query: {query}")
        lang = (lang or "en").split("-")[0]
        search_url = (
            f"https://{lang}.wikipedia.org/w/api.php?action=query&list=search&"
            f"srsearch={query}&format=json"
        )
        try:
            search_results = requests.get(search_url,
                                          timeout=5,
                                          headers={"User-Agent": self.USER_AGENT}
                                          ).json().get("query", {}).get("search", [])
        except Exception as e:
            LOG.error(f"Error fetching search results: {e}")
            search_results = []

        if not search_results:
            kwx = self.get_keyword_extractor(lang)
            keywords = kwx.extract(query, lang=lang)
            if keywords:
                fallback_query = max(keywords)
                if fallback_query and fallback_query != query:
                    LOG.debug(f"WikiSolver Fallback, new query: {fallback_query}")
                    return self.get_data(fallback_query, lang=lang)
            return {}


        LOG.debug(f"Matched {len(search_results)} Wikipedia pages, using top {top_k}")
        search_results = search_results[:top_k+1]

        # Prepare for parallel fetch and maintain original order
        wiki_pages = [None] * len(search_results)  # List to hold results in original order
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_idx = {
                executor.submit(self.get_page_data, str(r["pageid"]), lang): idx
                for idx, r in enumerate(search_results)
                if "(disambiguation)" not in r["title"]
            }

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]  # Get original index from future
                title, ans, img = future.result()
                if title and ans:
                    wiki_pages[idx] = (title, ans, img)

        wiki_pages = [s for s in wiki_pages if s is not None]
        return [{
            "title": title,
            "summary": wiki_content,
            "img": img,
            "score": 1.0 - (idx * 0.1)
        } for idx, (title, wiki_content, img) in enumerate(wiki_pages)]

    def query(self, query: str, lang: Optional[str] = None, k: int = 5) -> List[Tuple[str, float]]:
        """
        Searches the knowledge base for relevant documents or data.

        Args:
            query: The search string.
            lang: BCP-47 language code.
            k: The maximum number of results to return.

        Returns:
            List of tuples (content, score) for the top k matches.
        """
        return [(d["summary"], d["score"]) for d in self.get_data(query, lang, top_k=k)]


WIKIPEDIA_PERSONA = {
    "name": "Wikipedia",
    "solvers": [
        "ovos-solver-plugin-wikipedia",
        "ovos-solver-failure-plugin"
    ]
}

if __name__ == "__main__":
    LOG.set_level("ERROR")

    s = WikipediaSolver()
    for e in s.query("venus"):
        print(e)
    for e in s.query("mercury"):
        print(e)