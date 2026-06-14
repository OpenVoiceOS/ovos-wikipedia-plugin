"""
Unit tests for WikipediaRetrievalEngine and WikipediaToolbox.

Uses mocked HTTP calls (no network required) unless marked as integration tests.
"""
import unittest
from unittest.mock import patch, MagicMock

from ovos_wikipedia import WikipediaRetrievalEngine, WikipediaResult, WikipediaToolbox


def _make_engine(**cfg) -> WikipediaRetrievalEngine:
    return WikipediaRetrievalEngine(config=cfg or {})


def _fake_result(title="Test Page", summary="Test summary about the topic.", conf=0.8) -> WikipediaResult:
    return WikipediaResult(
        page_id="123",
        lang="en",
        title=title,
        summary=summary,
        conf=conf,
        query="test query",
    )


# ---------------------------------------------------------------------------
# WikipediaResult dataclass
# ---------------------------------------------------------------------------

class TestWikipediaResult(unittest.TestCase):

    def test_required_fields(self):
        r = WikipediaResult(page_id="1", lang="en", title="T", summary="S")
        self.assertEqual(r.page_id, "1")
        self.assertEqual(r.lang, "en")
        self.assertIsNone(r.image)
        self.assertIsNone(r.best_passage)
        self.assertEqual(r.conf, 0.5)

    def test_optional_fields(self):
        r = WikipediaResult(page_id="1", lang="en", title="T", summary="S",
                            image="http://img", conf=0.9, query="q", best_passage="p")
        self.assertEqual(r.image, "http://img")
        self.assertEqual(r.conf, 0.9)
        self.assertEqual(r.best_passage, "p")


# ---------------------------------------------------------------------------
# Plugin loading
# ---------------------------------------------------------------------------

class TestPluginLoading(unittest.TestCase):

    def test_no_plugins_when_config_empty(self):
        s = _make_engine()
        self.assertIsNone(s._extractive_qa)
        self.assertIsNone(s._reranker)
        self.assertIsNone(s._kword_extractor)

    def test_extractive_qa_loads_when_plugin_available(self):
        mock_cls = MagicMock(return_value=MagicMock())
        with patch("ovos_wikipedia.load_extractive_qa_plugin", return_value=mock_cls):
            s = WikipediaRetrievalEngine(config={"extractive_qa": "some-plugin"})
        self.assertIsNotNone(s._extractive_qa)

    def test_reranker_loads_when_plugin_available(self):
        mock_cls = MagicMock(return_value=MagicMock())
        with patch("ovos_wikipedia.load_reranker_plugin", return_value=mock_cls):
            s = WikipediaRetrievalEngine(config={"reranker": "some-plugin"})
        self.assertIsNotNone(s._reranker)

    def test_keyword_extractor_loads_when_plugin_available(self):
        mock_cls = MagicMock(return_value=MagicMock())
        with patch("ovos_wikipedia.load_keyword_extract_plugin", return_value=mock_cls):
            s = WikipediaRetrievalEngine(config={"keyword_extractor": "some-plugin"})
        self.assertIsNotNone(s._kword_extractor)

    def test_unknown_extractive_qa_plugin_returns_none(self):
        with patch("ovos_wikipedia.load_extractive_qa_plugin", return_value=None):
            s = WikipediaRetrievalEngine(config={"extractive_qa": "nonexistent-plugin"})
        self.assertIsNone(s._extractive_qa)

    def test_unknown_reranker_plugin_returns_none(self):
        with patch("ovos_wikipedia.load_reranker_plugin", return_value=None):
            s = WikipediaRetrievalEngine(config={"reranker": "nonexistent-plugin"})
        self.assertIsNone(s._reranker)

    def test_plugin_init_exception_returns_none(self):
        mock_cls = MagicMock(side_effect=RuntimeError("init failed"))
        with patch("ovos_wikipedia.load_extractive_qa_plugin", return_value=mock_cls):
            s = WikipediaRetrievalEngine(config={"extractive_qa": "bad-plugin"})
        self.assertIsNone(s._extractive_qa)


# ---------------------------------------------------------------------------
# _score_page (pure function)
# ---------------------------------------------------------------------------

class TestScorePage(unittest.TestCase):

    def test_perfect_title_match_scores_high(self):
        WikipediaRetrievalEngine._score_page.cache_clear()
        score = WikipediaRetrievalEngine._score_page("python", "Python", "Python is a programming language.", 0)
        self.assertGreater(score, 0.1)

    def test_unrelated_title_scores_low(self):
        WikipediaRetrievalEngine._score_page.cache_clear()
        score = WikipediaRetrievalEngine._score_page("python language", "Banana", "Banana is a tropical fruit.", 0)
        self.assertLess(score, 0.5)

    def test_later_index_scores_lower(self):
        WikipediaRetrievalEngine._score_page.cache_clear()
        score_first = WikipediaRetrievalEngine._score_page("python", "Python", "Python language.", 0)
        score_later = WikipediaRetrievalEngine._score_page("python", "Python", "Python language.", 3)
        self.assertGreater(score_first, score_later)

    def test_empty_query_does_not_crash(self):
        WikipediaRetrievalEngine._score_page.cache_clear()
        score = WikipediaRetrievalEngine._score_page("", "Python", "Python is a programming language.", 0)
        self.assertIsInstance(score, float)


# ---------------------------------------------------------------------------
# _rerank_pages
# ---------------------------------------------------------------------------

class TestRerankPages(unittest.TestCase):

    def setUp(self):
        self.s = _make_engine()

    def test_sorted_by_conf_descending_without_reranker(self):
        pages = [
            _fake_result(title="B", conf=0.3),
            _fake_result(title="A", conf=0.9),
            _fake_result(title="C", conf=0.6),
        ]
        result = self.s._rerank_pages("query", pages, "en")
        confs = [r.conf for r in result]
        self.assertEqual(confs, sorted(confs, reverse=True))

    def test_reranker_updates_conf_scores(self):
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [(0.95, "summary A"), (0.1, "summary B")]
        self.s._reranker = mock_reranker

        pages = [
            _fake_result(title="A", conf=0.3, summary="summary A"),
            _fake_result(title="B", conf=0.8, summary="summary B"),
        ]
        result = self.s._rerank_pages("query", pages, "en")
        self.assertEqual(result[0].conf, 0.95)
        self.assertEqual(result[1].conf, 0.1)

    def test_reranker_called_with_summaries(self):
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [(0.9, "s1"), (0.8, "s2")]
        self.s._reranker = mock_reranker

        p1 = _fake_result(summary="s1")
        p2 = _fake_result(summary="s2")
        self.s._rerank_pages("q", [p1, p2], "en")
        mock_reranker.rerank.assert_called_once_with("q", ["s1", "s2"], "en")

    def test_best_passage_used_over_summary_for_reranker(self):
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [(0.9, "passage1")]
        self.s._reranker = mock_reranker

        p = _fake_result(summary="full summary")
        p.best_passage = "passage1"
        self.s._rerank_pages("q", [p], "en")
        mock_reranker.rerank.assert_called_once_with("q", ["passage1"], "en")


# ---------------------------------------------------------------------------
# _get_page_data (mocked HTTP)
# ---------------------------------------------------------------------------

class TestGetPageData(unittest.TestCase):

    def _make_response(self, pid, title, extract, thumbnail=None):
        page = {"title": title, "extract": extract}
        if thumbnail:
            page["thumbnail"] = {"source": thumbnail}
        return {"query": {"pages": {pid: page}}}

    @patch("ovos_wikipedia.requests.get")
    def test_returns_wikipedia_result(self, mock_get):
        mock_get.return_value.json.return_value = self._make_response(
            "42", "Python (programming language)", "Python is a high-level language."
        )

        result = WikipediaRetrievalEngine._get_page_data("42", "en", "python", 0)
        self.assertIsInstance(result, WikipediaResult)
        self.assertEqual(result.title, "Python (programming language)")
        self.assertEqual(result.lang, "en")
        self.assertEqual(result.page_id, "42")

    @patch("ovos_wikipedia.requests.get")
    def test_disambiguation_page_returns_none(self, mock_get):
        mock_get.return_value.json.return_value = self._make_response(
            "99", "Mercury", "Mercury may refer to: the planet, the element, the god."
        )

        result = WikipediaRetrievalEngine._get_page_data("99", "en", "mercury", 0)
        self.assertIsNone(result)

    @patch("ovos_wikipedia.requests.get")
    def test_image_url_strips_thumb_path(self, mock_get):
        thumbnail = "https://upload.wikimedia.org/thumb/img/320px-img.jpg"
        mock_get.return_value.json.return_value = self._make_response(
            "7", "Eiffel Tower", "The Eiffel Tower is in Paris.", thumbnail=thumbnail
        )

        result = WikipediaRetrievalEngine._get_page_data("7", "en", "eiffel tower", 0)
        self.assertIsNotNone(result.image)
        self.assertNotIn("thumb", result.image)
        self.assertNotIn("320px-img.jpg", result.image)

    @patch("ovos_wikipedia.requests.get", side_effect=Exception("network error"))
    def test_network_error_returns_none(self, mock_get):

        result = WikipediaRetrievalEngine._get_page_data("1", "en", "q", 0)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# search (mocked HTTP)
# ---------------------------------------------------------------------------

_SEARCH_RESULTS = {
    "query": {
        "search": [
            {"pageid": 1001, "title": "Python (programming language)"},
            {"pageid": 1002, "title": "Python (snake)"},
        ]
    }
}

_PAGE_DATA_1001 = {
    "query": {"pages": {"1001": {"title": "Python (programming language)", "extract": "Python is a language."}}}
}
_PAGE_DATA_1002 = {
    "query": {"pages": {"1002": {"title": "Python (snake)", "extract": "Python is a large snake."}}}
}


class TestSearch(unittest.TestCase):

    def setUp(self):
        self.s = _make_engine()


    def _patch_requests(self, search_data, page_data_map):
        def side_effect(url, **kwargs):
            mock_resp = MagicMock()
            if "list=search" in url:
                mock_resp.json.return_value = search_data
            else:
                for pid, data in page_data_map.items():
                    if f"pageids={pid}" in url:
                        mock_resp.json.return_value = data
                        break
                else:
                    mock_resp.json.return_value = {"query": {"pages": {}}}
            return mock_resp
        return side_effect

    @patch("ovos_wikipedia.requests.get")
    def test_returns_list_of_results(self, mock_get):
        mock_get.side_effect = self._patch_requests(
            _SEARCH_RESULTS, {"1001": _PAGE_DATA_1001, "1002": _PAGE_DATA_1002}
        )
        results = self.s.search("python", "en", top_k=2)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], WikipediaResult)

    @patch("ovos_wikipedia.requests.get")
    def test_results_sorted_by_conf_descending(self, mock_get):
        mock_get.side_effect = self._patch_requests(
            _SEARCH_RESULTS, {"1001": _PAGE_DATA_1001, "1002": _PAGE_DATA_1002}
        )
        results = self.s.search("python", "en", top_k=2)
        confs = [r.conf for r in results]
        self.assertEqual(confs, sorted(confs, reverse=True))

    @patch("ovos_wikipedia.requests.get")
    def test_empty_search_returns_empty_list(self, mock_get):
        mock_get.return_value.json.return_value = {"query": {"search": []}}
        results = self.s.search("xyzzy_nonexistent_12345", "en")
        self.assertEqual(results, [])

    @patch("ovos_wikipedia.requests.get")
    def test_network_error_returns_empty_list(self, mock_get):
        mock_get.side_effect = Exception("timeout")
        results = self.s.search("python", "en")
        self.assertEqual(results, [])

    @patch("ovos_wikipedia.requests.get")
    def test_disambiguation_pages_are_skipped(self, mock_get):
        search_data = {"query": {"search": [{"pageid": 555, "title": "Mercury (disambiguation)"}]}}
        mock_get.side_effect = self._patch_requests(search_data, {})
        results = self.s.search("mercury", "en")
        self.assertEqual(results, [])

    @patch("ovos_wikipedia.requests.get")
    def test_extractive_qa_called_when_loaded(self, mock_get):
        mock_get.side_effect = self._patch_requests(
            _SEARCH_RESULTS, {"1001": _PAGE_DATA_1001, "1002": _PAGE_DATA_1002}
        )
        mock_qa = MagicMock()
        mock_qa.get_best_passage.return_value = "Python is a language."
        self.s._extractive_qa = mock_qa

        results = self.s.search("python programming", "en", top_k=2)
        self.assertTrue(mock_qa.get_best_passage.called)
        for r in results:
            self.assertEqual(r.best_passage, "Python is a language.")

    @patch("ovos_wikipedia.requests.get")
    def test_keyword_fallback_when_empty_results(self, mock_get):
        call_count = [0]

        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "list=search" in url:
                if call_count[0] == 0:
                    resp.json.return_value = {"query": {"search": []}}
                else:
                    resp.json.return_value = _SEARCH_RESULTS
                call_count[0] += 1
            else:
                resp.json.return_value = _PAGE_DATA_1001
            return resp

        mock_get.side_effect = side_effect

        mock_kw = MagicMock()
        mock_kw.extract.return_value = ["python programming language"]
        self.s._kword_extractor = mock_kw

        results = self.s.search("what is python used for programming", "en")
        self.assertGreater(len(results), 0)
        mock_kw.extract.assert_called_once()

    @patch("ovos_wikipedia.requests.get")
    def test_lang_code_stripped_to_two_chars(self, mock_get):
        mock_get.return_value.json.return_value = {"query": {"search": []}}
        self.s.search("python", "en-US")
        search_url = mock_get.call_args_list[0][0][0]
        self.assertIn("en.wikipedia.org", search_url)
        self.assertNotIn("en-US", search_url)


# ---------------------------------------------------------------------------
# query() method
# ---------------------------------------------------------------------------

class TestQuery(unittest.TestCase):

    def setUp(self):
        self.s = _make_engine()

    def test_returns_list_of_tuples(self):
        fake_pages = [
            _fake_result(summary="Summary one.", conf=0.9),
            _fake_result(summary="Summary two.", conf=0.7),
        ]
        with patch.object(self.s, "search", return_value=fake_pages):
            results = self.s.query("test query", "en", k=2)
        self.assertEqual(len(results), 2)
        for content, score in results:
            self.assertIsInstance(content, str)
            self.assertIsInstance(score, float)

    def test_best_passage_preferred_over_summary(self):
        page = _fake_result(summary="Full summary text.")
        page.best_passage = "Extracted passage."
        with patch.object(self.s, "search", return_value=[page]):
            results = self.s.query("q", "en")
        self.assertEqual(results[0][0], "Extracted passage.")

    def test_summary_used_when_no_best_passage(self):
        page = _fake_result(summary="Full summary text.")
        page.best_passage = None
        with patch.object(self.s, "search", return_value=[page]):
            results = self.s.query("q", "en")
        self.assertEqual(results[0][0], "Full summary text.")

    def test_empty_search_returns_empty_list(self):
        with patch.object(self.s, "search", return_value=[]):
            results = self.s.query("q", "en")
        self.assertEqual(results, [])

    def test_k_forwarded_to_search(self):
        with patch.object(self.s, "search", return_value=[]) as mock_search:
            self.s.query("q", "en", k=5)
        mock_search.assert_called_once_with("q", "en", 5)

    def test_default_k_is_one(self):
        with patch.object(self.s, "search", return_value=[]) as mock_search:
            self.s.query("q", "en")
        mock_search.assert_called_once_with("q", "en", 1)

    def test_units_accepted_and_ignored(self):
        fake_pages = [_fake_result(summary="Summary.", conf=0.8)]
        with patch.object(self.s, "search", return_value=fake_pages):
            results = self.s.query("q", "en", k=1, units="metric")
        self.assertEqual(len(results), 1)


# ---------------------------------------------------------------------------
# WikipediaToolbox
# ---------------------------------------------------------------------------

class TestWikipediaToolbox(unittest.TestCase):

    def test_discover_tools_returns_search_tool(self):
        tb = WikipediaToolbox()
        tools = tb.discover_tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "search_wikipedia")

    def test_search_wikipedia_returns_title_summary_pairs(self):
        tb = WikipediaToolbox()
        fake_pages = [
            _fake_result(title="Python", summary="A programming language."),
        ]
        with patch.object(tb.wiki, "search", return_value=fake_pages):
            from ovos_wikipedia import SearchWikipediaArgs
            output = tb.search_wikipedia(SearchWikipediaArgs(query="python", lang="en"))
        self.assertEqual(output.results, [("Python", "A programming language.")])

    def test_search_wikipedia_empty_results(self):
        tb = WikipediaToolbox()
        with patch.object(tb.wiki, "search", return_value=[]):
            from ovos_wikipedia import SearchWikipediaArgs
            output = tb.search_wikipedia(SearchWikipediaArgs(query="xyzzy", lang="en"))
        self.assertEqual(output.results, [])


if __name__ == "__main__":
    unittest.main()
