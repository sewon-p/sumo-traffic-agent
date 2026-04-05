import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag import _keyword_search, _load_regulations, enrich_prompt, format_context, search


class TestRegulationsLoad(unittest.TestCase):
    def test_loads_regulations(self):
        regs = _load_regulations()
        self.assertGreater(len(regs), 10)

    def test_regulation_fields(self):
        regs = _load_regulations()
        for reg in regs:
            self.assertIn("id", reg)
            self.assertIn("category", reg)
            self.assertIn("text", reg)


class TestKeywordSearch(unittest.TestCase):
    def test_school_zone(self):
        results = _keyword_search("학교 앞 어린이보호구역")
        categories = [r["category"] for r in results]
        self.assertIn("어린이보호구역", categories)

    def test_tunnel(self):
        results = _keyword_search("터널 내부 교통")
        categories = [r["category"] for r in results]
        self.assertIn("터널", categories)

    def test_construction(self):
        results = _keyword_search("공사구간")
        categories = [r["category"] for r in results]
        self.assertIn("공사구간", categories)

    def test_autonomous(self):
        results = _keyword_search("자율주행 시범구역")
        categories = [r["category"] for r in results]
        self.assertIn("자율주행시범구역", categories)

    def test_empty_query(self):
        results = _keyword_search("")
        self.assertEqual(len(results), 0)

    def test_top_k(self):
        results = _keyword_search("도로 교통", top_k=2)
        self.assertLessEqual(len(results), 2)


class TestEnrichPrompt(unittest.TestCase):
    def test_enriches_with_context(self):
        enriched = enrich_prompt("학교 앞 등하교 시간")
        self.assertIn("[참고 규정]", enriched)
        self.assertIn("학교 앞 등하교 시간", enriched)

    def test_no_match_returns_original(self):
        original = "xyzabc qqq zzz"
        # Force keyword-only search by passing invalid API key
        enriched = enrich_prompt(original, api_key="invalid")
        self.assertEqual(enriched, original)


class TestFormatContext(unittest.TestCase):
    def test_formats_regulations(self):
        regs = [{"law": "테스트법 제1조", "text": "테스트 규정입니다."}]
        result = format_context(regs)
        self.assertIn("[테스트법 제1조]", result)
        self.assertIn("테스트 규정입니다.", result)

    def test_empty_list(self):
        self.assertEqual(format_context([]), "")


if __name__ == "__main__":
    unittest.main()
